"""
agent.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from manabot.env import ObservationSpace
from manabot.infra import getLogger, AgentHypers
class Agent(nn.Module):
    """
    Agent that:
      1. Encodes game objects with typed embeddings.
      2. Allows objects to exchange information using GameObjectAttention.
      3. Adds relevant object information to the actions.
      4. Computes policy logits and a value estimate.
    """
    def __init__(self, observation_space: ObservationSpace, hypers: AgentHypers):
        super().__init__()
        self.observation_space = observation_space
        self.hypers = hypers
        self.logger = getLogger(__name__)

        # Extract dimensions from observation embedding configuration.
        enc = observation_space.encoder
        player_dim = enc.player_dim
        card_dim = enc.card_dim
        perm_dim = enc.permanent_dim
        action_dim = enc.action_dim  # this is 6 (5 action types + validity flag)
        self.max_focus_objects = enc.max_focus_objects
        embed_dim = hypers.hidden_dim

        # Typed object embeddings.
        self.player_embedding = ProjectionLayer(player_dim, embed_dim)
        self.card_embedding = ProjectionLayer(card_dim, embed_dim)
        self.perm_embedding = ProjectionLayer(perm_dim, embed_dim)
        # CHANGE: Build action_embedding to use only the first (action_dim - 1) features
        self.action_embedding = ProjectionLayer(action_dim - 1, embed_dim)
        self.logger.info(f"Player embedding: ({player_dim} -> {embed_dim})")
        self.logger.info(f"Card embedding: ({card_dim} -> {embed_dim})")
        self.logger.info(f"Perm embedding: ({perm_dim} -> {embed_dim})")
        self.logger.info(f"Action embedding: ({action_dim - 1} -> {embed_dim})")
        
        # Currently not using attention
        # Global game state processor.
        # num_heads = hypers.num_attention_heads
        #self.attention = GameObjectAttention(embed_dim, num_heads=num_heads)
        # logger.info(f"Attention: {embed_dim} -> {embed_dim}")
        
        # Action processing.
        actions_with_focus_dim = (self.max_focus_objects + 1) * embed_dim
        self.action_layer = ProjectionLayer(actions_with_focus_dim, embed_dim)
        self.logger.info(f"Action layer: ({actions_with_focus_dim} -> {embed_dim})")
        # Policy and value heads.
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1))
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            MaxPoolingLayer(dim=1),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1))
        )
        self.logger.info(f"Policy head: ({embed_dim} -> {1})")
        self.logger.info(f"Value head: ({embed_dim} -> 1)")
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        log = self.logger.getChild("forward")

        # Gather object embeddings, controller IDs, and validity mask.
        objects, is_agent, validity = self._gather_object_embeddings(obs)
        log.debug(f"Objects: {objects.shape}")
        log.debug(f"Is agent: {is_agent.shape}")
        log.debug(f"Validity: {validity.shape}")

        key_padding_mask = (validity == 0)  # [B, total_objs]
        log.debug(f"Key padding mask: {key_padding_mask.shape}")
        # post_attention_objects = self.attention(objects, is_agent, key_padding_mask=key_padding_mask)
        post_attention_objects = objects
        log.debug(f"Post attention objects: {post_attention_objects.shape}")

        informed_actions = self._gather_informed_actions(obs, post_attention_objects)
        log.debug(f"Informed actions: {informed_actions.shape}")

        # Policy branch: use informed action embeddings.
        logits = self.policy_head(informed_actions).squeeze(-1)  # [B, max_actions]
        # APPLY MASKING HERE so that logits for invalid actions are set to -1e8.
        logits = logits.masked_fill(obs["actions_valid"] == 0, -1e8)
        log.debug(f"Logits: {logits.shape}")
        # Value branch: aggregate global state via max pooling.
        value = self.value_head(post_attention_objects).squeeze(-1)  # [B]
        log.debug(f"Value: {value.shape}")
        return logits, value
    
    def _add_focus(self, 
                   actions: torch.Tensor, 
                   post_attention_objects: torch.Tensor, 
                   action_focus: torch.Tensor) -> torch.Tensor:
        """
        Incorporates focus object embeddings into the action embeddings by concatenating the
        original action embedding with the flattened focus embeddings.
        """
        log = self.logger.getChild("add_focus")

        B, max_actions, embed_dim = actions.shape
        log.debug(f"B: {B}")
        log.debug(f"max_actions: {max_actions}")
        log.debug(f"embed_dim: {embed_dim}")
        log.debug(f"post_attention_objects: {post_attention_objects.shape}")
        log.debug(f"action_focus: {action_focus.shape}")
        # Create a mask: valid (1.0) where index != -1, 0.0 otherwise.
        valid_mask = (action_focus != -1).unsqueeze(-1).float()  # [B, max_actions, max_focus_objects, 1]

        # Replace -1 with 0 to enable gathering.
        action_focus = action_focus.clone()  # avoid in-place modification
        action_focus[action_focus == -1] = 0
        
        log.debug(f"Valid mask: {valid_mask.shape}")

        # Expand action_focus for torch.gather.
        focus_indices = action_focus.unsqueeze(-1).expand(B, max_actions, self.max_focus_objects, embed_dim).long()  # [B, max_actions, max_focus_objects, embed_dim]
        log.debug(f"focus_indices: {focus_indices.shape}")

        post_attention_focus_objects = post_attention_objects.unsqueeze(1).expand(-1, max_actions, -1, -1)  # [B, max_actions, num_objects, embed_dim]
        focus_embeds = torch.gather(post_attention_focus_objects, 2, focus_indices)  # [B, max_actions, max_focus_objects, embed_dim]
        log.debug(f"focus_embeds: {focus_embeds.shape}")

        # Zero out embeddings where the focus index was invalid.
        focus_embeds = focus_embeds * valid_mask
        log.debug(f"valid_focus_embeds: {focus_embeds.shape}")

        # Concatenate the original action embedding with the flattened focus embeddings.
        focus_flat = focus_embeds.view(B, max_actions, -1)  # [B, max_actions, max_focus_objects * embed_dim]
        log.debug(f"focus_flat: {focus_flat.shape}")
        actions_with_focus = torch.cat([actions, focus_flat], dim=-1)  # [B, max_actions, (1 + max_focus_objects) * embed_dim]
        log.debug(f"actions_with_focus: {actions_with_focus.shape}")
        return actions_with_focus
    
    def _gather_informed_actions(self, obs: Dict[str, torch.Tensor], post_attention_objects: torch.Tensor) -> torch.Tensor:
        """
        Gathers and embeds actions from the observation.
        Applies validity masking to the action embeddings.
        """
        log = self.logger.getChild("gather_informed_actions")
        log.debug(f"Obs: {obs.keys()}")
        log.debug(f"Obs['actions']: {obs['actions'].shape}")
        # CHANGE: Use only the first (action_dim - 1) features (i.e. drop the validity flag)
        actions = self.action_embedding(obs["actions"][..., :-1])  # [B, max_actions, embed_dim]
        log.debug(f"Actions: {actions.shape}")
        valid_mask = obs["actions_valid"].unsqueeze(-1)  # [B, max_actions, 1]
        log.debug(f"Valid mask: {valid_mask.shape}")
        actions = actions * valid_mask
        log.debug(f"Actions after valid mask: {actions.shape}")
        actions_with_focus = self._add_focus(actions, post_attention_objects, obs["action_focus"])  # [B, max_actions, (1 + max_focus_objects)*embed_dim]
        log.debug(f"Actions with focus: {actions_with_focus.shape}")
        return self.action_layer(actions_with_focus)  # [B, max_actions, embed_dim]
    
    def _gather_object_embeddings(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gathers and encodes game objects from the observation and assembles their validity masks.
        Order: agent_player, opponent_player, agent_cards, opponent_cards, agent_permanents, opponent_permanents.
        """
        log = self.logger.getChild("gather_object_embeddings")
        device = obs["agent_player"].device
        log.debug(f"Device: {device}")
        log.debug(f"Obs['agent_player']: {obs['agent_player'].shape}")
        log.debug(f"Obs['opponent_player']: {obs['opponent_player'].shape}")
        log.debug(f"Obs['agent_cards']: {obs['agent_cards'].shape}")
        log.debug(f"Obs['opponent_cards']: {obs['opponent_cards'].shape}")
        
        enc_agent_player = self.player_embedding(obs["agent_player"])   # [B, N_agent, embed_dim]
        enc_opp_player = self.player_embedding(obs["opponent_player"])    # [B, N_opp, embed_dim]
        enc_agent_cards = self.card_embedding(obs["agent_cards"])         # [B, C_agent, embed_dim]
        enc_opp_cards = self.card_embedding(obs["opponent_cards"])          # [B, C_opp, embed_dim]
        enc_agent_perms = self.perm_embedding(obs["agent_permanents"])      # [B, P_agent, embed_dim]
        enc_opp_perms = self.perm_embedding(obs["opponent_permanents"])       # [B, P_opp, embed_dim]
        objects = torch.cat([
            enc_agent_player, enc_opp_player,
            enc_agent_cards, enc_opp_cards,
            enc_agent_perms, enc_opp_perms
        ], dim=1)  # [B, total_objs, embed_dim]


        agent_player_is_agent = torch.ones(enc_agent_player.shape[0], enc_agent_player.shape[1], dtype=torch.bool, device=device)
        opponent_player_is_agent = torch.zeros(enc_opp_player.shape[0], enc_opp_player.shape[1], dtype=torch.bool, device=device)
        agent_cards_is_agent = torch.ones(enc_agent_cards.shape[0], enc_agent_cards.shape[1], dtype=torch.bool, device=device)
        opp_cards_is_agent = torch.zeros(enc_opp_cards.shape[0], enc_opp_cards.shape[1], dtype=torch.bool, device=device)
        agent_perms_is_agent = torch.ones(enc_agent_perms.shape[0], enc_agent_perms.shape[1], dtype=torch.bool, device=device)
        opp_perms_is_agent = torch.zeros(enc_opp_perms.shape[0], enc_opp_perms.shape[1], dtype=torch.bool, device=device)
        is_agent = torch.cat([ 
            agent_player_is_agent,
            opponent_player_is_agent,
            agent_cards_is_agent,
            opp_cards_is_agent,
            agent_perms_is_agent,
            opp_perms_is_agent
        ], dim=1)  # [B, total_objs]

        validity = torch.cat([
            obs["agent_player_valid"], obs["opponent_player_valid"],
            obs["agent_cards_valid"], obs["opponent_cards_valid"],
            obs["agent_permanents_valid"], obs["opponent_permanents_valid"]
        ], dim=1)  # [B, total_objs]
        
        log.debug(f"Objects: {objects.shape}")
        log.debug(f"Is agent: {is_agent.shape}")
        log.debug(f"Validity: {validity.shape}")
        
        return objects, is_agent, validity
    
    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(obs)
            return value
    
    def get_action_and_value(self,
                             obs: Dict[str, torch.Tensor],
                             action: Optional[torch.Tensor] = None,
                             deterministic: bool = False
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log = self.logger.getChild("get_action_and_value")
        logits, value = self.forward(obs)
        if (obs["actions_valid"].sum(dim=-1) == 0).any():
            raise ValueError("No valid actions available")
        # NOTE: Masking has already been applied in forward()
        # Compute probabilities and create distribution
        probs = torch.softmax(logits, dim=-1)
        log.debug(f"Probs: {probs.shape}")
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log.debug(f"Action: {action.shape}")
        log.debug(f"Log prob: {dist.log_prob(action).shape}")
        log.debug(f"Entropy: {dist.entropy().shape}")
        return action, dist.log_prob(action), dist.entropy(), value

# -----------------------------------------------------------------------------
# Basic Building Blocks
# -----------------------------------------------------------------------------
class MaxPoolingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log = getLogger(__name__).getChild("forward")
        log.debug(f"X: {x.shape}")
        pooled, _ = torch.max(x, dim=self.dim)
        log.debug(f"Pooled: {pooled.shape}")
        return pooled


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            layer_init(nn.Linear(input_dim, output_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(output_dim, output_dim))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log = getLogger(__name__).getChild("forward")
        log.debug(f"X: {x.shape}")
        return self.projection(x)

# -----------------------------------------------------------------------------
# Game State Processor with Learned Perspective
# -----------------------------------------------------------------------------
class GameObjectAttention(nn.Module):
    """
    Processes the complete game state by:
      - Adding a learned perspective vector (with sign flip based on controller),
      - Applying multi-head attention,
      - And returning the context-rich output.
    """
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logger = getLogger(__name__).getChild("attention")
        self.logger.info(f"Making perspective vector of size {embedding_dim}")
        self.perspective = nn.Parameter(torch.randn(embedding_dim) / embedding_dim**0.5)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, num_heads * embedding_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(num_heads * embedding_dim, embedding_dim))
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, objects: torch.Tensor, is_agent: torch.Tensor, key_padding_mask: torch.BoolTensor) -> torch.Tensor:
        # objects: [B, total_objs, embedding_dim]
        # is_agent: [B, total_objs]
        log = self.logger.getChild("forward")
        log.debug(f"Objects: {objects.shape}")
        log.debug(f"Is agent: {is_agent.shape}")
        log.debug(f"Key padding mask: {key_padding_mask.shape}")
        
        perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)  # [B, total_objs, 1]
        log.debug(f"Perspective: {self.perspective.shape}")
        owned_objects = objects + perspective_scale * self.perspective
        log.debug(f"Owned objects: {owned_objects.shape}")

        attn_out, _ = self.mha(owned_objects, owned_objects, owned_objects, key_padding_mask=key_padding_mask)
        log.debug(f"Attn out: {attn_out.shape}")
        
        x = self.norm1(owned_objects + attn_out) # [B, total_objs, embedding_dim]
        log.debug(f"X: {x.shape}")

        mlp_out = self.mlp(x) # [B, total_objs, embedding_dim]
        log.debug(f"MLP out: {mlp_out.shape}")

        post_norm = self.norm2(x + mlp_out)  # [B, total_objs, embedding_dim]   
        log.debug(f"Post norm: {post_norm.shape}")
        # ZERO OUT outputs for masked (invalid) positions.
        mask = (~key_padding_mask).unsqueeze(-1).float()  # valid positions: 1.0; invalid: 0.0
        post_norm = post_norm * mask
        log.debug(f"Post norm after masking: {post_norm.shape}")
        return post_norm

def layer_init(layer: nn.Module, gain: int = 1, bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
