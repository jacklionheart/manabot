"""
agent.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from manabot.env import ObservationSpace
from manabot.infra import AgentHypers

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
        self.obs_space = observation_space
        self.hypers = hypers

        # Extract dimensions from observation embedding configuration.
        enc = observation_space.encoder
        player_dim = enc.player_dim
        card_dim = enc.card_dim
        perm_dim = enc.permanent_dim
        action_dim = enc.action_dim
        self.max_focus_objects = enc.max_focus_objects
        max_actions = enc.max_actions
        embed_dim = hypers.hidden_dim
        num_heads = hypers.num_attention_heads

        # Typed object embeddings.
        self.player_embedding = ProjectionLayer(player_dim, embed_dim)
        self.card_embedding = ProjectionLayer(card_dim, embed_dim)
        self.perm_embedding = ProjectionLayer(perm_dim, embed_dim)
        self.action_embedding = ProjectionLayer(action_dim, embed_dim)
            
        
        # Global game state processor.
        self.attention = GameObjectAttention(embed_dim, num_heads=num_heads)
        
        # Action processing.
        actions_with_focus_dim = (self.max_focus_objects + 1) * embed_dim
        self.action_layer = ProjectionLayer(actions_with_focus_dim, embed_dim)

        # Policy and value heads.
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, max_actions))
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            MaxPoolingLayer(dim=1),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1))
        )
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gather object embeddings, controller IDs, and validity mask.
        objects, is_agent, validity = self._gather_object_embeddings(obs)
        agent_id = obs["agent_player"][:, 0].long()  # [B]
        # Create a key_padding_mask: True for invalid objects.
        key_padding_mask = (validity == 0)  # [B, total_objs]
        post_attention_objects = self.attention(objects, is_agent, agent_id, key_padding_mask=key_padding_mask)
        
        informed_actions = self._gather_informed_actions(obs, post_attention_objects)
        
        # Policy branch: use informed action embeddings.
        logits = self.policy_head(informed_actions)  # [B, max_actions]
        # Value branch: aggregate global state via max pooling.
        value = self.value_head(post_attention_objects).squeeze(-1)  # [B]
        return logits, value
    
    def _add_focus(self, 
                   actions: torch.Tensor, 
                   post_attention_objects: torch.Tensor, 
                   action_focus: torch.Tensor) -> torch.Tensor:
        """
        Incorporates focus object embeddings into the action embeddings by concatenating the
        original action embedding with the flattened focus embeddings.
        
        Args:
            actions: Tensor of shape [B, max_actions, embed_dim]
            post_attention_objects: Tensor of shape [B, num_objects, embed_dim]
            action_focus: Tensor of shape [max_actions, max_focus_objects] or 
                          [B, max_actions, max_focus_objects] containing indices into post_attention_objects.
        
        Returns:
            Tensor of shape [B, max_actions, (1 + max_focus_objects) * embed_dim],
            where for each action the vector is:
                (action_embedding, focus1_embedding, focus2_embedding, ..., focusN_embedding)
            If any focus index is -1, its embedding is replaced with a zero vector.
        """
        B, max_actions, embed_dim = actions.shape

        # Create a mask: valid (1.0) where index != -1, 0.0 otherwise.
        valid_mask = (action_focus != -1).unsqueeze(-1).float()  # [B, max_actions, max_focus_objects, 1]

        # Expand action_focus for torch.gather.
        # post_attention_objects is of shape [B, num_objects, embed_dim]
        # We want to gather along dimension 1, so expand action_focus to have a trailing embed_dim.
        
        focus_indices = action_focus.unsqueeze(-1).expand(B, max_actions, self.max_focus_objects, embed_dim) # [B, max_actions, max_focus_objects, embed_dim]
        
        # Gather focus embeddings from post_attention_objects using the indices.
        # First expand into action dimenson.
        post_attention_focus_objects = post_attention_objects.unsqueeze(1).expand(-1, max_actions, -1, -1) # [B, max_actions, num_objects, embed_dim]
        focus_embeds = torch.gather(post_attention_focus_objects, 2, focus_indices)  # [B, max_actions, max_focus_objects, embed_dim]
        
        # Zero out embeddings where the focus index was invalid.
        focus_embeds = focus_embeds * valid_mask

        # Concatenate the original action embedding with the flattened focus embeddings.
        focus_flat = focus_embeds.view(B, max_actions, -1) # B, max_actions, max_focus_objects * embed_dim]
        actions_with_focus = torch.cat([actions, focus_flat], dim=-1) # B, max_actions, (1 + max_focus_objects) * embed_dim
        return actions_with_focus
    
    def _gather_informed_actions(self, obs: Dict[str, torch.Tensor], post_attention_objects: torch.Tensor) -> torch.Tensor:
        """
        Gathers and embeds actions from the observation.
        Applies validity masking to the action embeddings.
        """
        # Extract action features (ignoring the last validity bit).
        actions = self.action_embedder(obs["actions"][:, :, :-1])  # [B, max_actions, embed_dim]
        valid_mask = obs["actions_valid"].unsqueeze(-1)  # [B, max_actions, 1]
        actions = actions * valid_mask
        actions_with_focus = self._add_focus(actions, post_attention_objects, obs["action_focus"])  # [B, max_actions, (2*n+1)*embed_dim]
        return self.action_layer(actions_with_focus)  # [B, max_actions, embed_dim]
    
    def _gather_object_embeddings(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gathers and encodes game objects from the observation and assembles their validity masks.
        Assumes all inputs are batched.
        
        Order:
        1) agent_player (ctrl=0)
        2) opponent_player (ctrl=1)
        3) agent_cards (0)
        4) opponent_cards (1)
        5) agent_permanents (0)
        6) opponent_permanents (1)
        
        Returns:
            objects: [B, total_objs, embed_dim]
            controller_ids: [B, total_objs] (0 for agent, 1 for opponent)
            validity: [B, total_objs] (1 for valid, 0 for invalid)
        """
        # We assume the following keys in obs are already batched:
        # "agent_player": [B, N_agent, player_dim]
        # "agent_player_id": [B, 1]
        # "agent_player_valid": [B, 1]
        # and similarly for opponent_player, agent_cards, opponent_cards, agent_permanents, opponent_permanents.

        # Encode (objects, controlled_by_agent, valid) for each object. 
        # Always concat in this order across all 3 tensors:
        # 1) agent_player
        # 2) opponent_player
        # 3) agent_cards
        # 4) opponent_cards
        # 5) agent_permanents
        # 6) opponent_permanents
        
        # Concatenate all object embeddings along the slot dimension.
        # Encode each chunk using the corresponding projection layers.
        enc_agent_player = self.player_embedding(obs["agent_player"])   # [B, N_agent, embed_dim]
        enc_opp_player = self.player_embedding(obs["opponent_player"])         # [B, N_opp, embed_dim]
        enc_agent_cards = self.card_embedding(obs["agent_cards"])         # [B, C_agent, embed_dim]
        enc_opp_cards = self.card_embedding(obs["opponent_cards"])             # [B, C_opp, embed_dim]
        enc_agent_perms = self.perm_embedding(obs["agent_permanents"])         # [B, P_agent, embed_dim]
        enc_opp_perms = self.perm_embedding(obs["opponent_permanents"])             # [B, P_opp, embed_dim]
        objects = torch.cat([
            enc_agent_player, enc_opp_player,
            enc_agent_cards, enc_opp_cards,
            enc_agent_perms, enc_opp_perms
        ], dim=1)  # [B, total_objs, embed_dim]

        # Build a boolean mask for ownership:
        # For agent objects, set True; for opponent objects, set False.
        agent_player_is_agent = torch.ones_like(enc_agent_player, dtype=torch.bool)    # [B, 1]
        opponent_player_is_agent = torch.zeros_like(enc_opp_player, dtype=torch.bool)         # [B, 1]
        agent_cards_is_agent = torch.ones_like(enc_agent_cards, dtype=torch.bool)       # [B, C_agent]
        opp_cards_is_agent = torch.zeros_like(enc_opp_cards, dtype=torch.bool)          # [B, C_opp]
        agent_perms_is_agent = torch.ones_like(enc_agent_perms, dtype=torch.bool)       # [B, P_agent]
        opp_perms_is_agent = torch.zeros_like(enc_opp_perms, dtype=torch.bool)          # [B, P_opp]
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
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

# -----------------------------------------------------------------------------
# Basic Building Blocks
# -----------------------------------------------------------------------------
class MaxPoolingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = torch.max(x, dim=self.dim)
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
      
    Now accepts a key_padding_mask to ignore invalid object slots.
    """
    def __init__(self, embedding_dim: int, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.perspective = nn.Parameter(torch.randn(embedding_dim) / embedding_dim**0.5)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, num_heads * embedding_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(num_heads * embedding_dim, embedding_dim))
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, objects: torch.Tensor, is_agent: torch.Tensor,
                agent_id: torch.Tensor, key_padding_mask: torch.BoolTensor) -> torch.Tensor:
        # objects: [B, total_objs, embedding_dim]
        # is_agent: [B, total_objs]
        # agent_id: [B]
        perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)  # [B, total_objs, 1]
        # Add ownership information to the objects.is the b
        owned_objects = objects + perspective_scale * self.perspective
        attn_out, _ = self.mha(owned_objects, owned_objects, owned_objects, key_padding_mask=key_padding_mask)
        x = self.norm1(owned_objects + attn_out) # [B, total_objs, num_heads * embedding_dim]
        mlp_out = self.mlp(x) # [B, total_objs, embedding_dim]
        return self.norm2(x + mlp_out)  # [B, total_objs, embedding_dim]

def layer_init(layer: nn.Module, gain: int = 1, bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer