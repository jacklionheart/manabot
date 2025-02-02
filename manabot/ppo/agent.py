"""
agent.py
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers

logger = logging.getLogger(__name__)


def layer_init(layer: nn.Module, gain: int = 1, bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    A PPO Agent that processes multiple components of a Magic: The Gathering
    game state (global, players, cards, permanents) and outputs:
      - Logits for each action slot (discrete)
      - Value estimate for the current state

    Action masking:
      - Expects obs['actions'] to have shape (batch, max_actions, action_dim).
      - The final dimension (or a specified one) indicates valid=1 / invalid=0.
      - We set invalid actions' logits to -∞ before sampling.
    """

    def __init__(
        self,
        # This is an unbatched observation space.
        observation_space: ObservationSpace,
        hypers: AgentHypers = AgentHypers(),
    ):
        super().__init__()
        self.hypers = hypers
        self.observation_space = observation_space

        enc = observation_space.encoder

        # --------------------
        # 1. Global MLP
        # --------------------
        self.global_net = nn.Sequential(
            layer_init(nn.Linear(enc.shapes["global"][0], hypers.game_embedding_dim)),
            nn.LayerNorm(hypers.game_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),

            layer_init(nn.Linear(hypers.game_embedding_dim, hypers.game_embedding_dim)),
            nn.LayerNorm(hypers.game_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),
        )

        # --------------------
        # 2. Players MLP
        # We'll pool over 2 players dimension -> mean or sum
        # --------------------
        self.player_net = nn.Sequential(
            layer_init(nn.Linear(enc.shapes["players"][1], hypers.game_embedding_dim)),
            nn.LayerNorm(hypers.game_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),

            layer_init(nn.Linear(hypers.game_embedding_dim, hypers.game_embedding_dim)),
            nn.LayerNorm(hypers.game_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),
        )

        # --------------------
        # 3. Cards MLP
        # We'll pool over max_cards dimension
        # --------------------
        self.card_net = nn.Sequential(
            layer_init(nn.Linear(enc.shapes["cards"][1], hypers.battlefield_embedding_dim)),
            nn.LayerNorm(hypers.battlefield_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),

            layer_init(nn.Linear(hypers.battlefield_embedding_dim, hypers.battlefield_embedding_dim)),
            nn.LayerNorm(hypers.battlefield_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),
        )

        # --------------------
        # 4. Permanents MLP
        # We'll pool over max_permanents dimension
        # --------------------
        self.permanent_net = nn.Sequential(
            layer_init(nn.Linear(enc.shapes["permanents"][1], hypers.battlefield_embedding_dim)),
            nn.LayerNorm(hypers.battlefield_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),

            layer_init(nn.Linear(hypers.battlefield_embedding_dim, hypers.battlefield_embedding_dim)),
            nn.LayerNorm(hypers.battlefield_embedding_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),
        )

        # Combined dimension after pooling these sub-embeddings
        combined_dim = (
            hypers.game_embedding_dim  # global
            + hypers.game_embedding_dim  # players
            + hypers.battlefield_embedding_dim  # cards
            + hypers.battlefield_embedding_dim  # permanents
        )

        # --------------------
        # Final feature net
        # --------------------
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(combined_dim, hypers.hidden_dim)),
            nn.LayerNorm(hypers.hidden_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),

            layer_init(nn.Linear(hypers.hidden_dim, hypers.hidden_dim)),
            nn.LayerNorm(hypers.hidden_dim),
            nn.ReLU(),
            nn.Dropout(hypers.dropout_rate),
        )

        # --------------------
        # Policy & Value heads
        # enc.max_actions is the typical discrete action space dimension
        # --------------------
        self.actor = layer_init(nn.Linear(hypers.hidden_dim, enc.max_actions))
        self.critic = layer_init(nn.Linear(hypers.hidden_dim, 1))

    def _pool_sequence(self, x: torch.Tensor, network: nn.Module) -> torch.Tensor:
        """
        x shape: (batch, seq_len, feats)
        Flatten into (batch*seq_len, feats), apply network, 
        reshape to (batch, seq_len, -1), then average over seq_len.
        """
        batch_size, seq_len, feat_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, feat_dim)
        emb = network(x_flat)
        emb = emb.reshape(batch_size, seq_len, -1)
        emb = emb.mean(dim=1)  # or .sum(dim=1) if desired
        return emb

    def _extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert the dictionary of sub-tensors to a combined feature vector.
        (batch, global_dim) -> global_net
        (batch, 2, player_dim) -> player_net -> average pool
        (batch, max_cards, card_dim) -> card_net -> average pool
        (batch, max_permanents, permanent_dim) -> permanent_net -> average pool
        """
        # global: shape (batch, gdim)
        global_emb = self.global_net(obs["global"])

        # players: shape (batch, 2, pdim)
        player_emb = self._pool_sequence(obs["players"], self.player_net)

        # cards: shape (batch, max_cards, cdim)
        card_emb = self._pool_sequence(obs["cards"], self.card_net)

        # permanents: shape (batch, max_permanents, cdim)
        perm_emb = self._pool_sequence(obs["permanents"], self.permanent_net)

        # Concatenate final embedding
        combined = torch.cat([global_emb, player_emb, card_emb, perm_emb], dim=-1)
        return combined

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (logits, value).
        - logits: shape (batch, max_actions)
        - value: shape (batch,)
        """
        features = self._extract_features(obs)
        features = self.feature_net(features)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience method for retrieving value estimates with no gradients.
        """
        _, value = self(obs)
        return value

    def get_action_and_value(
            self,
            obs: Dict[str, torch.Tensor],
            action: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Main method to get an action (sampled or deterministic) and the value estimate.
            Returns (action, logprob, entropy, value).

            Action Masking:
            - We expect obs['actions'] to have shape (batch, max_actions, action_dim).
            - The final dimension indicates valid=1 or invalid=0.
            - Invalid actions are masked with large negative values instead of -inf
                to avoid NaN issues with log probabilities.
            """
            logits, value = self(obs)

            # Get action mask and apply it
            action_mask = obs["actions"][..., -1].bool()
            masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
            
            # Use large negative value instead of -inf to avoid NaN in log probs
            masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))

            # Create distribution and handle deterministic vs sampling
            dist = Categorical(logits=masked_logits)
            
            if action is None:
                if deterministic:
                    action = torch.argmax(masked_logits, dim=-1)
                else:
                    action = dist.sample()

            logprob = dist.log_prob(action)
            entropy = dist.entropy()

            return action, logprob, entropy, value