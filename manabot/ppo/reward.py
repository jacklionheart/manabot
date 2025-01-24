from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from manabot.data.data import Observation

class RewardPolicy(ABC):
    """Abstract base class for reward computation strategies."""
    
    @abstractmethod
    def compute(
        self,
        input_observation: Observation,
        action: int,
        output_observation: Observation,
        done: bool
    ) -> float:
        """Compute reward for a state transition.

        Args:
            old_obs: Observation before action
            action: Action taken
            new_obs: Observation after action
            done: Whether episode is complete

        Returns:
            float: Computed reward value
        """
        pass

class BasicRewardPolicy(RewardPolicy):
    """Basic reward policy focusing on game progression and winning.
    
    Rewards:
    - Small positive reward for advancing game state (playing cards, etc)
    - Medium reward for dealing damage
    - Large reward for winning
    - Negative rewards for losing or invalid actions
    """
    
    def __init__(
        self,
    
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.damage_scale = damage_scale
        self.progression_reward = progression_reward
    
    def compute(
        self,
        input_observation: Observation,
        action: int,
        output_observation: Observation,
        done: bool
    ) -> float:
        reward = 0.0
        
        # Terminal state rewards
        if done:
            # Check if we won by looking at life totals
            player_0_lost = output_observation.life_totals[0] <= 0
            player_1_lost = output_observation.life_totals[1] <= 0
            
            if player_1_lost and not player_0_lost:
                return self.win_reward
            elif player_0_lost:
                return self.lose_reward
        
        # Damage rewards
        damage_dealt = sum(
            input_observation.life_totals[i] - output_observation.life_totals[i]
            for i in range(2)
        )
        reward += damage_dealt * self.damage_scale
        
        # Game progression rewards
        if action != 0:  # If not passing
            reward += self.progression_reward
            
        return reward
    



    def to_tensors(self, card_embeddings: CardEmbeddings, num_permanents: int, num_actions: int, device: str = 'cuda') -> Dict[str, Tensor]:
        """Convert to tensor format with card embeddings and fixed sizes."""
        # Pad battlefield to fixed size
        padded_battlefield = self.battlefield + [ Permanent(card_id=0, owner_id=0, controller_id=0) for _ in range(num_permanents - len(self.battlefield)) ]
        
        permanent_tensor = torch.tensor([[
            # Embedding Key
            p.card_id,
            # Retained features
            p.owner_id,
            p.controller_id,
            int(p.tapped),
            int(p.attacking),
            int(p.blocking),
            int(p.summoning_sick),
            p.power,
            p.toughness,
            p.damage
        ] for p in padded_battlefield[:num_permanents]], device=device)

        # Embed card_ids for permanents
        permanent_tensor = card_embeddings.embed_field(permanent_tensor, field_idx=0)
    
        # Pad actions to fixed size
        padded_actions = self.action_options + [
            Action(source_id=0, target_id=0, valid=False)  # Pad with invalid action
            for _ in range(num_actions - len(self.action_options))
        ]
        action_tensor = torch.tensor([[
            a.source_id,
            a.target_id,
            int(a.valid)
        ] for a in padded_actions[:num_actions]], device=device)
        
        # First embed the source
        action_tensor = card_embeddings.embed_field(action_tensor, field_idx=0)  
        # Then embed target, which is now at index embedding_dim
        action_tensor = card_embeddings.embed_field(action_tensor, field_idx=card_embeddings.embedding_dim)
            
        # Core game state features
        game_tensor = torch.tensor([
            self.active_player,
            self.turn_number,
            int(self.step),
            *self.life_totals,
            int(self.action_type),
            self.lands_played_this_turn,
            int(self.game_over),
            int(self.won)
        ], device=device)

        return {
            'permanents': permanent_tensor,  # Shape: (num_permanents, embedding_dim + 9)
            'actions': action_tensor,        # Shape: (num_actions, 2*embedding_dim + 1)
            'game': game_tensor             # Shape: (10,)
        }

