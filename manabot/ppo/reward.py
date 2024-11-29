from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from manabot.env.data import Observation

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