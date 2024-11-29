
from typing import Tuple, Dict, List , Any, Protocol
from torch import Tensor
import numpy as np
from sympy import O

from manabot.env.data import Observation, ObservationSpace

class Env(Protocol):
    """Protocol defining required game environment interface"""

    def step(self, actions: np.ndarray) -> Tuple[Observation, float, Dict]:
        """Perform an action and receive the next state.
        
        Args:
            actions: Array of action indices, shape (num_envs,)
        
        Returns:
            observation: Next observation
            reward: Reward for the action
            info: Info dict
        """
        ...


    def reset(self) -> Observation: 
        """Reset the environment and return the initial observation"""
        ...

    def close(self): ...
    

    @property
    def observation_space(self) -> ObservationSpace: ...

    @property
    def device(self) -> str: ...