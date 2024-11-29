from typing import List, Tuple, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from manabot.env import Observation, ObservationSpace
import threading
from torch import Tensor
import torch

from manabot.env import Env


class VectorEnv():
    """Vectorized version of the MTG environment that runs multiple instances in parallel"""
    
    def __init__(self, envs: List[Env]):
        """
        Args:
            num_envs: Number of environments to run in parallel
            env_factory: Function that creates a new environment instance
            use_threading: If True, uses threading for parallel execution
        """
        self.envs = envs
        self.executor = ThreadPoolExecutor(max_workers=len(self.envs))
        self.lock = threading.Lock()
        
    @property
    def device(self) -> str:
        """Device to use for tensor operations"""
        return self.envs[0].device
    
    @property
    def observation_space(self) -> ObservationSpace:
        """Observation space of the environment"""
        return self.envs[0].observation_space
    
    def start(self) -> List[Observation]:
        """Start all environments"""
        observations = list(self.executor.map(lambda env: env.start(), self.envs))
        return observations
    
    def step(self, actions: np.ndarray) -> List[Tuple[Observation, float, Dict]]:
        """Take a step in all environments.
        
        Args:
            actions: Array of action indices, shape (num_envs,)
        
        Returns:
            observations: Dict of observation tensors for all envs
            rewards: Reward tensor, shape (num_envs,)
            dones: Done flags tensor, shape (num_envs,)
            infos: List of info dicts, one per env
        """
        assert len(actions) == len(self.envs)

        with self.lock:
            results = list(self.executor.map(
                lambda env_action: env_action[0].step(env_action[1]),
                zip(self.envs, actions)
            ))
                
        return results


        # # Unpack results
        # observations = [result[0] for result in results]
        # rewards = [result[1] for result in results]
        # dones = [result[2] for result in results]
        # infos = [result[3] for result in results]
        
        # # Convert to tensors
        # batched_obs = {
        #     k: torch.stack([obs[k] for obs in observations]).to(self.device)
        #     for k in observations[0].keys()
        # }
        # reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # done_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # return batched_obs, reward_tensor, done_tensor, infos
    
    def close(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown()
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()