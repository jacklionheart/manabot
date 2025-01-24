# manabot/env/vector_env.py

from typing import List, Tuple, Dict, Any
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor
import threading

from manabot.data import Observation
from .env import Env
from .config import PlayerConfig

class VectorEnv:
    """
    Vectorized version of the MTG environment that runs multiple managym.Env instances in parallel.
    """

    def __init__(self, envs: List[Env]):
        """
        Args:
            envs: Pre-created list of managym.Env instances. Each instance is a separate environment.
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs)
        self.lock = threading.Lock()

    def reset(
        self,
        player_configs_list: List[List[PlayerConfig]]
    ) -> List[Tuple[Observation, Dict[str, str]]]:
        """
        Reset each environment in parallel.

        Args:
            player_configs_list: For each environment, a list of PlayerConfigs
                                 (e.g., decklists for each player).
                                 So player_configs_list[i] is passed to envs[i].reset(...).

        Returns:
            A list of (Observation, info) tuples, one per environment.
        """
        assert len(player_configs_list) == self.num_envs, (
            "player_configs_list must have same length as number of envs."
        )

        with self.lock:
            results = list(self.executor.map(
                lambda idx_env: idx_env[0].reset(idx_env[1]),
                zip(self.envs, player_configs_list)
            ))
            # Each result is (Observation, info_dict)

        return results

    def step(
        self,
        actions: ndarray
    ) -> List[Tuple[Observation, float, bool, bool, Dict[str, str]]]:
        """
        Take a step in all environments in parallel.

        Args:
            actions: Array of actions with shape (num_envs,). Each action is an integer ID.

        Returns:
            For each environment, a tuple:
            (Observation, reward, terminated, truncated, info).
        """
        assert len(actions) == self.num_envs, (
            "Actions array length must match number of envs."
        )

        with self.lock:
            # step each environment with the corresponding action
            step_results = list(self.executor.map(
                lambda env_action: env_action[0].step(env_action[1]),
                zip(self.envs, actions)
            ))
            # Each step_result is (obs, reward, terminated, truncated, info)

        return step_results