import managym
from manabot.data import ObservationSpace

class Env:
    """
    A Pythonic Env wrapper around the C++ managym.Env object.
    Converts all inputs/outputs to/from manabot.env.data classes.
    """

    def __init__(self, skip_trivial: bool = False):
        self._cpp_env = managym.Env(skip_trivial)
        self.observation_space = ObservationSpace()

    def reset(self, player_configs):
        """
        Example usage:
           player_a = managym.PlayerConfig("Alice", {"Mountain": 40})
           player_b = managym.PlayerConfig("Bob", {"Forest": 40})
           obs = env.reset([player_a, player_b])
        Returns a manabot.env.data.Observation
        """
        cpp_obs, cpp_info = self._cpp_env.reset(player_configs)
        py_obs = self.observation_space.encoder.encode(cpp_obs)
        # info remains a dict of string->string, we can just pass it up
        return py_obs, cpp_info

    def step(self, action: int):
        """Apply an action (int) and return (Observation, reward, done, info)."""
        cpp_obs, reward, terminated, truncated, info = self._cpp_env.step(action)
        py_obs = self.observation_space.encoder.encode(cpp_obs)
        return py_obs, reward, terminated, truncated, info

