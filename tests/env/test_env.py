import unittest

from manabot.env import Env
from manabot.data import Observation
import managym  # The C++ module


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = Env(skip_trivial=False)
        self.player_a = managym.PlayerConfig("Alice", {"Mountain": 40})
        self.player_b = managym.PlayerConfig("Bob", {"Forest": 40})

    def test_reset(self):
        obs, info = self.env.reset([self.player_a, self.player_b])
        self.assertIsInstance(obs, Observation)
        self.assertFalse(obs.game_over)
        self.assertIn("Alice", [self.player_a.name, self.player_b.name])  # sanity check

    def test_step(self):
        obs, info = self.env.reset([self.player_a, self.player_b])
        # Pick an action index, e.g. 0
        next_obs, reward, done, truncated, step_info = self.env.step(0)
        self.assertIsInstance(next_obs, Observation)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(step_info, dict)

        # Possibly check that we haven't broken any field conversions
        self.assertTrue(next_obs.validate(), "Pythonic observation should pass validation")

    # If you also test your from_managym_observation logic directly:
    def test_observation_conversion(self):
        cpp_env = managym.Env()
        cpp_obs, cpp_info = cpp_env.reset([self.player_a, self.player_b])
        py_obs = Observation(cpp_obs)
        
        # Check that fields match
        self.assertEqual(py_obs.game_over, cpp_obs.game_over)
        self.assertEqual(py_obs.turn.turn_number, cpp_obs.turn.turn_number)
        # etc.

if __name__ == "__main__":
    unittest.main()
