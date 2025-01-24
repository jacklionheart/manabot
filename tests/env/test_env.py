import unittest

from manabot.env import Env
from manabot.data import Observation
import managym  # The C++ module


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.player_a = managym.PlayerConfig("Alice", {"Mountain": 20, "Grey Ogre": 40})
        self.player_b = managym.PlayerConfig("Bob", {"Forest": 20, "Llanowar Elves": 40})

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

    def test_game_completes(self):
        """
        Test that a game can run to completion and properly signals termination.
        
        This test verifies that:
        1. The environment can be reset and begin a game
        2. Actions can be taken successfully
        3. The game eventually reaches a terminal state
        4. Termination is properly signaled via done flag
        5. Final observation contains appropriate end-game information
        """
        
        # Reset and verify initial state
        obs, info = self.env.reset([self.player_a, self.player_b])
        self.assertFalse(obs.game_over)  # Game shouldn't be over at start
        self.assertIsNotNone(obs.action_space)  # Should have valid actions
        
        # Run for a maximum number of steps
        max_steps = 1000  # Generous limit to ensure game can complete
        steps_taken = 0
        game_completed = False
        
        while steps_taken < max_steps:
            # Always take the first available action (index 0)
            # This is sufficient for testing as the environment enforces valid actions
            next_obs, reward, terminated, truncated, info = self.env.step(0)
            steps_taken += 1
            
            # Check observation validity
            self.assertTrue(next_obs.validate())
            
            # If game is over, verify terminal state
            if terminated:
                game_completed = True
                self.assertTrue(next_obs.game_over)  # Game state should match termination
                self.assertIsNotNone(next_obs.won)   # Win/loss should be determined
                break
                
            # Even if not terminated, verify basic observation properties
            self.assertGreaterEqual(next_obs.turn.turn_number, 0)
            self.assertIsNotNone(next_obs.action_space)
        
        # Verify game completed successfully
        self.assertTrue(game_completed, 
                       f"Game did not complete within {max_steps} steps")
        self.assertLess(steps_taken, max_steps, 
                       "Game took maximum number of steps, might indicate infinite loop")
        
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
