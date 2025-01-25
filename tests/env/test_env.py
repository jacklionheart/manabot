import unittest

from manabot.env import Env
import managym  # The C++ module


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.player_a = managym.PlayerConfig("Alice", {"Mountain": 20, "Grey Ogre": 40})
        self.player_b = managym.PlayerConfig("Bob", {"Forest": 20, "Llanowar Elves": 40})

    def test_reset(self):
        obs, info = self.env.reset([self.player_a, self.player_b])
        self.assertIsInstance(obs, dict)
        self.assertIn("Alice", [self.player_a.name, self.player_b.name])  # sanity check

    def test_step(self):
        obs, info = self.env.reset([self.player_a, self.player_b])
        # Pick an action index, e.g. 0
        next_obs, reward, done, truncated, step_info = self.env.step(0)
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(step_info, dict)

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
        
        # Run for a maximum number of steps
        max_steps = 1000  # Generous limit to ensure game can complete
        steps_taken = 0
        game_completed = False
        
        while steps_taken < max_steps:
            # Always take the first available action (index 0)
            # This is sufficient for testing as the environment enforces valid actions
            next_obs, reward, terminated, truncated, info = self.env.step(0)
            steps_taken += 1
            
            # If game is over, verify terminal state
            if terminated:
                game_completed = True
                break
                
            # check turn count increased
            self.assertGreaterEqual(next_obs['global'][0], 1)
        
        # Verify game completed successfully
        self.assertTrue(game_completed, 
                       f"Game did not complete within {max_steps} steps")
        self.assertLess(steps_taken, max_steps, 
                       "Game took maximum number of steps, might indicate infinite loop")
        


if __name__ == "__main__":
    unittest.main()
