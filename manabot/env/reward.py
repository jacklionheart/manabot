from dataclasses import dataclass

from manabot.infra.hypers import RewardHypers
import managym

class Reward:
    def __init__(self, hypers: RewardHypers):
        self.hypers = hypers

    def compute(self, cpp_reward : int, last_obs : managym.Observation, new_obs : managym.Observation) -> float:
        
        agent = last_obs.players[0].player_index

        if self.hypers.trivial:
            return 1.0

        else:
            if new_obs.game_over:
                if new_obs.won == agent:
                    return self.hypers.win_reward
                else:
                    return self.hypers.lose_reward
            else:
                return cpp_reward 
