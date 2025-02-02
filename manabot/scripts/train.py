#!/usr/bin/env python
"""
train.py
Entry point for a quick training run of ManaBot using PPO.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from manabot.infra import Experiment, Hypers, MatchHypers
from manabot.env import ObservationSpace, VectorEnv, Match, Reward
from manabot.ppo import Agent, Trainer
import manabot.infra.hypers

manabot.infra.hypers.initialize()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert each config group to a proper Python object.
    # (If needed, these will be proper dataclass instances.)
    obs_config = OmegaConf.to_object(cfg.observation)
    train_config = OmegaConf.to_object(cfg.train)
    reward_config = OmegaConf.to_object(cfg.reward)
    agent_config = OmegaConf.to_object(cfg.agent)
    experiment_config = OmegaConf.to_object(cfg.experiment)
    match_config = OmegaConf.to_object(cfg.match)
    # Create the top-level hypers instance, now with match as a dataclass.
    hypers = Hypers(
        observation=obs_config,
        match=match_config,
        train=train_config,
        reward=reward_config,
        agent=agent_config,
        experiment=experiment_config
    )
    
    # Setup components
    experiment = Experiment(hypers.experiment, hypers)
    observation_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    
    # Create environment and agent
    env = VectorEnv(hypers.train.num_envs, match, observation_space, reward, device=experiment.device)
    agent = Agent(observation_space, hypers.agent)
    
    # Train
    trainer = Trainer(agent, experiment, env, hypers.train)
    trainer.train()

if __name__ == "__main__":
    main()
