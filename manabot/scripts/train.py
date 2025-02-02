#!/usr/bin/env python
"""
train.py
Entry point for a quick training run of ManaBot using PPO.
This script sets up the experiment, environment, agent, and trainer using the Hydra configuration,
overriding the training hyperparameters to use a small (CPU) quick-run setup.

Quick Training Config:
    - num_envs: 4
    - num_steps: 128
    - total_timesteps: 100000
    - batch_size: 512
    - learning_rate: 2.5e-4

This script uses the Trainer object directly (without subclassing).
"""

import hydra
from omegaconf import OmegaConf
from manabot.infra import Experiment, Hypers
from manabot.env.observation import ObservationSpace
from manabot.env.env import VectorEnv  # Note: VectorEnv is defined in manabot/env/env.py
from manabot.env.match import Match
from manabot.env.reward import Reward
from manabot.ppo.agent import Agent
from manabot.ppo.trainer import Trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Hypers) -> None:
    # --- Quick Training Configuration Overrides ---
    quick_train_config = {
        'num_envs': 4,        # Small enough for CPU
        'num_steps': 128,     # Standard PPO window
        'total_timesteps': 100000,  # ~1hr on MacBook
        'batch_size': 512,
        'learning_rate': 2.5e-4
    }
    cfg.train.num_envs = quick_train_config['num_envs']
    cfg.train.num_steps = quick_train_config['num_steps']
    cfg.train.total_timesteps = quick_train_config['total_timesteps']
    cfg.train.learning_rate = quick_train_config['learning_rate']
    
    # --- Experiment Setup ---
    experiment = Experiment(cfg.experiment)
    run_name, writer = experiment.setup()
    print(f"Starting training run: {run_name}")
    
    # --- Create Environment Components ---
    # Observation space from config (uses ObservationSpaceHypers)
    observation_space = ObservationSpace(cfg.observation)
    # Create the match configuration (the Match object accepts a MatchHypers)
    match = Match(cfg.match)
    # Create the reward computation object
    reward = Reward(cfg.reward)
    
    # --- Create Vectorized Environment ---
    env = VectorEnv(cfg.train.num_envs, match, observation_space, reward, device=experiment.device)
    
    # --- Create Agent ---
    agent = Agent(observation_space, cfg.agent)
    
    # --- Create Trainer ---
    trainer = Trainer(agent, experiment, env, cfg.train)
        
    # --- Start Training ---
    trainer.train()


if __name__ == "__main__":
    main()
