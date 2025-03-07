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

