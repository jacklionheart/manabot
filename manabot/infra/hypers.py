"""
hypers.py
Centralized hyperparameter configuration using Hydra.

This module defines a unified configuration system for all hyperparameters across the codebase,
organizing them into logical groups while maintaining type safety and easy CLI/YAML override support.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from omegaconf import MISSING

# Import existing hypers
from hydra.core.config_store import ConfigStore

@dataclass
class ObservationSpaceHypers:
    max_cards: int = 100
    max_permanents: int = 50
    max_actions: int = 20
    max_focus_objects: int = 2

@dataclass
class MatchHypers:
    """Parameters previously passed to the Match object."""
    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: dict = field(default_factory=lambda: {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    })
    villain_deck: dict = field(default_factory=lambda: {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    })

@dataclass
class ExperimentHypers:
    """Configuration for experiment tracking and environment setup."""
    exp_name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    track: bool = False
    wandb_project_name: str = "manabot"
    wandb_entity: Optional[str] = None

@dataclass
class AgentHypers:
    """
    Agent hyperparameters for network architecture.
    """
    # Dimensions for sub-embeddings
    game_embedding_dim: int = 8
    battlefield_embedding_dim: int = 8
    hidden_dim: int = 64

    # Dropout for more stable training
    dropout_rate: float = 0.1

@dataclass
class TrainHypers:
    """
    Training-related hyperparameters, e.g. total_timesteps, learning_rate, etc.
    """
    total_timesteps: int = 20_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    @property
    def batch_size(self) -> int:
        """Compute batch size from env and steps configuration."""
        return self.num_envs * self.num_steps
    
    @property
    def minibatch_size(self) -> int:
        """Compute minibatch size from batch size and number of minibatches."""
        return self.batch_size // self.num_minibatches

@dataclass
class RewardHypers:
    trivial: bool = False # If true, always return reward=1
    win_reward: float = 100.0
    lose_reward: float = -100.0



@dataclass
class Hypers:
    """
    Top-level configuration that composes all hyperparameters.
    
    Each field represents a major component of the system, with its own set of 
    hyperparameters. This structure allows for easy organization and override
    through either YAML configs or command line arguments.
    """
    defaults: list = field(default_factory=lambda: [
        "_self_",
        {"observation": "default"},
        {"match": "default"},
        {"train": "default"},
        {"reward": "default"},
        {"agent": "default"},
        {"experiment": "default"}
    ])
    
    # Main configuration groups - using composition over inheritance
    observation: ObservationSpaceHypers = field(default_factory=ObservationSpaceHypers)
    match: MatchHypers = field(default_factory=MatchHypers)
    train: TrainHypers = field(default_factory=TrainHypers)
    reward: RewardHypers = field(default_factory=RewardHypers)
    agent: AgentHypers = field(default_factory=AgentHypers)
    experiment: ExperimentHypers = field(default_factory=ExperimentHypers)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train.num_minibatches > self.train.num_envs:
            raise ValueError("num_minibatches cannot exceed num_envs")
        
        if self.observation.max_cards < 1:
            raise ValueError("max_cards must be positive")
            
        if self.observation.max_actions < 1:
            raise ValueError("max_actions must be positive")

def register_configs() -> None:
    """Register all configurations with Hydra's config store."""
    cs = ConfigStore.instance()
    
    # Register the main config schema
    cs.store(name="config", node=Hypers)
    
    
    # Environment
    cs.store(
        group="observation",
        name="default",
        node=ObservationSpaceHypers,
    )
    cs.store(
        group="match",
        name="default",
        node=MatchHypers,
    )
    cs.store(
        group="reward",
        name="default",
        node=RewardHypers,
    )

    # Model
    cs.store(
        group="train",
        name="default",
        node=TrainHypers,
    )
    cs.store(
        group="agent",
        name="default",
        node=AgentHypers,
    )

    # Infrastructure
    cs.store(
        group="experiment",
        name="default",
        node=ExperimentHypers,
    )

def initialize() -> None:

    """Initialize Hydra configuration system."""
    register_configs()