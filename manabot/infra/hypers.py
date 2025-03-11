"""
hypers.py
Centralized hyperparameter configuration using Hydra.

This module defines a unified configuration system for all hyperparameters across the codebase,
organizing them into logical groups while maintaining type safety and easy CLI/YAML override support.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Any, get_type_hints, List
from pathlib import Path

from hydra.core.config_store import ConfigStore

@dataclass
class ObservationSpaceHypers:
    max_cards_per_player: int = 100
    max_permanents_per_player: int = 50
    max_actions: int = 10
    max_focus_objects: int = 2

@dataclass
class MatchHypers:
    """Parameters previously passed to the Match object."""
    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: Dict[str, int] = field(default_factory=lambda: {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Grey Ogre": 18,
    })
    villain_deck: Dict[str, int] = field(default_factory=lambda: {
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
    wandb: bool = True
    wandb_project_name: str = "manabot"
    runs_dir: Path = field(default_factory=lambda: Path.home() / "manabot-runs")
    log_level: str = "INFO" 
    profiler_enabled: bool = False 

@dataclass
class AgentHypers:
    # Shared embedding space for GameObjects and Actions.
    hidden_dim: int = 64
    # Number of attention heads used in the GameObjectAttention layer.
    num_attention_heads: int = 4

@dataclass
class TrainHypers:
    """Training-related hyperparameters."""
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
    target_kl: float = float("inf")

@dataclass
class RewardHypers:
    trivial: bool = False
    managym: bool = False
    win_reward: float = 100.0
    lose_reward: float = -100.0

@dataclass
class Hypers:
    """Top-level configuration that composes all hyperparameters."""
    observation: ObservationSpaceHypers = field(default_factory=ObservationSpaceHypers)
    match: MatchHypers = field(default_factory=MatchHypers)
    train: TrainHypers = field(default_factory=TrainHypers)
    reward: RewardHypers = field(default_factory=RewardHypers)
    agent: AgentHypers = field(default_factory=AgentHypers)
    experiment: ExperimentHypers = field(default_factory=ExperimentHypers)
    
    def __post_init__(self):
        """Validate configuration after initialization."""    
        if self.observation.max_cards_per_player < 1:
            raise ValueError("max_cards_per_player must be positive")
            
        if self.observation.max_actions < 1:
            raise ValueError("max_actions must be positive")


def initialize() -> None:
    """Register configurations with Hydra's config store."""
    cs = ConfigStore.instance()
    
    # Register the main config structure
    cs.store(name="hypers", node=Hypers)
    
    # Register config groups
    cs.store(group="observation", name="default", node=ObservationSpaceHypers)
    cs.store(group="match", name="default", node=MatchHypers)
    cs.store(group="train", name="default", node=TrainHypers)
    cs.store(group="reward", name="default", node=RewardHypers)
    cs.store(group="agent", name="default", node=AgentHypers)
    cs.store(group="experiment", name="default", node=ExperimentHypers)

def add_hypers(parser, dataclass_type, prefix=''):
    """
    Automatically add arguments to an ArgumentParser based on a dataclass.
    Supports nested dataclasses with prefixed parameter names.
    
    Args:
        parser: argparse.ArgumentParser instance
        dataclass_type: The dataclass type to extract fields from
        prefix: Prefix for parameter names (used for nested dataclasses)
    """
    type_hints = get_type_hints(dataclass_type)
    default_instance = dataclass_type()  # Create instance to get default values
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = type_hints.get(field_name)
        assert field_type is not None
        default_value = getattr(default_instance, field_name)
        
        # Handle nested dataclasses recursively
        if field.type.__module__ == 'dataclasses' or (hasattr(field_type, '__origin__') and field_type.__origin__ is type):
            # This is likely a dataclass field
            if hasattr(default_value, '__dataclass_fields__'):
                nested_prefix = f"{prefix}{field_name}_" if prefix else f"{field_name}_"
                add_hypers(parser, type(default_value), nested_prefix)
                continue
        
        # Create parameter name with prefix
        param_name = f"{prefix}{field_name}"
        
        # Convert snake_case to kebab-case for CLI args
        arg_name = f"--{param_name.replace('_', '-')}"
        
        # Set appropriate type and action based on field type
        if field_type == bool:
            # For boolean fields, use store_true/store_false actions
            if default_value:
                parser.add_argument(f"--no-{param_name.replace('_', '-')}", 
                                    dest=param_name, action="store_false",
                                    help=f"Disable {field_name} (default: enabled)")
            else:
                parser.add_argument(arg_name, dest=param_name, action="store_true",
                                    help=f"Enable {field_name} (default: disabled)")
        else:
            if isinstance(default_value, (dict, list)) or field_type in (Dict, List, Path):
                assert False, f"Complex type {field_type} not supported for command line arguments"
                
            # Get docstring for the field if available
            field_doc = field.metadata.get('doc', f"{field_name} parameter")
            
            parser.add_argument(arg_name, dest=param_name, type=field_type, default=default_value,
                               help=f"{field_doc} (default: {default_value})")


def parse_hypers(args, dataclass_type, prefix=''):
    """
    Parse nested hyperparameters from command line arguments.
    
    Args:
        args: Arguments namespace from ArgumentParser
        dataclass_type: The dataclass type to construct
        prefix: Prefix used for parameter names (used for nested dataclasses)
        
    Returns:
        An instance of dataclass_type with values from args
    """
    type_hints = get_type_hints(dataclass_type)
    result = {}
    
    for field in fields(dataclass_type):
        field_name = field.name
        field_type = type_hints.get(field_name)
        assert field_type is not None
        # Handle nested dataclasses recursively
        if field.type.__module__ == 'dataclasses' or (hasattr(field_type, '__origin__') and field_type.__origin__ is type):
            if hasattr(dataclass_type().__getattribute__(field_name), '__dataclass_fields__'):
                nested_prefix = f"{prefix}{field_name}_" if prefix else f"{field_name}_"
                nested_value = parse_hypers(args, type(dataclass_type().__getattribute__(field_name)), nested_prefix)
                result[field_name] = nested_value
                continue
        
        # Get parameter name with prefix
        param_name = f"{prefix}{field_name}"
        
        # Check if the parameter was provided in the arguments
        if hasattr(args, param_name):
            result[field_name] = getattr(args, param_name)
    
    # Create and return an instance of the dataclass
    return dataclass_type(**result)
