"""
experiment.py
Experiment tracking and environment setup with proper config handling.
"""

import os
import time
import logging
from pathlib import Path
import random
from dataclasses import asdict
import torch
import wandb
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from .hypers import ExperimentHypers, Hypers

logger = logging.getLogger(__name__)
CODE_CONTEXT_ROOT = Path(os.getenv("CODE_CONTEXT_ROOT", str(Path.home() / "src")))

def flatten_config(cfg: dict, parent_key: str = '', sep: str = '/') -> dict:
    """Flatten nested dictionary with path-like keys."""
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Experiment:
    """Experiment tracking and environment setup."""
    def __init__(self, experiment_hypers: ExperimentHypers = ExperimentHypers(), full_hypers: Hypers = Hypers()):
        self.experiment_hypers = experiment_hypers
        self.full_hypers = full_hypers
        self.exp_name = experiment_hypers.exp_name
        self.seed = experiment_hypers.seed
        self.torch_deterministic = experiment_hypers.torch_deterministic
        self.device = experiment_hypers.device
        self.wandb_on = experiment_hypers.wandb
        self.tensorboard_on = experiment_hypers.tensorboard       
        self.wandb_project_name = experiment_hypers.wandb_project_name
        self.runs_dir = self.experiment_hypers.runs_dir / self.exp_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writer = None
        self.wandb_run = None

        self._setup_random()
        self._setup_tracking()

    def _get_flattened_config(self) -> dict:
        """Convert nested hypers to flat dict for wandb."""
        # Convert all dataclasses to dicts
        config_dict = {
            "observation": asdict(self.full_hypers.observation),
            "match": asdict(self.full_hypers.match),
            "train": asdict(self.full_hypers.train),
            "reward": asdict(self.full_hypers.reward),
            "agent": asdict(self.full_hypers.agent),
            "experiment": asdict(self.full_hypers.experiment),
        }
        # Flatten with path-like keys (e.g. "train/learning_rate")
        return flatten_config(config_dict)

    def _setup_tracking(self):
        """Setup experiment tracking with wandb and tensorboard."""
        run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        run_dir = self.runs_dir / run_name
        
        if self.wandb_on:
            try:
                # Get flattened config for wandb
                config = self._get_flattened_config()
                
                # Add some useful metadata
                config.update({
                    "seed": self.seed,
                    "device": self.device,
                    "runs_dir": str(self.runs_dir),
                    "code_context": str(CODE_CONTEXT_ROOT),
                })
                
                self.wandb_run = wandb.init(
                    project=self.wandb_project_name,
                    entity=None,
                    name=run_name,
                    dir=str(self.runs_dir),
                    config=config,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True,
                )
                logger.info(f"Initialized wandb run: {run_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_on = False
        
        if self.tensorboard_on:
            self.summary_writer = SummaryWriter(str(run_dir))

    def add_scalar(self, tag: str, value, step: int):
        """Write a value to tensorboard."""
        if self.summary_writer:
            self.summary_writer.add_scalar(tag, value, step)

    def _setup_random(self):
        """Setup random number generators."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

    def close(self):
        """Cleanup wandb and tensorboard resources."""
        if self.summary_writer:
            self.summary_writer.close()
            self.summary_writer = None
        
        if self.wandb_on and self.wandb_run:
            try:
                wandb.finish()
                self.wandb_run = None
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")