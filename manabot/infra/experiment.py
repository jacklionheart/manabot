import time
import random
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from .hypers import ExperimentHypers



class Experiment:
    """Experiment tracking and environment setup."""
    def __init__(self, hypers: ExperimentHypers = ExperimentHypers()):
        self.hypers = hypers
        self.exp_name = hypers.exp_name
        self.seed = hypers.seed
        self.torch_deterministic = hypers.torch_deterministic
        self.device = hypers.device
        self.track = hypers.track
        self.wandb_project_name = hypers.wandb_project_name
        self.wandb_entity = hypers.wandb_entity
        self.writer = SummaryWriter(f"runs/{self.exp_name}__{self.seed}__{self.device}__{int(time.time())}")

    def upgrade_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"

    def setup(self) -> tuple[str, SummaryWriter]:
        run_name, writer = self.setup_tracking()
        self.setup_random()
        return run_name, writer

    def setup_tracking(self) -> tuple[str, SummaryWriter]:
        """Setup experiment tracking with wandb and tensorboard."""
        run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        
        if self.track:
            import wandb
            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        
        writer = SummaryWriter(f"runs/{run_name}")
        return run_name, writer

    def setup_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic


