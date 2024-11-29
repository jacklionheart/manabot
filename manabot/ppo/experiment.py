import time
import random
from dataclasses import dataclass, asdict
import argparse
from typing import Optional
import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from manabot.ppo import Hypers

@dataclass
class Experiment:
    """Configuration for experiment tracking and environment setup."""
    exp_name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "manabot"
    wandb_entity: Optional[str] = None

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() and self.cuda else "cpu"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add experiment configuration arguments to parser."""
        parser.add_argument("--exp-name", type=str, default=cls.exp_name,
            help="the name of this experiment")
        parser.add_argument("--seed", type=int, default=cls.seed,
            help="seed of the experiment")
        parser.add_argument("--torch-deterministic", type=bool, default=cls.torch_deterministic,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
        parser.add_argument("--cuda", type=bool, default=cls.cuda,
            help="if toggled, cuda will be enabled by default")
        parser.add_argument("--track", type=bool, default=cls.track,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        parser.add_argument("--wandb-project-name", type=str, default=cls.wandb_project_name,
            help="the wandb's project name")
        parser.add_argument("--wandb-entity", type=str, default=cls.wandb_entity,
            help="the entity (team) of wandb's project")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Experiment':
        """Create an Experiment instance from parsed arguments."""
        return cls(
            exp_name=args.exp_name,
            seed=args.seed,
            torch_deterministic=args.torch_deterministic,
            cuda=args.cuda,
            track=args.track,
            wandb_project_name=args.wandb_project_name,
            wandb_entity=args.wandb_entity,
        )
    
    def setup(self, hypers: Hypers) -> tuple[str, SummaryWriter]:
        run_name, writer = self.setup_tracking(hypers)
        self.setup_random()
        return run_name, writer

    def setup_tracking(self, hypers: Hypers) -> tuple[str, SummaryWriter]:
        """Setup experiment tracking with wandb and tensorboard."""
        run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        
        if self.track:
            import wandb
            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                config={**asdict(self), **asdict(hypers)},
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in {**asdict(self), **asdict(hypers)}.items()])
            ),
        )
        return run_name, writer

    def setup_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic


