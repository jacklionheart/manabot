from dataclasses import dataclass
from typing import Optional
import argparse

@dataclass
class Hypers:   
    """Hyperparameters for PPO training."""
    # Observation space hyperparameters
    max_permanents: int = 15
    max_actions: int = 20
    
    # Training parameters
    total_timesteps: int = 20_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    
    # PPO specific parameters
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

    # Reward parameters
    win_reward: float = 100.0
    lose_reward: float = -100.0
    damage_scale: float = 1.0
    progression_reward: float = 1.0

    # Model hyperparameters
    game_embedding_dim: int = 4
    battlefield_embedding_dim: int = 4
    hidden_dim: int = 16


    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add hyperparameter arguments to parser."""
        # Training parameters
        parser.add_argument("--total-timesteps", type=int, default=cls.total_timesteps,
            help="total timesteps of the experiments")
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate,
            help="the learning rate of the optimizer")
        parser.add_argument("--num-envs", type=int, default=cls.num_envs,
            help="the number of parallel game environments")
        parser.add_argument("--num-steps", type=int, default=cls.num_steps,
            help="the number of steps to run in each environment per policy rollout")
        parser.add_argument("--anneal-lr", type=bool, default=cls.anneal_lr,
            help="Toggle learning rate annealing for policy and value networks")

        # PPO specific parameters
        parser.add_argument("--gamma", type=float, default=cls.gamma,
            help="the discount factor gamma")
        parser.add_argument("--gae-lambda", type=float, default=cls.gae_lambda,
            help="the lambda for the general advantage estimation")
        parser.add_argument("--num-minibatches", type=int, default=cls.num_minibatches,
            help="the number of mini-batches")
        parser.add_argument("--update-epochs", type=int, default=cls.update_epochs,
            help="the K epochs to update the policy")
        parser.add_argument("--norm-adv", type=bool, default=cls.norm_adv,
            help="Toggles advantages normalization")
        parser.add_argument("--clip-coef", type=float, default=cls.clip_coef,
            help="the surrogate clipping coefficient")
        parser.add_argument("--clip-vloss", type=bool, default=cls.clip_vloss,
            help="Toggles whether or not to use a clipped loss for the value function")
        parser.add_argument("--ent-coef", type=float, default=cls.ent_coef,
            help="coefficient of the entropy")
        parser.add_argument("--vf-coef", type=float, default=cls.vf_coef,
            help="coefficient of the value function")
        parser.add_argument("--max-grad-norm", type=float, default=cls.max_grad_norm,
            help="the maximum norm for the gradient clipping")
        parser.add_argument("--target-kl", type=float, default=cls.target_kl,
            help="the target KL divergence threshold")
        
        # Reward parameters
        parser.add_argument("--win-reward", type=float, default=cls.win_reward,
            help="reward for winning a game")
        parser.add_argument("--lose-reward", type=float, default=cls.lose_reward,
            help="reward for losing a game")
        parser.add_argument("--damage-scale", type=float, default=cls.damage_scale,
            help="scaling factor for damage dealt")
        parser.add_argument("--progression-reward", type=float, default=cls.progression_reward,
            help="reward for progressing the game state")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Hypers':
        """Create a Hypers instance from parsed arguments."""
        return cls(
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            anneal_lr=args.anneal_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            norm_adv=args.norm_adv,
            clip_coef=args.clip_coef,
            clip_vloss=args.clip_vloss,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            win_reward=args.win_reward,
            lose_reward=args.lose_reward,
            damage_scale=args.damage_scale,
            progression_reward=args.progression_reward,
        )
