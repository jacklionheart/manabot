"""
trainer.py

The primary interface for training is the Trainer class.

PPO training steps:
  - Collect trajectories (rollouts) using vectorized environments.
  - Compute advantages using GAE.
  - Perform PPO updates.
  - Save/load checkpoints.

This version uses a multi-agent buffer organized as (num_envs x num_players) queues.
"""

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple, List
import time
import wandb

import numpy as np
import psutil
import torch
import torch.nn as nn
import hydra

from manabot.infra import getLogger, Experiment, Hypers, TrainHypers
import manabot.env.observation
from manabot.env import ObservationSpace, VectorEnv, Match, Reward
from manabot.ppo.agent import Agent
import manabot.infra.hypers

from manabot.infra.profiler import Profiler

manabot.infra.hypers.initialize()


# -----------------------------------------------------------------------------
# Buffer Classes
# -----------------------------------------------------------------------------

class PPOBuffer:
    def __init__(self, device: str):
        self.device = device
        self.reset()
    def store(self, obs: dict, action: torch.Tensor, reward: torch.Tensor,
              value: torch.Tensor, logprob: torch.Tensor, done: torch.Tensor) -> None:
        # Ensure each observation tensor is stored properly
        for k, v in obs.items():
            # Stack tensors along a new dimension
            self.obs_buff[k].append(v.unsqueeze(0))
        self.actions_buf.append(action.unsqueeze(0))
        self.logprobs_buf.append(logprob.unsqueeze(0))
        self.rewards_buf.append(reward.unsqueeze(0))
        self.values_buf.append(value.unsqueeze(0))
        self.dones_buf.append(done.unsqueeze(0))

    def compute_advantages(self, gamma: float, gae_lambda: float, next_value: torch.Tensor, next_done: torch.Tensor):
        """
        Compute advantages using GAE for the transitions in this buffer.
        next_value: a scalar bootstrap value for this (env, player) pair.
        next_done: a scalar (0 or 1) indicating if the environment was done.
        """
        self.obs = {k: torch.cat(v, dim=0) for k, v in self.obs_buff.items()}
        self.actions = torch.cat(self.actions_buf, dim=0)
        self.logprobs = torch.cat(self.logprobs_buf, dim=0)
        self.rewards = torch.cat(self.rewards_buf, dim=0)
        self.values = torch.cat(self.values_buf, dim=0)
        self.dones = torch.cat(self.dones_buf, dim=0)

        T = self.rewards.shape[0]
        if T == 0:
            self.advantages = torch.tensor([], device=self.device)
            self.returns = torch.tensor([], device=self.device)
            return

        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                # For the last timestep, use the externally provided next_done flag.
                next_non_terminal = 1.0 - next_done.float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float()
                next_val = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        self.advantages = advantages
        self.returns = advantages + self.values

    def reset(self):
        from collections import defaultdict
        self.obs_buff = defaultdict(list)
        self.actions_buf = []
        self.logprobs_buf = []
        self.rewards_buf = []
        self.values_buf = []
        self.dones_buf = []
        self.obs = None
        self.actions = None
        self.logprobs = None
        self.rewards = None
        self.values = None
        self.dones = None

        self.advantages = None
        self.returns = None

class MultiAgentBuffer:
    """
    Maintains a separate PPOBuffer for each (env, player) pair.
    """
    def __init__(self, device: str, num_envs: int, num_players: int = 2):
        self.device = device
        self.num_envs = num_envs
        self.num_players = num_players
        self.buffers = {
            (env_idx, pid): PPOBuffer(device)
            for env_idx in range(num_envs)
            for pid in range(num_players)
        }

    def store(self, obs: dict, action: torch.Tensor, reward: torch.Tensor,
              value: torch.Tensor, logprob: torch.Tensor, done: torch.Tensor,
              actor_ids: torch.Tensor) -> None:
        # For each environment in the batch, store the transition in the buffer for the acting player.
        num_envs = action.shape[0]
        for i in range(num_envs):
            pid = int(actor_ids[i].item())
            key = (i, pid)
            single_obs = {k: v[i] for k, v in obs.items()}
            self.buffers[key].store(single_obs, action[i], reward[i],
                                      value[i], logprob[i], done[i])

    def compute_advantages(self, next_values: torch.Tensor, next_dones: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute advantages for each (env, player) buffer.
        next_values: tensor of shape (num_envs,) for each environment.
        next_dones: tensor of shape (num_envs,) for each environment.
        """
        for env in range(self.num_envs):
            for pid in range(self.num_players):
                buf = self.buffers[(env, pid)]
                if len(buf.actions_buf) == 0:
                    continue
                bootstrap_value = next_values[env] if not next_dones[env].item() else torch.tensor(0.0, device=self.device)
                buf.compute_advantages(gamma, gae_lambda, bootstrap_value, next_dones[env])

    def get_flattened(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten transitions from all (env, player) buffers into a single batch.
        Returns:
          merged_obs, merged_logprobs, merged_actions, merged_advantages, merged_returns, merged_values
        """
        all_obs = []
        all_logprobs = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_values = []

        for buf in self.buffers.values():
            assert buf.obs is not None
            assert buf.actions is not None
            assert buf.logprobs is not None
            assert buf.values is not None
            assert buf.advantages is not None
            assert buf.returns is not None

            if len(buf.actions) == 0:
                continue
            all_obs.append(buf.obs)
            all_logprobs.append(buf.logprobs)
            all_actions.append(buf.actions)
            all_advantages.append(buf.advantages)
            all_returns.append(buf.returns)
            all_values.append(buf.values)

        if not all_obs:
            raise ValueError("No valid transitions found in any buffer.")

        merged_obs = {}
        for k in all_obs[0].keys():
            merged_obs[k] = torch.cat([obs[k] for obs in all_obs], dim=0)

        merged_logprobs = torch.cat(all_logprobs, dim=0)
        merged_actions = torch.cat(all_actions, dim=0)
        merged_advantages = torch.cat(all_advantages, dim=0)
        merged_returns = torch.cat(all_returns, dim=0)
        merged_values = torch.cat(all_values, dim=0)

        return merged_obs, merged_logprobs, merged_actions, merged_advantages, merged_returns, merged_values
    
    def reset(self):
        for buf in self.buffers.values():
            buf.reset()


# -----------------------------------------------------------------------------
# Trainer Class
# -----------------------------------------------------------------------------

class Trainer:
    """
    PPO Trainer for manabot.

    Implements the training loop:
      1. Collect trajectories using vectorized environments.
      2. Compute advantages per (env, player) buffer.
      3. Merge buffers for a unified policy update.
      4. Run multiple update epochs with detailed logging.

    Also provides checkpoint saving/loading functionality.
    """
    def __init__(self, agent: Agent, experiment: Experiment,
                 env: VectorEnv, hypers: TrainHypers = TrainHypers()):
        self.agent = agent.to(experiment.device)
        self.experiment = experiment
        self.env = env
        self.hypers = hypers
        self.global_step = 0

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=hypers.learning_rate,
            eps=1e-5,
            weight_decay=0.01
        )

        self.logger = getLogger(__name__)

        self.multi_buffer = MultiAgentBuffer(experiment.device, hypers.num_envs, num_players=2)
        self.consecutive_invalid_batches = 0
        self.invalid_batch_threshold = 5
        self.wandb = self.experiment.wandb_run

        # Initialize the profiler
        self.profiler = self.experiment.profiler
        
        if self.wandb:
            self.wandb.summary.update({
                "max_episode_return": float("-inf"),
                "best_win_rate": 0.0,
                "time_to_converge": None,
            })
        self.logger.info("Trainer initialized.")

    def train(self) -> None:
        # Use context manager for root timer
        with self.profiler.track("train"):
            hypers = self.hypers
            env = self.env
            device = self.experiment.device
            batch_size = hypers.num_envs * hypers.num_steps
            minibatch_size = batch_size // hypers.num_minibatches
            num_updates = hypers.total_timesteps // batch_size
            self.start_time = time.time()

            self.logger.info("Resetting environment for training.")
            next_obs, _ = env.reset()
            next_done = torch.zeros(hypers.num_envs, dtype=torch.bool, device=device)

            prev_actor_ids = manabot.env.observation.get_agent_indices(next_obs)

            for update in range(1, num_updates + 1):
                if hypers.anneal_lr:
                    frac = 1.0 - (update - 1) / num_updates
                    lr_now = frac * hypers.learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_now
                    self.logger.info(f"Update {update}: Annealed LR set to {lr_now}")
                else:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(f"Update {update}: LR = {current_lr}")

                self.multi_buffer.reset()

                self.logger.info("Starting rollout data collection.")
                wandb.log({"rollout/step": 0}, step=self.global_step)
                
                with self.profiler.track("rollout"):
                    with self.profiler.track("step"):
                        for step in range(hypers.num_steps):
                            try:
                                next_obs, next_done, prev_actor_ids = self._rollout_step(next_obs, prev_actor_ids)
                                self.consecutive_invalid_batches = 0
                            except Exception as e:
                                self.consecutive_invalid_batches += 1
                                self.logger.error(f"Rollout step error at step {step}: {e}")
                                if self.consecutive_invalid_batches >= self.invalid_batch_threshold:
                                    raise RuntimeError(f"Failure during rollout; halting training: {e}")
                                else:
                                    self.logger.error("Skipping faulty rollout step.")
                    
                    with self.profiler.track("advantage"):
                        with torch.no_grad():
                            next_value = self.agent.get_value(next_obs)
                        self.multi_buffer.compute_advantages(next_value, next_done,
                                                            hypers.gamma, hypers.gae_lambda)

                        try:
                            obs, logprobs, actions, advantages, returns, values = self.multi_buffer.get_flattened()
                            self.logger.info(f"Flattened buffer has {logprobs.numel()} transitions.")
                        except ValueError as e:
                            self.logger.error(f"No valid transitions in buffers: {e}")
                            raise

                clipfracs = []
                approx_kl = 0.0
                inds = np.arange(batch_size)

                with self.profiler.track("gradient"):
                    for epoch in range(hypers.update_epochs):
                        np.random.shuffle(inds)
                        for start in range(0, batch_size, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = inds[start:end]
                            mb_obs = {k: v[mb_inds] for k, v in obs.items()}
                            mb_old_logprobs = logprobs[mb_inds]
                            mb_actions = actions[mb_inds]
                            mb_advantages = advantages[mb_inds]
                            mb_returns = returns[mb_inds]
                            mb_values = values[mb_inds]

                            approx_kl, clip_fraction = self._optimize_step(
                                mb_obs, mb_old_logprobs, mb_actions,
                                mb_advantages, mb_returns, mb_values
                            )
                            clipfracs.append(clip_fraction)

                            if update % 10 == 0:
                                self._log_system_metrics()

                            if hypers.target_kl != float("inf") and approx_kl > hypers.target_kl:
                                self.logger.info(f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f}")
                                break


                with torch.no_grad():
                    y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                sps = int(self.global_step / (time.time() - self.start_time))
                if self.wandb:
                    self.wandb.log({
                        "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "losses/explained_variance": explained_var,
                        "charts/SPS": sps
                    }, step=self.global_step)
                
                self.experiment.log_performance(step=self.global_step)

                time_since_start = time.time() - self.start_time
                self.logger.info(
                    f"Update {update}/{num_updates} | SPS: {sps} | Total time: {time_since_start:.2f}s"
                )

                self.logger.info(f"Buffer sizes: {[len(buf.actions_buf) for buf in self.multi_buffer.buffers.values()]}")

            env.close()
            if self.wandb:
                self.save()
            self.experiment.close()
            self.logger.info("Training completed.")

    def _rollout_step(self, next_obs: Dict[str, torch.Tensor],
                      actor_ids: torch.Tensor) -> Tuple[Dict[str, torch.Tensor],
                                                        torch.Tensor,
                                                        torch.Tensor]:
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)

        try:
            with self.profiler.track("env"):
                new_obs, reward, done, _, info = self.env.step(action)
        except Exception as e:
            self.logger.error(f"env.step() failed: {e}")
            raise e

        self.global_step += self.hypers.num_envs
        if not self._validate_obs(new_obs):
            raise RuntimeError("Invalid observation format detected; halting training.")
        
        self.multi_buffer.store(next_obs, action, reward, value, logprob, done, actor_ids)
        new_actor_ids = manabot.env.observation.get_agent_indices(new_obs)
        self.logger.debug(f"Rollout step completed; new actor_ids: {new_actor_ids}")
        return new_obs, done, new_actor_ids

    def _optimize_step(self, obs: Dict[str, torch.Tensor],
                       logprobs: torch.Tensor,
                       actions: torch.Tensor,
                       advantages: torch.Tensor,
                       returns: torch.Tensor,
                       values: torch.Tensor) -> Tuple[float, float]:
        hypers = self.hypers
        _, new_logprobs, entropy, new_values = self.agent.get_action_and_value(obs, actions)
        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - hypers.clip_coef, 1 + hypers.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        new_values = new_values.view(-1)
        if hypers.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(new_values - values, -hypers.clip_coef, hypers.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - hypers.ent_coef * entropy_loss + hypers.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), hypers.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item()
            approx_kl = max(approx_kl, 0.0)
            clip_fraction = (torch.abs(ratio - 1) > hypers.clip_coef).float().mean().item()

        if self.wandb:
            self.wandb.log({
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/approx_kl": approx_kl,
                "losses/clip_fraction": clip_fraction,
                "ppo/losses": {
                    "policy": pg_loss.item(),
                    "value": v_loss.item(),
                    "entropy": entropy_loss.item()
                },
                "ppo/metrics": {
                    "kl": approx_kl,
                    "clip_fraction": clip_fraction
                }
            }, step=self.global_step)

        self.logger.debug(f"Optimize step: approx_kl={approx_kl}, clip_fraction={clip_fraction}")
        return approx_kl, clip_fraction

    def _validate_obs(self, obs: dict) -> bool:
        expected_keys = set(self.env.observation_space.keys())
        if set(obs.keys()) != expected_keys:
            self.logger.error(f"Observation keys mismatch. Expected {expected_keys}, got {set(obs.keys())}")
            return False

        for k, v in obs.items():
            expected_shape = self.env.observation_space[k].shape
            if v.shape[1:] != expected_shape:
                self.logger.error(f"Observation shape mismatch for key {k}. "
                                  f"Expected {expected_shape} (inside batch), got {v.shape[1:]}")
                return False
        return True

    def _log_system_metrics(self):
        if not self.wandb:
            return

        metrics = {
            "system/memory_used": psutil.Process().memory_info().rss / (1024 * 1024),
            "system/cpu_percent": psutil.cpu_percent(),
            "system/steps_per_second": int(self.global_step / (time.time() - self.start_time))
        }
        if torch.cuda.is_available():
            metrics.update({
                "system/gpu_utilization": torch.cuda.utilization(),
                "system/gpu_memory_allocated": torch.cuda.memory_allocated() / (1024 * 1024),
                "system/gpu_memory_reserved": torch.cuda.memory_reserved() / (1024 * 1024)
            })
        wandb.log(metrics, step=self.global_step)
        self.logger.debug(f"Logged system metrics: {metrics}")


    def save(self) -> None:
        assert self.wandb is not None
        name = self.experiment.exp_name
        path = f"{name}.pt"  
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)
        
        # Create and log artifact
        artifact = wandb.Artifact(
            name=name,
            type="model",
            description=f"Latest model at step {self.global_step}",
        )
        artifact.add_file(path)
        self.wandb.log_artifact(artifact)


@hydra.main(version_base=None, config_path="../conf", config_name="local")
def main(cfg: DictConfig) -> None:
    obs_config = OmegaConf.to_object(cfg.observation)
    train_config = OmegaConf.to_object(cfg.train)
    reward_config = OmegaConf.to_object(cfg.reward)
    agent_config = OmegaConf.to_object(cfg.agent)
    experiment_config = OmegaConf.to_object(cfg.experiment)
    match_config = OmegaConf.to_object(cfg.match)
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