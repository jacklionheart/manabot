"""
trainer.py

The primary interface for training is the `Trainer` class.

PPO training steps:
  - Collect trajectories (rollouts) using vectorized environments.
  - Compute advantages.
  - Perform PPO updates.

Additional responsibilities:
  - Logging training metrics.
  - Saving and loading checkpoints.
"""

import time
import logging
from typing import Dict, Tuple, List

import numpy as np
import psutil
import torch
import torch.nn as nn
import wandb

from manabot.ppo.agent import Agent
from manabot.infra.experiment import Experiment
from manabot.infra.hypers import TrainHypers
from manabot.env import ObservationSpace, VectorEnv
import manabot.env.observation

logger = logging.getLogger("manabot.ppo.trainer")

# =============================================================================
#  Internal Classes
# =============================================================================

class PPOBuffer:
    """
    Buffer to store transitions for one rollout (for one player).
    Pre-allocated with fixed capacity (num_steps); if capacity is exceeded, a warning is logged.
    """
    def __init__(self, observation_space: ObservationSpace, num_steps: int, num_envs: int, device: str):
        # Determine the shape for each observation sub-tensor.
        shapes: List[Tuple[int, ...]] = []
        for shape in observation_space.values():
            if shape.shape is not None:
                shapes.append(shape.shape)
            else:
                raise ValueError(f"Observation space {shape} has no shape")
        
        # Pre-allocate tensors for the entire rollout.
        self.obs = {
            k: torch.zeros((num_steps, num_envs) + shapes[i],
                           dtype=torch.float32, device=device)
            for i, k in enumerate(observation_space.keys())
        }
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.advantages = None  # To be computed later
        self.returns = None     # To be computed later

        self.device = device
        self.num_steps = num_steps  # capacity per rollout
        self.num_envs = num_envs
        self.step_idx = 0

    def store(self, obs: Dict[str, torch.Tensor],
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    value: torch.Tensor,
                    logprob: torch.Tensor,
                    done: torch.Tensor) -> None:
        """Store a transition. If capacity is exceeded, log a warning and skip."""
        if self.step_idx >= self.num_steps:
            logger.warning("PPOBuffer overflow: dropping transition.")
            return

        with torch.no_grad():
            for k, v in obs.items():
                # Log shapes at first storage for debugging
                if self.step_idx == 0:
                    logger.debug(f"Storing obs key '{k}' with shape {v.shape}")
                self.obs[k][self.step_idx] = v
            self.actions[self.step_idx] = action
            self.rewards[self.step_idx] = reward
            self.values[self.step_idx] = value
            self.logprobs[self.step_idx] = logprob
            self.dones[self.step_idx] = done
        self.step_idx += 1

    def compute_advantages(self, next_value: torch.Tensor, next_done: torch.Tensor,
                           gamma: float, gae_lambda: float) -> None:
        """Compute advantages and returns using GAE."""
        with torch.no_grad():
            if self.step_idx == 0:
                self.advantages = self.rewards[:0]
                self.returns = self.values[:0]
                return

            self.advantages = torch.zeros_like(self.rewards[:self.step_idx])
            lastgaelam = torch.zeros(self.rewards.shape[1], device=self.device)
            for t in reversed(range(self.step_idx)):
                if t == self.step_idx - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1].float()
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                self.advantages[t] = lastgaelam
            self.returns = self.advantages + self.values[:self.step_idx]
            logger.debug("Computed advantages and returns for current buffer.")

    def get_flattened(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten the stored transitions for PPO updates."""
        if self.advantages is None:
            raise ValueError("Call compute_advantages() before get_flattened()")
        assert self.returns is not None

        b_obs = {k: v[:self.step_idx].reshape((-1,) + v.shape[2:]) for k, v in self.obs.items()}
        b_logprobs = self.logprobs[:self.step_idx].reshape(-1)
        b_actions = self.actions[:self.step_idx].reshape(-1)
        b_advantages = self.advantages[:self.step_idx].reshape(-1)
        b_returns = self.returns[:self.step_idx].reshape(-1)
        b_values = self.values[:self.step_idx].reshape(-1)
        logger.debug(f"Flattened buffer: {self.step_idx} transitions.")
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def reset(self) -> None:
        """Reset the buffer for the next rollout."""
        logger.debug("Resetting PPOBuffer.")
        self.step_idx = 0
        self.advantages = None
        self.returns = None


class MultiAgentBuffer:
    """
    Maintains separate PPOBuffer instances for each player.
    Transitions for a given environment are stored only in the buffer of the actor who took the move.
    """
    def __init__(self, observation_space: ObservationSpace, num_steps: int, num_envs: int,
                 device: str, num_players: int = 2):
        self.buffers: Dict[int, PPOBuffer] = {
            pid: PPOBuffer(observation_space, num_steps, num_envs, device)
            for pid in range(num_players)
        }

    def store(self, obs: Dict[str, torch.Tensor],
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    value: torch.Tensor,
                    logprob: torch.Tensor,
                    done: torch.Tensor,
                    actor_ids: torch.Tensor) -> None:
        """
        For each environment in the batch, store the transition in the buffer corresponding
        to the actor who took the action.
        """
        for pid, buffer in self.buffers.items():
            indices = (actor_ids == pid).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            sub_obs = {k: v[indices] for k, v in obs.items()}
            sub_action = action[indices]
            sub_reward = reward[indices]
            sub_value = value[indices]
            sub_logprob = logprob[indices]
            sub_done = done[indices]
            buffer.store(sub_obs, sub_action, sub_reward, sub_value, sub_logprob, sub_done)
            logger.debug(f"Stored {indices.numel()} transitions for player {pid}.")

    def compute_advantages(self, next_value: torch.Tensor, next_done: torch.Tensor,
                           gamma: float, gae_lambda: float) -> None:
        for pid, buffer in self.buffers.items():
            logger.debug(f"Computing advantages for player {pid} with {buffer.step_idx} transitions.")
            buffer.compute_advantages(next_value, next_done, gamma, gae_lambda)

    def get_flattened(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_list: Dict[str, List[torch.Tensor]] = {}
        logprobs_list = []
        actions_list = []
        advantages_list = []
        returns_list = []
        values_list = []
        for pid, buffer in self.buffers.items():
            try:
                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = buffer.get_flattened()
            except ValueError:
                continue
            if b_logprobs.numel() == 0:
                continue
            for key, tensor in b_obs.items():
                obs_list.setdefault(key, []).append(tensor)
            logprobs_list.append(b_logprobs)
            actions_list.append(b_actions)
            advantages_list.append(b_advantages)
            returns_list.append(b_returns)
            values_list.append(b_values)
            logger.info(f"Player {pid} buffer flattened with {b_logprobs.numel()} transitions.")
        if len(logprobs_list) == 0:
            raise ValueError("No valid transitions found in any buffer.")
        merged_obs = {k: torch.cat(tensors, dim=0) for k, tensors in obs_list.items()}
        merged_logprobs = torch.cat(logprobs_list, dim=0)
        merged_actions = torch.cat(actions_list, dim=0)
        merged_advantages = torch.cat(advantages_list, dim=0)
        merged_returns = torch.cat(returns_list, dim=0)
        merged_values = torch.cat(values_list, dim=0)
        return merged_obs, merged_logprobs, merged_actions, merged_advantages, merged_returns, merged_values

    def reset(self) -> None:
        for pid, buffer in self.buffers.items():
            logger.debug(f"Resetting buffer for player {pid}.")
            buffer.reset()


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    PPO Trainer for manabot.

    Implements the training loop:
      1. Collect trajectories (rollouts) using vectorized environments.
      2. Compute advantages per player buffer.
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
        self.global_step = 0  # global step counter

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=hypers.learning_rate,
            eps=1e-5,
            weight_decay=0.01
        )

        # Assume two players in the multi-agent setting.
        self.multi_buffer = MultiAgentBuffer(
            env.observation_space, hypers.num_steps, hypers.num_envs,
            experiment.device, num_players=2
        )

        self.consecutive_invalid_batches = 0
        self.invalid_batch_threshold = 5

        self.wandb = self.experiment.wandb_run

        if self.wandb:
            self.wandb.summary.update({
                "max_episode_return": float("-inf"),
                "best_win_rate": 0.0,
                "time_to_converge": None,
            })
        logger.info("Trainer initialized.")

    def train(self) -> None:
        """
        Execute the PPO training loop.
        """
        hypers = self.hypers
        env = self.env
        device = self.experiment.device
        batch_size = hypers.num_envs * hypers.num_steps
        minibatch_size = batch_size // hypers.num_minibatches
        num_updates = hypers.total_timesteps // batch_size  
        self.start_time = time.time()

        logger.info("Resetting environment for training.")
        next_obs, _ = env.reset()
        next_done = torch.zeros(hypers.num_envs, dtype=torch.bool, device=device)

        # Use original actor selection: always take players[:, 0, 0]
        prev_actor_ids = manabot.env.observation.get_agent_indices(next_obs)

        for update in range(1, num_updates + 1):
            # Anneal learning rate if enabled.
            if hypers.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr_now = frac * hypers.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_now
                logger.info(f"Update {update}: Annealed LR set to {lr_now}")
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Update {update}: LR = {current_lr}")

            self.multi_buffer.reset()

            ############################################################
            # Rollout data collection
            ############################################################
            for step in range(hypers.num_steps):
                try:
                    next_obs, next_done, prev_actor_ids = self._rollout_step(next_obs, prev_actor_ids)
                    self.consecutive_invalid_batches = 0
                except Exception as e:
                    self.consecutive_invalid_batches += 1
                    logger.error(f"Rollout step error at step {step}: {e}")
                    if self.consecutive_invalid_batches >= self.invalid_batch_threshold:
                        raise RuntimeError(f"Failure during rollout; halting training: {e}")
                    else:
                        logger.error("Skipping faulty rollout step.")
            
            ############################################################
            # Compute advantages
            ############################################################
            logger.info("Computing advantages for collected rollouts.")
            obs, logprobs, actions, advantages, returns, values = self._compute_advantages(next_obs, next_done)

            ############################################################
            # PPO updates
            ############################################################
            clipfracs = []
            approx_kl = 0.0

            inds = np.arange(batch_size)
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
                        logger.info(f"Early stopping at epoch {epoch} due to KL divergence {approx_kl:.4f}")
                        break

            with torch.no_grad():
                y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.experiment.add_scalar("charts/learning_rate", 
                                       self.optimizer.param_groups[0]["lr"], self.global_step)
            self.experiment.add_scalar("losses/explained_variance", explained_var, self.global_step)
            sps = int(self.global_step / (time.time() - self.start_time))
            self.experiment.add_scalar("charts/SPS", sps, self.global_step)

            logger.info(f"Update {update}/{num_updates} | SPS: {sps} | Buffer sizes: " +
                        f"{[buf.step_idx for buf in self.multi_buffer.buffers.values()]}")
        env.close()
        self.experiment.close()
        logger.info("Training completed.")

    def save_checkpoint(self, path: str) -> None:
        """Save the current state of the agent, optimizer, and hyperparameters."""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_hypers': self.hypers,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load the agent and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.experiment.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Checkpoint loaded. Note: TrainHypers are not automatically restored.")

    def _rollout_step(self, next_obs: Dict[str, torch.Tensor], actor_ids: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Execute a single rollout step in the environment.
        """
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)
        try:
            new_obs, reward, done, _, info = self.env.step(action)
            logger.debug(f"Step env.step() output: reward={reward}, done={done}")
        except Exception as e:
            logger.error(f"env.step() failed: {e}")
            raise e

        self.global_step += self.hypers.num_envs
        logger.debug(f"Global step updated to {self.global_step}")

        if not self._validate_obs(new_obs):
            raise RuntimeError("Invalid observation format detected; halting training.")

        self.multi_buffer.store(next_obs, action, reward, value, logprob, done, actor_ids)
        new_actor_ids = manabot.env.observation.get_agent_indices(new_obs)
        logger.info(f"Rollout step completed; new actor_ids: {new_actor_ids}")
        return new_obs, done, new_actor_ids

    def _compute_advantages(self, next_obs: Dict[str, torch.Tensor], next_done: torch.Tensor) -> Tuple[
            Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hypers = self.hypers
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs)
        logger.debug(f"Next value computed with shape: {next_value.shape}")

        self.multi_buffer.compute_advantages(next_value, next_done, hypers.gamma, hypers.gae_lambda)
        try:
            obs, logprobs, actions, advantages, returns, values = self.multi_buffer.get_flattened()
            logger.info(f"Flattened buffer has {logprobs.numel()} transitions.")
        except ValueError as e:
            logger.error(f"No valid transitions in buffers: {e}")
            raise
        return obs, logprobs, actions, advantages, returns, values

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
            clip_fraction = (torch.abs(ratio - 1) > hypers.clip_coef).float().mean().item()

        self.experiment.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.experiment.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.experiment.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.experiment.add_scalar("losses/approx_kl", approx_kl, self.global_step)
        self.experiment.add_scalar("losses/clip_fraction", clip_fraction, self.global_step)

        if self.wandb:
            self.wandb.log({
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

        logger.debug(f"Optimize step: approx_kl={approx_kl}, clip_fraction={clip_fraction}")
        return approx_kl, clip_fraction

    def _validate_obs(self, obs: dict) -> bool:
        expected_keys = set(self.env.observation_space.keys())
        if set(obs.keys()) != expected_keys:
            logger.error(f"Observation keys mismatch. Expected {expected_keys}, got {set(obs.keys())}")
            return False

        for k, v in obs.items():
            expected_shape = self.env.observation_space[k].shape
            if v.shape[1:] != expected_shape:
                logger.error(f"Observation shape mismatch for key {k}. Expected {expected_shape} (inside batch), got {v.shape[1:]}")
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
        logger.info(f"Logged system metrics: {metrics}")
