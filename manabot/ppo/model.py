
from typing import Dict, List, Optional, Tuple
from torch.nn import Module, Conv2d, Linear, Sequential, ReLU
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
import time
import torch
import torch.nn

from manabot.env import VectorEnv
from manabot.data import Observation
from manabot.ppo import Hypers
from manabot.ppo.experiment import Experiment
from manabot.data import ObservationSpace



def layer_init(module: Module) -> None:
    """Initialize weights for all linear layers."""
    if isinstance(module, Linear):
        torch.nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
class Agent(Module):
    """PPO agent for Magic gameplay."""
    
    def __init__(self, hypers: Hypers, device: str):
        
        observation_space = ObservationSpace(
            hypers.max_permanents,
            hypers.max_actions
        )
        
        # Game state encoders
        self.game_net = Sequential(
            Linear(observation_space.game_shape[0], hypers.game_embedding_dim),
            ReLU()
        )
        self.permanent_net = Sequential(
            Linear(observation_space.permanent_shape[0], hypers.battlefield_embedding_dim),
            ReLU()
        )
        
        # Combine into game state understanding
        self.feature_net = Sequential(
            Linear(hypers.game_embedding_dim + hypers.battlefield_embedding_dim, hypers.hidden_dim),
            ReLU(),
            Linear(hypers.hidden_dim, hypers.hidden_dim),
            ReLU()
        )
        
        # Then evaluate actions using that understanding
        self.action_scorer = Sequential(
            Linear(hypers.hidden_dim + observation_space.action_shape[0], hypers.hidden_dim),
            ReLU(),
            Linear(hypers.hidden_dim, 1)
        )
        
        # Policy and value heads
        self.actor = Linear(hypers.hidden_dim, observation_space.max_actions)
        self.critic = Linear(hypers.hidden_dim, 1)
        
        self.apply(layer_init)
        
        # Special initialization for actor head
        torch.nn.init.orthogonal_(self.actor.weight)
        torch.nn.init.zeros_(self.actor.bias)

        self.to(device)

    def _batch_observations(self, observations: List[Observation]) -> Dict[str, torch.Tensor]:
        # Convert each observation to tensors and stack them
        obs_tensors_list = [
            obs.to_tensors(
                self.observation_space.card_embeddings,
                self.observation_space.max_permanents,
                self.observation_space.max_actions,
                device=self.device
            )
            for obs in observations
        ]
        
        # Combine the dictionaries by stacking tensors
        return {
            key: torch.stack([obs[key] for obs in obs_tensors_list])
            for key in obs_tensors_list[0].keys()
        }

    def _extract_features(self, batched_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        game_features = self.game_net(batched_obs['game'])
        permanent_features = self.permanent_net(batched_obs['permanents']).mean(dim=1)
        combined = torch.cat([game_features, permanent_features], dim=-1)
        return self.feature_net(combined)

    def get_values(self, observations: List[Observation]) -> torch.Tensor:
        """Compute value function for batch of observations.
        
        Args:
            observations: List of observations to evaluate
            
        Return Shape:
            (len(observations),) - A tensor containing one value per observation
        """
        batched_obs = self._batch_observations(observations)
        features = self._extract_features(batched_obs)
        return self.critic(features).squeeze(-1)  # Shape: (len(observations),)

    def get_actions_and_values(
        self,
        observations: List[Observation],
        actions: Optional[torch.Tensor] = None,
        argmax: bool = False
    ) -> Tuple[torch.distributions.Categorical, Optional[torch.Tensor], torch.Tensor]:
        """Compute action distribution and value function for batch of observations.
        
        Args:
            observations: List of observations to evaluate
            action: Optional actions for computing log probs, shape: (len(observations),)
            argmax: If True, select most probable action
            
        Return Shapes:
            action_dist: Categorical distribution over actions for each observation
            selected_action: (len(observations),) if action was provided/selected, else None
            values: (len(observations),) tensor containing value estimates
        """
        batched_obs = self._batch_observations(observations)
        features = self._extract_features(batched_obs)
        
        # Get action logits and mask invalid actions
        action_logits = self.actor(features)
        action_mask = batched_obs['actions'][..., -1]  # Assume last dim is valid flag
        action_logits = action_logits.masked_fill(~action_mask.bool(), float('-inf'))
        
        # Create action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Select action if requested
        selected_action = None
        if action is not None:
            selected_action = action
        elif argmax:
            selected_action = torch.argmax(action_logits, dim=-1)
        
        values = self.critic(features).squeeze(-1)  # Shape: (len(observations),)
        
        return action_dist, selected_action, values

class Model():
    def __init__(self, hypers: Hypers, device: str):
        self.hypers = hypers
        self.device = device
        self.agent = Agent(hypers, device)

    def train(self, env: VectorEnv, experiment: Experiment, writer: SummaryWriter) -> None:
        hypers = self.hypers
        device = self.device
        num_steps = hypers.num_steps
        num_envs = hypers.num_envs

        """Execute PPO training loop."""
        optimizer = Adam(self.agent.parameters(), lr=hypers.learning_rate, eps=1e-5)

        # Storage buffers
        observations = [{
            'game': torch.zeros((num_steps, num_envs) + env.observation_space.shapes['game'], device=device),
            'permanents': torch.zeros((num_steps, num_envs) + env.observation_space.shapes['permanents'], device=device),
            'actions': torch.zeros((num_steps, num_envs) + env.observation_space.shapes['actions'], dtype=torch.long, device=device),
        } for _ in range(2)]  # One set of buffers per environment
        logprobs = torch.zeros((num_steps, num_envs), device=device)
        rewards = torch.zeros((num_steps, num_envs), device=device)
        dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        values = torch.zeros((num_steps, num_envs), device=device)

        # Initialize training
        global_step = 0
        start_time = time.time()
        next_obs = env.reset()

        num_updates = hypers.total_timesteps // hypers.batch_size
        for update in range(1, num_updates + 1):
            # Learning rate annealing
            if hypers.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * hypers.learning_rate

            # Collect rollout
            for step in range(num_steps):
                global_step += num_envs
                for env_idx, obs in enumerate(next_obs):
                    for key, tensor in observations[env_idx].items():
                        tensor[step, env_idx] = obs[key]

                action_dist, selected_action, value = self.agent.get_actions_and_value(next_obs)
                for env_idx, action in enumerate(selected_action):
                    logprobs[step, env_idx] = action_dist[env_idx].log_prob(action)
                    values[step, env_idx] = value[env_idx]

                # Step ALL envs forward in parallel
                next_obs, reward, done, info = env.step([a.item() for a in selected_action])
                for env_idx, (r, d) in enumerate(zip(reward, done)):
                    rewards[step, env_idx] = r
                    dones[step, env_idx] = d

                # Log episode info
                for idx, item in enumerate(info):
                    if "episode" in item.keys():
                        player_idx = idx % 2
                        writer.add_scalar(f"charts/episodic_return-player{player_idx}", item["episode"]["r"], global_step)
                        writer.add_scalar(f"charts/episodic_length-player{player_idx}", item["episode"]["l"], global_step)

            # Compute returns and advantages
            with torch.no_grad():
                next_values = self.agent.get_values(next_obs)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - dones[:, t]
                        nextvalues = next_values
                    else:
                        nextnonterminal = 1.0 - dones[:, t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[:, t] + hypers.gamma * nextvalues * nextnonterminal - values[:, t]
                    advantages[:, t] = lastgaelam = delta + hypers.gamma * hypers.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # PPO Update
            for env_idx in range(num_envs):
                b_obs = {k: v[:, env_idx] for k, v in observations[env_idx].items()}
                b_logprobs = logprobs[:, env_idx]
                b_actions = observations[env_idx]['actions'][:, env_idx]
                b_advantages = advantages[:, env_idx]
                b_returns = returns[:, env_idx]
                b_values = values[:, env_idx]

                clipfracs = []
                for epoch in range(hypers.update_epochs):
                    np.random.shuffle(b_inds := np.arange(hypers.batch_size))
                    for start in range(0, hypers.batch_size, hypers.minibatch_size):
                        end = start + hypers.minibatch_size
                        mb_inds = b_inds[start:end]

                        action_dist, _, entropy, new_value = self.agent.get_actions_and_value([b_obs], b_actions[mb_inds])
                        logratio = action_dist.log_prob(b_actions[mb_inds]) - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        # Policy gradient loss
                        pg_loss = torch.max(
                            -b_advantages[mb_inds] * ratio,
                            -b_advantages[mb_inds] * torch.clamp(
                                ratio,
                                1 - hypers.clip_coef,
                                1 + hypers.clip_coef,
                            ),
                        ).mean()

                        # Value loss
                        v_loss = 0.5 * ((new_value.squeeze(-1) - b_returns[mb_inds]) ** 2).mean()

                        # Combined loss
                        loss = pg_loss - hypers.ent_coef * entropy.mean() + v_loss * hypers.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), hypers.max_grad_norm)
                        optimizer.step()

                        clipfracs.append(((ratio - 1.0).abs() > hypers.clip_coef).float().mean().item())

            # Log metrics
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
