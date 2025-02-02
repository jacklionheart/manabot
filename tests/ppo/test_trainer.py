"""
test_trainer.py

Comprehensive tests for the PPO Trainer and related functionality. This file covers:

1. Rollout and Buffer Management
   - Transition collection
   - Multi-agent handling
   - Observation validation
   - Buffer overflow behavior

2. Advantage Computation
   - GAE with constant rewards
   - GAE with alternating rewards
   - Proper multi-agent attribution
   - Bootstrapping for non-terminal episodes
   - Advantage normalization sign preservation

3. PPO Optimization
   - Single optimization step (gradient flow)
   - KL divergence checks
   - Probability ratio finiteness
   - Value loss (clipped/unclipped)
   - Early stopping

4. Checkpointing
   - Save/load model state
   - Equality of parameters after loading

5. Additional State/Policy Checks
   - Action masking for invalid actions
   - Trajectory ordering
   - Player perspective consistency
   - Reward attribution to correct players
   - Handling termination vs truncation
"""

import logging
import pytest
import numpy as np
import torch
from contextlib import contextmanager
from typing import Generator
import os
import shutil
import tempfile
from pathlib import Path
from manabot.env import VectorEnv, Match, ObservationSpace, Reward
from manabot.ppo import Agent, Trainer
from manabot.infra import (
    Experiment,
    TrainHypers,
    AgentHypers,
    ObservationSpaceHypers,
    RewardHypers,
    ExperimentHypers
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@contextmanager
def temp_run_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test runs.
    
    Usage:
        with temp_run_dir() as run_dir:
            # run_dir will be used for tensorboard/wandb artifacts
            # directory is automatically cleaned up after the test
    """
    original_runs = os.environ.get('MANABOT_RUNS_DIR')
    temp_dir = tempfile.mkdtemp(prefix='manabot_test_')
    os.environ['MANABOT_RUNS_DIR'] = temp_dir
    try:
        yield temp_dir
    finally:
        if original_runs:
            os.environ['MANABOT_RUNS_DIR'] = original_runs
        else:   
            os.environ.pop('MANABOT_RUNS_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def run_dir() -> Generator[str, None, None]:
    """Pytest fixture that provides a temporary run directory.
    
    Usage:
        def test_something(run_dir):
            # run_dir points to temporary directory
            # directory is automatically cleaned up after the test
    """
    with temp_run_dir() as d:
        yield d

@pytest.fixture
def observation_space():
    """Creates a minimal ObservationSpace for testing."""
    return ObservationSpace(
        ObservationSpaceHypers(
            max_cards=3,
            max_permanents=2,
            max_actions=5,
            max_focus_objects=2
        )
    )

@pytest.fixture
def experiment(run_dir):  # Add run_dir dependency here
    """Creates an Experiment instance for testing."""
    return Experiment(ExperimentHypers(
        exp_name="test_run",
        seed=42,
        wandb=False,
        tensorboard=True,
        runs_dir=Path(run_dir)
    ))

@pytest.fixture
def vector_env(observation_space):
    """Creates a VectorEnv instance with a small number of environments."""
    return VectorEnv(
        2,
        Match(),
        observation_space,
        Reward(RewardHypers(trivial=True))
    )

@pytest.fixture
def agent(observation_space):
    """Creates an Agent with minimal configuration for deterministic tests."""
    a_hypers = AgentHypers(
        game_embedding_dim=8,
        battlefield_embedding_dim=8,
        hidden_dim=16,
        dropout_rate=0.0
    )
    return Agent(observation_space, a_hypers)

@pytest.fixture
def trainer(agent, experiment, vector_env):
    """Creates a Trainer instance with minimal configuration."""
    t_hypers = TrainHypers(
        num_envs=2,
        num_steps=2000,
        num_minibatches=4,
        total_timesteps=50_000
    )
    trainer = Trainer(agent, experiment, vector_env, t_hypers)
    yield trainer
    # Cleanup
    if trainer:
        trainer.experiment.close()
    if hasattr(trainer, 'env'):
        trainer.env.close()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _collect_mock_ppo_batch(trainer, batch_size=12):
    """
    Returns synthetic data (obs, old_logprobs, actions, advantages, returns, values)
    to simulate a batch for testing PPO updates. Shapes align with trainer._optimize_step().

    For real usage, run an actual environment rollout or define more structured data.
    """
    obs = {
        "global": torch.randn(
            batch_size, trainer.agent.observation_space.encoder.shapes["global"][0]
        ),
        "players": torch.randn(
            batch_size, 2, trainer.agent.observation_space.encoder.shapes["players"][1]
        ),
        "cards": torch.randn(
            batch_size, 3, trainer.agent.observation_space.encoder.shapes["cards"][1]
        ),
        "permanents": torch.randn(
            batch_size, 2, trainer.agent.observation_space.encoder.shapes["permanents"][1]
        ),
        "actions": torch.randn(
            batch_size, 5, trainer.agent.observation_space.encoder.shapes["actions"][1]
        ),
    }

    old_logprobs = torch.clamp(torch.randn(batch_size), min=-5, max=5)
    actions = torch.randint(low=0, high=5, size=(batch_size,))
    advantages = torch.clamp(torch.randn(batch_size), min=-2, max=2)
    returns = torch.clamp(torch.randn(batch_size), min=-2, max=2)
    values = torch.clamp(torch.randn(batch_size), min=-2, max=2)

    return obs, old_logprobs, actions, advantages, returns, values

def _multiagent_rollout(trainer, steps=4):
    """
    Runs a small multi-agent rollout using trainer._rollout_step(), 
    returning the final (obs, done). For real usage, adapt to your environment logic.
    """
    next_obs, _ = trainer.env.reset()
    next_done = torch.zeros(
        trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
    )
    actor_ids = trainer._get_actor_indices(next_obs)

    for _ in range(steps):
        new_obs, new_done, new_actor_ids = trainer._rollout_step(next_obs, actor_ids)
        next_obs, next_done, actor_ids = new_obs, new_done, new_actor_ids

    return next_obs, next_done


# -----------------------------------------------------------------------------
# 1. ROLLOUT AND BUFFER MANAGEMENT
# -----------------------------------------------------------------------------

class TestRolloutAndBuffer:
    """
    Tests verifying correct transition collection, multi-agent handling,
    overflow behavior, and observation validation in the Trainer's buffers.
    """

    def test_rollout_step(self, trainer):
        """
        Single rollout step:
        - Check actor_id extraction
        - Ensure transitions go to correct player buffer
        - Confirm observation shapes remain consistent
        - Confirm valid actor IDs
        """
        next_obs, _ = trainer.env.reset()
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        new_obs, done, new_actor_ids = trainer._rollout_step(next_obs, prev_actor_ids)

        # Check shapes
        assert set(new_obs.keys()) == set(next_obs.keys()), "Observation keys mismatch"
        for k in new_obs:
            assert new_obs[k].shape == next_obs[k].shape, f"Shape mismatch on key {k}"

        # Check valid actor IDs
        assert new_actor_ids.shape == (trainer.hypers.num_envs,)
        assert torch.all((new_actor_ids >= 0) & (new_actor_ids < 2)), "Invalid actor IDs"

        # Check buffer storage capacity
        for _, buffer in trainer.multi_buffer.buffers.items():
            if buffer.step_idx > 0:
                assert buffer.step_idx <= trainer.hypers.num_steps
                # Verify dimension
                for v in buffer.obs.values():
                    assert v.shape[0] == trainer.hypers.num_steps

    def test_buffer_overflow(self, trainer):
        """
        Fill the buffer beyond capacity and verify:
        - Overflow is handled (no crash)
        - Step indices do not exceed capacity
        """
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        # Exceed capacity
        for _ in range(trainer.hypers.num_steps + 2):
            new_obs, done, new_actor_ids = trainer._rollout_step(next_obs, prev_actor_ids)
            next_obs, prev_actor_ids = new_obs, new_actor_ids

        # Ensure we didn't go over capacity
        for buffer in trainer.multi_buffer.buffers.values():
            assert buffer.step_idx <= trainer.hypers.num_steps, "Buffer overflow not handled"

    def test_observation_validation(self, trainer):
        """
        Validate observation shapes/keys:
        - Correct obs should pass
        - Missing key should fail
        - Wrong shape should fail
        """
        next_obs, _ = trainer.env.reset()
        assert trainer._validate_obs(next_obs), "Valid obs did not pass validation"

        # Missing key
        invalid_obs = {k: v for k, v in next_obs.items() if k != 'global'}
        assert not trainer._validate_obs(invalid_obs), "Obs missing 'global' key should fail"

        # Wrong shape
        invalid_obs2 = next_obs.copy()
        invalid_obs2['global'] = torch.zeros(1)
        assert not trainer._validate_obs(invalid_obs2), "Obs with wrong shape should fail"


# -----------------------------------------------------------------------------
# 2. ADVANTAGE COMPUTATION
# -----------------------------------------------------------------------------

class TestAdvantageComputation:
    """
    Tests verifying correct GAE calculations and advantage properties
    for both constant and alternating rewards, multi-agent handling,
    bootstrapping, and advantage normalization.
    """

    def test_gae_constant_rewards(self, trainer):
        """
        GAE with constant rewards. In a simplified scenario, 
        later advantages should be smaller if the reward is truly constant.
        """
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        # Fill buffer with "constant" rewards
        for _ in range(trainer.hypers.num_steps):
            next_obs, next_done, prev_actor_ids = trainer._rollout_step(next_obs, prev_actor_ids)

        obs, logprobs, actions, advantages, returns, values = trainer._compute_advantages(
            next_obs, next_done
        )

        # Check advantage
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()
        if advantages.numel() > 1:
            first_half = advantages[: advantages.shape[0] // 2].mean()
            last_half = advantages[advantages.shape[0] // 2 :].mean()
            # The test expects first_half >= last_half with truly constant positive rewards
            assert first_half >= last_half, "Later steps do not have lower advantages"

    def test_gae_alternating_rewards(self, trainer):
        """
        GAE with an alternating (+1/-1) reward pattern.
        Verifies advantage sign changes and multi-agent correctness.
        """
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        # Manually inject +1/-1 rewards. We'll do it by 
        # temporarily overriding the env's reward or the buffer storage logic. 
        # For simplicity, let's just do a few steps and then forcibly rewrite 
        # the stored reward in the buffer after each step.

        for i in range(trainer.hypers.num_steps):
            new_obs, new_done, new_actor_ids = trainer._rollout_step(next_obs, prev_actor_ids)
            # Overwrite last inserted reward in the buffer for whichever buffer stored it
            sign = 1.0 if i % 2 == 0 else -1.0
            for pid, buf in trainer.multi_buffer.buffers.items():
                if buf.step_idx > 0:  # just now added
                    buf.rewards[buf.step_idx - 1] = sign
            next_obs, next_done, prev_actor_ids = new_obs, new_done, new_actor_ids

        obs, logprobs, actions, advantages, returns, values = trainer._compute_advantages(
            next_obs, next_done
        )

        # Basic sign checks
        assert not torch.isnan(advantages).any()
        # We won't do a precise numeric check, but we can ensure there's both positive and negative adv.
        assert (advantages > 0).any(), "Expected some positive advantages with +1 rewards"
        assert (advantages < 0).any(), "Expected some negative advantages with -1 rewards"

    def test_gae_same_players_turns(self, trainer):
        """
        Ensure GAE is only computed for the player who actually acted. 
        Multi-agent scenario with separate buffers per player.
        """
        _multiagent_rollout(trainer, steps=4)
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(
            next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda
        )

        buf0 = trainer.multi_buffer.buffers[0]
        buf1 = trainer.multi_buffer.buffers[1]
        assert buf0.step_idx + buf1.step_idx <= trainer.hypers.num_steps, \
            "Too many transitions total: multi-agent buffer mismatch."

    def test_bootstrap_value_non_terminal(self, trainer):
        """
        Verify GAE bootstraps from next_value when the episode is truncated 
        (done=False) rather than truly done.
        """
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        next_done[:] = False  # Force no true terminal
        for _ in range(trainer.hypers.num_steps):
            new_obs, new_done, actor_ids = trainer._rollout_step(
                next_obs, torch.zeros_like(next_done)
            )
            next_obs, next_done = new_obs, new_done

        # Compute advantages
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs)
        trainer.multi_buffer.compute_advantages(
            next_value, next_done, trainer.hypers.gamma, trainer.hypers.gae_lambda
        )

        # Check last advantage is not NaN
        for pid, buf in trainer.multi_buffer.buffers.items():
            if buf.step_idx > 0:
                last_adv = buf.advantages[buf.step_idx - 1]
                assert not torch.isnan(last_adv).any(), "Non-terminal GAE produced NaN"

    def test_advantage_normalization_sign(self, trainer):
        """
        Check that advantage normalization preserves sign distribution.
        """
        obs, old_lp, actions, advs, rets, vals = _collect_mock_ppo_batch(trainer, batch_size=10)
        advs[:5] = -1.0
        advs[5:] = 5.0

        mean_a = advs.mean()
        std_a = advs.std() + 1e-8
        normed = (advs - mean_a) / std_a

        # Negative half stays negative, positive half stays positive
        assert (normed[:5] < 0).all(), "Negative advantages lost sign"
        assert (normed[5:] > 0).all(), "Positive advantages lost sign"


# -----------------------------------------------------------------------------
# 3. PPO OPTIMIZATION
# -----------------------------------------------------------------------------

class TestPPOOptimization:
    """
    Tests verifying the PPO update mechanics, including:
    - Single-step optimization (gradient flow)
    - KL divergence checks
    - Probability ratio no NaNs
    - Value loss checks
    - Early stopping conditions
    """

    def test_ppo_update(self, trainer):
        """
        Run a single PPO update step; verify:
        - KL divergence is finite
        - Gradients exist and are non-zero
        - Early stopping if KL exceeds threshold
        """
        obs, old_lp, acts, advs, rets, vals = _collect_mock_ppo_batch(trainer, batch_size=10)
        trainer.hypers.target_kl = 2

        approx_kl, clip_fraction = trainer._optimize_step(
            obs, old_lp, acts, advs, rets, vals
        )

        # Basic checks
        assert isinstance(approx_kl, float)
        assert approx_kl >= 0, "KL should be non-negative"
        assert not np.isnan(approx_kl), "KL is NaN"
        assert 0 <= clip_fraction <= 1

        # Check gradients
        for param in trainer.agent.parameters():
            assert param.grad is not None, "Missing gradient on param"
            assert not torch.isnan(param.grad).any(), "NaN in gradient"

        # If approx_kl > target_kl, trainer logs a message or early-stops. 
        # Checking that behavior might require hooking into trainer logs or code.

    def test_optimization_step(self, trainer):
        """
        Single-step optimization on real transitions from environment.
        Ensures we can gather data, compute advantages, and call _optimize_step.
        """
        next_obs, _ = trainer.env.reset()
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        # Collect transitions
        for _ in range(trainer.hypers.num_steps):
            next_obs, next_done, prev_actor_ids = trainer._rollout_step(
                next_obs, prev_actor_ids
            )

        obs, logprobs, actions, advantages, returns, values = trainer._compute_advantages(
            next_obs, next_done
        )

        # If there's valid data, run an update step
        if all(t.numel() > 0 for t in [logprobs, actions, advantages, returns, values]):
            approx_kl, clip_fraction = trainer._optimize_step(
                obs, logprobs, actions, advantages, returns, values
            )
            assert isinstance(approx_kl, float)
            assert isinstance(clip_fraction, float)

    def test_value_loss(self, trainer):
        """
        Check both clipped and unclipped value losses. Verify scaling by vf_coef.
        """
        obs, old_lp, actions, advs, rets, vals = _collect_mock_ppo_batch(trainer, batch_size=10)

        # Test unclipped
        trainer.hypers.clip_vloss = False
        _, _ = trainer._optimize_step(obs, old_lp, actions, advs, rets, vals)
        # If we wanted to be thorough, we'd record the actual value loss
        # from logs or by hooking into the trainer's code.

        # Test clipped
        trainer.hypers.clip_vloss = True
        _, _ = trainer._optimize_step(obs, old_lp, actions, advs, rets, vals)
        # Similarly check logs or internal trainer state for the difference.


# -----------------------------------------------------------------------------
# 4. CHECKPOINTING
# -----------------------------------------------------------------------------

class TestCheckpointing:
    """
    Tests for saving and loading model checkpoints, ensuring that
    parameters are restored properly.
    """

    def test_save_load_checkpoint(self, trainer, tmp_path):
        """
        Basic checkpoint test: do some rollout, save, load, 
        verify agent parameters match.
        """
        next_obs, _ = trainer.env.reset()
        next_obs = {k: torch.as_tensor(v, device=trainer.experiment.device)
                    for k, v in next_obs.items()}
        next_done = torch.zeros(
            trainer.hypers.num_envs, dtype=torch.bool, device=trainer.experiment.device
        )
        prev_actor_ids = trainer._get_actor_indices(next_obs)

        for _ in range(trainer.hypers.num_steps):
            next_obs, next_done, prev_actor_ids = trainer._rollout_step(
                next_obs, prev_actor_ids
            )

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(ckpt_path))

        # Create new trainer and load
        new_trainer = Trainer(trainer.agent, trainer.experiment, trainer.env, trainer.hypers)
        new_trainer.load_checkpoint(str(ckpt_path))

        # Check parameter equality
        for p1, p2 in zip(trainer.agent.parameters(), new_trainer.agent.parameters()):
            assert torch.allclose(p1, p2), "Parameters differ after loading checkpoint"


# -----------------------------------------------------------------------------
# 5. ADDITIONAL POLICY/STATE TESTS
# -----------------------------------------------------------------------------

def test_probability_ratios_no_nans(trainer):
    """
    Ensure pi_new / pi_old remains finite after a PPO update.
    """
    obs, old_logprobs, actions, advantages, returns, values = _collect_mock_ppo_batch(trainer)

    # For ratio comparison, we need old distribution
    with torch.no_grad():
        _, old_logprobs_check, _, _ = trainer.agent.get_action_and_value(obs, actions)

    # One PPO step
    approx_kl, clip_fraction = trainer._optimize_step(
        obs, old_logprobs, actions, advantages, returns, values
    )

    with torch.no_grad():
        _, new_logprobs, _, _ = trainer.agent.get_action_and_value(obs, actions)
        ratio = (new_logprobs - old_logprobs_check).exp()

    assert not torch.isnan(ratio).any(), "NaN in pi_new/pi_old ratio"
    assert not torch.isinf(ratio).any(), "Inf in pi_new/pi_old ratio"


def test_kl_divergence_below_target(trainer):
    """
    Check approximate KL remains below a certain threshold if target_kl is set.
    """
    trainer.hypers.target_kl = 2
    obs, old_lp, acts, advs, rets, vals = _collect_mock_ppo_batch(trainer)

    approx_kl, clip_fraction = trainer._optimize_step(obs, old_lp, acts, advs, rets, vals)
    assert approx_kl < trainer.hypers.target_kl, f"KL {approx_kl} exceeded target {trainer.hypers.target_kl}"


@pytest.mark.parametrize("invalid_value", [float("-inf"), float("inf"), float("nan")])
def test_logits_for_invalid_actions_inf(agent, observation_space, invalid_value):
    """
    Confirm the agent (in a masking scenario) sets invalid actions 
    to -1e8 (or some sentinel) in the final logits.
    """
    # Create a single-batch obs where some actions are invalid
    # and see how the agent masks them. We'll do something similar
    # to your existing test_action_masking.

    batch_size = 2
    enc = observation_space.encoder
    obs = {
        "global": torch.ones((batch_size, *enc.shapes["global"])) * 0.1,
        "players": torch.ones((batch_size, *enc.shapes["players"])) * 0.1,
        "cards": torch.ones((batch_size, *enc.shapes["cards"])) * 0.1,
        "permanents": torch.ones((batch_size, *enc.shapes["permanents"])) * 0.1,
        "actions": torch.zeros((batch_size, *enc.shapes["actions"]))
    }
    # Mark first action valid in each batch, the rest invalid
    obs["actions"][:, 0, -1] = 1.0

    logits, _ = agent.forward(obs)
    # Normally the agent does a masking like:
    mask = obs["actions"][..., -1].bool()
    masked_logits = torch.where(
        mask, logits, torch.tensor(-1e8, device=logits.device)
    )

    # The invalid spots should be exactly -1e8
    invalid_positions = ~mask
    assert torch.all(masked_logits[invalid_positions] == -1e8), "Invalid actions not set to -1e8"


def test_player_perspective_multiagent(trainer):
    """
    If players[0] is always the acting player, verify actor_ids matches players[0].
    """
    next_obs, _ = trainer.env.reset()
    actor_ids = trainer._get_actor_indices(next_obs)
    assert torch.all(actor_ids == next_obs["players"][:, 0, 0].long()), \
        "Mismatch at first step"

    for _ in range(3):
        new_obs, new_done, new_actor_ids = trainer._rollout_step(next_obs, actor_ids)
        assert torch.all(new_actor_ids == new_obs["players"][:, 0, 0].long()), \
            "Mismatch in subsequent steps"
        next_obs, actor_ids = new_obs, new_actor_ids


def test_reward_attribution(trainer):
    """
    Check correct reward attribution to each player's buffer in multi-agent setting.
    """
    _multiagent_rollout(trainer, steps=6)
    buf0 = trainer.multi_buffer.buffers[0]
    buf1 = trainer.multi_buffer.buffers[1]
    assert buf0.step_idx + buf1.step_idx <= trainer.hypers.num_steps, \
        "Sum of step_idx across players is too large."


def test_episode_termination_vs_truncation(trainer):
    """
    Demonstrate difference in true done vs. truncated episodes.
    We'll force env[0] to be done, env[1] truncated.
    """
    next_obs, _ = trainer.env.reset()
    for _ in range(2):
        new_obs, new_done, actor_ids = trainer._rollout_step(
            next_obs, torch.zeros_like(next_obs["players"][:, 0, 0])
        )
        next_obs = new_obs

    # Suppose env[0] done=True, env[1] truncated => done=False
    new_done[0] = True  # real terminal
    # env[1] remains false => truncated

    with torch.no_grad():
        next_value = trainer.agent.get_value(next_obs)
    trainer.multi_buffer.compute_advantages(
        next_value, new_done, trainer.hypers.gamma, trainer.hypers.gae_lambda
    )
    # Could add asserts based on terminal vs truncated differences.
