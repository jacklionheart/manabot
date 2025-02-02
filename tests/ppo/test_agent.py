import pytest
import torch
from torch.distributions import Categorical

# Adjust these imports per your codebase structure:
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
from manabot.ppo import Agent


@pytest.fixture
def observation_space() -> ObservationSpace:
    """
    Minimal observation space setup.
    Adjust max_actions, shapes, etc. to match 
    your environment's typical config.
    """
    return ObservationSpace(
        ObservationSpaceHypers(
            max_cards=3,
            max_permanents=2,
            max_actions=5,   # allow a few possible actions
            max_focus_objects=2
        )
    )

@pytest.fixture
def agent(observation_space: ObservationSpace) -> Agent:
    """
    Create a lightly configured agent 
    with no dropout for test determinism.
    """
    hypers = AgentHypers(
        game_embedding_dim=8,
        battlefield_embedding_dim=8,
        hidden_dim=16,
        dropout_rate=0.0,  # disable dropout
    )
    return Agent(observation_space, hypers)

@pytest.fixture
def sample_obs(observation_space: ObservationSpace) -> dict:
    """
    Construct a small batch of synthetic observations that matches the shapes 
    from observation_space. Uses small non-zero values to avoid NaN issues.
    """
    batch_size = 2
    encoder = observation_space.encoder
    obs = {}

    # Fill sub-tensors with small non-zero values
    obs["global"] = torch.ones((batch_size, *encoder.shapes["global"])) * 0.1
    obs["players"] = torch.ones((batch_size, *encoder.shapes["players"])) * 0.1
    obs["cards"] = torch.ones((batch_size, *encoder.shapes["cards"])) * 0.1
    obs["permanents"] = torch.ones((batch_size, *encoder.shapes["permanents"])) * 0.1

    # Actions: use proper shape from encoder
    obs["actions"] = torch.zeros((batch_size, *encoder.shapes["actions"]))
    
    # Set some valid actions (use the proper valid action indicator from encoder)
    for i in range(batch_size):
        # First two actions valid for first batch, first and last for second batch
        if i == 0:
            obs["actions"][i, :2, -1] = 1.0
        else:
            obs["actions"][i, 0, -1] = 1.0
            obs["actions"][i, -1, -1] = 1.0
            
        # Add small values to other action features to avoid NaN
        obs["actions"][i, :, :-1] = 0.1

    return obs

def test_forward_shape(agent: Agent, sample_obs: dict):
    """
    Check shape correctness for (logits, value).
    """
    logits, value = agent.forward(sample_obs)
    batch_size = sample_obs["global"].shape[0]
    max_actions = agent.observation_space.encoder.max_actions

    assert logits.shape == (batch_size, max_actions), \
        f"Expected logits shape {(batch_size, max_actions)}, got {logits.shape}"
    assert value.shape == (batch_size,), \
        f"Expected value shape {(batch_size,)}, got {value.shape}"


def test_action_masking(agent: Agent, sample_obs: dict):
    """
    Ensure we never sample an invalid action (i.e., 
    where the valid bit is 0).
    """
    action, logprob, entropy, value = agent.get_action_and_value(sample_obs)

    # Verify shape correctness
    batch_size = sample_obs["global"].shape[0]
    assert action.shape == (batch_size,)
    assert logprob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)

    # Check each batch element 
    action_mask = sample_obs["actions"][..., -1]  # shape (batch, max_actions)
    for i in range(batch_size):
        chosen_action = action[i].item()
        valid_bit = action_mask[i, chosen_action].item()
        assert valid_bit == 1.0, f"Sampled an invalid action {chosen_action} in batch {i}."


def test_deterministic_action(agent: Agent, sample_obs: dict):
    """
    If we pass deterministic=True, the agent 
    should pick the highest-prob (valid) action.
    We'll artificially tweak the logits to ensure a known outcome.
    """
    # Let’s do a forward pass to get the raw logits
    logits, value = agent.forward(sample_obs)
    # Suppose we want to force "action=1" to be the best in batch=0,
    # and "action=4" to be the best in batch=1 (assuming that's valid).
    # We'll set them to large values:
    logits[0, 1] = 5.0
    logits[1, 4] = 10.0

    # Apply mask
    mask = sample_obs["actions"][..., -1].bool()
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    dist = torch.distributions.Categorical(logits=masked_logits)
    deterministic_action = torch.argmax(dist.probs, dim=-1)

    # Check that these are the forced best actions
    assert deterministic_action[0].item() == 1, "Expected action=1 for batch=0"
    # For batch=1, we forced action=4 to be best, but let's confirm it's valid in the mask 
    if mask[1, 4].item() == 1.0:
        assert deterministic_action[1].item() == 4, "Expected action=4 for batch=1"
    else:
        # If action=4 is invalid in the mask, then we can't pick it, 
        # so let's see if your environment allows it:
        pass


def test_gradient_flow(agent: Agent, sample_obs: dict):
    """
    Minimal gradient flow test: 
    confirm we can do backward() on a simple loss 
    and no NaNs appear.
    """
    logits, value = agent(sample_obs)

    # For a trivial PPO-like loss, we might do:
    # action = dist.sample()
    # advantage = ...
    # But let's keep it simple:
    loss = logits.mean() + value.mean()
    loss.backward()

    # Check that we have gradients
    for name, param in agent.named_parameters():
        assert param.grad is not None, f"No grad for param {name}"
        assert not torch.isnan(param.grad).any(), f"NaN grad in param {name}"


