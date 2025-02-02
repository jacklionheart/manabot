"""
test_observation.py
Tests for data structures and observation encoding.

This test suite verifies:
1. Parity between Python and C++ enums for game state representation
2. Observation encoding for the neural network
3. Observation space sampling and validation
"""

import pytest
import numpy as np
import managym
import torch
from typing import Dict, Tuple

from manabot.env.observation import (
    ObservationSpace,
    ObservationSpaceHypers,
    ObservationEncoder,
    PhaseEnum,
    StepEnum,
    ActionEnum,
    ZoneEnum,
)

# Fixtures for common test setup
@pytest.fixture(scope="session")
def hypers():
    """Create an encoder with minimal dimensions for testing.
    
    We use session scope since the encoder is stateless and can be reused.
    Dimensions are chosen to be small enough for fast testing but large
    enough to catch potential issues.
    """
    return ObservationSpaceHypers(
        max_cards=3,
        max_permanents=3,
        max_actions=2,
        max_focus_objects=2,
    )

@pytest.fixture(scope="session")
def encoder(hypers):
    """Create an observation encoder for testing."""
    return ObservationEncoder(hypers)

@pytest.fixture(scope="session")
def observation_space(hypers):
    """Create an observation space using our test encoder."""
    return ObservationSpace(hypers)

@pytest.fixture(scope="session")
def player_configs() -> Tuple[managym.PlayerConfig, managym.PlayerConfig]:
    """Create consistent player configurations for testing.
    
    Using simple decks with known properties makes tests more predictable
    and easier to debug.
    """
    player_a = managym.PlayerConfig("Alice", {"Mountain": 20, "Grey Ogre": 40})
    player_b = managym.PlayerConfig("Bob", {"Forest": 20, "Llanowar Elves": 40})
    return player_a, player_b

@pytest.fixture
def env():
    """Create a fresh environment for each test.
    
    We use function scope to ensure each test starts with a clean environment.
    """
    env = managym.Env()
    yield env

@pytest.fixture
def initial_observation(env, player_configs) -> Tuple[managym.Observation, Dict]:
    """Get the initial observation from a fresh environment."""
    return env.reset(list(player_configs))

class TestEnumParity:
    """Verify that our Python enums match the C++ implementation exactly."""
    
    @pytest.mark.parametrize("py_enum, cpp_enum", [
        (ZoneEnum.LIBRARY, managym.ZoneEnum.LIBRARY),
        (ZoneEnum.HAND, managym.ZoneEnum.HAND),
        (ZoneEnum.BATTLEFIELD, managym.ZoneEnum.BATTLEFIELD),
        (ZoneEnum.GRAVEYARD, managym.ZoneEnum.GRAVEYARD),
        (ZoneEnum.STACK, managym.ZoneEnum.STACK),
        (ZoneEnum.EXILE, managym.ZoneEnum.EXILE),
        (ZoneEnum.COMMAND, managym.ZoneEnum.COMMAND),
    ])
    def test_zone_enum(self, py_enum, cpp_enum):
        """Test zone enum parity one value at a time for clear error reporting."""
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize("py_enum, cpp_enum", [
        (PhaseEnum.BEGINNING, managym.PhaseEnum.BEGINNING),
        (PhaseEnum.PRECOMBAT_MAIN, managym.PhaseEnum.PRECOMBAT_MAIN),
        (PhaseEnum.COMBAT, managym.PhaseEnum.COMBAT),
        (PhaseEnum.POSTCOMBAT_MAIN, managym.PhaseEnum.POSTCOMBAT_MAIN),
        (PhaseEnum.ENDING, managym.PhaseEnum.ENDING),
    ])
    def test_phase_enum(self, py_enum, cpp_enum):
        """Test phase enum parity."""
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize("py_enum, cpp_enum", [
        (StepEnum.BEGINNING_UNTAP, managym.StepEnum.BEGINNING_UNTAP),
        (StepEnum.BEGINNING_UPKEEP, managym.StepEnum.BEGINNING_UPKEEP),
        (StepEnum.BEGINNING_DRAW, managym.StepEnum.BEGINNING_DRAW),
        (StepEnum.PRECOMBAT_MAIN_STEP, managym.StepEnum.PRECOMBAT_MAIN_STEP),
        (StepEnum.COMBAT_BEGIN, managym.StepEnum.COMBAT_BEGIN),
        (StepEnum.COMBAT_DECLARE_ATTACKERS, managym.StepEnum.COMBAT_DECLARE_ATTACKERS),
        (StepEnum.COMBAT_DECLARE_BLOCKERS, managym.StepEnum.COMBAT_DECLARE_BLOCKERS),
        (StepEnum.COMBAT_DAMAGE, managym.StepEnum.COMBAT_DAMAGE),
        (StepEnum.COMBAT_END, managym.StepEnum.COMBAT_END),
        (StepEnum.POSTCOMBAT_MAIN_STEP, managym.StepEnum.POSTCOMBAT_MAIN_STEP),
        (StepEnum.ENDING_END, managym.StepEnum.ENDING_END),
        (StepEnum.ENDING_CLEANUP, managym.StepEnum.ENDING_CLEANUP),
    ])
    def test_step_enum(self, py_enum, cpp_enum):
        """Test step enum parity."""
        assert int(cpp_enum) == py_enum

    @pytest.mark.parametrize("py_enum, cpp_enum", [
        (ActionEnum.PRIORITY_PLAY_LAND, managym.ActionEnum.PRIORITY_PLAY_LAND),
        (ActionEnum.PRIORITY_CAST_SPELL, managym.ActionEnum.PRIORITY_CAST_SPELL),
        (ActionEnum.PRIORITY_PASS_PRIORITY, managym.ActionEnum.PRIORITY_PASS_PRIORITY),
        (ActionEnum.DECLARE_ATTACKER, managym.ActionEnum.DECLARE_ATTACKER),
        (ActionEnum.DECLARE_BLOCKER, managym.ActionEnum.DECLARE_BLOCKER),
    ])
    def test_action_enum(self, py_enum, cpp_enum):
        """Test action enum parity."""
        assert int(cpp_enum) == py_enum

class TestObservationEncoder:
    """Test the encoding of game state observations into tensors."""

    def test_shapes(self, encoder):
        """Verify that encoder produces tensors with correct shapes."""
        shapes = encoder.shapes
        
        # Global state shape includes turn number, phase/step one-hot, and game outcome
        assert shapes['global'] == (1 + len(PhaseEnum) + len(StepEnum) + 2,)
        
        assert shapes['players'] == (2, 5 + len(ZoneEnum))
        
        # Verify card, permanent, and action shapes match encoder settings
        assert shapes['cards'] == (3, encoder.card_dim)
        assert shapes['permanents'] == (3, encoder.permanent_dim)
        assert shapes['actions'] == (2, encoder.action_dim)

    def test_encode_from_managym(self, encoder, initial_observation):
        """Test encoding of actual managym observations."""
        cpp_obs, _ = initial_observation
        tensors = encoder.encode(cpp_obs)
        
        # Verify dictionary structure
        assert isinstance(tensors, dict)
        assert set(tensors.keys()) == {'global', 'players', 'cards', 'permanents', 'actions'}
        
        # Verify shapes match encoder specifications
        for name, shape in encoder.shapes.items():
            assert tensors[name].shape == shape
            # Verify no NaN/Inf values
            assert not np.isnan(tensors[name]).any(), f"NaN values in {name}"
            assert not np.isinf(tensors[name]).any(), f"Inf values in {name}"

    def test_encode_focus_object(self, encoder, initial_observation):
        """Test encoding of individual game objects (players, cards, permanents)."""
        cpp_obs, _ = initial_observation
        
        # Test player focus encoding
        focus = encoder.encode_focus_object(cpp_obs, 0)
        assert len(focus) == encoder.focus_dim
        assert focus[2] == 20.0, "Initial life total should be 20"
        
        # Test card focus encoding (if cards exist)
        if cpp_obs.cards:
            card_id = next(iter(cpp_obs.cards.keys()))
            focus = encoder.encode_focus_object(cpp_obs, card_id)
            assert len(focus) == encoder.focus_dim
            assert not np.isnan(focus).any(), "NaN values in card encoding"
        
        # Test permanent focus encoding (if permanents exist)
        if cpp_obs.permanents:
            perm_id = next(iter(cpp_obs.permanents.keys()))
            focus = encoder.encode_focus_object(cpp_obs, perm_id)
            assert len(focus) == encoder.focus_dim
            assert not np.isnan(focus).any(), "NaN values in permanent encoding"

        # Test invalid object handling
        focus = encoder.encode_focus_object(cpp_obs, 99999)
        assert np.all(focus == 0), "Invalid object should encode to zeros"

class TestObservationSpace:
    """Test the observation space specification and sampling."""

    def test_sample(self, observation_space, encoder):
        """Test observation sampling and bounds."""
        sample = observation_space.sample()
        
        # Verify dictionary structure
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {'global', 'players', 'cards', 'permanents', 'actions'}
        
        # Verify shapes match encoder
        for name, shape in encoder.shapes.items():
            assert sample[name].shape == shape
            # Verify no NaN/Inf values
            assert not np.isnan(sample[name]).any(), f"NaN values in {name}"
            assert not np.isinf(sample[name]).any(), f"Inf values in {name}"


    def test_contains(self, observation_space):
        """Test observation space membership checking."""
        # Valid sample should be contained
        sample = observation_space.sample()
        assert observation_space.contains(sample)

        # Test invalid shapes
        bad_sample = {k: v[:-1] for k, v in sample.items()}
        assert not observation_space.contains(bad_sample)


    def test_encode_from_managym(self, observation_space, initial_observation):
        """Test that encoded observations are valid members of the space."""
        cpp_obs, _ = initial_observation
        tensors = observation_space.encode(cpp_obs)
        assert observation_space.contains(tensors)

if __name__ == '__main__':
    pytest.main([__file__])