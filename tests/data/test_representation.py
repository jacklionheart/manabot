import unittest
import warnings
import numpy as np
import torch
from typing import Dict

from manabot.data import (
    Observation,
    Player,
    Card,
    CardTypes,
    ManaCost,
    Permanent,
    Turn,
    PhaseEnum,
    StepEnum,
    ActionSpace,
    Action,
    ActionEnum,
    ActionSpaceEnum,
    ZoneEnum,
    InputTensorSpace
)


class TestRepresentation(unittest.TestCase):
    def setUp(self):
        """Create a small InputTensorSpace for testing."""
        self.obs_space = InputTensorSpace(
            num_players=2,
            max_cards=3,
            max_permanents=3,
            max_actions=2,
            max_focus_objects=2
        )
        # We'll use the automatic dimension calculations rather than hard-coding them

    def _make_minimal_observation(self) -> Observation:
        """Create a minimal but complete test observation."""
        py_obs = Observation()

        # Top-level state
        py_obs.game_over = False
        py_obs.won = False
        py_obs.turn = Turn(
            turn_number=3,
            phase=PhaseEnum.PRECOMBAT_MAIN,
            step=StepEnum.PRECOMBAT_MAIN_STEP,
            active_player_id=0,
            agent_player_id=0
        )

        # Players with distinct features to test encoding
        py_obs.players = {
            0: Player(id=0, player_index=0, is_agent=True, is_active=True, 
                     life=20, zone_counts=[1,0,0,0,0,0,0]),
            1: Player(id=1, player_index=1, is_agent=False, is_active=False, 
                     life=18, zone_counts=[1,1,0,0,0,0,0])
        }

        # Cards with varied zones and types
        card_types1 = CardTypes(
            is_castable=True, is_permanent=False, is_non_land_permanent=False,
            is_non_creature_permanent=False, is_spell=True, is_creature=True,
            is_land=False, is_planeswalker=False, is_enchantment=False,
            is_artifact=False, is_kindred=False, is_battle=False
        )
        card_types2 = CardTypes(
            is_castable=False, is_permanent=True, is_non_land_permanent=False,
            is_non_creature_permanent=True, is_spell=False, is_creature=False,
            is_land=True, is_planeswalker=False, is_enchantment=False,
            is_artifact=False, is_kindred=False, is_battle=False
        )
        
        py_obs.cards = {
            101: Card(zone=ZoneEnum.HAND, owner_id=0, id=101, registry_key=9999,
                     power=2, toughness=2, card_types=card_types1,
                     mana_cost=ManaCost(cost=[1,0,0,0,0], mana_value=1)),
            102: Card(zone=ZoneEnum.BATTLEFIELD, owner_id=1, id=102, registry_key=8888,
                     power=0, toughness=0, card_types=card_types2,
                     mana_cost=ManaCost(cost=[0,0,0], mana_value=0))
        }

        # Permanents with different characteristics
        py_obs.permanents = {
            200: Permanent(id=200, controller_id=0, tapped=False, damage=0,
                         is_creature=True, is_land=False, is_summoning_sick=False),
            201: Permanent(id=201, controller_id=1, tapped=True, damage=1,
                         is_creature=False, is_land=True, is_summoning_sick=True)
        }

        # Action space with focus objects of different types
        py_obs.action_space = ActionSpace(
            action_space_type=ActionSpaceEnum.PRIORITY,
            actions=[
                Action(action_type=ActionEnum.PRIORITY_CAST_SPELL, focus=[101]),  # focus on a card
                Action(action_type=ActionEnum.DECLARE_ATTACKER, focus=[200]),     # focus on a permanent
                Action(action_type=ActionEnum.PRIORITY_PASS_PRIORITY, focus=[0])  # focus on a player
            ],
            focus=[]
        )

        return py_obs

    def test_shape_dictionary(self):
        """Verify that the shapes property provides correct tensor shapes."""
        shapes = self.obs_space.shapes
        
        # Check that we have exactly the expected keys
        expected_keys = {'global', 'players', 'cards', 'permanents', 'actions'}
        self.assertEqual(set(shapes.keys()), expected_keys)
        
        # Verify each shape matches our configured dimensions
        self.assertEqual(shapes['global'], (self.obs_space.global_dim,))
        self.assertEqual(shapes['players'], (self.obs_space.num_players, self.obs_space.player_dim))
        self.assertEqual(shapes['cards'], (self.obs_space.max_cards, self.obs_space.card_dim))
        self.assertEqual(shapes['permanents'], (self.obs_space.max_permanents, self.obs_space.permanent_dim))
        self.assertEqual(shapes['actions'], (self.obs_space.max_actions, self.obs_space.action_dim))

    def test_encode_focus_object(self):
        """Test that focus objects are encoded correctly based on their type."""
        obs = self._make_minimal_observation()
        
        # Test player encoding (id=0)
        player_encoding = self.obs_space.encode_focus_object(obs, 0)
        self.assertEqual(len(player_encoding), self.obs_space.focus_object_dim)
        
        # Player features should be at start, other sections should be zero
        self.assertEqual(player_encoding[0], 20.0)  # life total
        self.assertEqual(player_encoding[1], 1.0)   # is_active
        self.assertEqual(player_encoding[2], 1.0)   # is_agent
        self.assertEqual(player_encoding[3], 1.0)   # First zone count
        
        # Verify card and permanent sections are zero for player encoding
        card_start = self.obs_space.player_dim
        perm_start = card_start + self.obs_space.card_dim
        self.assertTrue(np.all(player_encoding[card_start:perm_start] == 0))
        self.assertTrue(np.all(player_encoding[perm_start:] == 0))
        
        # Test card encoding (id=101)
        card_encoding = self.obs_space.encode_focus_object(obs, 101)
        self.assertEqual(len(card_encoding), self.obs_space.focus_object_dim)
        
        # Player section should be zero
        self.assertTrue(np.all(card_encoding[:card_start] == 0))
        
        # Card features should be present in middle section
        card_section = card_encoding[card_start:perm_start]
        zone_idx = ZoneEnum.HAND.value
        self.assertEqual(card_section[zone_idx], 1.0)  # Zone one-hot for HAND
        offset = self.obs_space.zone_count
        self.assertEqual(card_section[offset], 0.0)    # owner_id
        self.assertEqual(card_section[offset + 1], 2.0)  # power
        self.assertEqual(card_section[offset + 2], 2.0)  # toughness
        
        # Permanent section should be zero
        self.assertTrue(np.all(card_encoding[perm_start:] == 0))
        
        # Test permanent encoding (id=200)
        perm_encoding = self.obs_space.encode_focus_object(obs, 200)
        self.assertEqual(len(perm_encoding), self.obs_space.focus_object_dim)
        
        # Earlier sections should be zero
        self.assertTrue(np.all(perm_encoding[:perm_start] == 0))
        
        # Permanent features should be present
        perm_section = perm_encoding[perm_start:]
        self.assertEqual(perm_section[0], 0.0)  # controller_id
        self.assertEqual(perm_section[1], 0.0)  # not tapped
        self.assertEqual(perm_section[5], 0.0)  # not summoning sick
        self.assertEqual(perm_section[7], 1.0)  # is_creature

        # Test invalid ID
        invalid_encoding = self.obs_space.encode_focus_object(obs, 99999)
        self.assertTrue(np.all(invalid_encoding == 0))
        
        # Test card encoding
        card_encoding = self.obs_space.encode_focus_object(obs, 101)  # card id 101
        self.assertEqual(len(card_encoding), self.obs_space.focus_object_dim)
        # Verify card section is populated (power=2)
        offset = self.obs_space.player_dim
        self.assertEqual(card_encoding[offset + self.obs_space.zone_count + 2], 2.0)
        
        # Test permanent encoding
        perm_encoding = self.obs_space.encode_focus_object(obs, 200)  # permanent id 200
        self.assertEqual(len(perm_encoding), self.obs_space.focus_object_dim)
        # Verify permanent section is populated (controller_id=0)
        offset = self.obs_space.player_dim + self.obs_space.card_dim
        self.assertEqual(perm_encoding[offset], 0.0)

    def test_shape_validation(self):
        """Test that shape validation catches mismatches and allows correct shapes."""
        obs = self._make_minimal_observation()
        
        # Test successful encoding
        encoded = self.obs_space.encode_full_observation(obs)
        for name, shape in self.obs_space.shapes.items():
            self.assertEqual(encoded[name].shape, shape)
            
        # Test overflow warnings
        obs_overflow = self._make_minimal_observation()
        # Add extra cards beyond max_cards
        for i in range(10):
            obs_overflow.cards[1000 + i] = obs_overflow.cards[101]
        
        with self.assertWarns(Warning):
            encoded = self.obs_space.encode_full_observation(obs_overflow)
            self.assertEqual(encoded['cards'].shape, self.obs_space.shapes['cards'])

    def test_action_encoding(self):
        """Test that actions are encoded with their focus objects correctly."""
        obs = self._make_minimal_observation()
        encoded = self.obs_space.encode_full_observation(obs)
        
        # Get the actions tensor
        actions_tensor = encoded['actions']
        
        # Check shape includes space for focus objects
        expected_action_dim = (self.obs_space.action_count + 
                             (self.obs_space.max_focus_objects * self.obs_space.focus_object_dim))
        self.assertEqual(actions_tensor.shape, (self.obs_space.max_actions, expected_action_dim))
        
        # Verify action type one-hot encoding
        first_action = actions_tensor[0]
        action_type_onehot = first_action[:self.obs_space.action_count]
        self.assertEqual(torch.argmax(action_type_onehot).item(), 
                        ActionEnum.PRIORITY_CAST_SPELL)

    def test_invalid_focus_object(self):
        """Test handling of invalid focus object IDs."""
        obs = self._make_minimal_observation()
        
        # Test encoding of non-existent object ID
        invalid_encoding = self.obs_space.encode_focus_object(obs, 99999)
        self.assertTrue(np.all(invalid_encoding == 0))
        
        # Add an action with invalid focus
        obs.action_space.actions.append(
            Action(action_type=ActionEnum.PRIORITY_PASS_PRIORITY, focus=[99999])
        )
        
        # Should encode without error, invalid focus becomes zero vector
        encoded = self.obs_space.encode_full_observation(obs)
        self.assertIn('actions', encoded)