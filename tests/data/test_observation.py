"""Tests for data structures and observation encoding."""

import unittest

import numpy as np
import managym

from manabot.data import (
    ObservationSpace,
    ObservationEncoder,
    PhaseEnum,
    StepEnum,
    ActionEnum,
    ZoneEnum,
)
from manabot.data.observation import MAX_LIFE, MAX_TURNS

class TestEnumParity(unittest.TestCase):
    """Compare Python enums with managym C++ enums."""
    
    def test_zone_enum(self):
        self.assertEqual(int(managym.ZoneEnum.LIBRARY), ZoneEnum.LIBRARY)
        self.assertEqual(int(managym.ZoneEnum.HAND), ZoneEnum.HAND)
        self.assertEqual(int(managym.ZoneEnum.BATTLEFIELD), ZoneEnum.BATTLEFIELD)
        self.assertEqual(int(managym.ZoneEnum.GRAVEYARD), ZoneEnum.GRAVEYARD)
        self.assertEqual(int(managym.ZoneEnum.STACK), ZoneEnum.STACK)
        self.assertEqual(int(managym.ZoneEnum.EXILE), ZoneEnum.EXILE)
        self.assertEqual(int(managym.ZoneEnum.COMMAND), ZoneEnum.COMMAND)

    def test_phase_enum(self):
        self.assertEqual(int(managym.PhaseEnum.BEGINNING), PhaseEnum.BEGINNING)
        self.assertEqual(int(managym.PhaseEnum.PRECOMBAT_MAIN), PhaseEnum.PRECOMBAT_MAIN)
        self.assertEqual(int(managym.PhaseEnum.COMBAT), PhaseEnum.COMBAT)
        self.assertEqual(int(managym.PhaseEnum.POSTCOMBAT_MAIN), PhaseEnum.POSTCOMBAT_MAIN)
        self.assertEqual(int(managym.PhaseEnum.ENDING), PhaseEnum.ENDING)

    def test_step_enum(self): 
        self.assertEqual(int(managym.StepEnum.BEGINNING_UNTAP), StepEnum.BEGINNING_UNTAP)
        self.assertEqual(int(managym.StepEnum.BEGINNING_UPKEEP), StepEnum.BEGINNING_UPKEEP)
        self.assertEqual(int(managym.StepEnum.BEGINNING_DRAW), StepEnum.BEGINNING_DRAW)
        self.assertEqual(int(managym.StepEnum.PRECOMBAT_MAIN_STEP), StepEnum.PRECOMBAT_MAIN_STEP)
        self.assertEqual(int(managym.StepEnum.COMBAT_BEGIN), StepEnum.COMBAT_BEGIN)
        self.assertEqual(int(managym.StepEnum.COMBAT_DECLARE_ATTACKERS), StepEnum.COMBAT_DECLARE_ATTACKERS)
        self.assertEqual(int(managym.StepEnum.COMBAT_DECLARE_BLOCKERS), StepEnum.COMBAT_DECLARE_BLOCKERS)
        self.assertEqual(int(managym.StepEnum.COMBAT_DAMAGE), StepEnum.COMBAT_DAMAGE)
        self.assertEqual(int(managym.StepEnum.COMBAT_END), StepEnum.COMBAT_END)
        self.assertEqual(int(managym.StepEnum.POSTCOMBAT_MAIN_STEP), StepEnum.POSTCOMBAT_MAIN_STEP)
        self.assertEqual(int(managym.StepEnum.ENDING_END), StepEnum.ENDING_END)
        self.assertEqual(int(managym.StepEnum.ENDING_CLEANUP), StepEnum.ENDING_CLEANUP)

    def test_action_enum(self):
        self.assertEqual(int(managym.ActionEnum.PRIORITY_PLAY_LAND), ActionEnum.PRIORITY_PLAY_LAND)
        self.assertEqual(int(managym.ActionEnum.PRIORITY_CAST_SPELL), ActionEnum.PRIORITY_CAST_SPELL)
        self.assertEqual(int(managym.ActionEnum.PRIORITY_PASS_PRIORITY), ActionEnum.PRIORITY_PASS_PRIORITY)
        self.assertEqual(int(managym.ActionEnum.DECLARE_ATTACKER), ActionEnum.DECLARE_ATTACKER)
        self.assertEqual(int(managym.ActionEnum.DECLARE_BLOCKER), ActionEnum.DECLARE_BLOCKER)

class TestObservationEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = ObservationEncoder(
            num_players=2,
            max_cards=3,
            max_permanents=3,
            max_actions=2,
            max_focus_objects=2,
        )
        
        self.env = managym.Env()
        self.player_a = managym.PlayerConfig("Alice", {"Mountain": 20, "Grey Ogre": 40})
        self.player_b = managym.PlayerConfig("Bob", {"Forest": 20, "Llanowar Elves": 40})

    def test_shapes(self):
        shapes = self.encoder.shapes
        self.assertEqual(shapes['global'], (1 + len(PhaseEnum) + len(StepEnum) + 2,))
        self.assertEqual(shapes['players'], (2, 1 + 2 + len(ZoneEnum)))
        self.assertEqual(shapes['cards'], (3, self.encoder.card_dim))
        self.assertEqual(shapes['permanents'], (3, self.encoder.permanent_dim))
        self.assertEqual(shapes['actions'], (2, self.encoder.action_dim))

    def test_encode_from_managym(self):
        cpp_obs, _ = self.env.reset([self.player_a, self.player_b])
        tensors = self.encoder.encode(cpp_obs)
        
        self.assertIsInstance(tensors, dict)
        self.assertEqual(set(tensors.keys()), {'global', 'players', 'cards', 'permanents', 'actions'})
        
        for name, shape in self.encoder.shapes.items():
            self.assertEqual(tensors[name].shape, shape)

        # Basic value checks
        self.assertEqual(tensors['global'][0], 1.0)  # First turn  
        self.assertGreaterEqual(tensors['players'][0,0], 20.0)  # Starting life
        self.assertLessEqual(tensors['players'][0,0], MAX_LIFE)  # Life within bounds

    def test_encode_focus_object(self):
        cpp_obs, _ = self.env.reset([self.player_a, self.player_b])
        
        # Player focus
        focus = self.encoder.encode_focus_object(cpp_obs, 0)
        self.assertEqual(len(focus), self.encoder.focus_dim)
        self.assertEqual(focus[0], 20.0)  # Life total
        
        # Card focus 
        card_id = next(iter(cpp_obs.cards.keys()))
        focus = self.encoder.encode_focus_object(cpp_obs, card_id)
        self.assertEqual(len(focus), self.encoder.focus_dim)
        
        # Permanent focus (if battlefield has permanents)
        if cpp_obs.permanents:
            perm_id = next(iter(cpp_obs.permanents.keys()))
            focus = self.encoder.encode_focus_object(cpp_obs, perm_id)
            self.assertEqual(len(focus), self.encoder.focus_dim)

        # Invalid object should return zeros
        focus = self.encoder.encode_focus_object(cpp_obs, 99999)
        self.assertTrue(np.all(focus == 0))

class TestObservationSpace(unittest.TestCase):
    def setUp(self):
        self.encoder = ObservationEncoder(
            num_players=2,
            max_cards=3,
            max_permanents=3,
            max_actions=2, 
            max_focus_objects=2,
        )
        self.space = ObservationSpace(self.encoder)
        self.env = managym.Env()
        self.player_a = managym.PlayerConfig("Alice", {"Mountain": 20, "Grey Ogre": 40}) 
        self.player_b = managym.PlayerConfig("Bob", {"Forest": 20, "Llanowar Elves": 40})

    def test_sample(self):
        sample = self.space.sample()
        
        self.assertIsInstance(sample, dict)
        self.assertEqual(set(sample.keys()), {'global', 'players', 'cards', 'permanents', 'actions'})
        
        # Check shapes and bounds
        for name, shape in self.encoder.shapes.items():
            self.assertEqual(sample[name].shape, shape)

        self.assertLessEqual(sample['global'][0], MAX_TURNS)
        self.assertLessEqual(np.max(sample['players'][:, 0]), MAX_LIFE)

    def test_contains(self):
        sample = self.space.sample()
        self.assertTrue(self.space.contains(sample))

        # Invalid shapes
        bad_sample = {k: v[:-1] for k, v in sample.items()}
        self.assertFalse(self.space.contains(bad_sample))


    def test_encode_from_managym(self):
        cpp_obs, _ = self.env.reset([self.player_a, self.player_b])
        tensors = self.encoder.encode(cpp_obs)
        self.assertTrue(self.space.contains(tensors))

if __name__ == '__main__':
    unittest.main()