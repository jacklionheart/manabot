# test_data.py

import unittest
from typing import Dict
from dataclasses import dataclass, field

from .data import *

def create_minimal_observation() -> Observation:
    """
    Helper to build a minimal but valid Observation for testing.
    This single function can be reused in many test cases,
    reducing maintenance overhead if the environment structure changes.
    """
    # Create 1 player
    players = {
        1: PlayerData(
            player_id=1, player_index=0, is_active=True, has_priority=True,
            life=20, zone_counts=[40, 7, 3, 0, 0, 0, 0], mana=[2, 1, 0, 0, 3, 0]
        )
    }

    # 2 objects: one in hand as a card, one on battlefield as a permanent
    objects = {
        100: BaseObjectData(
            id=100, zone=ZoneType.HAND, owner_id=1, controller_id=1
        ),
        101: BaseObjectData(
            id=101, zone=ZoneType.BATTLEFIELD, owner_id=1, controller_id=1
        )
    }
    cards = {
        100: CardData(
            id=100, registry_key=1001,
            mana_cost=ManaCost(cost=[0, 1, 0, 0, 1, 0], mana_value=2)  # W+R
        )
    }
    permanents = {
        101: PermanentData(
            id=101, tapped=False,
            power=2, toughness=2, damage=0, is_summoning_sick=False
        )
    }
    stackobjs = {}  # none in this minimal example

    turn = TurnData(
        turn_number=1,
        phase=Phase.UNTAP,    # e.g. Phase.UNTAP, if stored as int
        step=Step.UNTAP,     # e.g. Step.UNTAP
        active_player_id=1,
        priority_player_id=1
    )
    action_space = ActionSpace(
        action_space_type=ActionSpaceType.PRIORITY,
        # For demonstration, let's say we're in PRIORITY
        actions=[ActionOption(action_type=ActionType.CAST_SPELL, focus=[100])],  # e.g. CAST_SPELL
        focus=[]
    )

    return Observation(
        game_over=False, won=False,
        turn=turn,
        action_space=action_space,
        players=players,
        objects=objects,
        cards=cards,
        permanents=permanents,
        stackobjs={}
    )


class TestDataBasic(unittest.TestCase):
    """
    Basic tests to ensure Observation and data structures
    remain valid across refactors.
    """

    def test_minimal_game_state_validation(self):
        """Ensure the minimal example is valid."""
        obs = create_minimal_observation()
        self.assertTrue(obs.validate(), "Observation should be valid with minimal data.")

    def test_zone_types(self):
        """Check that zone references in BaseObjectData are consistent."""
        obs = create_minimal_observation()
        # id=100 is in HAND
        self.assertEqual(obs.objects[100].zone, ZoneType.HAND)
        # id=101 is on BATTLEFIELD
        self.assertEqual(obs.objects[101].zone, ZoneType.BATTLEFIELD)

    def test_card_data(self):
        """Check that CardData is properly stored."""
        obs = create_minimal_observation()
        self.assertIn(100, obs.cards)
        card = obs.cards[100]
        self.assertEqual(card.registry_key, 1001)
        self.assertEqual(card.mana_cost.mana_value, 2)

    def test_permanent_data(self):
        """Check that PermanentData is properly stored."""
        obs = create_minimal_observation()
        self.assertIn(101, obs.permanents)
        perm = obs.permanents[101]
        self.assertFalse(perm.tapped)
        self.assertEqual(perm.power, 2)
        self.assertEqual(perm.toughness, 2)
        self.assertEqual(perm.damage, 0)

    def test_action_space(self):
        """Check that we can store and retrieve actions properly."""
        obs = create_minimal_observation()
        self.assertFalse(obs.game_over)
        self.assertEqual(len(obs.action_space.actions), 1)
        # The action in the minimal example:
        act = obs.action_space.actions[0]
        self.assertEqual(act.action_type, 3)  # e.g. CAST_SPELL
        self.assertEqual(act.focus, [100])


class TestRepresentationBasic(unittest.TestCase):
    """
    Basic flow tests to illustrate how data is encoded
    and how toggles in FeatureConfig affect results.
    """

    def setUp(self):
        self.obs = create_minimal_observation()

    def test_default_config_encoding(self):
        """Encode the minimal GameState with default config."""
        encoder = RepresentationEncoder()
        result = encoder.encode_game_state(self.obs)
        self.assertIsInstance(result, np.ndarray)

        # We have 2 objects, so shape = (2, some_feature_dim)
        # No padding by default => shape[0] = 2
        self.assertEqual(result.shape[0], 2)

        # By default, everything is int-based if no embeddings are used.
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_with_padding(self):
        """Test max_objects padding/truncation."""
        config = FeatureConfig(max_objects=5)
        encoder = RepresentationEncoder(config)
        mat = encoder.encode_game_state(self.obs)
        # We have 2 objects, but padded to 5 => shape[0] = 5
        self.assertEqual(mat.shape[0], 5)

    def test_registry_embeddings(self):
        """Use a CardEmbedding to embed registry_key as a vector."""
        # Suppose we have an 8D embedding
        emb = CardEmbedding(
            embedding_dim=8,
            embedding_map={
                1001: np.random.randn(8).astype(np.float32),
                2000: np.random.randn(8).astype(np.float32)
            }
        )
        config = FeatureConfig(
            registry_embeddings=True,
            card_embedding=emb
        )
        encoder = RepresentationEncoder(config)
        mat = encoder.encode_game_state(self.obs)

        # Now we have a float array because of embeddings
        self.assertTrue(np.issubdtype(mat.dtype, np.floating))
        # The shape's second dimension is bigger because we replaced registry_key with an 8D embedding
        # Let's just check we have shape[0] = 2
        self.assertEqual(mat.shape[0], 2)

    def test_conflict_registry_encoding(self):
        """Ensure we can't do one-hot and embeddings at the same time."""
        with self.assertRaises(ValueError):
            FeatureConfig(registry_one_hot=True, registry_embeddings=True)

    def test_multi_hot_subtypes(self):
        """If we had subtypes, they'd be multi-hot in the final array."""
        # We'll simulate that object 100 had subtypes, e.g. "Creature"
        # In real code, you'd store that in the game_state's CardData, etc.
        # For demonstration, we won't do a full environment refactor.
        # We'll just check that the code path doesn't break with subtypes.

        config = FeatureConfig(subtype_multihot=True)
        encoder = RepresentationEncoder(config)
        # We don't actually have subtypes in the minimal environment,
        # so the multi-hot part will be all zeros appended.
        mat = encoder.encode_game_state(self.obs)
        # Should have 2 objects, each with new subtypes dimension
        self.assertEqual(mat.shape[0], 2)
        # We can also do a quick check on the shape if needed.

    def test_example_flow(self):
        """Demonstrate a typical usage in a single test."""
        config = FeatureConfig(include_card_features=True, max_objects=4)
        encoder = RepresentationEncoder(config)
        mat = encoder.encode_game_state(self.obs)
        print("Encoded object array shape:", mat.shape)
        print(mat)


if __name__ == "__main__":
    unittest.main()
