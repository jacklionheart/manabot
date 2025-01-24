# tests/env/test_data.py
import unittest

import managym

# Our Python-side enumerations/data classes
from manabot.data import (
    ZoneEnum,
    PhaseEnum,
    StepEnum,
    ActionEnum,
    ActionSpaceEnum,
    Turn,
    Player,
    CardTypes,
    ManaCost,
    Card,
    Permanent,
    Action,
    ActionSpace,
    Observation,
)

# -----------------------------------------------------------------------------
# 1) Enum Tests
# -----------------------------------------------------------------------------
class TestEnumParity(unittest.TestCase):
    """
    Compare Python-side enum values (manabot.data.*Enum) with the pybind C++ side (managym.*Enum).
    Because the pybind enumerations aren't iterable, we do explicit checks.
    """

    def test_zone_enum(self):
        # Check each known ZoneEnum entry
        # Instead of list(managym.ZoneEnum), we do explicit name-based checks:
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

    def test_action_space_enum(self):
        self.assertEqual(int(managym.ActionSpaceEnum.GAME_OVER), ActionSpaceEnum.GAME_OVER)
        self.assertEqual(int(managym.ActionSpaceEnum.PRIORITY), ActionSpaceEnum.PRIORITY)
        self.assertEqual(int(managym.ActionSpaceEnum.DECLARE_ATTACKER), ActionSpaceEnum.DECLARE_ATTACKER)
        self.assertEqual(int(managym.ActionSpaceEnum.DECLARE_BLOCKER), ActionSpaceEnum.DECLARE_BLOCKER)


# -----------------------------------------------------------------------------
# 2) Structural Field Tests
# -----------------------------------------------------------------------------
# We define a map from "C++ data struct name" to "(Python class, fields we expect)".
# Then we confirm each field is present in the pybind class as well.
# -----------------------------------------------------------------------------
STRUCT_FIELD_MAP = {
    # managym.Player -> manabot.data.Player
    "Player": (
        managym.Player,  # the pybind11 struct/class
        Player,          # Python dataclass
        ["id", "player_index", "is_agent", "is_active", "life", "zone_counts"]
    ),
    # managym.Turn -> manabot.data.Turn
    "Turn": (
        managym.Turn,
        Turn,
        ["turn_number", "phase", "step", "active_player_id", "agent_player_id"]
    ),
    # managym.ManaCost -> manabot.data.ManaCost
    "ManaCost": (
        managym.ManaCost,
        ManaCost,
        ["cost", "mana_value"]
    ),
    # managym.CardTypes -> manabot.data.CardTypes
    "CardTypes": (
        managym.CardTypes,
        CardTypes,
        [
            "is_castable",
            "is_permanent",
            "is_non_land_permanent",
            "is_non_creature_permanent",
            "is_spell",
            "is_creature",
            "is_land",
            "is_planeswalker",
            "is_enchantment",
            "is_artifact",
            "is_kindred",
            "is_battle",
        ]
    ),
    # managym.Card -> manabot.data.Card
    "Card": (
        managym.Card,
        Card,
        [
            "zone",
            "owner_id",
            "id",
            "registry_key",
            "power",
            "toughness",
            "card_types",
            "mana_cost",
        ]
    ),
    # managym.Permanent -> manabot.data.Permanent
    "Permanent": (
        managym.Permanent,
        Permanent,
        [
            "id",
            "controller_id",
            "tapped",
            "damage",
            "is_creature",
            "is_land",
            "is_summoning_sick",
        ]
    ),
    # managym.Action -> manabot.data.Action
    "Action": (
        managym.Action,
        Action,
        [
            "action_type",
            "focus",
        ]
    ),
    # managym.ActionSpace -> manabot.data.ActionSpace
    "ActionSpace": (
        managym.ActionSpace,
        ActionSpace,
        [
            "action_space_type",
            "actions",
            "focus",
        ]
    ),
    # managym.Observation -> manabot.data.Observation
    "Observation": (
        managym.Observation,
        Observation,
        [
            "game_over",
            "won",
            "turn",
            "action_space",
            "players",
            "cards",
            "permanents",
        ]
    ),
}


class TestStructFields(unittest.TestCase):
    def test_struct_fields_match(self):
        """
        For each data structure, verify that the Python dataclass has exactly the
        expected fields, and that the pybind C++ class also has them as attributes.
        """
        for struct_name, (cxx_class, py_class, field_list) in STRUCT_FIELD_MAP.items():
            with self.subTest(struct_name=struct_name):
                # 1) Python dataclass should have exactly these fields
                python_fields = set(py_class.__dataclass_fields__.keys())
                expected = set(field_list)
                self.assertEqual(
                    python_fields,
                    expected,
                    f"Python {py_class.__name__} fields differ from expected {field_list}",
                )

                # 2) The C++ side (pybind) should have each as an attribute
                for field_name in field_list:
                    self.assertTrue(
                        hasattr(cxx_class, field_name),
                        f"C++ class {struct_name} missing attribute '{field_name}'",
                    )


# -----------------------------------------------------------------------------
# 3) Round-Trip / Observation Tests
# -----------------------------------------------------------------------------
# We test that we can construct a managym.Observation in various ways and
# that the Python "Observation" dataclass can parse it consistently.
# -----------------------------------------------------------------------------
class TestObservationRoundTrip(unittest.TestCase):
    def test_empty_observation(self):
        """
        Create a default-constructed managym.Observation, check that Python sees
        the correct default values.
        """
        cpp_obs = managym.Observation()
        # Let's just confirm a few defaults
        self.assertFalse(cpp_obs.game_over)
        self.assertFalse(cpp_obs.won)
        self.assertEqual(len(cpp_obs.players), 0)
        self.assertEqual(len(cpp_obs.cards), 0)
        self.assertEqual(len(cpp_obs.permanents), 0)

        # Convert to Python side
        py_obs = Observation(cpp_obs)
        self.assertFalse(py_obs.game_over)
        self.assertFalse(py_obs.won)
        self.assertEqual(len(py_obs.players), 0)
        self.assertEqual(len(py_obs.cards), 0)
        self.assertEqual(len(py_obs.permanents), 0)

    def test_mocked_observation(self):
        """
        In principle, you could fill fields in a managym.Observation, then
        pass it to the Python constructor. We'll do a minimal field fill here
        if pybind allows direct writes. If not, just read them as is.
        """
        cpp_obs = managym.Observation()
        # Try to set a field or two if pybind exposes them readwrite:
        cpp_obs.game_over = True
        cpp_obs.won = True

        # Convert to Python
        py_obs = Observation(cpp_obs)
        self.assertTrue(py_obs.game_over)
        self.assertTrue(py_obs.won)

        # Validate function from manabot.data
        self.assertTrue(py_obs.validate(), "Pythonic observation should pass validation")

    def test_env_reset_observation(self):
        """
        Test that resetting the environment yields a consistent Observation.
        We'll do a quick check of the top-level fields. We won't assume that
        the game is 'game_over' or not, because it depends on actual game logic.
        """
        env = managym.Env(skip_trivial=False)
        player_a = managym.PlayerConfig("Alice", {"Mountain": 40})
        player_b = managym.PlayerConfig("Bob", {"Forest": 40})

        obs, info = env.reset([player_a, player_b])
        self.assertIsInstance(obs, managym.Observation)
        self.assertIsInstance(info, dict)

        # We can do a quick Pythonic mirror if we want:
        py_obs = Observation(obs)
        self.assertIn(py_obs.turn.turn_number, range(0, 1000))  # Just a sanity check
        self.assertFalse(py_obs.game_over)  # Typically not game-over immediately
        self.assertTrue(py_obs.validate(), "Fresh observation should validate successfully")


###############################################################################
# 4) If you want extended field-by-field testing of each data class
###############################################################################
class TestComprehensiveDataStructs(unittest.TestCase):
    """
    A more verbose set of tests verifying each data struct's contents.
    You can adapt these to your liking if you want to “exhaustively test every field.”
    """

    def test_player_data_defaults(self):
        p = managym.Player()
        self.assertEqual(p.id, 0)
        self.assertEqual(p.player_index, 0)
        self.assertFalse(p.is_agent)
        self.assertFalse(p.is_active)
        self.assertEqual(p.life, 20)
        # zone_counts should be length 7
        self.assertEqual(len(p.zone_counts), 7)

    def test_turn_data_defaults(self):
        t = managym.Turn()
        self.assertEqual(t.turn_number, 0)
        # Just confirm we can read them; we won't guess the defaults for phase/step
        self.assertIn(t.phase.value, range(0, 10))
        self.assertIn(t.step.value, range(0, 20))
        self.assertIn(t.active_player_id, range(-1, 10))
        self.assertIn(t.agent_player_id, range(-1, 10))

    def test_card_types_defaults(self):
        ctype = managym.CardTypes()
        self.assertFalse(ctype.is_castable)
        self.assertFalse(ctype.is_permanent)
        # etc. - Just check each boolean:
        self.assertFalse(ctype.is_non_land_permanent)
        self.assertFalse(ctype.is_non_creature_permanent)
        self.assertFalse(ctype.is_spell)
        self.assertFalse(ctype.is_creature)
        self.assertFalse(ctype.is_land)
        self.assertFalse(ctype.is_planeswalker)
        self.assertFalse(ctype.is_enchantment)
        self.assertFalse(ctype.is_artifact)
        self.assertFalse(ctype.is_kindred)
        self.assertFalse(ctype.is_battle)

    def test_card_defaults(self):
        c = managym.Card()
        self.assertEqual(c.zone.value, ZoneEnum.LIBRARY.value)  # possibly 0
        self.assertEqual(c.owner_id, 0)
        self.assertEqual(c.id, 0)
        self.assertEqual(c.registry_key, 0)
        self.assertEqual(c.power, 0)
        self.assertEqual(c.toughness, 0)
        # c.card_types is a managym.CardTypes
        self.assertIsInstance(c.card_types, managym.CardTypes)
        # c.mana_cost is managym.ManaCost
        self.assertIsInstance(c.mana_cost, managym.ManaCost)

    def test_permanent_defaults(self):
        p = managym.Permanent()
        self.assertEqual(p.id, 0)
        self.assertEqual(p.controller_id, 0)
        self.assertFalse(p.tapped)
        self.assertEqual(p.damage, 0)
        self.assertFalse(p.is_creature)
        self.assertFalse(p.is_land)
        self.assertFalse(p.is_summoning_sick)

    def test_action_defaults(self):
        a = managym.Action()
        self.assertEqual(a.action_type.value, ActionEnum.PRIORITY_PLAY_LAND.value)  # default is 0
        self.assertEqual(a.focus, [])

    def test_action_space_defaults(self):
        aspace = managym.ActionSpace()
        self.assertEqual(aspace.action_space_type.value, ActionSpaceEnum.GAME_OVER.value)
        self.assertEqual(aspace.actions, [])
        self.assertEqual(aspace.focus, [])

    def test_observation_defaults(self):
        obs = managym.Observation()
        self.assertFalse(obs.game_over)
        self.assertFalse(obs.won)
        self.assertIsInstance(obs.turn, managym.Turn)
        self.assertIsInstance(obs.action_space, managym.ActionSpace)
        self.assertEqual(obs.players, {})
        self.assertEqual(obs.cards, {})
        self.assertEqual(obs.permanents, {})


if __name__ == "__main__":
    unittest.main()
