"""
observation.py
Defines a Gymnasium-compatible observation space from managym observations.
"""

from enum import IntEnum
from typing import Dict
import numpy as np
from gymnasium import spaces
import managym

# -----------------------------------------------------------------------------
# Value Limits (still available if you need them, but no longer enforced in spaces)
# -----------------------------------------------------------------------------
MAX_LIFE = 100
MAX_TURNS = 1000
MAX_ZONE_SIZE = 1000
MAX_DAMAGE = 100
MAX_MANA = 20
MAX_POWER = 20

# -----------------------------------------------------------------------------
# Game Enums - mirror managym for validation
# -----------------------------------------------------------------------------
class PhaseEnum(IntEnum):
    """Mirrors managym.PhaseEnum for validation."""
    BEGINNING = 0
    PRECOMBAT_MAIN = 1
    COMBAT = 2
    POSTCOMBAT_MAIN = 3
    ENDING = 4

class StepEnum(IntEnum):
    """Mirrors managym.StepEnum for validation."""
    BEGINNING_UNTAP = 0
    BEGINNING_UPKEEP = 1
    BEGINNING_DRAW = 2
    PRECOMBAT_MAIN_STEP = 3
    COMBAT_BEGIN = 4
    COMBAT_DECLARE_ATTACKERS = 5
    COMBAT_DECLARE_BLOCKERS = 6
    COMBAT_DAMAGE = 7
    COMBAT_END = 8
    POSTCOMBAT_MAIN_STEP = 9
    ENDING_END = 10
    ENDING_CLEANUP = 11

class ActionEnum(IntEnum):
    """Mirrors managym.ActionEnum for validation."""
    PRIORITY_PLAY_LAND = 0
    PRIORITY_CAST_SPELL = 1
    PRIORITY_PASS_PRIORITY = 2
    DECLARE_ATTACKER = 3
    DECLARE_BLOCKER = 4

class ZoneEnum(IntEnum):
    """Mirrors managym.ZoneEnum for validation."""
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6

# -----------------------------------------------------------------------------
# Observation Encoding
# -----------------------------------------------------------------------------

class ObservationEncoder:
    """Converts managym observations to tensors."""
    def __init__(
        self,
        num_players: int = 2,
        max_cards: int = 100,
        max_permanents: int = 50,
        max_actions: int = 20,
        max_focus_objects: int = 2,
    ):
        self.num_players = num_players
        self.max_cards = max_cards
        self.max_permanents = max_permanents
        self.max_actions = max_actions
        self.max_focus_objects = max_focus_objects

        self.num_phases = len(PhaseEnum)
        self.num_steps = len(StepEnum)
        self.num_zones = len(ZoneEnum)
        self.num_actions = len(ActionEnum)

        # Feature dimensions
        self.player_dim = 1 + 2 + self.num_zones  # life + is_active + is_agent + zone_counts
        self.card_dim = self.num_zones + 1 + 2 + 1 + 6  # zone + owner + power/tough + mana + 6 type flags
        self.permanent_dim = 6  # controller, tapped, damage, summoning_sick, is_land, is_creature
        self.focus_dim = self.player_dim + self.card_dim + self.permanent_dim
        self.action_dim = self.num_actions + (max_focus_objects * self.focus_dim)

    @property
    def shapes(self) -> Dict[str, tuple]:
        return {
            'global': (1 + self.num_phases + self.num_steps + 2,),
            'players': (self.num_players, self.player_dim),
            'cards': (self.max_cards, self.card_dim),
            'permanents': (self.max_permanents, self.permanent_dim),
            'actions': (self.max_actions, self.action_dim)
        }

    def encode(self, obs: managym.Observation) -> Dict[str, np.ndarray]:
        return {
            'global': self._encode_global(obs),
            'players': self._encode_players(obs),
            'cards': self._encode_cards(obs),
            'permanents': self._encode_permanents(obs),
            'actions': self._encode_actions(obs)
        }

    def _encode_global(self, obs: managym.Observation) -> np.ndarray:
        if not hasattr(obs, 'turn'):
            raise ValueError("Observation missing turn data")
            
        arr = np.zeros(self.shapes['global'], dtype=np.float32)
        i = 0

        # First turn remains 1.0 by default
        arr[i] = float(obs.turn.turn_number)
        i += 1

        phase_val = int(obs.turn.phase) & 0xFF
        if 0 <= phase_val < self.num_phases:
            arr[i + phase_val] = 1.0
        i += self.num_phases

        step_val = int(obs.turn.step) & 0xFF
        if 0 <= step_val < self.num_steps:
            arr[i + step_val] = 1.0
        i += self.num_steps

        arr[i] = float(obs.game_over)
        arr[i + 1] = float(obs.won)
        return arr

    def _encode_players(self, obs: managym.Observation) -> np.ndarray:
        if not hasattr(obs, 'players'):
            raise ValueError("Observation missing players")
            
        arr = np.zeros(self.shapes['players'], dtype=np.float32)
        for idx, (_, player) in enumerate(sorted(obs.players.items())):
            if idx >= self.num_players:
                break
            arr[idx] = self._encode_player_features(player)
        return arr
    
    def _encode_player_features(self, player: managym.Player) -> np.ndarray:
        arr = np.zeros(self.player_dim, dtype=np.float32)
        i = 0

        arr[i] = float(player.life)
        i += 1
        arr[i] = float(player.is_active)
        i += 1
        arr[i] = float(player.is_agent)
        i += 1

        for z in range(min(len(player.zone_counts), self.num_zones)):
            arr[i + z] = float(player.zone_counts[z])
        return arr

    def _encode_cards(self, obs: managym.Observation) -> np.ndarray:
        if not hasattr(obs, 'cards'):
            raise ValueError("Observation missing cards")
            
        arr = np.zeros(self.shapes['cards'], dtype=np.float32)
        for idx, (_, card) in enumerate(sorted(obs.cards.items())):
            if idx >= self.max_cards:
                break
            arr[idx] = self._encode_card_features(card)
        return arr
    
    def _encode_card_features(self, card: managym.Card) -> np.ndarray:
        arr = np.zeros(self.card_dim, dtype=np.float32)
        i = 0

        zone_val = int(card.zone) & 0xFF
        if 0 <= zone_val < self.num_zones:
            arr[i + zone_val] = 1.0
        i += self.num_zones

        arr[i] = float(card.owner_id); i += 1
        arr[i] = float(card.power); i += 1
        arr[i] = float(card.toughness); i += 1
        arr[i] = float(card.mana_cost.mana_value); i += 1

        arr[i] = float(card.card_types.is_land); i += 1
        arr[i] = float(card.card_types.is_creature); i += 1
        arr[i] = float(card.card_types.is_artifact); i += 1
        arr[i] = float(card.card_types.is_enchantment); i += 1
        arr[i] = float(card.card_types.is_planeswalker); i += 1
        arr[i] = float(card.card_types.is_battle)
        return arr

    def _encode_permanents(self, obs: managym.Observation) -> np.ndarray:
        if not hasattr(obs, 'permanents'):
            raise ValueError("Observation missing permanents")
            
        arr = np.zeros(self.shapes['permanents'], dtype=np.float32)
        for idx, (_, perm) in enumerate(sorted(obs.permanents.items())):
            if idx >= self.max_permanents:
                break
            arr[idx] = self._encode_permanent_features(perm)
        return arr
    
    def _encode_permanent_features(self, perm: managym.Permanent) -> np.ndarray:
        arr = np.zeros(self.permanent_dim, dtype=np.float32)
        i = 0

        arr[i] = float(perm.controller_id)
        i += 1
        arr[i] = float(perm.tapped)
        i += 1
        arr[i] = float(perm.damage)
        i += 1
        arr[i] = float(perm.is_summoning_sick)
        i += 1
        arr[i] = float(perm.is_land)
        i += 1
        arr[i] = float(perm.is_creature)
        return arr

    def encode_focus_object(self, obs: managym.Observation, obj_id: int) -> np.ndarray:
        """
        Returns the raw (unbounded) features for player/card/permanent. 
        No normalization, no bounding.
        """
        arr = np.zeros(self.focus_dim, dtype=np.float32)
        
        # If it's a player
        if obj_id in obs.players:
            p = obs.players[obj_id]
            arr[0] = float(p.life)
            arr[1] = float(p.is_active)
            arr[2] = float(p.is_agent)
            for z in range(min(len(p.zone_counts), self.num_zones)):
                arr[3 + z] = float(p.zone_counts[z])
            return arr

        # If it's a card
        elif obj_id in obs.cards:
            c = obs.cards[obj_id]
            zone_val = int(c.zone) & 0xFF
            if 0 <= zone_val < self.num_zones:
                arr[self.player_dim + zone_val] = 1.0

            i = self.player_dim + self.num_zones
            arr[i] = float(c.owner_id); i += 1
            arr[i] = float(c.power); i += 1
            arr[i] = float(c.toughness); i += 1
            arr[i] = float(c.mana_cost.mana_value); i += 1
            
            arr[i] = float(c.card_types.is_land); i += 1
            arr[i] = float(c.card_types.is_creature); i += 1
            arr[i] = float(c.card_types.is_artifact); i += 1
            arr[i] = float(c.card_types.is_enchantment); i += 1
            arr[i] = float(c.card_types.is_planeswalker); i += 1
            arr[i] = float(c.card_types.is_battle)
            return arr

        # If it's a permanent
        elif obj_id in obs.permanents:
            pm = obs.permanents[obj_id]
            offset = self.player_dim + self.card_dim
            arr[offset + 0] = float(pm.controller_id)
            arr[offset + 1] = float(pm.tapped)
            arr[offset + 2] = float(pm.damage)
            arr[offset + 3] = float(pm.is_summoning_sick)
            arr[offset + 4] = float(pm.is_land)
            arr[offset + 5] = float(pm.is_creature)
            return arr

        # If no match, return zero
        return arr

    def _encode_actions(self, obs: managym.Observation) -> np.ndarray:
        arr = np.zeros(self.shapes['actions'], dtype=np.float32)
        
        for idx, action in enumerate(obs.action_space.actions[:self.max_actions]):
            if idx >= self.max_actions:
                break
            arr[idx] = self._encode_action(obs, action)
        return arr
    
    def _encode_action(self, obs: managym.Observation, action: managym.Action) -> np.ndarray:
        arr = np.zeros(self.action_dim, dtype=np.float32)
        
        action_type = int(action.action_type)
        if 0 <= action_type < self.num_actions:
            arr[action_type] = 1.0
            
        for f_idx, focus_id in enumerate(action.focus[:self.max_focus_objects]):
            offset = self.num_actions + (f_idx * self.focus_dim)
            arr[offset:offset + self.focus_dim] = self.encode_focus_object(obs, focus_id)
        return arr


class ObservationSpace(spaces.Space):
    def __init__(self, encoder: ObservationEncoder):
        super().__init__(shape=None, dtype=None)
        self.encoder = encoder
        
        # Instead of bounding these arrays, we let them be unbounded. 
        # This ensures .contains(...) won't fail on large numeric values.
        self.space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=encoder.shapes['global'],
                dtype=np.float32
            ),
            'players': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=encoder.shapes['players'],
                dtype=np.float32
            ),
            'cards': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=encoder.shapes['cards'],
                dtype=np.float32
            ),
            'permanents': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=encoder.shapes['permanents'],
                dtype=np.float32
            ),
            'actions': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=encoder.shapes['actions'],
                dtype=np.float32
            )
        })

    def sample(self, mask=None) -> dict:
        """Samples from each sub-space. (Random values in [-∞, ∞] is not well-defined, 
           so in practice this might generate NaNs. But it satisfies the 'no bound' approach.)"""
        return self.space.sample(mask)

    def contains(self, x: Dict[str, np.ndarray]) -> bool:
        """True if x fits the shape of each sub-space. We have no numeric bounds to check."""
        return self.space.contains(x)

    @property
    def shape(self) -> tuple | None:
        return self.space.shape
