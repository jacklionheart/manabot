# observation.py
# Defines data structures that are returned as part of the observations from the environment.
#
# All of these classes mirror the C++ structs in managym.
# We only implement a one-way "from C++" conversion for bridging the environment states.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

import managym


# ---------------------------------------------------------------------
# Python Enums referencing C++ enums
# ---------------------------------------------------------------------
class ZoneEnum(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6

class PhaseEnum(IntEnum):
    BEGINNING = 0
    PRECOMBAT_MAIN = 1
    COMBAT = 2
    POSTCOMBAT_MAIN = 3
    ENDING = 4

class StepEnum(IntEnum):
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
    PRIORITY_PLAY_LAND = 0
    PRIORITY_CAST_SPELL = 1  
    PRIORITY_PASS_PRIORITY = 2
    DECLARE_ATTACKER = 3
    DECLARE_BLOCKER = 4

class ActionSpaceEnum(IntEnum):
    GAME_OVER = 0
    PRIORITY = 1
    DECLARE_ATTACKER = 2
    DECLARE_BLOCKER = 3

# ---------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------
@dataclass
class Turn:
    turn_number: int
    phase: PhaseEnum
    step: StepEnum
    active_player_id: int
    agent_player_id: int

@dataclass
class Player:
    id: int
    player_index: int
    is_agent: bool
    is_active: bool
    life: int
    zone_counts: List[int]

@dataclass
class CardTypes:
    is_castable: bool
    is_permanent: bool
    is_non_land_permanent: bool
    is_non_creature_permanent: bool
    is_spell: bool
    is_creature: bool
    is_land: bool
    is_planeswalker: bool
    is_enchantment: bool
    is_artifact: bool
    is_kindred: bool
    is_battle: bool

@dataclass
class ManaCost:
    cost: List[int]   # e.g. [W, U, B, R, G, C]
    mana_value: int

@dataclass
class Card:
    zone: ZoneEnum
    owner_id: int
    id: int
    registry_key: int
    power: int
    toughness: int
    card_types: CardTypes
    mana_cost: ManaCost

@dataclass
class Permanent:
    id: int
    controller_id: int
    tapped: bool
    damage: int
    is_creature: bool
    is_land: bool
    is_summoning_sick: bool

@dataclass
class Action:
    action_type: ActionEnum
    focus: List[int]

@dataclass
class ActionSpace:
    action_space_type: ActionSpaceEnum
    actions: List[Action]
    focus: List[int]

@dataclass
class Observation:
    """A Pythonic snapshot of the entire game state."""
    game_over: bool = False
    won: bool = False 
    turn: Turn = field(default_factory=lambda: Turn(0, PhaseEnum.BEGINNING, StepEnum.BEGINNING_UNTAP, -1, -1))
    action_space: ActionSpace = field(default_factory=lambda: ActionSpace(ActionSpaceEnum.GAME_OVER, [], []))
    players: Dict[int, Player] = field(default_factory=dict)
    cards: Dict[int, Card] = field(default_factory=dict)
    permanents: Dict[int, Permanent] = field(default_factory=dict)

    def __init__(self, cxx_obs=None):
        super().__init__()
        if cxx_obs is not None:
            self.game_over = cxx_obs.game_over
            self.won = cxx_obs.won
            
            # Convert Turn
            self.turn = Turn(
                turn_number=cxx_obs.turn.turn_number,
                phase=PhaseEnum(int(cxx_obs.turn.phase) & 0xFF),  # Mask to get base enum value
                step=StepEnum(int(cxx_obs.turn.step) & 0xFF),
                active_player_id=cxx_obs.turn.active_player_id,
                agent_player_id=cxx_obs.turn.agent_player_id
            )

            # Convert ActionSpace
            actions = [
                Action(
                    action_type=ActionEnum(int(act_cpp.action_type) & 0xFF),
                    focus=list(act_cpp.focus)
                )
                for act_cpp in cxx_obs.action_space.actions
            ]
            self.action_space = ActionSpace(
                action_space_type=ActionSpaceEnum(int(cxx_obs.action_space.action_space_type) & 0xFF),
                actions=actions,
                focus=list(cxx_obs.action_space.focus)
            )

            # Convert Players
            self.players = {
                pid: Player(
                    id=p_cpp.id,
                    player_index=p_cpp.player_index,
                    is_agent=p_cpp.is_agent,
                    is_active=p_cpp.is_active,
                    life=p_cpp.life,
                    zone_counts=list(p_cpp.zone_counts)
                )
                for pid, p_cpp in cxx_obs.players.items()
            }

            # Convert Cards  
            self.cards = {}
            for cid, c_cpp in cxx_obs.cards.items():
                card_types = CardTypes(
                    is_castable=c_cpp.card_types.is_castable,
                    is_permanent=c_cpp.card_types.is_permanent,
                    is_non_land_permanent=c_cpp.card_types.is_non_land_permanent,
                    is_non_creature_permanent=c_cpp.card_types.is_non_creature_permanent,
                    is_spell=c_cpp.card_types.is_spell,
                    is_creature=c_cpp.card_types.is_creature,
                    is_land=c_cpp.card_types.is_land,
                    is_planeswalker=c_cpp.card_types.is_planeswalker,
                    is_enchantment=c_cpp.card_types.is_enchantment,
                    is_artifact=c_cpp.card_types.is_artifact,
                    is_kindred=c_cpp.card_types.is_kindred,
                    is_battle=c_cpp.card_types.is_battle,
                )
                mana_cost = ManaCost(
                    cost=list(c_cpp.mana_cost.cost),
                    mana_value=c_cpp.mana_cost.mana_value
                )
                self.cards[cid] = Card(
                    zone=ZoneEnum(int(c_cpp.zone) & 0xFF),
                    owner_id=c_cpp.owner_id,
                    id=c_cpp.id,
                    registry_key=c_cpp.registry_key,
                    power=c_cpp.power,
                    toughness=c_cpp.toughness,
                    card_types=card_types,
                    mana_cost=mana_cost
                )

            # Convert Permanents
            self.permanents = {
                pid: Permanent(
                    id=p_cpp.id,
                    controller_id=p_cpp.controller_id,
                    tapped=p_cpp.tapped,
                    damage=p_cpp.damage,
                    is_creature=p_cpp.is_creature,
                    is_land=p_cpp.is_land,
                    is_summoning_sick=p_cpp.is_summoning_sick
                )
                for pid, p_cpp in cxx_obs.permanents.items()
            }

    def validate(self) -> bool:
        # Basic consistency checks
        if self.turn.turn_number < 0:
            return False
        for pid, player_data in self.players.items():
            if player_data.id != pid:
                return False
        for cid, card_data in self.cards.items():
            if card_data.id != cid:
                return False
        for pid, perm_data in self.permanents.items():
            if perm_data.id != pid:
                return False
        return True