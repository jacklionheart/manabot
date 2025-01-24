# observation.py 
# schema for the game state data returned by managym.

"""
EDITING INSTRUCTIONS:    
Each data structure mirrors exactly a corresponding managym C++ structure; existing tests should mostly ensure they keep in sync.
Prioritize brevity and transparency when editing; 
ensure it remains an easy reference guide to raw data structures. 
Keep comments minimal and organizational.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

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

class ZoneEnum(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    STACK = 4
    EXILE = 5
    COMMAND = 6

# -----------------------------------------------------------------------------
#  Data structures
# -----------------------------------------------------------------------------

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
    cost: List[int]
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
    game_over: bool = False
    won: bool = False
    turn: Turn = field(default_factory=lambda: Turn(
        turn_number=0,
        phase=PhaseEnum.BEGINNING,
        step=StepEnum.BEGINNING_UNTAP,
        active_player_id=-1,
        agent_player_id=-1
    ))
    action_space: ActionSpace = field(default_factory=lambda: ActionSpace(
        action_space_type=ActionSpaceEnum.GAME_OVER, actions=[], focus=[]
    ))
    players: Dict[int, Player] = field(default_factory=dict)
    cards: Dict[int, Card] = field(default_factory=dict)
    permanents: Dict[int, Permanent] = field(default_factory=dict)

    def __init__(self, cxx_obs=None):
        super().__init__()
        if cxx_obs is not None:
            self.game_over = cxx_obs.game_over
            self.won = cxx_obs.won
            self.turn = Turn(
                turn_number=cxx_obs.turn.turn_number,
                phase=PhaseEnum(int(cxx_obs.turn.phase) & 0xFF),
                step=StepEnum(int(cxx_obs.turn.step) & 0xFF),
                active_player_id=cxx_obs.turn.active_player_id,
                agent_player_id=cxx_obs.turn.agent_player_id
            )
            a_actions = [
                Action(
                    action_type=ActionEnum(int(a.action_type) & 0xFF),
                    focus=list(a.focus)
                )
                for a in cxx_obs.action_space.actions
            ]
            self.action_space = ActionSpace(
                action_space_type=ActionSpaceEnum(int(cxx_obs.action_space.action_space_type) & 0xFF),
                actions=a_actions,
                focus=list(cxx_obs.action_space.focus)
            )
            self.players = {}
            for pid, p_cpp in cxx_obs.players.items():
                pdat = Player(
                    id=p_cpp.id,
                    player_index=p_cpp.player_index,
                    is_agent=p_cpp.is_agent,
                    is_active=p_cpp.is_active,
                    life=p_cpp.life,
                    zone_counts=list(p_cpp.zone_counts)
                )
                self.players[pid] = pdat

            self.cards = {}
            for cid, c_cpp in cxx_obs.cards.items():
                ctype = CardTypes(
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
                    is_battle=c_cpp.card_types.is_battle
                )
                cmana = ManaCost(
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
                    card_types=ctype,
                    mana_cost=cmana
                )

            self.permanents = {}
            for pid, pp in cxx_obs.permanents.items():
                self.permanents[pid] = Permanent(
                    id=pp.id,
                    controller_id=pp.controller_id,
                    tapped=pp.tapped,
                    damage=pp.damage,
                    is_creature=pp.is_creature,
                    is_land=pp.is_land,
                    is_summoning_sick=pp.is_summoning_sick
                )

    def validate(self) -> bool:
        # Minimal checks
        for pid, pdat in self.players.items():
            if pdat.id != pid:
                return False
        for cid, cdat in self.cards.items():
            if cdat.id != cid:
                return False
        for pid, pdat in self.permanents.items():
            if pdat.id != pid:
                return False
        return True
