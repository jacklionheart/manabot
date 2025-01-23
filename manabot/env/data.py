
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

# ----- Raw 

# ------------------- Enumerations -------------------
class ZoneType(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    EXILE = 4
    STACK = 5
    COMMAND = 6

class Phase(IntEnum):
    BEGINNING = 0
    PRECOMBAT_MAIN = 1
    COMBAT = 2
    POSTCOMBAT_MAIN = 3
    ENDING = 4


class Step(IntEnum):
    # Beginning Phase Steps
    BEGINNING_UNTAP = 0
    BEGINNING_UPKEEP = 1
    BEGINNING_DRAW = 2
    
    # Main Phase Steps
    PRECOMBAT_MAIN_STEP = 3

    # Combat Phase Steps
    COMBAT_BEGIN = 4
    COMBAT_DECLARE_ATTACKERS = 5
    COMBAT_DECLARE_BLOCKERS = 6
    COMBAT_DAMAGE = 7
    COMBAT_END = 8

    # Post-Combat Main Phase Steps
    POSTCOMBAT_MAIN_STEP = 9

    # Ending Phase Steps
    ENDING_END = 10
    ENDING_CLEANUP = 11

class ActionType(IntEnum):
    PASS = 0
    DECLARE_ATTACKER = 1
    DECLARE_BLOCKER = 2
    CAST_SPELL = 3
    PLAY_LAND = 4

class ActionSpaceType(IntEnum):
    GAME_OVER = 0
    PRIORITY = 1
    DECLARE_ATTACKER = 2
    DECLARE_BLOCKER = 3

# ------------------- Turn data ----------------------
@dataclass
class TurnData:
    turn_number: int
    phase: Phase
    step: Step
    active_player_id: int
    priority_player_id: int

# ------------------- Player data --------------------
@dataclass
class PlayerData:
    """
    For each player_id => PlayerData
    """
    # Unique ID for the player
    id: int
    # Index (0 for first, 1 for second, etc.)
    player_index: int
    # Flags
    is_active: bool = False
    has_priority: bool = False

    # Basic stats
    life: int = 20
    # For debugging or partial info: how many objects in each zone?
    zone_counts: List[int] = field(default_factory=lambda: [0]*7)  # [LIB, HAND, BATTLEFIELD, ...]
    # Mana pool: [C, W, U, B, R, G]
    mana: List[int] = field(default_factory=lambda: [0]*6)

# ------------------- Base & ManaCost ----------------
@dataclass
class BaseObjectData:
    """Core fields for objects in the game."""
    id: int
    zone: ZoneType
    owner_id: int
    controller_id: int

@dataclass
class ManaCost:
    """
    Example for a card's cost: WUBRGC order
    For instance, [1, 0, 0, 0, 2, 0] would mean 1 White, 2 Red
    """
    cost: List[int] = field(default_factory=lambda: [0]*6)
    # For debugging or quick checks
    mana_value: int = 0

# ------------------- Card data ----------------------
@dataclass
class CardData:
    """For card objects specifically (like in someone's hand or library)."""
    id: int
    registry_key: int
    mana_cost: ManaCost

# ------------------- Permanent data -----------------
@dataclass
class PermanentData:
    """Battlefield permanents."""
    id: int
    tapped: bool = False
    power: int = 0
    toughness: int = 0
    damage: int = 0
    is_summoning_sick: bool = False

# ------------------- Stack data ---------------------
@dataclass
class StackData:
    """For objects on the stack (spell or ability)."""
    id: int
    is_spell: bool
    is_ability: bool

# ------------------- Actions ------------------------
@dataclass
class ActionOption:
    action_type: ActionType
    # references to objects or players
    focus: List[int] = field(default_factory=list)

@dataclass
class ActionSpace:
    """
    A set of valid actions for a given decision point.
    """
    action_space_type: ActionSpaceType
    actions: List[ActionOption] = field(default_factory=list)
    # Possibly store a focus list for which objects or players matter
    focus: List[int] = field(default_factory=list)

# ------------------- Full GameState -----------------
@dataclass
class Observation:
    """
    Normalized state:
      - players
      - objects (base)
      - cards (if relevant)
      - permanents
      - stackobjs
      - turn
      - action_space
    """
    game_over: bool
    won: bool
    turn: TurnData
    
    # Fix the mutable default
    action_space: ActionSpace = field(
        default_factory=lambda: ActionSpace(
            action_space_type=ActionSpaceType.GAME_OVER
        )
    )
    
    # These were already correctly using field()
    players: Dict[int, PlayerData] = field(default_factory=dict)
    objects: Dict[int, BaseObjectData] = field(default_factory=dict)
    cards: Dict[int, CardData] = field(default_factory=dict)
    permanents: Dict[int, PermanentData] = field(default_factory=dict)
    stackobjs: Dict[int, StackData] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Basic validation placeholder. 
        E.g. check references, ID ranges, etc.
        """
        # Example checks:
        # 1) Turn data
        if self.turn.turn_number < 0: return False
        # 2) Players
        for pid, pdat in self.players.items():
            if pdat.id != pid:
                return False
        # 3) Objects
        for oid, obj in self.objects.items():
            if obj.id != oid:
                return False
            if not (0 <= obj.zone.value < 7):
                return False
        # 4) Cards
        for cid, cdat in self.cards.items():
            if cdat.id != cid:
                return False
        # 5) Perms
        for poid, pdat in self.permanents.items():
            if pdat.id != poid:
                return False
        # 6) Stack
        for soid, sdat in self.stackobjs.items():
            if sdat.id != soid:
                return False
        # 7) Action space
        # ...
        return True


# ------------------- Representation -------------------

@dataclass
class CardEmbedding:
    """
    A learnable or precomputed embedding map from card registry_key -> vector.
    Typically you'd load it from a model or train it end-to-end.
    """
    embedding_dim: int
    embedding_map: Dict[int, np.ndarray] = field(default_factory=dict)  # registry_key -> vector

    def get(self, registry_key: int) -> np.ndarray:
        """
        Return the embedding vector for this card. 
        If missing, return a zero vector or random vector.
        """
        if registry_key in self.embedding_map:
            return self.embedding_map[registry_key]
        else:
            return np.zeros(self.embedding_dim, dtype=np.float32)

@dataclass
class FeatureConfig:
    """
    Configuration toggles for which features to include in the final arrays,
    plus optional max sizes for array padding, and 
    how to handle registry_key encodings or subtypes, etc.
    """
    include_card_features: bool = True
    include_permanent_features: bool = True
    include_stack_features: bool = True

    # Maximum number of objects we can handle in a single state representation.
    max_objects: Optional[int] = None

    # Registry Key Handling
    registry_one_hot: bool = False    # If True, one-hot encode registry_key
    registry_embeddings: bool = False # If True, use CardEmbedding
    # We'll store the CardEmbedding if needed:
    card_embedding: Optional[CardEmbedding] = None

    # If we do one-hot or multi-hot for subtypes, we define them here.
    # E.g. a known set of card subtypes: "Creature", "Instant", "Sorcery", ...
    # We'll do a small example:
    known_subtypes: List[str] = field(default_factory=lambda: ["Creature", "Artifact", "Enchantment", "Instant", "Sorcery"])
    subtype_multihot: bool = False  # If True, we incorporate multi-hot vectors

    def __post_init__(self):
        # Basic validation if registry_one_hot + registry_embeddings are both true => conflict
        if self.registry_one_hot and self.registry_embeddings:
            raise ValueError("Cannot use both one-hot registry_key and embeddings at the same time.")

@dataclass
class GameObject:
    """
    Denormalized object representation (outer_join). 
    We'll add potential subtype info for multi-hot encoding if desired.
    """
    id: int
    zone_id: int
    owner_id: int
    controller_id: int

    # Type flags
    is_card: bool = False
    is_permanent: bool = False
    is_stack_object: bool = False

    # Card fields
    registry_key: Optional[int] = None
    mana_cost: Optional[List[int]] = None   # [C, W, U, B, R, G]
    mana_value: Optional[int] = None

    # For multi-hot subtypes (example usage)
    # E.g. ["Creature", "Sorcery"] -> we store indices for each
    subtypes: List[str] = field(default_factory=list)

    # Permanent fields
    tapped: Optional[bool] = None
    power: Optional[int] = None
    toughness: Optional[int] = None
    damage: Optional[int] = None
    summoning_sick: Optional[bool] = None

    # Stack fields
    is_spell: Optional[bool] = None
    is_ability: Optional[bool] = None

    def to_array(self, config: FeatureConfig) -> np.ndarray:
        """Convert this GameObject to a numeric array."""
        parts = []
        
        # Base fields (always 7 elements)
        base_fields = np.array([
            self.id,
            self.zone_id,
            self.owner_id,
            self.controller_id,
            int(self.is_card),
            int(self.is_permanent),
            int(self.is_stack_object)
        ], dtype=np.int32)
        parts.append(base_fields)

        # Registry encoding
        if self.is_card and self.registry_key is not None:
            if config.registry_embeddings and config.card_embedding:
                registry_part = config.card_embedding.get(self.registry_key)
            elif config.registry_one_hot:
                arr_size = 2000  # Make this configurable
                registry_part = np.zeros(arr_size, dtype=np.int32)
                if self.registry_key < arr_size:
                    registry_part[self.registry_key] = 1
            else:
                registry_part = np.array([self.registry_key], dtype=np.int32)
        else:
            if config.registry_embeddings and config.card_embedding:
                registry_part = np.zeros(config.card_embedding.embedding_dim, dtype=np.float32)
            elif config.registry_one_hot:
                registry_part = np.zeros(2000, dtype=np.int32)
            else:
                registry_part = np.array([-1], dtype=np.int32)
        parts.append(registry_part)

        # Card features
        card_features = []
        if config.include_card_features:
            # Mana cost (6 elements)
            card_features.extend(self.mana_cost if self.mana_cost else [-1]*6)
            # Mana value (1 element)
            card_features.append(self.mana_value if self.mana_value is not None else -1)
            # Subtypes if enabled
            if config.subtype_multihot:
                multi_hot = [0] * len(config.known_subtypes)
                for stype in self.subtypes:
                    if stype in config.known_subtypes:
                        multi_hot[config.known_subtypes.index(stype)] = 1
                card_features.extend(multi_hot)
        parts.append(np.array(card_features, dtype=np.int32))

        # Permanent features (5 elements)
        if config.include_permanent_features:
            perm_features = [
                int(self.tapped) if self.tapped is not None else -1,
                self.power if self.power is not None else -1,
                self.toughness if self.toughness is not None else -1,
                self.damage if self.damage is not None else -1,
                int(self.summoning_sick) if self.summoning_sick is not None else -1
            ]
            parts.append(np.array(perm_features, dtype=np.int32))

        # Stack features (2 elements)
        if config.include_stack_features:
            stack_features = [
                int(self.is_spell) if self.is_spell is not None else -1,
                int(self.is_ability) if self.is_ability is not None else -1
            ]
            parts.append(np.array(stack_features, dtype=np.int32))

        # Combine all parts
        if config.registry_embeddings and config.card_embedding:
            # Convert all to float32 if using embeddings
            parts = [p.astype(np.float32) for p in parts]
            
        return np.concatenate(parts)


class RepresentationEncoder:
    """
    High-level orchestrator for converting a Observation into 
    denormalized objects + arrays suitable for ML or RL.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def denormalize_objects(self, obs: Observation) -> List[GameObject]:
        """
        Outer-join approach: for each id in gs.objects,
        we gather data from (cards, permanents, stackobjs).
        Then we produce a single GameObject.
        We might also guess subtypes if you want to store them somewhere in CardData.
        """
        out = []
        for oid, base_obj in obs.objects.items():
            # Basic fields
            dn = GameObject(
                id=oid,
                zone_id=base_obj.zone.value,
                owner_id=base_obj.owner_id,
                controller_id=base_obj.controller_id
            )

            # If it's in cards
            if oid in obs.cards:
                dn.is_card = True
                cdat = obs.cards[oid]
                dn.registry_key = cdat.registry_key
                dn.mana_cost = cdat.mana_cost.cost
                dn.mana_value = cdat.mana_cost.mana_value
                # Suppose you store subtypes in cdat as cdat.subtypes
                # dn.subtypes = cdat.subtypes or []

            # If it's in permanents
            if oid in obs.permanents:
                dn.is_permanent = True
                pdat = obs.permanents[oid]
                dn.tapped = pdat.tapped
                dn.power = pdat.power
                dn.toughness = pdat.toughness
                dn.damage = pdat.damage
                dn.summoning_sick = pdat.is_summoning_sick

            # If it's in stackobjs
            if oid in obs.stackobjs:
                dn.is_stack_object = True
                sdat = obs.stackobjs[oid]
                dn.is_spell   = sdat.is_spell
                dn.is_ability = sdat.is_ability
            
            out.append(dn)
        return out

    def encode_objects(self, dn_objects: List[GameObject]) -> np.ndarray:
        """Converts a list of GameObject to a 2D NumPy array, with optional padding."""
        if not dn_objects:
            return np.zeros((0,0), dtype=np.int32)
        
        arrays = [obj.to_array(self.config) for obj in dn_objects]
        
        # Debug info
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            # If shapes don't match, print debug info
            print("Array shapes:", shapes)
            print("Arrays:", arrays)
            raise ValueError(f"Arrays have inconsistent shapes: {shapes}")
            
        try:
            mat = np.stack(arrays, axis=0)
        except ValueError as e:
            print("Failed to stack arrays:")
            for i, arr in enumerate(arrays):
                print(f"Array {i} shape: {arr.shape}, type: {arr.dtype}")
                print(arr)
            raise

        # If we have a max_objects, pad or truncate
        if self.config.max_objects is not None:
            if mat.shape[0] < self.config.max_objects:
                pad_size = self.config.max_objects - mat.shape[0]
                pad_shape = (pad_size, mat.shape[1])
                fill_val  = -1
                # If using embeddings => mat may be float
                if mat.dtype == np.float32:
                    fill_val = -1.0
                pad_block = np.full(pad_shape, fill_val, dtype=mat.dtype)
                mat = np.concatenate([mat, pad_block], axis=0)
            else:
                mat = mat[: self.config.max_objects]
        return mat

    def encode_game_state(self, obs: Observation) -> np.ndarray:
        """
        Example method that returns an array of shape [num_objects, feature_dim].
        If you want a dictionary with multiple arrays (like players, actions, etc.),
        you can expand below or create separate methods.
        """
        dn_objects = self.denormalize_objects(obs)
        return self.encode_objects(dn_objects)


def example_representation_flow(obs: Observation):
    """
    Demonstration of how to call the RepresentationEncoder with 
    one-hot or embedding approach for registry_keys.
    """
    # Suppose we do an embedding approach with dimension 8
    card_emb = CardEmbedding(
        embedding_dim=8,
        embedding_map={
            1001: np.random.randn(8).astype(np.float32),
            1002: np.random.randn(8).astype(np.float32),
        }
    )

    config = FeatureConfig(
        include_card_features=True,
        include_permanent_features=True,
        include_stack_features=True,
        max_objects=10,
        registry_one_hot=False,
        registry_embeddings=True,
        card_embedding=card_emb,
        subtype_multihot=True
    )

    encoder = RepresentationEncoder(config)
    obj_mat = encoder.encode_game_state(gs)
    print("Object array shape:", obj_mat.shape)
    print(obj_mat)
    return obj_mat
