"""
representation.py

Feature encoding logic for manabot Observations.
Denormalizes the data and converts it to numeric arrays
for neural network consumption.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
    
from .observation import Observation, ZoneEnum, CardTypes

@dataclass
class CardEmbedding:
    """
    Holds an embedding_map from registry_key -> np.ndarray.
    """
    embedding_dim: int
    embedding_map: Dict[int, np.ndarray] = field(default_factory=dict)

    def get(self, registry_key: int) -> np.ndarray:
        return self.embedding_map.get(registry_key, np.zeros(self.embedding_dim, dtype=np.float32))


@dataclass
class FeatureConfig:
    """
    Toggle options for how we encode the game state into arrays.
    """
    include_card_features: bool = True
    include_permanent_features: bool = True
    max_objects: Optional[int] = None
    registry_one_hot: bool = False
    registry_embeddings: bool = False
    card_embedding: Optional[CardEmbedding] = None

    def __post_init__(self):
        if self.registry_one_hot and self.registry_embeddings:
            raise ValueError("Cannot use both one-hot registry_key and embeddings at the same time.")


@dataclass
class GameObject:
    """
    Denormalized object combining Card + (optional) Permanent fields.
    """
    id: int
    zone: ZoneEnum
    owner_id: int
    registry_key: int
    power: int
    toughness: int
    mana_cost: List[int] = field(default_factory=lambda: [0]*6)
    mana_value: int = 0
    card_type: CardTypes = field(default_factory=lambda: CardTypes(
        is_castable=False,
        is_permanent=False,
        is_non_land_permanent=False,
        is_non_creature_permanent=False,
        is_spell=False,
        is_creature=False,
        is_land=False,
        is_planeswalker=False,
        is_enchantment=False,
        is_artifact=False,
        is_kindred=False,
        is_battle=False,
    ))

    controller_id: Optional[int] = None
    tapped: bool = False
    damage: int = 0
    is_summoning_sick: bool = False

    def to_array(self, config: FeatureConfig) -> np.ndarray:
        """
        Convert this object to a numeric feature vector based on config.
        """
        import numpy as np

        base_fields = [
            self.id,
            int(self.zone),
            self.owner_id,
            self.power,
            self.toughness,
            self.damage,
            self.controller_id if self.controller_id is not None else -1
        ]

        # Registry encoding
        if config.registry_embeddings and config.card_embedding:
            reg_part = config.card_embedding.get(self.registry_key)
        elif config.registry_one_hot:
            # Example: a fixed 2000-size array
            arr_size = 2000
            reg_part = np.zeros(arr_size, dtype=np.int32)
            if 0 <= self.registry_key < arr_size:
                reg_part[self.registry_key] = 1
        else:
            reg_part = np.array([self.registry_key], dtype=np.int32)

        card_features = []
        if config.include_card_features:
            card_features.extend(self.mana_cost)  # 6 elements
            card_features.append(self.mana_value)
            card_features.append(int(self.card_type.is_creature))
            card_features.append(int(self.card_type.is_land))
            card_features.append(int(self.card_type.is_permanent))

        perm_features = []
        if config.include_permanent_features and self.controller_id is not None:
            perm_features.append(int(self.tapped))
            perm_features.append(int(self.is_summoning_sick))
        else:
            perm_features.extend([0, 0])

        parts = [
            np.array(base_fields, dtype=np.int32),
            reg_part,
            np.array(card_features, dtype=np.int32) if card_features else np.array([], dtype=np.int32),
            np.array(perm_features, dtype=np.int32)
        ]

        # If using float embeddings, cast everything to float so it can be concatenated
        if config.registry_embeddings and config.card_embedding:
            float_parts = []
            for p in parts:
                if p.dtype != np.float32:
                    p = p.astype(np.float32)
                float_parts.append(p)
            return np.concatenate(float_parts, axis=0)
        else:
            return np.concatenate(parts, axis=0)


class RepresentationEncoder:
    """
    Converts Observations into a 2D array of shape [num_objects, feature_dim].
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def denormalize_objects(self, obs: Observation) -> List[GameObject]:
        dn_objects = []
        for cid, cdat in obs.cards.items():
            go = GameObject(
                id=cdat.id,
                zone=cdat.zone,
                owner_id=cdat.owner_id,
                registry_key=cdat.registry_key,
                power=cdat.power,
                toughness=cdat.toughness,
                mana_cost=cdat.mana_cost.cost,
                mana_value=cdat.mana_cost.mana_value,
                card_type=cdat.card_types
            )
            # If on battlefield, overlay Permanent
            if cdat.zone == ZoneEnum.BATTLEFIELD and cid in obs.permanents:
                pdat = obs.permanents[cid]
                go.controller_id = pdat.controller_id
                go.tapped = pdat.tapped
                go.damage = pdat.damage
                go.is_summoning_sick = pdat.is_summoning_sick

            dn_objects.append(go)

        return dn_objects

    def encode_objects(self, dn_objects: List[GameObject]) -> np.ndarray:
        import numpy as np
        if not dn_objects:
            return np.zeros((0, 0), dtype=np.int32)

        arrays = [obj.to_array(self.config) for obj in dn_objects]
        shape_set = {arr.shape for arr in arrays}
        if len(shape_set) != 1:
            raise ValueError(f"Inconsistent shapes among objects: {shape_set}")

        mat = np.stack(arrays, axis=0)

        # Optionally pad
        if self.config.max_objects is not None and self.config.max_objects > 0:
            if mat.shape[0] < self.config.max_objects:
                diff = self.config.max_objects - mat.shape[0]
                fill_val = -1
                if mat.dtype == np.float32:
                    fill_val = -1.0
                pad_block = np.full((diff, mat.shape[1]), fill_val, dtype=mat.dtype)
                mat = np.concatenate([mat, pad_block], axis=0)
            else:
                mat = mat[: self.config.max_objects]

        return mat

    def encode_game_state(self, obs: Observation) -> np.ndarray:
        dn_objs = self.denormalize_objects(obs)
        return self.encode_objects(dn_objs)
