# manabot/data/observation.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional
from torch import Tensor

import numpy as np
import torch
from zmq import device

from manabot.odld.model import card_embeddings

class Zone(IntEnum):
    LIBRARY = 0
    HAND = 1
    BATTLEFIELD = 2
    GRAVEYARD = 3
    EXILE = 4
    STACK = 5
    COMMAND = 6

class Step(IntEnum):
    UNTAP = 0
    UPKEEP = 1
    DRAW = 2
    MAIN1 = 3
    COMBAT_BEGIN = 4
    COMBAT_ATTACK = 5
    COMBAT_BLOCK = 6 
    COMBAT_DAMAGE = 7
    COMBAT_END = 8
    MAIN2 = 9
    END = 10
    CLEANUP = 11


class ActionType(IntEnum):
    PASS = 0
    DECLARE_ATTACKER = 1
    DECLARE_BLOCKER = 2
    CAST_SPELL = 3
    ACTIVATE_ABILITY = 4
    PLAY_LAND = 5


@dataclass
class ActionOption:
    target_id: int  # What the action affects
    source_id: int  # What's taking the action 
    valid: bool  # Whether action is legal

@dataclass
class Permanent:
    """Single permanent instance in a game state"""
    card_id: int
    owner_id: int
    controller_id: int
    tapped: bool = False
    attacking: bool = False
    blocking: bool = False
    summoning_sick: bool = False
    power: int = 0
    toughness: int = 0
    damage: int = 0

@dataclass
class CardEmbeddings:
    """Mapping of card IDs to their embeddings"""
    card_to_embedding: Dict[int, Tensor]
    embedding_dim: int
    pad_idx: int = 0

    def embed_field(self, tensor: Tensor, field_idx: int) -> Tensor:
        """
        Replace a field with embeddings and concatenate with remaining features
        
        Args:
            tensor: The tensor containing card IDs
            field_idx: Index of the column containing card IDs to embed
            
        Returns:
            Tensor with field replaced by embeddings and concatenated with other features
        """
        # Get the indices that need to be embedded
        card_ids = tensor[:, field_idx]
        
        # Create embeddings
        embedded = torch.stack([
            self.card_to_embedding.get(int(id), self.card_to_embedding[self.pad_idx])
            for id in card_ids
        ])
        
        # Remove the original column and concatenate with embedding
        return torch.cat([tensor[:, :field_idx], embedded, tensor[:, field_idx+1:]], dim=1)
    
@dataclass
class ObservationSpace:
    """Defines the shape and structure of observations."""
    card_embeddings: CardEmbeddings

    max_permanents: int
    max_actions: int

    @property
    def game_shape(self) -> torch.Size:
        """Shape of game state tensor."""
        return torch.Size([9])  # Fixed game features from Observation
        
    @property
    def permanent_shape(self) -> torch.Size:
        """Shape of a single permanent (including embeddings)."""
        return torch.Size([self.card_embeddings.embedding_dim + 9])  # embedding + permanent features
        
    @property
    def action_shape(self) -> torch.Size:
        """Shape of a single action (including embeddings)."""
        return torch.Size([2 * self.card_embeddings.embedding_dim + 1])  # 2 embeddings + valid flag

    @property
    def shapes(self) -> Dict[str, torch.Size]:
        """Complete observation shapes."""
        return {
            'game': self.game_shape,
            'permanents': torch.Size([self.max_permanents, *self.permanent_shape]),
            'actions': torch.Size([self.max_actions, *self.action_shape])
        }
    
    @property
    def shape(self) -> torch.Size:
        """Shape of all observation tensors."""
        return torch.Size([sum(shape.numel() for shape in self.shapes.values())])
    
    

@dataclass
class Observation:
    """Complete game state snapshot"""
    # Core game state
    game_over: bool
    won: bool
    active_player: int
    turn_number: int
    lands_played_this_turn: int
    step: Step
    life_totals: List[int]

    battlefield: List[Permanent]
    action_options: List[ActionOption]

    action_type: ActionType

    def to_tensors(self, card_embeddings: CardEmbeddings, num_permanents: int, num_actions: int, device: str = 'cuda') -> Dict[str, Tensor]:
        """Convert to tensor format with card embeddings and fixed sizes."""
        # Pad battlefield to fixed size
        padded_battlefield = self.battlefield + [ Permanent(card_id=0, owner_id=0, controller_id=0) for _ in range(num_permanents - len(self.battlefield)) ]
        
        permanent_tensor = torch.tensor([[
            # Embedding Key
            p.card_id,
            # Retained features
            p.owner_id,
            p.controller_id,
            int(p.tapped),
            int(p.attacking),
            int(p.blocking),
            int(p.summoning_sick),
            p.power,
            p.toughness,
            p.damage
        ] for p in padded_battlefield[:num_permanents]], device=device)

        # Embed card_ids for permanents
        permanent_tensor = card_embeddings.embed_field(permanent_tensor, field_idx=0)
    
        # Pad actions to fixed size
        padded_actions = self.action_options + [
            ActionOption(source_id=0, target_id=0, valid=False)  # Pad with invalid action
            for _ in range(num_actions - len(self.action_options))
        ]
        action_tensor = torch.tensor([[
            a.source_id,
            a.target_id,
            int(a.valid)
        ] for a in padded_actions[:num_actions]], device=device)
        
        # First embed the source
        action_tensor = card_embeddings.embed_field(action_tensor, field_idx=0)  
        # Then embed target, which is now at index embedding_dim
        action_tensor = card_embeddings.embed_field(action_tensor, field_idx=card_embeddings.embedding_dim)
            
        # Core game state features
        game_tensor = torch.tensor([
            self.active_player,
            self.turn_number,
            int(self.step),
            *self.life_totals,
            int(self.action_type),
            self.lands_played_this_turn,
            int(self.game_over),
            int(self.won)
        ], device=device)

        return {
            'permanents': permanent_tensor,  # Shape: (num_permanents, embedding_dim + 9)
            'actions': action_tensor,        # Shape: (num_actions, 2*embedding_dim + 1)
            'game': game_tensor             # Shape: (10,)
        }

