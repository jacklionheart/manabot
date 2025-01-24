from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings
import numpy as np
import torch

from .observation import (
    Observation,
    PhaseEnum,
    StepEnum,
    ActionEnum,
    ZoneEnum
)

from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings
import numpy as np
import torch

from .observation import (
    Observation,
    PhaseEnum,
    StepEnum,
    ActionEnum,
    ZoneEnum
)

@dataclass
class InputTensorSpace:
    """
    InputTensorSpace defines the tensor representation for neural network input.
    The representation consists of five main components:
    - global: Game-level state (turn, phase, etc.)
    - players: Per-player features
    - cards: Card features across all zones
    - permanents: Features for objects on the battlefield
    - actions: Available actions and their focus objects

    Focus objects (referenced by actions) are represented as concatenated 
    player/card/permanent features, with only the relevant section populated.
    """
    # Basic capacities
    num_players: int = 2
    max_cards: int = 100
    max_permanents: int = 50
    max_actions: int = 20
    max_focus_objects: int = 2

    # Enumeration sizes (derived from game rules)
    phase_count: int = len(PhaseEnum)    # 5
    step_count: int = len(StepEnum)      # 12
    action_count: int = len(ActionEnum)  # 5
    zone_count: int = len(ZoneEnum)      # 7

    @property
    def global_dim(self) -> int:
        """
        Number of features in the global state vector:
        - turn_number (1)
        - phase one-hot (phase_count)
        - step one-hot (step_count)
        - game_over, won flags (2)
        """
        return 1 + self.phase_count + self.step_count + 2

    @property
    def player_dim(self) -> int:
        """
        Features per player:
        - life total (1)
        - is_active flag (1)
        - is_agent flag (1)
        - zone_counts array (zone_count)
        """
        return 1 + 1 + 1 + self.zone_count

    @property
    def card_dim(self) -> int:
        """
        Features per card:
        - zone one-hot (zone_count)
        - owner_id (1)
        - power, toughness (2)
        - mana_value (1)
        - type flags (6): land, creature, artifact, enchantment, planeswalker, battle
        """
        return self.zone_count + 1 + 2 + 1 + 6

    @property
    def permanent_dim(self) -> int:
        """
        Features per permanent:
        - controller_id (1)
        - tapped flag (1)
        - damage (1)
        - power, toughness (2)
        - summoning_sick flag (1)
        - type flags (4): land, creature, artifact, enchantment
        """
        return 1 + 1 + 1 + 2 + 1 + 4

    @property
    def focus_object_dim(self) -> int:
        """
        Dimension for a focus object, which could be any game object.
        We concatenate space for all types, filling only the relevant section:
        - player features
        - card features
        - permanent features
        """
        return self.player_dim + self.card_dim + self.permanent_dim

    @property
    def action_dim(self) -> int:
        """
        Features per action:
        - action_type one-hot (action_count)
        - focus objects (max_focus_objects * focus_object_dim)
        """
        return self.action_count + (self.max_focus_objects * self.focus_object_dim)

    @property 
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns expected shapes for each tensor in the encoded observation.
        """
        return {
            'global': (self.global_dim,),
            'players': (self.num_players, self.player_dim),
            'cards': (self.max_cards, self.card_dim),
            'permanents': (self.max_permanents, self.permanent_dim),
            'actions': (self.max_actions, self.action_dim)
        }

    def encode_global(self, obs: Observation) -> torch.Tensor:
        """
        Encode global game state features.
        
        Returns:
            torch.Tensor with shape (global_dim,) containing turn number,
            phase/step one-hots, and game state flags.
        
        Raises:
            ValueError: If the encoded tensor shape doesn't match expected shape.
        """
        arr = np.zeros(self.shapes['global'], dtype=np.float32)
        i = 0
        
        # Turn number
        arr[i] = float(obs.turn.turn_number)
        i += 1
        
        # Phase one-hot
        phase_hot = np.zeros(self.phase_count, dtype=np.float32)
        if 0 <= obs.turn.phase < self.phase_count:
            phase_hot[obs.turn.phase] = 1.0
        arr[i:i + self.phase_count] = phase_hot
        i += self.phase_count
        
        # Step one-hot
        step_hot = np.zeros(self.step_count, dtype=np.float32)
        if 0 <= obs.turn.step < self.step_count:
            step_hot[obs.turn.step] = 1.0
        arr[i:i + self.step_count] = step_hot
        i += self.step_count
        
        # Game state flags
        arr[i] = 1.0 if obs.game_over else 0.0
        i += 1
        arr[i] = 1.0 if obs.won else 0.0
        
        tensor = torch.tensor(arr, dtype=torch.float32)
        expected_shape = self.shapes['global']
        if tensor.shape != expected_shape:
            raise ValueError(f"Global tensor shape mismatch: got {tensor.shape}, expected {expected_shape}")
        return tensor

    def encode_players(self, obs: Observation) -> torch.Tensor:
        """
        Encode player features into a fixed-size tensor.
        
        Returns:
            torch.Tensor with shape (num_players, player_dim) containing
            encoded features for each player.
            
        Raises:
            ValueError: If the encoded tensor shape doesn't match expected shape.
        """
        arr = np.zeros(self.shapes['players'], dtype=np.float32)
        for idx, (pid, player) in enumerate(obs.players.items()):
            if idx >= self.num_players:
                warnings.warn(f"Too many players ({len(obs.players)}); truncating to {self.num_players}.")
                break
                
            row = np.zeros(self.player_dim, dtype=np.float32)
            i = 0
            row[i] = float(player.life)
            i += 1
            row[i] = 1.0 if player.is_active else 0.0
            i += 1
            row[i] = 1.0 if player.is_agent else 0.0
            i += 1
            row[i:i + self.zone_count] = player.zone_counts[:self.zone_count]
            arr[idx] = row
            
        tensor = torch.tensor(arr, dtype=torch.float32)
        expected_shape = self.shapes['players']
        if tensor.shape != expected_shape:
            raise ValueError(f"Players tensor shape mismatch: got {tensor.shape}, expected {expected_shape}")
        return tensor

    def encode_cards(self, obs: Observation) -> torch.Tensor:
        """
        Encode card features into a fixed-size tensor.
        
        Returns:
            torch.Tensor with shape (max_cards, card_dim) containing
            encoded features for each card.
            
        Raises:
            ValueError: If the encoded tensor shape doesn't match expected shape.
        """
        arr = np.zeros(self.shapes['cards'], dtype=np.float32)
        card_ids = sorted(obs.cards.keys())
        if len(card_ids) > self.max_cards:
            warnings.warn(f"Too many cards ({len(card_ids)}); truncating to {self.max_cards}.")
            
        for idx, cid in enumerate(card_ids[:self.max_cards]):
            card = obs.cards[cid]
            row = np.zeros(self.card_dim, dtype=np.float32)
            i = 0
            
            # Zone one-hot
            zone_hot = np.zeros(self.zone_count, dtype=np.float32)
            if 0 <= card.zone < self.zone_count:
                zone_hot[card.zone] = 1.0
            row[i:i + self.zone_count] = zone_hot
            i += self.zone_count
            
            # Basic properties
            row[i] = float(card.owner_id)
            i += 1
            row[i] = float(card.power)
            i += 1
            row[i] = float(card.toughness)
            i += 1
            row[i] = float(card.mana_cost.mana_value)
            i += 1
            
            # Type flags
            row[i] = 1.0 if card.card_types.is_land else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_creature else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_artifact else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_enchantment else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_planeswalker else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_battle else 0.0
            
            arr[idx] = row
            
        tensor = torch.tensor(arr, dtype=torch.float32)
        expected_shape = self.shapes['cards']
        if tensor.shape != expected_shape:
            raise ValueError(f"Cards tensor shape mismatch: got {tensor.shape}, expected {expected_shape}")
        return tensor

    def encode_permanents(self, obs: Observation) -> torch.Tensor:
        """
        Encode permanent features into a fixed-size tensor.
        
        Returns:
            torch.Tensor with shape (max_permanents, permanent_dim) containing
            encoded features for each permanent.
            
        Raises:
            ValueError: If the encoded tensor shape doesn't match expected shape.
        """
        arr = np.zeros(self.shapes['permanents'], dtype=np.float32)
        perm_ids = sorted(obs.permanents.keys())
        if len(perm_ids) > self.max_permanents:
            warnings.warn(f"Too many permanents ({len(perm_ids)}); truncating to {self.max_permanents}.")
            
        for idx, pid in enumerate(perm_ids[:self.max_permanents]):
            perm = obs.permanents[pid]
            row = np.zeros(self.permanent_dim, dtype=np.float32)
            i = 0
            
            # Basic properties
            row[i] = float(perm.controller_id)
            i += 1
            row[i] = 1.0 if perm.tapped else 0.0
            i += 1
            row[i] = float(perm.damage)
            i += 1
            row[i] = 0.0  # power placeholder
            i += 1
            row[i] = 0.0  # toughness placeholder
            i += 1
            row[i] = 1.0 if perm.is_summoning_sick else 0.0
            i += 1
            
            # Type flags
            row[i] = 1.0 if perm.is_land else 0.0
            i += 1
            row[i] = 1.0 if perm.is_creature else 0.0
            i += 1
            row[i] = 0.0  # is_artifact placeholder
            i += 1
            row[i] = 0.0  # is_enchantment placeholder
            
            arr[idx] = row
            
        tensor = torch.tensor(arr, dtype=torch.float32)
        expected_shape = self.shapes['permanents']
        if tensor.shape != expected_shape:
            raise ValueError(f"Permanents tensor shape mismatch: got {tensor.shape}, expected {expected_shape}")
        return tensor

    def encode_focus_object(self, obs: Observation, obj_id: int) -> np.ndarray:
        """
        Encode a single focus object (player/card/permanent) into the universal format.
        The encoding has three sections in this order:
        1. Player features (if obj_id is a player)
        2. Card features (if obj_id is a card)
        3. Permanent features (if obj_id is a permanent)
        Only one section will be populated; others remain zero.

        Returns:
            np.ndarray: Focus object encoding with shape (focus_object_dim,)
        """
        row = np.zeros(self.focus_object_dim, dtype=np.float32)
        
        # Try encoding as player (offset = 0)
        if obj_id in obs.players:
            player = obs.players[obj_id]
            i = 0  # Player section starts at beginning
            
            # Life total
            row[i] = float(player.life)
            i += 1
            # Activity flags
            row[i] = 1.0 if player.is_active else 0.0
            i += 1
            row[i] = 1.0 if player.is_agent else 0.0
            i += 1
            # Zone counts
            row[i:i + self.zone_count] = player.zone_counts[:self.zone_count]
            
        # Try encoding as card (offset = player_dim)
        elif obj_id in obs.cards:
            card = obs.cards[obj_id]
            i = self.player_dim  # Card section starts after player section
            
            # Zone one-hot
            zone_hot = np.zeros(self.zone_count, dtype=np.float32)
            if 0 <= card.zone < self.zone_count:
                zone_hot[card.zone] = 1.0
            row[i:i + self.zone_count] = zone_hot
            i += self.zone_count
            
            # Basic properties
            row[i] = float(card.owner_id)
            i += 1
            row[i] = float(card.power)
            i += 1
            row[i] = float(card.toughness)
            i += 1
            row[i] = float(card.mana_cost.mana_value)
            i += 1
            
            # Type flags
            row[i] = 1.0 if card.card_types.is_land else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_creature else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_artifact else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_enchantment else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_planeswalker else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_battle else 0.0
            
        # Try encoding as permanent (offset = player_dim + card_dim)
        elif obj_id in obs.permanents:
            permanent = obs.permanents[obj_id]
            i = self.player_dim + self.card_dim  # Permanent section starts after player and card sections
            
            # Controller and state
            row[i] = float(permanent.controller_id)
            i += 1
            row[i] = 1.0 if permanent.tapped else 0.0
            i += 1
            row[i] = float(permanent.damage)
            i += 1
            row[i] = 0.0  # power placeholder
            i += 1
            row[i] = 0.0  # toughness placeholder
            i += 1
            row[i] = 1.0 if permanent.is_summoning_sick else 0.0
            i += 1
            
            # Type flags
            row[i] = 1.0 if permanent.is_land else 0.0
            i += 1
            row[i] = 1.0 if permanent.is_creature else 0.0
            i += 1
            row[i] = 0.0  # is_artifact placeholder
            i += 1
            row[i] = 0.0  # is_enchantment placeholder
            
        return row
            
        offset += self.player_dim
        
        # Try encoding as card
        if obj_id in obs.cards:
            card = obs.cards[obj_id]
            i = offset
            zone_hot = np.zeros(self.zone_count, dtype=np.float32)
            if 0 <= card.zone < self.zone_count:
                zone_hot[card.zone] = 1.0
            row[i:i + self.zone_count] = zone_hot
            i += self.zone_count
            row[i] = float(card.owner_id)
            i += 1
            row[i] = float(card.power)
            i += 1
            row[i] = float(card.toughness)
            i += 1
            row[i] = float(card.mana_cost.mana_value)
            i += 1
            row[i] = 1.0 if card.card_types.is_land else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_creature else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_artifact else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_enchantment else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_planeswalker else 0.0
            i += 1
            row[i] = 1.0 if card.card_types.is_battle else 0.0
            return row
            
        offset += self.card_dim
        
        # Try encoding as permanent
        if obj_id in obs.permanents:
            perm = obs.permanents[obj_id]
            i = offset
            row[i] = float(perm.controller_id)
            i += 1
            row[i] = 1.0 if perm.tapped else 0.0
            i += 1
            row[i] = float(perm.damage)
            i += 1
            row[i] = 0.0  # power placeholder
            i += 1
            row[i] = 0.0  # toughness placeholder
            i += 1
            row[i] = 1.0 if perm.is_summoning_sick else 0.0
            i += 1
            row[i] = 1.0 if perm.is_land else 0.0
            i += 1
            row[i] = 1.0 if perm.is_creature else 0.0
            i += 1
            row[i] = 0.0  # is_artifact placeholder
            i += 1
            row[i] = 0.0  # is_enchantment placeholder
            
        return row

    def encode_actions(self, obs: Observation) -> torch.Tensor:
        """
        Encode available actions and their focus objects into a fixed-size tensor.
        
        Returns:
            torch.Tensor with shape (max_actions, action_dim) containing
            encoded features for each action and its focus objects.
            
        Raises:
            ValueError: If the encoded tensor shape doesn't match expected shape.
        """
        arr = np.zeros(self.shapes['actions'], dtype=np.float32)
        
        for i, action in enumerate(obs.action_space.actions[:self.max_actions]):
            if i >= self.max_actions:
                warnings.warn(f"Too many actions ({len(obs.action_space.actions)}); truncating to {self.max_actions}.")
                break
                
            # Action type one-hot
            action_hot = np.zeros(self.action_count, dtype=np.float32)
            if 0 <= action.action_type < self.action_count:
                action_hot[action.action_type] = 1.0
            arr[i, :self.action_count] = action_hot
            
            # Encode focus objects
            for f_idx, focus_id in enumerate(action.focus[:self.max_focus_objects]):
                if f_idx >= self.max_focus_objects:
                    warnings.warn(f"Action {i} has too many focus objects; truncating to {self.max_focus_objects}.")
                    break
                    
                # Calculate offset into the action tensor for this focus object
                focus_offset = self.action_count + (f_idx * self.focus_object_dim)
                
                # Encode the focus object and place it in the action tensor
                focus_encoding = self.encode_focus_object(obs, focus_id)
                arr[i, focus_offset:focus_offset + self.focus_object_dim] = focus_encoding
                
        tensor = torch.tensor(arr, dtype=torch.float32)
        expected_shape = self.shapes['actions']
        if tensor.shape != expected_shape:
            raise ValueError(f"Actions tensor shape mismatch: got {tensor.shape}, expected {expected_shape}")
        return tensor

    def encode_full_observation(self, obs: Observation) -> Dict[str, torch.Tensor]:
        """
        Encode a complete observation into a dictionary of tensors.
        
        Returns:
            Dictionary with keys matching self.shapes, containing encoded tensors
            for each aspect of the game state.
            
        Raises:
            ValueError: If any encoded tensor doesn't match its expected shape.
            
        Note:
            This method verifies that all encoded tensors match their expected
            shapes from self.shapes before returning. Individual encoding methods
            also perform their own shape validation.
        """
        tensors = {
            'global': self.encode_global(obs),
            'players': self.encode_players(obs),
            'cards': self.encode_cards(obs),
            'permanents': self.encode_permanents(obs),
            'actions': self.encode_actions(obs)
        }
        
        # Double-check all shapes match expected shapes
        for name, tensor in tensors.items():
            expected_shape = self.shapes[name]
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in encode_full_observation for {name}: "
                    f"got {tensor.shape}, expected {expected_shape}"
                )
                
        return tensors

    def get_total_dims(self) -> Dict[str, int]:
        """
        Calculate total dimensions (number of elements) for each tensor type.
        Useful for network architecture design.
        """
        return {
            name: int(np.prod(shape)) 
            for name, shape in self.shapes.items()
        }

    def debug_string(self) -> str:
        """Generate a debug representation of the tensor space configuration."""
        lines = [
            "=== InputTensorSpace Configuration ===",
            "Capacity:",
            f"  Players: {self.num_players}",
            f"  Cards: {self.max_cards}",
            f"  Permanents: {self.max_permanents}",
            f"  Actions: {self.max_actions}",
            f"  Focus Objects per Action: {self.max_focus_objects}",
            "",
            "Feature Dimensions:",
            f"  Global: {self.global_dim}",
            f"  Player: {self.player_dim}",
            f"  Card: {self.card_dim}",
            f"  Permanent: {self.permanent_dim}",
            f"  Focus Object: {self.focus_object_dim}",
            f"  Action: {self.action_dim}",
            "",
            "Output Shapes:"
        ]
        for name, shape in self.shapes.items():
            lines.append(f"  {name}: {shape}")
            
        return "\n".join(lines)