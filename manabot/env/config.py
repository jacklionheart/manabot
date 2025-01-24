from dataclasses import dataclass
from typing import Dict

import managym

@dataclass
class PlayerConfig:
    """Python-side player config that we pass into manabot Env."""
    name: str
    deck: Dict[str, int]

    def to_cpp(self) -> "managym.PlayerConfig":
        """Convert to the underlying C++/pybind PlayerConfig."""
        return managym.PlayerConfig(self.name, self.deck)
