"""
match.py
Defines configuration for a Magic: The Gathering match between two players.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any
import json
import argparse

import managym

from manabot.infra.hypers import MatchHypers

@dataclass
class Match:
    """Python-side player configuration that we pass into manabot Env.
    
    This represents a match between two players (hero and villain) with their
    respective decks. The default deck is a simple two-color aggressive deck.
    """
    hero: str
    villain: str
    hero_deck: Dict[str, int]
    villain_deck: Dict[str, int] 
    hypers: MatchHypers

    def __init__(self, hypers: MatchHypers = MatchHypers()):
        self.hypers = hypers
        self.hero = hypers.hero
        self.villain = hypers.villain
        self.hero_deck = deepcopy(hypers.hero_deck)
        self.villain_deck = deepcopy(hypers.villain_deck)

    def to_cpp_hero(self) -> "managym.PlayerConfig":
        """Convert hero configuration to C++ format."""
        return managym.PlayerConfig(self.hero, self.hero_deck)
    
    def to_cpp_villain(self) -> "managym.PlayerConfig":
        """Convert villain configuration to C++ format."""
        return managym.PlayerConfig(self.villain, self.villain_deck)
    
    def to_cpp(self) -> "list[managym.PlayerConfig]":
        """Convert entire match configuration to C++ format."""
        return [self.to_cpp_hero(), self.to_cpp_villain()]

    def __str__(self) -> str:
        """Return a human-readable string representation of the match."""
        return f"Match({self.hero} vs {self.villain})"

    
def parse_deck(deck_str: str) -> Dict[str, int]:
    """Parse a deck string into a dictionary of card counts.
    
    Accepts either:
    1. JSON string: '{"Mountain": 12, "Forest": 12}'
    2. Simple format: 'Mountain:12,Forest:12'
    """
    try:
        # First try JSON format
        return json.loads(deck_str)
    except json.JSONDecodeError:
        # Fall back to simple format
        deck = {}
        for pair in deck_str.split(','):
            if ':' not in pair:
                raise ValueError(f"Invalid deck format: {deck_str}")
            card, count = pair.split(':')
            deck[card.strip()] = int(count)
        return deck
