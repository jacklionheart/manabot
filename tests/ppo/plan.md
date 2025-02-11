# Testing and Evaluation Plan for manabot PPO (Revised)

## 1. Model Behavioral Analysis

### Key Metrics to Track
- Land play rate when possible (lands in hand / lands played)
- Attack frequency when able (creatures that could attack / did attack)
- Blocking decisions (available blockers / blocks declared)

### Implementation
```python
def log_behavioral_metrics(env_info, step):
    """Extract and log behavioral metrics from env info"""
    if 'action_stats' in env_info:
        stats = env_info['action_stats']
        # Log land play rate
        if stats['lands_in_hand'] > 0:
            land_play_rate = stats['lands_played'] / stats['lands_in_hand']
            writer.add_scalar("behavior/land_play_rate", land_play_rate, step)
        
        # Log attack rate
        if stats['possible_attackers'] > 0:
            attack_rate = stats['actual_attackers'] / stats['possible_attackers']
            writer.add_scalar("behavior/attack_rate", attack_rate, step)
```

## 2. Policy Comparison Tool (scripts/compare_agents.py)

### Head-to-Head Evaluation
```python
class PolicyComparator:
    def __init__(self, env, agent1, agent2, min_games=100):
        self.env = env
        self.agents = {0: agent1, 1: agent2}
        self.results = []
        self.min_games = min_games
        
    def evaluate(self):
        """Run full evaluation suite"""
        self.play_games(self.min_games)
        return {
            'win_rate': self.compute_win_rate(),
            'elo_diff': self.compute_elo_difference(),
            'avg_game_length': np.mean([r['length'] for r in self.results])
        }
        
    def compute_elo_difference(self):
        """Basic ELO calculation based on win rates"""
        win_rate = self.compute_win_rate()
        return -400 * math.log10(1/win_rate - 1)
```

### Minimum Requirements
- 100 games for initial comparison
- Track both win rates and game lengths
- Basic ELO implementation
- Support for saving/loading comparison results
