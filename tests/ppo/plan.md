# Testing and Evaluation Plan for manabot PPO (Revised)

## 1. Core Testing Suite (test_trainer.py)

### Rollout and Buffer Management
- **Objective:** Verify correct transition collection and multi-agent handling
- **Key Tests:**
  ```python
  def test_rollout_step():
      # Verify for a single rollout step:
      # - Correct actor_id extraction from obs
      # - Transitions routed to correct player buffer
      # - Tensor shapes match ObservationSpace
      # - Action selection uses correct player's policy
  
  def test_buffer_overflow():
      # Fill buffer beyond capacity
      # Verify warning is logged
      # Check transitions are dropped correctly
  
  def test_observation_validation():
      # Test _validate_obs with:
      # - Valid observations
      # - Missing keys
      # - Wrong tensor shapes
      # Verify consecutive invalid batch counting
  ```

### Advantage Computation
- **Objective:** Validate GAE calculations with known scenarios
- **Key Tests:**
  ```python
  def test_gae_constant_rewards():
      # Set up rollout with constant rewards (e.g., 1.0)
      # Calculate expected GAE values by hand
      # Compare against computed advantages
  
  def test_gae_alternating_rewards():
      # Create rollout with alternating rewards (+1/-1)
      # Verify advantage signs and magnitudes
      # Check proper player attribution
  ```

### PPO Optimization
- **Objective:** Verify PPO loss computation and gradient flow
- **Key Tests:**
  ```python
  def test_ppo_update():
      # Run single optimization step
      # Verify:
      # - KL divergence is finite
      # - Gradients exist and are non-zero
      # - Early stopping works if KL exceeds threshold
  
  def test_value_loss():
      # Check both clipped and unclipped value losses
      # Verify proper scaling by vf_coef
  ```

## 2. Quick Training Script (scripts/train_quick.py)

### Configuration
```python
QUICK_TRAIN_CONFIG = {
    'num_envs': 4,        # Small enough for CPU
    'num_steps': 128,     # Standard PPO window
    'total_timesteps': 100000,  # ~1hr on MacBook
    'batch_size': 512,
    'learning_rate': 2.5e-4
}
```

### Success Metrics
- Policy loss should decrease by ≥20% over first 50k steps
- Value loss should show downward trend
- Explained variance should increase above 0
- No GPU required; should run on CPU in ~1 hour

### Checkpointing
- Save every 10k steps
- Verify can load and resume training
- Test checkpoint contains:
  - Agent state
  - Optimizer state
  - Training step count

## 3. Logging Infrastructure

### TensorBoard Metrics
- Per-update metrics:
  - Learning rate
  - Policy/value/entropy losses
  - KL divergence
  - Advantage statistics
  - SPS (steps per second)

### Per-Player Statistics
- Episodic returns and lengths
- Buffer sizes and utilization
- Mean values and advantages

### Error Handling
- Log warnings for buffer overflow
- Error after N consecutive invalid batches
- Track observation validation failures

## 4. Model Behavioral Analysis

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

## 5. Policy Comparison Tool (scripts/compare_agents.py)

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

## 6. External Service Integration

### TensorBoard
```bash
# Local viewing
tensorboard --logdir=runs

# Remote setup (if needed)
tensorboard --logdir=runs --port=6006 --bind_all
```

### Weights & Biases
Configuration in experiment.py:
```python
if self.track:
    wandb.init(
        project=self.wandb_project_name,
        entity=self.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
    )
```

---

## Implementation Priority Order:
1. Core testing suite
2. Quick training script
3. Basic behavioral metrics
4. Policy comparison tool
5. Extended logging/analysis
