"""
evaluate.py
Simplified and consolidated evaluation package for manabot.

This module provides:
1. Loading models from wandb
2. Player abstractions for model inference
3. Basic game simulation and statistics tracking
4. Action distribution and decision analysis
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import wandb
import threading
from collections import defaultdict, Counter

from manabot.ppo.agent import Agent
from manabot.env import Env, Match, Reward, ObservationSpace
from manabot.env.observation import get_agent_indices
from manabot.infra.hypers import MatchHypers, RewardHypers, add_hypers, parse_hypers
from manabot.infra.log import getLogger

logger = getLogger(__name__)

# -----------------------------------------------------------------------------
# Evaluation Hyperparameters
# -----------------------------------------------------------------------------

@dataclass
class EvaluationHypers:
    """Hyperparameters for model evaluation."""
    hero: str = "quick_train"
    villain: str = "random"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchHypers = field(default_factory=MatchHypers)  # Match configuration


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

    
def load_model_from_wandb(
    artifact_name: str,
    version: str = "latest", 
    project: Optional[str] = None,
    device: str = "cpu"
) -> Agent:
    """
    Load a trained model from wandb artifacts with minimal wandb interaction.
    
    Args:
        artifact_name: Name of the experiment (e.g. "quick_train")
        version: Version string (e.g. "v3" or "latest")
        project: Wandb project name
        device: Device to load model on ("cpu" or "cuda")
        
    Returns:
        Loaded agent model ready for inference
    """
    try:
        # Force online mode to ensure artifact can be fetched
        os.environ['WANDB_MODE'] = 'online'
        
        # Use a silent API object without starting a run
        api = wandb.Api()
        artifact_path = f"{project or 'manabot'}/{artifact_name + "_latest.pt"}:{version}"
        logger.info(f"Loading artifact: {artifact_path}")
        
        try:
            artifact = api.artifact(artifact_path)
        except Exception as e:
            logger.warning(f"Error loading artifact {artifact_path}: {e}")
            # Try with a more specific path format
            artifact_path = f"{project or 'manabot'}/{artifact_name}_latest.pt:{version}"
            logger.info(f"Trying alternative artifact path: {artifact_path}")
            artifact = api.artifact(artifact_path)
            
        artifact_dir = artifact.download("/tmp")
        
        # Directly use the expected filename pattern based on the save method
        potential_paths = [
            os.path.join(artifact_dir, f"{artifact_name}.pt"),
            os.path.join(artifact_dir, f"{artifact_name}_latest.pt"),
        ]
        
        # Also look for any .pt files
        pt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        potential_paths.extend([os.path.join(artifact_dir, f) for f in pt_files])
        
        # Find the first valid checkpoint file
        checkpoint_path = None
        for path in potential_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError(f"No .pt files found in the artifact at {artifact_dir}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Debug: print the checkpoint keys
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Look for model state dict with flexible key names
        state_dict_key = None
        for key in ['agent_state_dict', 'model_state_dict', 'state_dict']:
            if key in checkpoint:
                state_dict_key = key
                break
                
        if state_dict_key is None:
            raise ValueError(f"Checkpoint does not contain a state dictionary. Keys found: {list(checkpoint.keys())}")
            
        logger.info(f"Using state dictionary from key: {state_dict_key}")
            
        # Create observation space and agent
        from manabot.infra.hypers import ObservationSpaceHypers, AgentHypers
        
        # Create default observation space and agent hyperparameters
        obs_hypers = ObservationSpaceHypers()
        agent_hypers = AgentHypers()
        
        # Create observation space and agent with default hyperparameters
        obs_space = ObservationSpace(obs_hypers)
        agent = Agent(obs_space, agent_hypers)
        logger.info("Created model with default hyperparameters")
        
        # Load model weights
        agent.load_state_dict(checkpoint[state_dict_key])
        agent.eval()
        agent = agent.to(device)
        
        # See if we have information about training steps
        if 'global_step' in checkpoint:
            logger.info(f"Model was trained for {checkpoint['global_step']} steps")
        
        logger.info(f"Successfully loaded model")
        return agent
    except Exception as e:
        logger.error(f"Error loading model from wandb: {e}")
        # If the exception relates to artifact not found, show clear message
        if "not found" in str(e).lower():
            logger.error(f"Could not find artifact '{artifact_name}'. Check if the name is correct and the artifact exists.")
        # If the exception relates to model loading, print more details
        elif "state dictionary" in str(e) or "state_dict" in str(e):
            logger.error("The model file was found but its structure doesn't match expectations.")
            logger.error("This could happen if the model was saved with a different format or version.")
        raise

# -----------------------------------------------------------------------------
# Player Classes
# -----------------------------------------------------------------------------

class PlayerType(Enum):
    """Types of players for evaluation."""
    MODEL = "model"
    RANDOM = "random"
    RULE_BASED = "rule_based"  # For future rule-based players

class Player:
    """Base player class for evaluation."""
    def __init__(self, name: str, player_type: PlayerType):
        self.name = name
        self.player_type = player_type
        self.device = "cpu"
        self.wins = 0
        self.games = 0
        self.action_history = []  # Track actions for analysis
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from observation."""
        raise NotImplementedError
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.wins / self.games if self.games > 0 else 0.0
    
    def record_result(self, won: bool) -> None:
        """Record game result."""
        self.games += 1
        if won:
            self.wins += 1
    
    def record_action(self, action: int, obs: Dict[str, np.ndarray]) -> None:
        """Record an action for later analysis."""
        self.action_history.append((action, obs.get("actions", []).shape[0]))
    
    def get_action_distribution(self) -> Dict[str, float]:
        """Get distribution of action types."""
        if not self.action_history:
            return {}
        
        action_counts = Counter([action for action, _ in self.action_history])
        total = len(self.action_history)
        return {f"action_{action}": count/total for action, count in action_counts.items()}
    
    def reset_history(self) -> None:
        """Reset action history."""
        self.action_history = []
    
    def to(self, device: str) -> 'Player':
        """Move player to specified device."""
        self.device = device
        return self

class ModelPlayer(Player):
    """Player that uses a trained model for inference."""
    def __init__(
        self, 
        name: str, 
        agent: Agent, 
        deterministic: bool = True,
        record_logits: bool = False
    ):
        super().__init__(name, PlayerType.MODEL)
        self.agent = agent
        self.deterministic = deterministic
        self.device = next(agent.parameters()).device
        self.record_logits = record_logits
        self.logits_history = [] if record_logits else None
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from model."""
        # Convert numpy arrays to tensors with batch dimension
        tensor_obs = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }
        
        # Get action from model
        with torch.no_grad():
            logits, _ = self.agent(tensor_obs)
            if self.record_logits:
                assert self.logits_history is not None
                self.logits_history.append(logits.detach().cpu().numpy())
                
            action, _, _, _ = self.agent.get_action_and_value(
                tensor_obs, deterministic=self.deterministic)
            
            # Record the action for analysis
            action_value = action.item()
            self.record_action(action_value, obs)
            return action_value
    
    def get_action_confidence(self) -> Dict[str, float]:
        """Get statistics about action confidence."""
        if not self.logits_history:
            return {}
        
        # Calculate softmax probabilities
        probs = []
        for logits in self.logits_history:
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs.append(exp_logits / np.sum(exp_logits, axis=1, keepdims=True))
        
        # Calculate statistics
        chosen_probs = [p.max() for p in probs]
        return {
            "mean_confidence": np.mean(chosen_probs),
            "min_confidence": np.min(chosen_probs),
            "max_confidence": np.max(chosen_probs),
        }
    
    def to(self, device: str) -> 'ModelPlayer':
        """Move player and model to device."""
        super().to(device)
        self.agent = self.agent.to(device)
        return self

class RandomPlayer(Player):
    """Player that selects random valid actions."""
    def __init__(self, name: str):
        super().__init__(name, PlayerType.RANDOM)
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get random valid action."""
        valid_actions = np.where(obs["actions_valid"] > 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        
        action = int(np.random.choice(valid_actions))
        self.record_action(action, obs)
        return action

# -----------------------------------------------------------------------------
# Game Statistics
# -----------------------------------------------------------------------------

class GameOutcome(Enum):
    """Possible game outcomes."""
    HERO_WIN = "hero_win"
    VILLAIN_WIN = "villain_win"
    TIMEOUT = "timeout"

class GameStats:
    """Game statistics tracking with enhanced analysis capabilities."""
    
    def __init__(self):
        self.games = []
        self.hero_wins = 0
        self.villain_wins = 0
        self.timeouts = 0
        self.total_steps = 0
        self.total_duration = 0
        self.lock = threading.Lock()

        # Enhanced tracking
        self.steps_to_win = []  # Track steps taken in winning games
        self.phase_distributions = defaultdict(int)  # Track game phases
        self.game_records = []  # Detailed game records
    
    def record_game(
        self, 
        outcome: GameOutcome, 
        steps: int, 
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a completed game with optional metadata.
        
        Args:
            outcome: Game outcome (hero win, villain win, timeout)
            steps: Number of steps in the game
            duration: Time taken for the game in seconds
            metadata: Optional additional game data for analysis
        """
        game_record = {
            "outcome": outcome,
            "steps": steps,
            "duration": duration
        }
        
        if metadata:
            game_record.update(metadata)
        
        with self.lock:
            self.games.append(game_record)
            self.game_records.append(game_record)
        
            if outcome == GameOutcome.HERO_WIN:
                self.hero_wins += 1
                self.steps_to_win.append(steps)
            elif outcome == GameOutcome.VILLAIN_WIN:
                self.villain_wins += 1
            else:
                self.timeouts += 1
                
            self.total_steps += steps
            self.total_duration += duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_games = len(self.games)
        if total_games == 0:
            return {"total_games": 0}
            
        return {
            "total_games": total_games,
            "hero_wins": self.hero_wins,
            "villain_wins": self.villain_wins,
            "timeouts": self.timeouts,
            "hero_win_rate": self.hero_wins / total_games,
            "villain_win_rate": self.villain_wins / total_games,
            "avg_steps": self.total_steps / total_games,
            "avg_duration": self.total_duration / total_games,
            "avg_steps_to_win": np.mean(self.steps_to_win) if self.steps_to_win else 0,
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of game patterns."""
        if not self.games:
            return {}
            
        # Analyze turn distribution
        turn_counts = [g["steps"] for g in self.games]
        hero_win_steps = [g["steps"] for g in self.games 
                          if g["outcome"] == GameOutcome.HERO_WIN]
        villain_win_steps = [g["steps"] for g in self.games 
                             if g["outcome"] == GameOutcome.VILLAIN_WIN]
        
        analysis = {
            "turn_distribution": {
                "min": min(turn_counts),
                "max": max(turn_counts),
                "p25": np.percentile(turn_counts, 25),
                "p50": np.percentile(turn_counts, 50),
                "p75": np.percentile(turn_counts, 75),
            },
            "hero_win_turn_distribution": self._get_percentiles(hero_win_steps),
            "villain_win_turn_distribution": self._get_percentiles(villain_win_steps),
            "early_game_win_rate": self._calculate_win_rate_by_turn_range(0, 10),
            "mid_game_win_rate": self._calculate_win_rate_by_turn_range(11, 25),
            "late_game_win_rate": self._calculate_win_rate_by_turn_range(26, float('inf')),
        }
        
        return analysis
    
    def _get_percentiles(self, values: List[int]) -> Dict[str, Any]:
        """Calculate percentiles for a list of values."""
        if not values:
            return {"count": 0}
        
        fvalues = [float(v) for v in values]

        return {
            "count": len(fvalues),
            "min": min(fvalues),
            "max": max(fvalues),
            "p25": np.percentile(fvalues, 25),
            "p50": np.percentile(fvalues, 50),
            "p75": np.percentile(fvalues, 75),
        }
    
    def _calculate_win_rate_by_turn_range(self, min_turn: int, max_turn: int) -> float:
        """Calculate win rate for games that ended within a turn range."""
        games_in_range = [g for g in self.games 
                         if min_turn <= g["steps"] <= max_turn]
        
        if not games_in_range:
            return 0.0
            
        hero_wins = sum(1 for g in games_in_range 
                        if g["outcome"] == GameOutcome.HERO_WIN)
        return hero_wins / len(games_in_range)
    
    def log_to_wandb(self, run) -> None:
        """Log summary statistics and visualizations to wandb."""
        summary = self.get_summary()
        
        # Basic metrics
        metrics = {
            "eval/hero_win_rate": summary["hero_win_rate"],
            "eval/villain_win_rate": summary["villain_win_rate"],
            "eval/timeout_rate": summary["timeouts"] / summary["total_games"],
            "eval/avg_steps": summary["avg_steps"],
            "eval/avg_duration": summary["avg_duration"],
        }
        
        if self.steps_to_win:
            metrics["eval/avg_steps_to_win"] = summary["avg_steps_to_win"]
        
        run.log(metrics)
        
        # Game records table
        games_table = wandb.Table(
            columns=["outcome", "steps", "duration"],
            data=[[g["outcome"].value, g["steps"], g["duration"]] for g in self.games]
        )
        run.log({"eval/games": games_table})
        
        # Turn distribution histogram
        if self.games:
            turn_counts = [g["steps"] for g in self.games]
            turn_histogram = np.histogram(turn_counts, bins=20)
            run.log({
                "eval/turn_distribution": wandb.Histogram(
                    np_histogram=turn_histogram
                )
            })
            
            # Win rates by turn
            detailed = self.get_detailed_analysis()
            run.log({
                "eval/early_game_win_rate": detailed["early_game_win_rate"],
                "eval/mid_game_win_rate": detailed["mid_game_win_rate"],
                "eval/late_game_win_rate": detailed["late_game_win_rate"],
            })

# -----------------------------------------------------------------------------
# Outcome Determination
# -----------------------------------------------------------------------------

def determine_outcome(
    info: dict, 
    last_obs: dict, 
    turn_count: int, 
    max_steps: int
) -> GameOutcome:
    """
    Determine the outcome of a game with robust fallback logic.
    
    Args:
        info: Information dictionary from environment step
        last_obs: Last observation from environment
        turn_count: Current turn count
        max_steps: Maximum steps before timeout
        
    Returns:
        Game outcome (HERO_WIN, VILLAIN_WIN, or TIMEOUT)
    """
    # Check for timeout first
    if turn_count >= max_steps:
        return GameOutcome.TIMEOUT
        
    # Primary source: winner field in info dict
    if "winner" in info:
        return GameOutcome.HERO_WIN if info["winner"] == 0 else GameOutcome.VILLAIN_WIN
        
    # Fallback: game_over and won fields in observation
    if last_obs.get("game_over", False):
        return GameOutcome.HERO_WIN if last_obs.get("won", -1) == 0 else GameOutcome.VILLAIN_WIN
        
    # Fallback: Check life totals if available
    hero_life = _extract_life(last_obs, player_index=0)
    villain_life = _extract_life(last_obs, player_index=1)
    
    if hero_life is not None and villain_life is not None:
        if hero_life <= 0:
            return GameOutcome.VILLAIN_WIN
        if villain_life <= 0:
            return GameOutcome.HERO_WIN
    
    # If we can't determine a winner, consider it a timeout
    return GameOutcome.TIMEOUT

def _extract_life(obs: dict, player_index: int) -> Optional[float]:
    """Extract life total for a player from observation if possible."""
    try:
        # This is a simplified example - adapt based on your actual observation structure
        if player_index == 0 and "agent_player" in obs:
            return obs["agent_player"][0, 2]  # Assuming life is at index 2
        elif player_index == 1 and "opponent_player" in obs:
            return obs["opponent_player"][0, 2]  # Assuming life is at index 2
    except (IndexError, KeyError):
        pass
    return None

# -----------------------------------------------------------------------------
# Main Evaluator
# -----------------------------------------------------------------------------

def evaluate_models(
    hero_player: Player,
    villain_player: Player,
    eval_hypers: Optional[EvaluationHypers] = None,
) -> GameStats:
    """
    Evaluate models in parallel by running multiple games simultaneously.
    
    Args:
        hero_player: Player for hero position
        villain_player: Player for villain position
        eval_hypers: Evaluation hyperparameters
        
    Returns:
        GameStats with results
    """
    logger = getLogger(__name__).getChild("evaluate_parallel")
    
    # Set up hyperparameters
    eval_hypers = eval_hypers or EvaluationHypers()
    
    # Determine number of threads (capped by games and CPU cores)
    num_threads = min(eval_hypers.num_threads, eval_hypers.num_games)
    logger.info(f"Starting evaluation: {hero_player.name} vs {villain_player.name}")
    logger.info(f"Running {eval_hypers.num_games} games with max {eval_hypers.max_steps} steps each")
    logger.info(f"Using {num_threads} parallel threads")
    
    # Create shared stats object and counter
    stats = GameStats()
    completed_games = 0
    completed_lock = threading.Lock()
    
    # Start timing
    start_time = time.time()
    
    # Worker function that runs games until target count is reached
    def worker_thread(thread_id):
        # Create environment for this thread
        match = Match(eval_hypers.match)
        observation_space = ObservationSpace()
        reward = Reward(RewardHypers())
        env = Env(match, observation_space, reward, auto_reset=False)
        
        nonlocal completed_games
        thread_games = 0
        
        while True:
            # Check if we've completed enough games
            with completed_lock:
                if completed_games >= eval_hypers.num_games:
                    break
                # Claim this game
                game_id = completed_games
                completed_games += 1
            
            # Simulate a single game
            outcome, steps, duration = _simulate_game(
                env, hero_player, villain_player, eval_hypers.max_steps)
            
            # Record game with minimal metadata
            metadata = {
                "thread_id": thread_id,
            }
            
            stats.record_game(outcome, steps, duration, metadata)
            
            # Update thread counter
            thread_games += 1
            
            # Log progress occasionally
            if thread_games % 5 == 0:
                logger.info(f"Thread {thread_id}: Completed {thread_games} games")
        
        # Clean up
        env.close()
        logger.info(f"Thread {thread_id} completed {thread_games} games")
    
    # Create and start worker threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=worker_thread,
            args=(thread_id,)
        )
        threads.append(thread)
        thread.start()
    
    # Monitor progress while threads are running
    while any(thread.is_alive() for thread in threads):
        with completed_lock:
            current = completed_games
        
        # Progress update every 5 seconds
        if current < eval_hypers.num_games:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = eval_hypers.num_games - current
            eta = remaining / rate if rate > 0 else "unknown"
            
            if isinstance(eta, float):
                eta_str = f"{eta:.1f} seconds"
            else:
                eta_str = str(eta)
                
            logger.info(f"Progress: {current}/{eval_hypers.num_games} games completed "
                       f"({current / eval_hypers.num_games:.1%}) | "
                       f"Rate: {rate:.2f} games/sec | ETA: {eta_str}")
        
        # Sleep to avoid busy waiting
        time.sleep(5)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # End timing
    total_time = time.time() - start_time
    
    # Log results
    summary = stats.get_summary()
    logger.info(f"Evaluation complete: {summary['total_games']} games in {total_time:.2f} seconds")
    logger.info(f"Overall performance: {summary['total_games'] / total_time:.2f} games/second")
    logger.info(f"Hero wins: {summary['hero_wins']} ({summary['hero_win_rate']:.2%})")
    logger.info(f"Villain wins: {summary['villain_wins']} ({summary['villain_win_rate']:.2%})")
    logger.info(f"Timeouts: {summary['timeouts']} ({summary['timeouts'] / summary['total_games']:.2%})")
    logger.info(f"Average steps per game: {summary['avg_steps']:.1f}")
    
    detailed = stats.get_detailed_analysis()
    logger.info("Detailed Analysis:")
    logger.info(f"Early game win rate: {detailed['early_game_win_rate']:.2f}")
    logger.info(f"Mid game win rate: {detailed['mid_game_win_rate']:.2f}")
    logger.info(f"Late game win rate: {detailed['late_game_win_rate']:.2f}")
    
    return stats

def _simulate_game(
    env: Env,
    hero_player: Player,
    villain_player: Player,
    max_steps: int
) -> Tuple[GameOutcome, int, float]:
    """
    Simulate a single game between two players.
    
    Args:
        env: Game environment
        hero_player: Player for hero position
        villain_player: Player for villain position
        max_steps: Maximum steps before timeout
        
    Returns:
        Tuple of (outcome, steps, duration)
    """
    # Reset environment
    start_time = time.time()
    obs, info = env.reset()
    done = False
    turn_count = 0
    
    # Track game state
    last_obs = obs
    
    # Main game loop
    while not done and turn_count < max_steps:
        # Get active player index from observation
        active_player_index = get_agent_indices(
            {k: torch.tensor(v)[None, ...] for k, v in obs.items()}
        )[0].item()
        
        # Select the appropriate player
        player = hero_player if active_player_index == 0 else villain_player
        
        # Get action from player
        try:
            action = player.get_action(obs)
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            # Fallback to random action if model fails
            valid_actions = np.where(obs["actions_valid"] > 0)[0]
            if len(valid_actions) == 0:
                logger.error("No valid actions available")
                break
            action = int(np.random.choice(valid_actions))
        
        # Step environment
        try:
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_obs = new_obs
            turn_count += 1
            
            # Update observation for next step
            obs = new_obs
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            break
    
    # Calculate duration
    if not done:
        logger.warning("Game did not complete")

    duration = time.time() - start_time
    
    # Determine outcome
    outcome = determine_outcome(info, last_obs, turn_count, max_steps)
    
    return outcome, turn_count, duration

# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

def main():
    """Command line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate manabot models")
    
    # Automatically add arguments from EvaluationHypers
    add_hypers(parser, EvaluationHypers)    
    args = parser.parse_args()
    eval_hypers = parse_hypers(args, EvaluationHypers)
    assert isinstance(eval_hypers, EvaluationHypers)
    
    logger.info(f"Loading hero model: {eval_hypers.hero}")
    hero_agent = load_model_from_wandb(eval_hypers.hero, device="cpu")
    
    if eval_hypers.villain.lower() == "random":
        logger.info("Using random opponent")
        villain_player = RandomPlayer("RandomVillain")
    else:
        logger.info(f"Loading villain model: {eval_hypers.villain}")
        villain_agent = load_model_from_wandb(eval_hypers.villain, device="cpu")
        villain_player = ModelPlayer(f"Model_{eval_hypers.villain}", villain_agent)
    
    hero_player = ModelPlayer(
        f"Model_{eval_hypers.hero}", 
        hero_agent, 
    )
    
    # Run evaluation
    evaluate_models(
        hero_player=hero_player,
        villain_player=villain_player,
        eval_hypers=eval_hypers,
    )
    
if __name__ == "__main__":
    main()