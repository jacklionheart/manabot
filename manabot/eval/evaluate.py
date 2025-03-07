"""
evaluate.py
Simplified and consolidated evaluation package for manabot.

This module provides:
1. Loading models from wandb
2. Player abstractions for model inference
3. Basic game simulation and statistics tracking
"""

import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import wandb

from manabot.ppo.agent import Agent
from manabot.env import Env, Match, Reward, ObservationSpace
from manabot.env.observation import get_agent_indices
from manabot.infra.hypers import MatchHypers, RewardHypers, ObservationSpaceHypers
from manabot.infra.log import getLogger

logger = getLogger(__name__)

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
    Load a trained model from wandb artifacts.
    
    Args:
        artifact_name: Name of the artifact (e.g. "experiment_name_model")
        version: Version string (e.g. "v3" or "latest")
        project: Wandb project name
        device: Device to load model on ("cpu" or "cuda")
        
    Returns:
        Loaded agent model ready for inference
    """
    try:
        # Initialize wandb if needed (anonymous mode is fine for loading)
        if wandb.run is None:
            wandb.init(
                project=project,
                name="model_loading",
                job_type="model_loading",
                mode="offline"
            )
        
        # Get artifact
        artifact_ref = f"{artifact_name}:{version}"
        logger.info(f"Loading artifact: {artifact_ref}")
        artifact = wandb.use_artifact(artifact_ref)
        artifact_dir = artifact.download()
        
        # Find model file
        pt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError("No .pt files found in the artifact")
        checkpoint_path = os.path.join(artifact_dir, pt_files[0])
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get hyperparameters
        train_hypers = checkpoint.get('train_hypers')
        if train_hypers is None:
            raise ValueError("Checkpoint does not contain 'train_hypers'")
        
        # Create observation space and agent
        obs_space = ObservationSpace(train_hypers.observation)
        agent = Agent(obs_space, train_hypers.agent)
        
        # Load model weights and set to evaluation mode
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.eval()
        agent = agent.to(device)
        
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return agent
        
    except Exception as e:
        logger.error(f"Error loading model from wandb: {e}")
        raise

# -----------------------------------------------------------------------------
# Player Classes
# -----------------------------------------------------------------------------

class PlayerType(Enum):
    """Types of players for evaluation."""
    MODEL = "model"
    RANDOM = "random"

class Player:
    """Base player class for evaluation."""
    
    def __init__(self, name: str, player_type: PlayerType):
        self.name = name
        self.player_type = player_type
        self.device = "cpu"
        self.wins = 0
        self.games = 0
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from observation."""
        raise NotImplementedError
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games == 0:
            return 0.0
        return self.wins / self.games
    
    def record_result(self, won: bool) -> None:
        """Record game result."""
        self.games += 1
        if won:
            self.wins += 1
    
    def to(self, device: str) -> 'Player':
        """Move player to specified device."""
        self.device = device
        return self

class ModelPlayer(Player):
    """Player that uses a trained model for inference."""
    
    def __init__(self, name: str, agent: Agent, deterministic: bool = True):
        super().__init__(name, PlayerType.MODEL)
        self.agent = agent
        self.deterministic = deterministic
        self.device = next(agent.parameters()).device
    
    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Get action from model."""
        # Convert numpy arrays to tensors with batch dimension
        tensor_obs = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }
        
        # Get action from model
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                tensor_obs, deterministic=self.deterministic)
            return action.item()
    
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
        return int(np.random.choice(valid_actions))

# -----------------------------------------------------------------------------
# Game Statistics
# -----------------------------------------------------------------------------

class GameOutcome(Enum):
    """Possible game outcomes."""
    HERO_WIN = "hero_win"
    VILLAIN_WIN = "villain_win"
    TIMEOUT = "timeout"

class GameStats:
    """Simple game statistics tracking."""
    
    def __init__(self):
        self.games = []
        self.hero_wins = 0
        self.villain_wins = 0
        self.timeouts = 0
        self.total_turns = 0
        self.total_duration = 0
    
    def record_game(self, outcome: GameOutcome, turns: int, duration: float) -> None:
        """Record a completed game."""
        self.games.append({
            "outcome": outcome,
            "turns": turns,
            "duration": duration
        })
        
        if outcome == GameOutcome.HERO_WIN:
            self.hero_wins += 1
        elif outcome == GameOutcome.VILLAIN_WIN:
            self.villain_wins += 1
        else:
            self.timeouts += 1
            
        self.total_turns += turns
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
            "avg_turns": self.total_turns / total_games,
            "avg_duration": self.total_duration / total_games,
        }
    
    def log_to_wandb(self, run: wandb.Run) -> None:
        """Log summary statistics to wandb."""
        summary = self.get_summary()
        run.log({
            "eval/hero_win_rate": summary["hero_win_rate"],
            "eval/avg_turns": summary["avg_turns"],
            "eval/avg_duration": summary["avg_duration"],
        })
        
        # Create table of all games
        games_table = wandb.Table(
            columns=["outcome", "turns", "duration"],
            data=[[g["outcome"].value, g["turns"], g["duration"]] for g in self.games]
        )
        run.log({"eval/games": games_table})

# -----------------------------------------------------------------------------
# Main Evaluator
# -----------------------------------------------------------------------------

def evaluate_models(
    hero_player: Player,
    villain_player: Player,
    num_games: int,
    match_hypers: Optional[MatchHypers] = None,
    max_turns: int = 200,
    log_to_wandb: bool = False,
    experiment_name: Optional[str] = None
) -> GameStats:
    """
    Evaluate models by simulating games between them.
    
    Args:
        hero_player: Player for hero position (player index 0)
        villain_player: Player for villain position (player index 1)
        num_games: Number of games to simulate
        match_hypers: Match configuration
        max_turns: Maximum turns before timeout
        log_to_wandb: Whether to log results to wandb
        experiment_name: Name for tracking
        
    Returns:
        GameStats with results
    """
    logger = getLogger(__name__).getChild("evaluate_models")
    # Setup
    device = next(hero_player.agent.parameters()).device if isinstance(hero_player, ModelPlayer) else "cpu"
    match_hypers = match_hypers or MatchHypers()
    experiment_name = experiment_name or f"{hero_player.name}_vs_{villain_player.name}"
    
    # Create environment components
    match = Match(match_hypers)
    observation_space = ObservationSpace()  # Default hyperparameters
    reward = Reward(RewardHypers())
    env = Env(match, observation_space, reward, auto_reset=False)
    
    # Tracking
    stats = GameStats()
    wandb_run = None
    
    # Setup wandb if requested
    if log_to_wandb:
        try:
            wandb_run = wandb.init(
                project="manabot-evaluation",
                name=experiment_name,
                config={
                    "hero_player": hero_player.name,
                    "hero_type": hero_player.player_type.value,
                    "villain_player": villain_player.name,
                    "villain_type": villain_player.player_type.value,
                    "num_games": num_games,
                    "max_turns": max_turns,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            log_to_wandb = False
    
    logger.info(f"Starting evaluation: {hero_player.name} vs {villain_player.name}")
    logger.info(f"Running {num_games} games with max {max_turns} turns each")
    
    # Run games
    for game_id in range(num_games):
        # Run a single game
        outcome, turns, duration = _simulate_game(
            env, hero_player, villain_player, max_turns)
        
        # Record results
        stats.record_game(outcome, turns, duration)
        
        # Update player records
        hero_player.record_result(outcome == GameOutcome.HERO_WIN)
        villain_player.record_result(outcome == GameOutcome.VILLAIN_WIN)
        
        # Log progress
        if (game_id + 1) % 10 == 0 or game_id == num_games - 1:
            summary = stats.get_summary()
            logger.info(f"Completed {game_id+1}/{num_games} games | "
                       f"Hero win rate: {summary['hero_win_rate']:.2f} | "
                       f"Avg turns: {summary['avg_turns']:.1f}")
    
    # Log final results
    summary = stats.get_summary()
    logger.info(f"Evaluation complete: {summary['total_games']} games")
    logger.info(f"Hero wins: {summary['hero_wins']} ({summary['hero_win_rate']:.2%})")
    logger.info(f"Villain wins: {summary['villain_wins']} ({summary['villain_win_rate']:.2%})")
    logger.info(f"Average turns: {summary['avg_turns']:.1f}")
    
    # Log to wandb if active
    if log_to_wandb and wandb_run:
        stats.log_to_wandb(wandb_run)
        wandb.finish()
    
    env.close()
    return stats

def _simulate_game(
    env: Env,
    hero_player: Player,
    villain_player: Player,
    max_turns: int
) -> Tuple[GameOutcome, int, float]:
    """
    Simulate a single game between two players.
    
    Returns:
        Tuple of (outcome, turns, duration)
    """
    # Reset environment
    start_time = time.time()
    obs, info = env.reset()
    done = False
    turn_count = 0
    
    # Track game state
    last_obs = obs
    
    # Main game loop
    while not done and turn_count < max_turns:
        # Get active player index from observation
        active_player_index = get_agent_indices(
            {k: torch.tensor(v)[None, ...] for k, v in obs.items()}
        )[0].item()
        
        # Select the appropriate player
        player = hero_player if active_player_index == 0 else villain_player
        
        # Get action from player
        action = player.get_action(obs)
        
        # Step environment
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_obs = new_obs
        turn_count += 1
        
        # Update observation for next step
        obs = new_obs
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Determine outcome
    if turn_count >= max_turns:
        outcome = GameOutcome.TIMEOUT
    elif "winner" in info:
        outcome = GameOutcome.HERO_WIN if info["winner"] == 0 else GameOutcome.VILLAIN_WIN
    else:
        # Fallback logic - check if game is over and who won
        if last_obs.get("game_over", False):
            if last_obs.get("won", -1) == 0:
                outcome = GameOutcome.HERO_WIN
            else:
                outcome = GameOutcome.VILLAIN_WIN
        else:
            # If we can't determine, use a heuristic like life totals
            outcome = GameOutcome.TIMEOUT
    
    return outcome, turn_count, duration

# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

def main():
    """Command line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate manabot models")
    parser.add_argument("--hero", type=str, required=True, help="Hero model artifact name")
    parser.add_argument("--villain", type=str, required=True, help="Villain model artifact name")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--max-turns", type=int, default=200, help="Maximum turns per game")
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    args = parser.parse_args()
    
    # Load models
    logger.info(f"Loading hero model: {args.hero}")
    hero_agent = load_model_from_wandb(args.hero, device=args.device)
    
    if args.villain.lower() == "random":
        logger.info("Using random opponent")
        villain_player = RandomPlayer("RandomVillain")
    else:
        logger.info(f"Loading villain model: {args.villain}")
        villain_agent = load_model_from_wandb(args.villain, device=args.device)
        villain_player = ModelPlayer(f"Model_{args.villain}", villain_agent)
    
    hero_player = ModelPlayer(f"Model_{args.hero}", hero_agent)
    
    # Run evaluation
    stats = evaluate_models(
        hero_player=hero_player,
        villain_player=villain_player,
        num_games=args.games,
        max_turns=args.max_turns,
        log_to_wandb=args.wandb,
        experiment_name=f"{args.hero}_vs_{args.villain}"
    )
    
    # Print summary
    summary = stats.get_summary()
    print("\n=== Evaluation Results ===")
    print(f"Total games: {summary['total_games']}")
    print(f"Hero wins: {summary['hero_wins']} ({summary['hero_win_rate']:.2%})")
    print(f"Villain wins: {summary['villain_wins']} ({summary['villain_win_rate']:.2%})")
    print(f"Average turns: {summary['avg_turns']:.1f}")
    print(f"Average game duration: {summary['avg_duration']:.2f}s")

if __name__ == "__main__":
    main()