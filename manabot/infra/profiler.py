"""
profiler.py
Hierarchical performance tracker with context-manager API and automatic nesting.

This module provides a Profiler class that enables tracking execution time across
various parts of your code in a hierarchical manner using relative labels. Key features include:

1. Context-manager API for clean, exception-safe timing
2. Automatic nesting of timers based on execution context
3. Node caching and time accumulation across repeated operations
4. Support for retrieving statistics including percentages of parent/total time,
   call counts, and detailed percentile statistics (min, max, mean, p5, p95, etc.)
5. Low overhead when disabled

Usage example:
    profiler = Profiler(enabled=True)
    with profiler.track("rollout"):
        # time rollout code
        with profiler.track("step"):  # Note: this is relative to "rollout"
            # time environment step call
            with profiler.track("env"):  # Note: this is relative to "rollout/step"
                # time environment operation
                ...
    stats = profiler.get_stats()  # returns a nested dict of timings
"""

import time
import random
from contextlib import contextmanager
from typing import Optional, Dict, List
import numpy as np

class TimingNode:
    def __init__(self, label: str, parent: Optional["TimingNode"] = None, max_samples: int = 100):
        self.label = label              # local label only
        self.parent = parent
        self.previous_total = 0.0       # accumulated time from previous runs
        self.start_time = None          # current start time (None if not running)
        self.children: Dict[str, "TimingNode"] = {}  # keyed by local label
        self.count = 0                  # number of times this node was entered
        self.durations: List[float] = []  # list to store durations of each call
        self.max_samples = max_samples  # maximum number of duration samples to keep

    def start(self):
        if self.start_time is not None:
            raise RuntimeError(f"Timer '{self.label}' is already running.")
        self.start_time = time.perf_counter()
        self.count += 1

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError(f"Timer '{self.label}' was not started!")
        elapsed = time.perf_counter() - self.start_time
        self.previous_total += elapsed
        
        # Add duration, but maintain max_samples limit
        if len(self.durations) >= self.max_samples:
            # Replace a random existing sample (reservoir sampling)
            idx = random.randint(0, self.max_samples-1)
            self.durations[idx] = elapsed
        else:
            self.durations.append(elapsed)
            
        self.start_time = None
        return elapsed

    def running_total(self) -> float:
        """Return total accumulated time plus running time if still active."""
        if self.start_time is not None:
            return self.previous_total + (time.perf_counter() - self.start_time)
        return self.previous_total

class Profiler:
    """
    Hierarchical profiler with context-manager API, auto nesting, and comprehensive statistics.
    Uses relative labels for simpler usage.

    Usage example:
        profiler = Profiler(enabled=True)
        with profiler.track("rollout"):
            # time rollout code
            with profiler.track("step"):  # Relative to "rollout"
                # time environment step call
                with profiler.track("env"):  # Relative to "rollout/step"
                    # time environment operation
                    ...
        stats = profiler.get_stats()  # returns a nested dict of timings

    The profiler caches nodes by full path so that repeated uses accumulate times.
    """
    def __init__(self, enabled: bool = True, max_samples: int = 100):
        self.enabled = enabled
        self.max_samples = max_samples
        self.root = TimingNode("", max_samples=max_samples)
        self.root.start()  # start root at initialization
        self.stack = [self.root]
        self.node_cache: Dict[str, TimingNode] = {"": self.root}

    @contextmanager
    def track(self, label: str):
        """
        Use as a context manager to time a code block. The provided label is automatically
        nested under the current active node. Labels are relative to the current context.
        If the same label is used again in the same position, its time is accumulated.
        """
        if not self.enabled:
            yield
            return

        # Get the current parent node
        parent = self.stack[-1]
        parent_path = self._get_full_path(parent)
        
        # Compute the full path for this node
        full_label = f"{parent_path}/{label}" if parent_path else label
        
        # Retrieve or create the node
        if full_label in self.node_cache:
            node = self.node_cache[full_label]
            if node.start_time is not None:
                raise RuntimeError(f"Node '{full_label}' is already running.")
            node.start()
        else:
            node = TimingNode(label, parent, max_samples=self.max_samples)
            node.start()
            self.node_cache[full_label] = node
            parent.children[label] = node
        
        self.stack.append(node)
        try:
            yield
        finally:
            node.stop()
            self.stack.pop()

    def _get_full_path(self, node: TimingNode) -> str:
        """Return the full hierarchical path for a node."""
        parts = []
        while node.parent is not None:
            parts.append(node.label)
            node = node.parent
        return "/".join(reversed(parts))

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary mapping each node's full hierarchical path (excluding root)
        to its timing statistics:
            - total_time: The running total time.
            - pct_of_parent: Percentage of parent's time.
            - pct_of_total: Percentage of overall total time.
            - count: Number of times this node was entered.
            - min: Minimum call duration.
            - max: Maximum call duration.
            - mean: Mean call duration.
            - p5: 5th percentile of call durations in samples.
            - p95: 95th percentile of call durations in samples.
        """
        total = self.root.running_total()
        stats = {}

        def recurse(node: TimingNode, path: str, parent_total: float):
            for child_label, child in node.children.items():
                full_path = f"{path}/{child_label}" if path else child_label
                node_total = child.running_total()
                
                # Compute statistics if there are any recorded durations
                if child.count > 0 and child.durations:
                    durations = np.array(child.durations)
                    min_val = float(np.min(durations))
                    max_val = float(np.max(durations))
                    mean_val = float(np.mean(durations))
                    p5 = float(np.percentile(durations, 5))
                    p95 = float(np.percentile(durations, 95))
                else:
                    min_val = max_val = mean_val = p5 = p95 = 0.0
                
                stats[full_path] = {
                    "total_time": node_total,
                    "pct_of_parent": (node_total / parent_total * 100) if parent_total > 0 else 0,
                    "pct_of_total": (node_total / total * 100) if total > 0 else 0,
                    "count": child.count,
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "p5": p5,
                    "p95": p95,
                }
                recurse(child, full_path, node_total)
        
        recurse(self.root, "", self.root.running_total())
        return stats
        
    def reset(self):
        """Reset the profiler to its initial state."""
        self.root = TimingNode("", max_samples=self.max_samples)
        self.root.start()
        self.stack = [self.root]
        self.node_cache = {"": self.root}