# manabot/infra/perf.py

import time
from typing import Optional, Dict, Any
class TimingNode:
    def __init__(self, label: str, parent: Optional["TimingNode"] = None):
        self.label = label
        self.parent = parent
        self.total_time = 0.0
        self.start_time = None
        self.children = []

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError(f"Timer {self.label} was not started!")
        elapsed = time.perf_counter() - self.start_time
        self.total_time += elapsed
        self.start_time = None
        return elapsed

class HierarchicalPerformanceTracker:
    """
    A hierarchical performance tracker. Calls to start(label) create a new timing node
    as a child of the currently active node. All child nodes must be stopped before
    stopping their parent.
    
    After running, get_stats() returns a flat dictionary where each key is the full hierarchical
    path (e.g. "train/rollout/step/env") and the value is a dict with:
      - total_time: The total elapsed time for that node.
      - pct_of_parent: That time as a percentage of its parent's total time.
      - pct_of_total: That time as a percentage of the root node's total time.
    """
    def __init__(self):
        self.root = TimingNode("root", None)
        self.stack = [self.root]

    def start(self, label: str):
        parent = self.stack[-1]
        node = TimingNode(label, parent)
        parent.children.append(node)
        node.start()
        self.stack.append(node)

    def stop(self, label: str):
        if not self.stack:
            raise RuntimeError("No active timer to stop.")
        node = self.stack.pop()
        if node.label != label:
            raise RuntimeError(f"Timer mismatch: expected to stop '{node.label}' but got '{label}'.")
        return node.stop()

    def get_stats(self) -> dict:
        total_root = self.root.total_time
        stats = {}

        def recurse(node: TimingNode, prefix: str, parent_time: float):
            if node != self.root:
                path = f"{prefix}/{node.label}" if prefix else node.label
                stats[path] = {
                    "total_time": node.total_time,
                    "pct_of_parent": (node.total_time / parent_time * 100) if parent_time > 0 else 0,
                    "pct_of_total": (node.total_time / total_root * 100) if total_root > 0 else 0,
                }
                next_prefix = path
            else:
                next_prefix = ""
            for child in node.children:
                recurse(child, next_prefix, node.total_time)
        recurse(self.root, "", self.root.total_time)
        return stats
