"""
perf.py
Hierarchical performance tracker with explicit parent specification.

When starting a node:
  - If the current active node's full path is empty (i.e. at root), the label must be local (without '/').
  - Otherwise, the label must be of the form "P/child" where P exactly matches the current active node’s full path.
If the provided parent portion does not match, a ValueError is raised.

When stopping a timer, you may supply either the local label or the full hierarchical label.
The tracker checks that the current active node matches the provided label.
"""

import time
from typing import Optional
class TimingNode:
    def __init__(self, label: str, parent: Optional["TimingNode"] = None):
        self.label = label  # local label only
        self.parent = parent
        self.previous_total = 0.0
        self.start_time = None
        self.children = []

    def start(self):
        self.start_time = time.perf_counter()
    
    def runningTotal(self):
        if self.start_time is None:
            return self.previous_total
        return self.previous_total + (time.perf_counter() - self.start_time)

    def stop(self):
        if self.start_time is None:
            raise RuntimeError(f"Timer '{self.label}' was not started!")
        elapsed = time.perf_counter() - self.start_time
        self.previous_total += elapsed
        self.start_time = None
        return elapsed

class PerformanceTracker:
    """
    Hierarchical performance tracker with explicit parent specification.

    API:
      - startRoot(): Resets and starts the root timer (its full path is defined as the empty string).
      - start(label): 
            • If the current active node's full path is empty, label must be local (no '/').
            • Otherwise, label must be of the form "P/child" where P equals the current active node’s full path.
      - stop(label): Stops the most recently started timer. The supplied label can be either:
            • a local label (which must match the node’s local label), or 
            • a full hierarchical label (which must equal the node’s full path).
      - get_stats(): Returns a dict mapping each node’s full hierarchical path to its timing stats.
    """
    def __init__(self):
        self.startRoot()

    def startRoot(self):
        # Create a root node whose full path is defined as "".
        self.root = TimingNode("")
        self.root.start()
        self.stack = [self.root]

    def _get_full_path(self, node: TimingNode) -> str:
        parts = []
        while node is not None and node.parent is not None:
            parts.append(node.label)
            node = node.parent
        return "/".join(reversed(parts))

    def start(self, label: str):
        current_full = self._get_full_path(self.stack[-1])
        if current_full == "":
            # At the root level, label must be local (no '/').
            if "/" in label:
                raise ValueError("At the root level, label must be a local label without '/'.")
            final_label = label
        else:
            # When nested, label must be of the form "current_full/child".
            if "/" not in label:
                raise ValueError("When not at the root, label must include the parent's full path (e.g. 'parent/child').")
            segments = label.split("/")
            expected_parent = "/".join(segments[:-1])
            local_label = segments[-1]
            if current_full != expected_parent:
                raise ValueError(
                    f"Parent mismatch: current active node is '{current_full}', "
                    f"but label expects parent '{expected_parent}' in '{label}'."
                )
            final_label = local_label

        parent = self.stack[-1]
        if parent.start_time is None:
            raise RuntimeError(f"Cannot start child '{final_label}' because the parent is not running.")
        node = TimingNode(final_label, parent)
        parent.children.append(node)
        node.start()
        self.stack.append(node)

    def stop(self, label: str):
        if not self.stack:
            raise RuntimeError("No active timer to stop.")
        node = self.stack.pop()
        # Determine whether to compare full path or local label.
        if "/" in label:
            full = self._get_full_path(node)
            if full != label:
                raise ValueError(f"Timer mismatch: expected to stop '{full}' but got '{label}'.")
        else:
            if node.label != label:
                raise ValueError(f"Timer mismatch: expected to stop '{node.label}' but got '{label}'.")
        return node.stop()

    def reset(self):
        self.startRoot()

    def get_stats(self) -> dict:
        total = self.root.runningTotal()

        stats = {}

        def recurse(node: TimingNode, path: str, parent_time: float):
            if node.parent is not None:  # skip root
                current_path = f"{path}/{node.label}" if path else node.label
                stats[current_path] = {
                    "total_time": node.runningTotal(),
                    "pct_of_parent": (node.runningTotal() / parent_time * 100) if parent_time > 0 else 0,
                    "pct_of_total": (node.runningTotal() / total * 100) if total > 0 else 0,
                }
                new_path = current_path
            else:
                new_path = ""
            for child in node.children:
                recurse(child, new_path, node.runningTotal())
        recurse(self.root, "", self.root.runningTotal())
        return stats
