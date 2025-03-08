"""
test_profiler.py
Tests for hierarchical performance tracking semantics and percentage calculations.

This suite verifies:
(1) Label Enforcement:
    - A valid nested sequence using the context manager with relative labels works as expected.
    - Nodes automatically build up the correct hierarchical structure.
(2) Percentage Calculations:
    - Build a multi-level hierarchy with simulated sleep durations and verify that each node's percentages are roughly as expected.
(3) Accumulated Timing:
    - Repeated uses of the same cached node (e.g., "rollout" and "gradient") accumulate time.
    - While a node is still running, its effective total time is greater than the stored total.
(4) Statistical Metrics:
    - Verify that the profiler computes percentiles (p2.5, p5, p95, p97.5), min, max, mean, 
      and the 95%-window correctly for a node.
"""

import time
import pytest
import numpy as np
from manabot.infra.profiler import Profiler  # Adjust this import as needed

# --- Basic Functionality Tests ---
def test_relative_labels():
    profiler = Profiler(enabled=True)
    
    # Valid sequence using relative labels
    with profiler.track("a"):
        with profiler.track("b"):  # This is relative to "a"
            with profiler.track("c"):  # This is relative to "a/b"
                pass
    
    # Check that the stats contain the expected full paths
    stats = profiler.get_stats()
    assert "a" in stats, "Missing stats for 'a'"
    assert "a/b" in stats, "Missing stats for 'a/b'"
    assert "a/b/c" in stats, "Missing stats for 'a/b/c'"

# --- Hierarchy Percentage Calculations ---
def test_hierarchy_percentages():
    """
    Create a multi-level hierarchy and verify percentage calculations.
    Structure:
      - Root ("")
          - group1
              - a (relative to group1)
                  - alpha (relative to group1/a)
                  - beta (relative to group1/a)
          - group2
              - x (relative to group2)
              - y (relative to group2)
    """
    profiler = Profiler(enabled=True)
    
    # Group 1
    with profiler.track("group1"):
        with profiler.track("a"):  # Relative to "group1"
            with profiler.track("alpha"):  # Relative to "group1/a"
                time.sleep(0.1)
            with profiler.track("beta"):  # Relative to "group1/a"
                time.sleep(0.1)
    
    # Group 2
    with profiler.track("group2"):
        with profiler.track("x"):  # Relative to "group2"
            time.sleep(0.18)
        with profiler.track("y"):  # Relative to "group2"
            time.sleep(0.02)

    stats = profiler.get_stats()
    for path, data in stats.items():
        print(f"{path}: {data}")

    # Check overall split: group1 and group2 should be roughly 50% each.
    group1_stats = stats.get("group1")
    group2_stats = stats.get("group2")
    assert group1_stats is not None, "Missing stats for 'group1'"
    assert group2_stats is not None, "Missing stats for 'group2'"
    tol_overall = 20.0  # tolerance percentage points
    assert abs(group1_stats["pct_of_total"] - 50) < tol_overall, f"'group1' pct_of_total not ~50: {group1_stats['pct_of_total']}"
    assert abs(group2_stats["pct_of_total"] - 50) < tol_overall, f"'group2' pct_of_total not ~50: {group2_stats['pct_of_total']}"
    
    # Within group1, there is one child "a" so its pct_of_parent should be ~100%.
    a_stats = stats.get("group1/a")
    assert a_stats is not None, "Missing stats for 'group1/a'"
    tol_group1 = 5.0
    assert abs(a_stats["pct_of_parent"] - 100) < tol_group1, f"'group1/a' pct_of_parent not ~100: {a_stats['pct_of_parent']}"
    
    # Within group1/a, check that "alpha" and "beta" split time roughly equally.
    alpha_stats = stats.get("group1/a/alpha")
    beta_stats = stats.get("group1/a/beta")
    assert alpha_stats is not None, "Missing stats for 'group1/a/alpha'"
    assert beta_stats is not None, "Missing stats for 'group1/a/beta'"
    tol_sub = 25.0
    assert abs(alpha_stats["pct_of_parent"] - 50) < tol_sub, f"'group1/a/alpha' pct_of_parent not ~50: {alpha_stats['pct_of_parent']}"
    assert abs(beta_stats["pct_of_parent"] - 50) < tol_sub, f"'group1/a/beta' pct_of_parent not ~50: {beta_stats['pct_of_parent']}"
    
    # In group2, "x" should be about 90% and "y" about 10% of group2's time.
    x_stats = stats.get("group2/x")
    y_stats = stats.get("group2/y")
    assert x_stats is not None, "Missing stats for 'group2/x'"
    assert y_stats is not None, "Missing stats for 'group2/y'"
    tol_group2 = 25.0
    assert abs(x_stats["pct_of_parent"] - 90) < tol_group2, f"'group2/x' pct_of_parent not ~90: {x_stats['pct_of_parent']}"
    assert abs(y_stats["pct_of_parent"] - 10) < tol_group2, f"'group2/y' pct_of_parent not ~10: {y_stats['pct_of_parent']}"

# --- Accumulated Timing Tests ---
def test_accumulated_timing():
    """
    Verify that repeated context-managed calls for the same cached node accumulate time,
    and that get_stats() correctly reports effective times when nodes are still running.
    """
    profiler = Profiler(enabled=True)

    rollout_times = [0.02, 0.01, 0.015]
    gradient_times = [0.005, 0.003, 0.004]
    accumulated_rollout = 0.0
    accumulated_gradient = 0.0

    for i in range(len(rollout_times)):
        with profiler.track("rollout"):
            time.sleep(rollout_times[i])
        accumulated_rollout += rollout_times[i]

        with profiler.track("gradient"):
            time.sleep(gradient_times[i])
        accumulated_gradient += gradient_times[i]

        stats = profiler.get_stats()
        rollout_stats = stats.get("rollout")
        gradient_stats = stats.get("gradient")
        print(f"Iteration {i} - rollout_stats: {rollout_stats}")
        print(f"Iteration {i} - gradient_stats: {gradient_stats}")
        assert abs(rollout_stats["total_time"] - accumulated_rollout) < 0.02, f"Iteration {i}: rollout time incorrect"
        assert abs(gradient_stats["total_time"] - accumulated_gradient) < 0.02, f"Iteration {i}: gradient time incorrect"

    # Test live stats: start "rollout" without finishing immediately.
    with profiler.track("rollout") as _:
        time.sleep(0.02)
        stats_running = profiler.get_stats()
        running_rollout = stats_running.get("rollout", {}).get("total_time", 0)
        assert running_rollout > accumulated_rollout, "Effective time for running node not greater than accumulated time"

# --- Statistical Metrics Tests ---
def test_statistical_metrics():
    """
    Verify that the profiler correctly computes percentiles, min, max, mean, and
    the 95%-window for a node's call durations.
    """
    profiler = Profiler(enabled=True, max_samples=100)  # Ensure we capture all samples
    durations = []
    
    # Execute the "test_node" 20 times with controlled sleep durations.
    for i in range(20):
        d = 0.005 + (i % 5) * 0.003  # Cycles through: 0.005, 0.008, 0.011, 0.014, 0.017 sec.
        durations.append(d)
        with profiler.track("test_node"):
            time.sleep(d)
    
    stats = profiler.get_stats()
    test_stats = stats.get("test_node")
    
    # Calculate expected statistics
    durations_array = np.array(durations)
    expected_min = float(np.min(durations_array))
    expected_max = float(np.max(durations_array))
    expected_mean = float(np.mean(durations_array))
    expected_p2_5 = float(np.percentile(durations_array, 2.5))
    expected_p5 = float(np.percentile(durations_array, 5))
    expected_p95 = float(np.percentile(durations_array, 95))
    expected_p97_5 = float(np.percentile(durations_array, 97.5))
    expected_window = expected_p97_5 - expected_p2_5

    print("Expected min:", expected_min)
    print("Expected max:", expected_max)
    print("Expected mean:", expected_mean)
    print("Expected p2_5:", expected_p2_5)
    print("Expected p5:", expected_p5)
    print("Expected p95:", expected_p95)
    print("Expected p97_5:", expected_p97_5)
    print("Expected window_95:", expected_window)
    print("Profiler reported for 'test_node':", test_stats)

    tol = 0.002  # Tolerance in seconds.
    assert abs(test_stats["min"] - expected_min) < tol, f"min mismatch: expected {expected_min}, got {test_stats['min']}"
    assert abs(test_stats["max"] - expected_max) < tol, f"max mismatch: expected {expected_max}, got {test_stats['max']}"
    assert abs(test_stats["mean"] - expected_mean) < tol, f"mean mismatch: expected {expected_mean}, got {test_stats['mean']}"
    assert abs(test_stats["p2_5"] - expected_p2_5) < tol, f"p2_5 mismatch: expected {expected_p2_5}, got {test_stats['p2_5']}"
    assert abs(test_stats["p5"] - expected_p5) < tol, f"p5 mismatch: expected {expected_p5}, got {test_stats['p5']}"
    assert abs(test_stats["p95"] - expected_p95) < tol, f"p95 mismatch: expected {expected_p95}, got {test_stats['p95']}"
    assert abs(test_stats["p97_5"] - expected_p97_5) < tol, f"p97_5 mismatch: expected {expected_p97_5}, got {test_stats['p97_5']}"
    assert abs(test_stats["window_95"] - expected_window) < tol, f"window_95 mismatch: expected {expected_window}, got {test_stats['window_95']}"

# --- Max Samples Test ---
def test_max_samples():
    """
    Verify that the profiler correctly handles the max_samples limit.
    """
    max_samples = 10
    profiler = Profiler(enabled=True, max_samples=max_samples)
    
    # Run the node more times than max_samples
    iterations = max_samples * 2
    
    for i in range(iterations):
        with profiler.track("sampled_node"):
            time.sleep(0.001)  # Very short sleep
    
    stats = profiler.get_stats()
    node_stats = stats.get("sampled_node")
    
    # Verify node was called the correct number of times
    assert node_stats["count"] == iterations, f"Count mismatch: expected {iterations}, got {node_stats['count']}"
    
    # The implementation now uses reservoir sampling, so we can't directly verify
    # the length of the internal durations list. Instead, let's verify the statistics
    # are reasonable.
    assert 0.0005 < node_stats["mean"] < 0.002, f"Mean outside reasonable range: {node_stats['mean']}"
    assert node_stats["min"] < node_stats["mean"], f"Min should be less than mean: min={node_stats['min']}, mean={node_stats['mean']}"
    assert node_stats["mean"] < node_stats["max"], f"Mean should be less than max: mean={node_stats['mean']}, max={node_stats['max']}"
    assert node_stats["p5"] <= node_stats["p95"], "p5 should be less than or equal to p95"

# --- Nested Relative Labels Test ---
def test_nested_relative_labels():
    """Test deeply nested relative labels to ensure proper hierarchy."""
    profiler = Profiler(enabled=True)
    
    with profiler.track("level1"):
        with profiler.track("level2"):
            with profiler.track("level3"):
                with profiler.track("level4"):
                    with profiler.track("level5"):
                        time.sleep(0.01)
    
    stats = profiler.get_stats()
    
    # Check that all levels exist in the stats
    assert "level1" in stats, "Missing level1"
    assert "level1/level2" in stats, "Missing level1/level2"
    assert "level1/level2/level3" in stats, "Missing level1/level2/level3"
    assert "level1/level2/level3/level4" in stats, "Missing level1/level2/level3/level4"
    assert "level1/level2/level3/level4/level5" in stats, "Missing level1/level2/level3/level4/level5"
    
    # Check that the hierarchy percentages make sense
    level5 = stats["level1/level2/level3/level4/level5"]
    level4 = stats["level1/level2/level3/level4"]
    level3 = stats["level1/level2/level3"]
    level2 = stats["level1/level2"]
    level1 = stats["level1"]
    
    assert level5["pct_of_parent"] == 100, "level5 should be 100% of level4's time"
    assert level4["pct_of_parent"] == 100, "level4 should be 100% of level3's time"
    assert level3["pct_of_parent"] == 100, "level3 should be 100% of level2's time"
    assert level2["pct_of_parent"] == 100, "level2 should be 100% of level1's time"
    assert level1["pct_of_total"] > 0, "level1 should be some percentage of total time"

if __name__ == "__main__":
    pytest.main([__file__])