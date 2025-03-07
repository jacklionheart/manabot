"""
test_perf.py
Tests for hierarchical performance tracking semantics and percentage calculations.

This suite verifies two key goals:

(1) Label Enforcement:
    - A valid sequence:
         startRoot(), start("a"), start("a/b"), stop("a/b"), stop("a"), start("z")
      should work.
    - An invalid sequence:
         startRoot(), start("a"), then start("b")
      should raise a ValueError because when nested the label must include the parent's full path.

(2) Percentage Calculations:
    - Build a multi-level hierarchy (at least three levels) with simulated sleep durations.
    - Verify that both each node’s percentage of its parent’s time ("pct_of_parent") and
      its percentage of the overall total ("pct_of_total") are roughly as expected.
"""

import time
import pytest
from manabot.infra.perf import PerformanceTracker

def test_label_enforcement():
    tracker = PerformanceTracker()
    tracker.startRoot()  # Resets tracker; active node full path is ""
    
    # Valid: at root, local label "a" is allowed.
    tracker.start("a")  # now full path "a"
    # When nested, must supply full parent's name.
    tracker.start("a/b")  # valid: active node full path "a" matches parent part "a"
    tracker.stop("a/b")   # You can call stop with the full hierarchical label...
    tracker.stop("a")     # ...or with the local label.
    # At root again, local label "z" is allowed.
    tracker.start("z")
    tracker.stop("z")
    
    # Now test an invalid sequence:
    tracker.reset()
    tracker.start("a")  # full path "a"
    # While active node is "a", calling start("b") (without the "a/" prefix) should fail.
    with pytest.raises(ValueError) as exc_info:
        tracker.start("b")
    assert "must include the parent's full path" in str(exc_info.value)
    # Clean up: stop "a".
    tracker.stop("a")

def test_hierarchy_percentages():
    """
    Create a multi-level hierarchy and verify percentage calculations.

    Structure:
      - Root (full path: "")
          - group1 (local label "group1")
              - group1/a (start as "group1/a")
                  - group1/a/alpha: sleep ~0.1 sec
                  - group1/a/beta: sleep ~0.1 sec
          - group2 (local label "group2")
              - group2/x: sleep ~0.18 sec (approx 90% of group2's time)
              - group2/y: sleep ~0.02 sec (approx 10% of group2's time)

    Expectations:
      - group1 and group2 should each account for roughly 50% of the overall time.
      - In group1, since "a" is the only child, its pct_of_parent should be ~100%.
        Within group1/a, "alpha" and "beta" should split time roughly equally.
      - In group2, "x" should be about 90% and "y" about 10% of group2's time.
    """
    tracker = PerformanceTracker()
    tracker.startRoot()  # root active node full path is ""
    
    # Group 1
    tracker.start("group1")           # full path "group1"
    tracker.start("group1/a")           # full path "group1/a"
    tracker.start("group1/a/alpha")     # full path "group1/a/alpha"
    time.sleep(0.1)
    tracker.stop("group1/a/alpha")
    tracker.start("group1/a/beta")      # full path "group1/a/beta"
    time.sleep(0.1)
    tracker.stop("group1/a/beta")
    tracker.stop("a")  # Stop "group1/a" using local label.
    tracker.stop("group1")  # Stop group1 using full label "group1" (since at root, local label is expected).
    
    # Group 2
    tracker.start("group2")           # full path "group2"
    tracker.start("group2/x")           # full path "group2/x"
    time.sleep(0.18)
    tracker.stop("group2/x")
    tracker.start("group2/y")           # full path "group2/y"
    time.sleep(0.02)
    tracker.stop("group2/y")
    tracker.stop("group2")
    
    # Finally, stop the root timer.
    # Since the root's local label is "" (empty), we stop with an empty string.
    tracker.stop("")
    
    stats = tracker.get_stats()
    # For debugging, print the stats:
    for path, data in stats.items():
        print(f"{path}: {data}")
    
    # Check overall split: group1 and group2 should each be roughly 50% of total.
    group1_stats = stats.get("group1")
    group2_stats = stats.get("group2")
    assert group1_stats is not None, "Missing stats for 'group1'"
    assert group2_stats is not None, "Missing stats for 'group2'"
    tol_overall = 20.0  # tolerance in percentage points
    assert abs(group1_stats["pct_of_total"] - 50) < tol_overall, f"'group1' pct_of_total not ~50: {group1_stats['pct_of_total']}"
    assert abs(group2_stats["pct_of_total"] - 50) < tol_overall, f"'group2' pct_of_total not ~50: {group2_stats['pct_of_total']}"
    
    # Within group1, there is one child "a" so its pct_of_parent should be 100%.
    a_stats = stats.get("group1/a")
    assert a_stats is not None, "Missing stats for 'group1/a'"
    tol_group1 = 5.0
    assert abs(a_stats["pct_of_parent"] - 100) < tol_group1, f"'group1/a' pct_of_parent not ~100: {a_stats['pct_of_parent']}"
    
    # Within group1/a, check that "alpha" and "beta" split time roughly 50/50.
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

if __name__ == "__main__":
    pytest.main([__file__])
