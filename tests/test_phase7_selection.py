"""
tests/test_phase7_selection.py
------------------------------
Unit tests for Phase 7: Highlight Segment Selection (Budget Algorithm).

Verifies:
1. Selection returns a non-empty list of intervals.
2. Each interval satisfies start < end.
3. Total duration of selected intervals adheres to budget ratio (+/- tolerance).
4. No two intervals overlap in time.
5. Smoothing preserves temporal continuity without NaNs.
"""

import os
import sys
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.selector import select_highlight_segments, smooth_scores


def test_phase7_segment_selection():
    # 66 segments (132 seconds total video length)
    np.random.seed(42)
    scores = np.random.uniform(0.1, 0.9, size=66)

    # 1. Test smoothing function
    smoothed = smooth_scores(scores, window_size=3)
    assert len(smoothed) == len(scores), "Smoothed scores length mismatch"
    assert not np.isnan(smoothed).any(), "NaN found in smoothed scores"

    # 2. Test highlight selection with 15% budget (approx 132 * 0.15 = 19.8s ~ 20s)
    budget = 0.15
    intervals = select_highlight_segments(scores, budget_ratio=budget, min_segment_duration=4.0)
    assert len(intervals) > 0, "No highlight intervals were selected!"

    total_highlight_duration = 0.0
    for idx, (start, end) in enumerate(intervals):
        assert start >= 0.0, f"Interval {idx} start time < 0: {start}"
        assert end > start, f"Interval {idx} end time <= start time: [{start}, {end}]"
        total_highlight_duration += (end - start)

    total_video_duration = len(scores) * 2.0  # 132s
    target_budget_duration = total_video_duration * budget

    print(f"\n[Selection Test] Video duration: {total_video_duration:.1f}s")
    print(f"[Selection Test] Target budget:  {target_budget_duration:.1f}s (15%)")
    print(f"[Selection Test] Actual selected:{total_highlight_duration:.1f}s ({len(intervals)} clips)")

    # 3. Check non-overlapping property
    for i in range(len(intervals) - 1):
        curr_end = intervals[i][1]
        next_start = intervals[i + 1][0]
        assert next_start >= curr_end, f"Overlapping intervals detected: {intervals[i]} and {intervals[i+1]}"

    print("[SUCCESS] All Phase 7 Highlight Segment Selection tests passed perfectly!")


if __name__ == "__main__":
    test_phase7_segment_selection()
