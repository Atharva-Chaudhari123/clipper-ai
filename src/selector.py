"""
src/selector.py
---------------
Phase 7: Highlight Segment Selection (Budget-based Top-K / Shot Selection).

Converts continuous 1D importance score predictions into a cohesive list
of video highlight intervals [(start_sec, end_sec), ...] based on a target duration budget
(e.g., 15% of total video duration or a fixed duration like 30 seconds).
"""

import numpy as np


def smooth_scores(scores: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Applies moving average temporal smoothing to segment scores.
    Window size 3 over 2-second segments smooths over a 6-second receptive field,
    preventing isolated single-frame flickering in highlight clips.

    Parameters:
        scores (np.ndarray): 1D array of predicted segment importance scores.
        window_size (int): Size of the moving average filter window (default: 3).

    Returns:
        np.ndarray: 1D array of smoothed scores of the same shape.
    """
    if len(scores) < window_size:
        return scores
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="same")
    return smoothed


def select_highlight_segments(
    scores: np.ndarray,
    budget_ratio: float = 0.15,
    min_segment_duration: float = 4.0,
    segment_duration: float = 2.0
) -> list[tuple[float, float]]:
    """
    Selects top highlight intervals from predicted segment scores under a duration budget.

    Parameters:
        scores (np.ndarray): 1D array of predicted scores for each 2-second segment.
        budget_ratio (float): Fraction of original video length to include in highlight (default: 0.15 = 15%).
        min_segment_duration (float): Minimum duration in seconds for any kept clip (default: 4.0s = 2 segments).
        segment_duration (float): Duration in seconds of each TVSum segment (default: 2.0s).

    Returns:
        list[tuple[float, float]]: List of non-overlapping highlight intervals [(start_sec, end_sec), ...].
    """
    num_segments = len(scores)
    if num_segments == 0:
        return []

    smoothed = smooth_scores(scores, window_size=3)

    # Calculate target number of segments based on budget ratio
    total_budget_segments = max(1, int(round(num_segments * budget_ratio)))

    # Rank segments by smoothed importance score
    sorted_indices = np.argsort(-smoothed)
    selected_mask = np.zeros(num_segments, dtype=bool)

    # Select top segments
    top_indices = sorted_indices[:total_budget_segments]
    selected_mask[top_indices] = True

    # Group contiguous True values into intervals
    intervals = []
    in_interval = False
    start_idx = 0

    for i in range(num_segments):
        if selected_mask[i] and not in_interval:
            in_interval = True
            start_idx = i
        elif not selected_mask[i] and in_interval:
            in_interval = False
            intervals.append((start_idx * segment_duration, i * segment_duration))

    if in_interval:
        intervals.append((start_idx * segment_duration, num_segments * segment_duration))

    # Merge intervals separated by <= 2.0 seconds to produce natural, smooth viewing
    merged_intervals = []
    for start, end in intervals:
        if not merged_intervals:
            merged_intervals.append((start, end))
        else:
            prev_start, prev_end = merged_intervals[-1]
            if start - prev_end <= 2.0:
                merged_intervals[-1] = (prev_start, end)
            else:
                merged_intervals.append((start, end))

    # Filter out very brief intervals shorter than min_segment_duration unless it's the only one
    filtered = [
        (s, e) for (s, e) in merged_intervals
        if (e - s) >= min_segment_duration
    ]
    if not filtered and merged_intervals:
        filtered = merged_intervals

    return filtered
