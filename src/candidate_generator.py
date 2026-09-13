"""
src/candidate_generator.py
--------------------------
Phase 13: Candidate Window Generator for Long Videos.

Combines visual saliency peaks from FastVisualScreener with speech activity
intervals from AudioPreScreener to generate bounded candidate highlight windows
(typically 20–60 seconds). Merges overlapping windows and applies duration/density
caps so expensive Whisper transcription runs ONLY on promising regions.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("CandidateGenerator")


def generate_candidate_windows(
    duration_sec: float,
    visual_scores: np.ndarray,
    segment_times: List[float],
    speech_intervals: Optional[List[Tuple[float, float]]] = None,
    target_window_duration: float = 40.0,
    min_window_duration: float = 20.0,
    max_window_duration: float = 75.0,
    merge_gap_threshold: float = 6.0,
    max_candidates: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generates a curated list of candidate windows for targeted transcription and scoring.

    Args:
        duration_sec: Total duration of video in seconds
        visual_scores: 1D array of visual activity/importance scores
        segment_times: Timestamps corresponding to visual_scores
        speech_intervals: List of (start_sec, end_sec) speech active intervals (or None)
        target_window_duration: Initial target duration for each window (default: 40s)
        min_window_duration: Minimum duration of a kept candidate window (default: 20s)
        max_window_duration: Maximum duration of a kept candidate window (default: 75s)
        merge_gap_threshold: Gap threshold (seconds) below which adjacent windows are merged
        max_candidates: Maximum number of candidate windows to return

    Returns:
        List of dicts: [
            {"candidate_id": int, "start": float, "end": float, "duration": float, "pre_score": float}
        ]
    """
    if duration_sec <= 0.0 or len(visual_scores) == 0:
        return []

    # If video is already short (<= 60s), entire video is a single candidate
    if duration_sec <= 65.0:
        return [{
            "candidate_id": 1,
            "start": 0.0,
            "end": round(duration_sec, 2),
            "duration": round(duration_sec, 2),
            "pre_score": float(np.mean(visual_scores)) if len(visual_scores) > 0 else 0.5
        }]

    # Determine default max_candidates based on video length
    if max_candidates is None:
        if duration_sec <= 600:       # <= 10 min
            max_candidates = 30
        elif duration_sec <= 1800:    # <= 30 min
            max_candidates = 60
        else:                         # 60+ min
            max_candidates = 100

    # Create speech presence mask across segments if speech detected
    num_segments = len(visual_scores)
    speech_boost = np.ones(num_segments, dtype=np.float32)
    has_speech = speech_intervals is not None and len(speech_intervals) > 0

    if has_speech:
        speech_mask = np.zeros(num_segments, dtype=bool)
        for i, t in enumerate(segment_times):
            # Check if segment center t falls inside any speech interval
            for sp_start, sp_end in speech_intervals:
                if sp_start - 1.0 <= t <= sp_end + 1.0:
                    speech_mask[i] = True
                    break
        # Boost segments with speech activity, dampen completely silent segments
        speech_boost = np.where(speech_mask, 1.25, 0.75)

    # Combined pre-screening score curve
    combined_scores = visual_scores * speech_boost

    # Identify local peaks
    sorted_indices = np.argsort(-combined_scores)
    half_win = target_window_duration / 2.0

    raw_windows: List[Tuple[float, float, float]] = []  # (start, end, score)
    selected_centers = []

    min_center_dist = target_window_duration * 0.7
    for idx in sorted_indices:
        center_t = segment_times[idx]
        # Suppress centers too close to already selected centers
        if any(abs(center_t - c) < min_center_dist for c in selected_centers):
            continue

        start_t = max(0.0, center_t - half_win)
        end_t = min(duration_sec, center_t + half_win)
        score = float(combined_scores[idx])

        raw_windows.append((start_t, end_t, score))
        selected_centers.append(center_t)

        if len(raw_windows) >= max_candidates:
            break

    # Sort windows chronologically
    raw_windows.sort(key=lambda x: x[0])

    # Merge overlapping or close windows
    merged: List[Dict[str, Any]] = []
    for s, e, score in raw_windows:
        if not merged:
            merged.append({"start": s, "end": e, "scores": [score]})
        else:
            prev = merged[-1]
            if s <= prev["end"] + merge_gap_threshold:
                # Merge if resulting duration does not exceed max_window_duration
                merged_dur = max(prev["end"], e) - prev["start"]
                if merged_dur <= max_window_duration:
                    prev["end"] = max(prev["end"], e)
                    prev["scores"].append(score)
                else:
                    # Non-overlapping adjacent window
                    s_adj = max(s, prev["end"])
                    if e - s_adj >= min_window_duration * 0.7:
                        merged.append({"start": s_adj, "end": e, "scores": [score]})
            else:
                merged.append({"start": s, "end": e, "scores": [score]})


    # Finalize and filter duration
    final_candidates: List[Dict[str, Any]] = []
    for item in merged:
        s = round(item["start"], 2)
        e = round(item["end"], 2)
        dur = round(e - s, 2)
        if dur >= min_window_duration:
            # Clamp if slightly over max
            if dur > max_window_duration:
                e = round(s + max_window_duration, 2)
                dur = round(e - s, 2)
            avg_score = float(np.mean(item["scores"]))
            final_candidates.append({
                "start": s,
                "end": e,
                "duration": dur,
                "pre_score": round(avg_score, 4)
            })

    # Rank by pre_score and cap to max_candidates
    final_candidates.sort(key=lambda x: x["pre_score"], reverse=True)
    if len(final_candidates) > max_candidates:
        final_candidates = final_candidates[:max_candidates]

    # Re-sort chronologically for downstream processing
    final_candidates.sort(key=lambda x: x["start"])

    # Assign IDs
    for i, cand in enumerate(final_candidates):
        cand["candidate_id"] = i + 1

    total_cand_time = sum(c["duration"] for c in final_candidates)
    reduction_pct = 100.0 * (1.0 - (total_cand_time / duration_sec)) if duration_sec > 0 else 0.0

    logger.info(
        f"Generated {len(final_candidates)} candidate windows "
        f"({total_cand_time:.1f}s candidate time vs {duration_sec:.1f}s video, "
        f"{reduction_pct:.1f}% transcription time reduction!)"
    )

    return final_candidates
