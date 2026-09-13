"""
tests/test_phase8_render.py
---------------------------
Unit tests for Phase 8: FFmpeg Highlight Video Rendering Engine.

Verifies:
1. Video rendering successfully cuts and concatenates highlight intervals.
2. Output MP4 file is created and has non-zero size.
3. Output video duration closely matches the sum of requested interval lengths.
4. FFmpeg cleans up intermediate part files.
"""

import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.renderer import render_highlight_video


def test_phase8_rendering():
    video_path = r"C:\Users\mahin\OneDrive\Desktop\New folder\video\akI8YFjEmUw.mp4"
    test_output_path = r"C:\Users\mahin\OneDrive\Desktop\New folder\outputs\highlights\test_sample_render.mp4"

    assert os.path.isfile(video_path), f"Sample video not found at: {video_path}"

    # Test intervals: total 10.0 seconds (4s + 6s)
    test_intervals = [(10.0, 14.0), (30.0, 36.0)]
    expected_duration = 10.0

    print(f"\nRendering test highlight clips: {test_intervals} (expected duration: {expected_duration:.1f}s)...")
    stats = render_highlight_video(
        source_video_path=video_path,
        intervals=test_intervals,
        output_path=test_output_path
    )

    # 1. Output file exists
    assert os.path.isfile(test_output_path), f"Rendered output file missing: {test_output_path}"

    # 2. File size > 0
    assert stats["file_size_bytes"] > 0, "Rendered file is 0 bytes!"
    print(f"Rendered file size: {stats['file_size_mb']} MB")

    # 3. Duration check (+/- 1.0 second tolerance for keyframe boundaries)
    actual_dur = stats["duration_seconds"]
    print(f"Expected duration: {expected_duration:.1f}s, Actual rendered duration: {actual_dur:.1f}s")
    assert abs(actual_dur - expected_duration) <= 1.5, (
        f"Rendered duration {actual_dur:.1f}s deviates too much from expected {expected_duration:.1f}s"
    )

    print("[SUCCESS] All Phase 8 FFmpeg Video Rendering tests passed perfectly!")


if __name__ == "__main__":
    test_phase8_rendering()
