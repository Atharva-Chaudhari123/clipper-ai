"""
tests/test_phase2_sampling.py
-----------------------------
Unit tests for Phase 2: TVSum Segment-Center Frame Sampling.

Verifies:
1. Frames directory exists and contains extracted image files.
2. metadata.csv exists and is non-empty.
3. Number of extracted frames matches the number of annotation segments.
4. Metadata segment indices are strictly continuous (0, 1, 2, ..., N-1).
5. Sample times strictly follow the formula: sample_time = 2*i + 1.0.
6. Annotation intervals strictly follow: [2*i, 2*(i+1)).
7. No missing frame files referenced in metadata.
8. No duplicate segment indices in metadata.
"""

import os
import sys
import csv
import glob
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.extract_frames import find_annotation_record


def test_phase2_frame_sampling():
    video_id = "akI8YFjEmUw"
    base_frames_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames"
    video_frames_dir = os.path.join(base_frames_dir, video_id)
    annotations_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations"

    # Load ground-truth annotation for comparison
    annot_record = find_annotation_record(video_id, annotations_dir)
    expected_segments = len(annot_record["label"])

    # 1. Verify frames directory and files exist
    assert os.path.isdir(video_frames_dir), f"Frames directory does not exist: {video_frames_dir}"
    jpg_files = sorted(glob.glob(os.path.join(video_frames_dir, "segment_*.jpg")))
    assert len(jpg_files) > 0, "No JPEG frame files found in frames directory"

    # 2. Verify metadata.csv exists and is readable
    metadata_path = os.path.join(video_frames_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"metadata.csv missing at: {metadata_path}"

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, "metadata.csv is empty"

    # 3. Verify number of frames equals number of annotation segments
    assert len(jpg_files) == expected_segments, (
        f"Frame count mismatch: found {len(jpg_files)} frames, expected {expected_segments} segments"
    )
    assert len(rows) == expected_segments, (
        f"Metadata row count mismatch: found {len(rows)} rows, expected {expected_segments} segments"
    )

    # 4. Verify metadata segment indices are strictly continuous [0, 1, 2, ..., N-1]
    segment_indices = [int(r["segment_index"]) for r in rows]
    expected_indices = list(range(expected_segments))
    assert segment_indices == expected_indices, "Segment indices are not strictly continuous"

    # 5. Verify sample times follow formula: 2*i + 1.0
    for i, r in enumerate(rows):
        expected_sample_time = 2.0 * i + 1.0
        actual_sample_time = float(r["sample_time"])
        assert np.isclose(actual_sample_time, expected_sample_time, atol=1e-2), (
            f"Segment {i}: expected sample time {expected_sample_time}, got {actual_sample_time}"
        )

    # 6. Verify annotation intervals follow formula: [2*i, 2*(i+1))
    for i, r in enumerate(rows):
        expected_start = 2.0 * i
        expected_end = 2.0 * (i + 1)
        actual_start = float(r["annotation_start"])
        actual_end = float(r["annotation_end"])
        assert np.isclose(actual_start, expected_start, atol=1e-2), (
            f"Segment {i}: expected start {expected_start}, got {actual_start}"
        )
        assert np.isclose(actual_end, expected_end, atol=1e-2), (
            f"Segment {i}: expected end {expected_end}, got {actual_end}"
        )

    # 7. Verify no missing frame files (every row points to a valid file on disk)
    for r in rows:
        frame_file = os.path.join(video_frames_dir, r["frame_path"])
        assert os.path.isfile(frame_file), f"Referenced frame file does not exist: {frame_file}"
        assert os.path.getsize(frame_file) > 0, f"Frame file is empty (0 bytes): {frame_file}"

    # 8. Verify no duplicate segment indices
    assert len(segment_indices) == len(set(segment_indices)), "Duplicate segment indices found in metadata"

    print("\n[SUCCESS] All 8 Phase 2 Frame Sampling tests passed perfectly!")


if __name__ == "__main__":
    test_phase2_frame_sampling()
