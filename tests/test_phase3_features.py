"""
tests/test_phase3_features.py
-----------------------------
Unit tests for Phase 3: Pretrained ResNet-50 Feature Extraction.

Verifies:
1. Feature file exists (data/processed/features/<video_id>.pt).
2. Feature file loads successfully with torch.load.
3. Feature tensor has exactly 2 dimensions (ndim == 2).
4. Number of rows equals number of extracted frames (66).
5. Feature dimension equals 2048 (shape[1] == 2048).
6. No NaN values in the feature tensor.
7. No Inf values in the feature tensor.
8. Metadata CSV exists (data/processed/features/<video_id>_metadata.csv).
9. Metadata row count equals feature row count (66).
10. Segment indices in metadata are strictly continuous (0, 1, 2, ..., N-1).
11. Feature tensor dtype is valid (torch.float32).
"""

import os
import sys
import csv
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_phase3_features():
    video_id = "akI8YFjEmUw"
    features_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features"
    frames_dir = os.path.join(r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames", video_id)

    feature_file = os.path.join(features_dir, f"{video_id}.pt")
    metadata_file = os.path.join(features_dir, f"{video_id}_metadata.csv")
    frame_meta_file = os.path.join(frames_dir, "metadata.csv")

    # Read original frame count from Phase 2 metadata
    assert os.path.isfile(frame_meta_file), f"Phase 2 frame metadata missing at: {frame_meta_file}"
    with open(frame_meta_file, "r", encoding="utf-8") as f:
        frame_rows = list(csv.DictReader(f))
    expected_frames = len(frame_rows)

    # 1. Feature file exists
    assert os.path.isfile(feature_file), f"Feature file not found: {feature_file}"

    # 2. Feature file loads successfully
    features = torch.load(feature_file)

    # 3. Feature tensor has exactly 2 dimensions
    assert features.ndim == 2, f"Expected 2D tensor [N, 2048], got {features.ndim}D shape: {features.shape}"

    # 4. Number of rows equals number of extracted frames
    assert features.shape[0] == expected_frames, (
        f"Row count mismatch: expected {expected_frames}, got {features.shape[0]}"
    )

    # 5. Feature dimension equals 2048
    assert features.shape[1] == 2048, (
        f"Feature dimension mismatch: expected 2048, got {features.shape[1]}"
    )

    # 6. No NaN values
    assert not torch.isnan(features).any().item(), "Found NaN values in feature tensor!"

    # 7. No Inf values
    assert not torch.isinf(features).any().item(), "Found Inf values in feature tensor!"

    # 8. Metadata exists
    assert os.path.isfile(metadata_file), f"Feature metadata file missing: {metadata_file}"

    # 9. Metadata row count equals feature row count
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))
    assert len(meta_rows) == features.shape[0], (
        f"Metadata row count ({len(meta_rows)}) != feature count ({features.shape[0]})"
    )

    # 10. Segment indices are continuous
    seg_indices = [int(r["segment_index"]) for r in meta_rows]
    expected_indices = list(range(expected_frames))
    assert seg_indices == expected_indices, f"Segment indices are not continuous: {seg_indices[:5]}..."

    # 11. Feature dtype is valid
    assert features.dtype == torch.float32, f"Expected torch.float32, got {features.dtype}"

    print(f"\n[SUCCESS] All 11 Phase 3 Feature Extraction tests passed perfectly!")
    print(f"Verified feature shape: {list(features.shape)} (dtype: {features.dtype})")


if __name__ == "__main__":
    test_phase3_features()
