"""
tests/test_phase4_dataset.py
----------------------------
Unit tests for Phase 4: TVSum Dataset and DataLoader with Temporal Masking.

Verifies:
1. TVSumDataset loads items correctly from annotations and cached features.
2. Item dictionary contains expected keys ('vid', 'features', 'targets', 'seq_len').
3. Targets are bounded strictly in [0.0, 1.0].
4. DataLoader collate_fn pads sequences accurately to max_len.
5. Collate mask has True for valid frames and False for padding.
6. Masked lengths match actual unpadded lengths.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TVSumDataset, tvsum_collate_fn


def test_phase4_dataset():
    annotations_file = r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations\tvsum_train_release.jsonl"
    features_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features"

    dataset = TVSumDataset(annotation_file=annotations_file, features_dir=features_dir)
    assert len(dataset) > 0, "Dataset is empty!"

    # 1. Test single item loading
    item = dataset[0]
    assert "vid" in item
    assert "features" in item
    assert "targets" in item
    assert "seq_len" in item

    features = item["features"]
    targets = item["targets"]
    assert features.ndim == 2, f"Expected 2D features [T, 2048], got {features.shape}"
    assert features.shape[1] == 2048, f"Expected feature dim 2048, got {features.shape[1]}"
    assert targets.ndim == 1, f"Expected 1D targets [T], got {targets.shape}"
    assert features.shape[0] == targets.shape[0], f"Feature/target length mismatch: {features.shape[0]} vs {targets.shape[0]}"

    # 2. Check target bounds [0.0, 1.0]
    assert (targets >= 0.0).all().item(), "Found target < 0.0"
    assert (targets <= 1.0).all().item(), "Found target > 1.0"
    assert not torch.isnan(targets).any().item(), "Target contains NaN"

    # 3. Test DataLoader and collate_fn with small batch
    batch_size = 4
    # Take first 4 items only if available
    sample_indices = list(range(min(4, len(dataset))))
    subset = torch.utils.data.Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=tvsum_collate_fn)

    batch = next(iter(loader))
    b_features = batch["features"]
    b_targets = batch["targets"]
    b_mask = batch["mask"]
    b_lengths = batch["lengths"]

    max_len = max(b_lengths).item()
    assert b_features.shape == (len(subset), max_len, 2048), f"Unexpected padded features shape: {b_features.shape}"
    assert b_targets.shape == (len(subset), max_len), f"Unexpected padded targets shape: {b_targets.shape}"
    assert b_mask.shape == (len(subset), max_len), f"Unexpected mask shape: {b_mask.shape}"

    # Verify mask validity
    for i, length in enumerate(b_lengths):
        assert b_mask[i, :length].all().item(), f"Mask should be True for valid frames in item {i}"
        assert not b_mask[i, length:].any().item(), f"Mask should be False for padding in item {i}"

    print(f"\n[SUCCESS] All Phase 4 TVSum Dataset & DataLoader tests passed perfectly!")
    print(f"Verified batch features shape: {list(b_features.shape)}, mask shape: {list(b_mask.shape)}")


if __name__ == "__main__":
    test_phase4_dataset()
