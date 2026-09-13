"""
src/dataset.py
--------------
Phase 4: TVSum PyTorch Dataset and DataLoader with Temporal Masking.

Provides a robust PyTorch Dataset and custom collate function for loading:
- Pre-extracted ResNet-50 visual feature sequences [T, 2048]
- Normalized human importance ground-truth ratings [T] in [0.0, 1.0]
- Temporal sequence masking to handle variable-length videos in mini-batches
"""

import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_normalized_ground_truth(labels_2d: list[list[float]]) -> torch.Tensor:
    """
    Converts 2D annotator ratings [N_segments, 20] into normalized 1D ground-truth scores [N_segments].
    Ratings are from 1.0 to 5.0. Formula: (mean_score - 1.0) / 4.0

    Parameters:
        labels_2d (list[list[float]]): 2D list of shape [N_segments, 20].

    Returns:
        torch.Tensor: 1D float32 tensor of shape [N_segments] bounded in [0.0, 1.0].
    """
    arr = np.array(labels_2d, dtype=np.float32)
    mean_scores = np.mean(arr, axis=1)  # Mean across 20 human annotators
    norm_scores = (mean_scores - 1.0) / 4.0
    norm_scores = np.clip(norm_scores, 0.0, 1.0)
    return torch.tensor(norm_scores, dtype=torch.float32)


class TVSumDataset(Dataset):
    """
    PyTorch Dataset for TVSum video features and ground-truth importance scores.

    Parameters:
        annotation_file (str): Path to JSONL annotation file (train or val).
        features_dir (str): Path to directory containing cached <video_id>.pt files.
    """

    def __init__(self, annotation_file: str, features_dir: str):
        super(TVSumDataset, self).__init__()
        self.annotation_file = annotation_file
        self.features_dir = features_dir

        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not os.path.isdir(features_dir):
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        # Load all video records from JSONL
        self.records = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        print(f"Loaded {len(self.records)} records from {os.path.basename(annotation_file)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        vid = record["vid"]

        feature_file = os.path.join(self.features_dir, f"{vid}.pt")
        if not os.path.isfile(feature_file):
            raise FileNotFoundError(f"Cached feature file missing for video '{vid}' at: {feature_file}")

        # Load cached 2048-d feature tensor: shape [T, 2048]
        features = torch.load(feature_file)
        if features.dtype != torch.float32:
            features = features.float()

        # Compute normalized ground-truth target: shape [T]
        targets = get_normalized_ground_truth(record["label"])

        # Temporal length check
        num_features = features.shape[0]
        num_targets = targets.shape[0]

        if num_features != num_targets:
            # Align by truncating to minimum length if minor 1-frame edge boundary difference occurs
            min_len = min(num_features, num_targets)
            features = features[:min_len]
            targets = targets[:min_len]

        return {
            "vid": vid,
            "query": record.get("query", ""),
            "domain": record.get("domain", ""),
            "features": features,          # Shape: [T, 2048]
            "targets": targets,            # Shape: [T]
            "seq_len": features.shape[0]   # Scalar integer
        }


def tvsum_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for batching variable-length temporal sequences.
    Pads feature sequences with 0 to max_len in the batch, and creates a boolean mask.

    Parameters:
        batch (list[dict]): List of items from TVSumDataset.

    Returns:
        dict:
            - 'features': Tensor of shape [B, max_len, 2048]
            - 'targets':  Tensor of shape [B, max_len]
            - 'mask':     Bool Tensor of shape [B, max_len] (True for valid positions, False for padding)
            - 'lengths':  Long Tensor of shape [B] containing original unpadded lengths
            - 'vids':     List of video ID strings [B]
    """
    batch_size = len(batch)
    lengths = [item["seq_len"] for item in batch]
    max_len = max(lengths)
    feature_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    vids = []

    for i, item in enumerate(batch):
        t = item["seq_len"]
        padded_features[i, :t, :] = item["features"]
        padded_targets[i, :t] = item["targets"]
        mask[i, :t] = True
        vids.append(item["vid"])

    return {
        "features": padded_features,               # [B, max_len, 2048]
        "targets": padded_targets,                 # [B, max_len]
        "mask": mask,                              # [B, max_len]
        "lengths": torch.tensor(lengths, dtype=torch.long), # [B]
        "vids": vids
    }
