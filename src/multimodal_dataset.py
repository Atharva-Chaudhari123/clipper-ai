"""
src/multimodal_dataset.py
-------------------------
PyTorch Dataset and DataLoader Collate Function for Multimodal Video Highlights.

Loads aligned pairs of:
1. ResNet-50 visual features [T, 2048]
2. Whisper-derived text embeddings [T, 384]
3. Normalized TVSum human importance scores [T] in [0.0, 1.0]
4. Boolean padding masks [B, max_len] for variable-length batching
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List

from src.dataset import get_normalized_ground_truth


class MultimodalTVSumDataset(Dataset):
    """
    Multimodal TVSum Dataset loading both visual and text features.
    """
    def __init__(
        self,
        annotation_file: str,
        visual_features_dir: str,
        text_features_dir: str
    ):
        super(MultimodalTVSumDataset, self).__init__()
        self.annotation_file = annotation_file
        self.visual_features_dir = visual_features_dir
        self.text_features_dir = text_features_dir

        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        if not os.path.isdir(visual_features_dir):
            raise FileNotFoundError(f"Visual features directory not found: {visual_features_dir}")
        if not os.path.isdir(text_features_dir):
            raise FileNotFoundError(f"Text features directory not found: {text_features_dir}")

        self.records = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        vid = record["vid"]

        v_file = os.path.join(self.visual_features_dir, f"{vid}.pt")
        t_file = os.path.join(self.text_features_dir, f"{vid}.pt")

        if not os.path.isfile(v_file):
            raise FileNotFoundError(f"Visual feature file missing for {vid} at {v_file}")
        if not os.path.isfile(t_file):
            raise FileNotFoundError(f"Text feature file missing for {vid} at {t_file}")

        visual_feats = torch.load(v_file).float()  # [T_v, 2048]
        text_feats = torch.load(t_file).float()    # [T_t, 384]
        targets = get_normalized_ground_truth(record["label"])  # [T_gt]

        # Harmonize lengths across modalities and ground truth
        min_len = min(visual_feats.shape[0], text_feats.shape[0], targets.shape[0])
        visual_feats = visual_feats[:min_len]
        text_feats = text_feats[:min_len]
        targets = targets[:min_len]

        return {
            "vid": vid,
            "query": record.get("query", ""),
            "domain": record.get("domain", ""),
            "visual_features": visual_feats,  # [T, 2048]
            "text_features": text_feats,      # [T, 384]
            "targets": targets,                # [T]
            "seq_len": min_len
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable-length multimodal sequences into batched tensors with boolean masks.
    """
    batch_size = len(batch)
    lengths = [item["seq_len"] for item in batch]
    max_len = max(lengths)

    v_dim = batch[0]["visual_features"].shape[1]  # 2048
    t_dim = batch[0]["text_features"].shape[1]    # 384

    padded_v = torch.zeros(batch_size, max_len, v_dim, dtype=torch.float32)
    padded_t = torch.zeros(batch_size, max_len, t_dim, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    vids = []

    for i, item in enumerate(batch):
        t = item["seq_len"]
        padded_v[i, :t, :] = item["visual_features"]
        padded_t[i, :t, :] = item["text_features"]
        padded_targets[i, :t] = item["targets"]
        mask[i, :t] = True
        vids.append(item["vid"])

    return {
        "visual_features": padded_v,                        # [B, max_len, 2048]
        "text_features": padded_t,                          # [B, max_len, 384]
        "targets": padded_targets,                          # [B, max_len]
        "mask": mask,                                       # [B, max_len]
        "lengths": torch.tensor(lengths, dtype=torch.long), # [B]
        "vids": vids
    }
