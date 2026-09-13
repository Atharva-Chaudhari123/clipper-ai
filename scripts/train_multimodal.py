"""
scripts/train_multimodal.py
---------------------------
Trains the Multimodal Highlight BiLSTM on the TVSum dataset using:
- 2048-D frozen ResNet-50 visual features
- 384-D Whisper text transcript embeddings
- Human importance score ground truth
- Masked MSE loss and validation tracking (Spearman rho & Kendall tau)

Saves best checkpoint to:
models/best_multimodal_bilstm.pt
(Preserves models/best_bilstm.pt baseline untouched)
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, kendalltau
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.multimodal_model import MultimodalHighlightBiLSTM
from src.multimodal_dataset import MultimodalTVSumDataset, multimodal_collate_fn

TRAIN_ANNOTATIONS = os.path.join(BASE_DIR, "annotations", "tvsum_train_release.jsonl")
VAL_ANNOTATIONS = os.path.join(BASE_DIR, "annotations", "tvsum_val_release.jsonl")
VISUAL_FEATS_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
TEXT_FEATS_DIR = os.path.join(BASE_DIR, "data", "processed", "text_features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_multimodal_bilstm.pt")


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes MSE loss only over unpadded positions."""
    diff = (preds - targets) ** 2
    masked_diff = diff * mask.float()
    return masked_diff.sum() / torch.clamp(mask.float().sum(), min=1.0)


def evaluate(model, dataloader, device):
    """Evaluates validation loss, Spearman rho, and Kendall tau on held-out validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    spearman_scores = []
    kendall_scores = []

    with torch.no_grad():
        for batch in dataloader:
            v = batch["visual_features"].to(device)
            t = batch["text_features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            preds = model(v, t, mask=mask)
            loss = masked_mse_loss(preds, targets, mask)
            total_loss += loss.item()
            num_batches += 1

            # Compute rank correlations for each video in the batch
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, l in enumerate(lengths):
                p_seq = preds_np[i, :l]
                t_seq = targets_np[i, :l]

                if np.std(p_seq) > 1e-6 and np.std(t_seq) > 1e-6:
                    rho, _ = spearmanr(p_seq, t_seq)
                    tau, _ = kendalltau(p_seq, t_seq)
                    if not np.isnan(rho):
                        spearman_scores.append(rho)
                    if not np.isnan(tau):
                        kendall_scores.append(tau)

    avg_loss = total_loss / max(num_batches, 1)
    mean_spearman = float(np.mean(spearman_scores)) if spearman_scores else 0.0
    mean_kendall = float(np.mean(kendall_scores)) if kendall_scores else 0.0

    return avg_loss, mean_spearman, mean_kendall


def train_multimodal(
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
):
    print("=" * 65)
    print("MULTIMODAL HIGHLIGHT BiLSTM TRAINING (VISUAL + WHISPER TEXT)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Datasets and Loaders
    train_dataset = MultimodalTVSumDataset(TRAIN_ANNOTATIONS, VISUAL_FEATS_DIR, TEXT_FEATS_DIR)
    val_dataset = MultimodalTVSumDataset(VAL_ANNOTATIONS, VISUAL_FEATS_DIR, TEXT_FEATS_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=multimodal_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn
    )

    print(f"Train samples: {len(train_dataset)} videos (Batches: {len(train_loader)})")
    print(f"Val samples:   {len(val_dataset)} videos (Batches: {len(val_loader)})\n")

    # Model definition
    model = MultimodalHighlightBiLSTM(
        visual_dim=2048,
        text_dim=384,
        proj_visual_dim=512,
        proj_text_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(f"Multimodal Model Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_spearman = 0.0
    best_kendall = 0.0
    best_epoch = 0

    print("\nStarting training loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Spearman':>10} | {'Kendall':>10} | {'Status':>8}")
    print("-" * 65)

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            v = batch["visual_features"].to(device)
            t = batch["text_features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(v, t, mask=mask)
            loss = masked_mse_loss(preds, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        val_loss, val_spearman, val_kendall = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_spearman = val_spearman
            best_kendall = val_kendall
            best_epoch = epoch
            status = "BEST *"

            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_spearman": val_spearman,
                "val_kendall": val_kendall,
                "config": {
                    "visual_dim": 2048,
                    "text_dim": 384,
                    "proj_visual_dim": 512,
                    "proj_text_dim": 256,
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "dropout": 0.3
                }
            }, BEST_MODEL_PATH)

        print(f"{epoch:5d} | {avg_train_loss:10.5f} | {val_loss:10.5f} | {val_spearman:10.4f} | {val_kendall:10.4f} | {status:>8}")

    total_time = time.time() - start_train_time
    print("-" * 65)
    print(f"Training completed in {total_time:.1f}s.")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss:     {best_val_loss:.5f}")
    print(f"Best Spearman Rho: {best_spearman:.4f}")
    print(f"Best Kendall Tau:  {best_kendall:.4f}")
    print(f"Checkpoint saved:  {BEST_MODEL_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    train_multimodal()
