"""
src/train.py
------------
Phase 6: Model Training and Evaluation on TVSum Dataset.

Trains the VideoHighlightBiLSTM network on the 40 TVSum training videos
and evaluates on the 10 validation videos using:
- Masked Mean Squared Error (MSE) loss
- Spearman's rank correlation coefficient (rho)
- Kendall's tau rank correlation coefficient (tau)

Saves the best performing model checkpoint to models/best_bilstm.pt.
"""

import os
import sys
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, kendalltau
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TVSumDataset, tvsum_collate_fn
from src.lstm_model import VideoHighlightBiLSTM, compute_masked_mse_loss


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> tuple[float, float, float]:
    """
    Evaluates model on validation set. Computes average MSE loss, Spearman rho, and Kendall tau.

    Parameters:
        model: VideoHighlightBiLSTM model
        val_loader: DataLoader for validation split
        device: torch device

    Returns:
        tuple[float, float, float]: (avg_val_loss, avg_spearman_rho, avg_kendall_tau)
    """
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    spearman_scores = []
    kendall_scores = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            preds = model(features)
            loss = compute_masked_mse_loss(preds, targets, mask)
            total_val_loss += loss.item()
            num_batches += 1

            # Compute rank correlation per video sequence
            preds_cpu = preds.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            for i in range(len(lengths)):
                L = lengths[i].item()
                p_seq = preds_cpu[i, :L]
                t_seq = targets_cpu[i, :L]

                # Avoid zero variance edge case
                if np.std(p_seq) > 1e-6 and np.std(t_seq) > 1e-6:
                    rho, _ = spearmanr(p_seq, t_seq)
                    tau, _ = kendalltau(p_seq, t_seq)
                    if not np.isnan(rho):
                        spearman_scores.append(rho)
                    if not np.isnan(tau):
                        kendall_scores.append(tau)

    avg_loss = total_val_loss / max(1, num_batches)
    avg_rho = float(np.mean(spearman_scores)) if spearman_scores else 0.0
    avg_tau = float(np.mean(kendall_scores)) if kendall_scores else 0.0
    return avg_loss, avg_rho, avg_tau


def train_bilstm(
    annotations_dir: str = r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations",
    features_dir: str = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features",
    models_dir: str = r"C:\Users\mahin\OneDrive\Desktop\New folder\models",
    num_epochs: int = 25,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3
) -> str:
    """
    Main training loop for VideoHighlightBiLSTM.
    """
    os.makedirs(models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("TVSUM BI-LSTM TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device:            {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs:            {num_epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Hidden Dim:        {hidden_dim} (x2 Bidirectional = {hidden_dim * 2})")
    print(f"LSTM Layers:       {num_layers}")
    print("=" * 60)

    train_jsonl = os.path.join(annotations_dir, "tvsum_train_release.jsonl")
    val_jsonl = os.path.join(annotations_dir, "tvsum_val_release.jsonl")

    train_dataset = TVSumDataset(annotation_file=train_jsonl, features_dir=features_dir)
    val_dataset = TVSumDataset(annotation_file=val_jsonl, features_dir=features_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tvsum_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=tvsum_collate_fn)

    model = VideoHighlightBiLSTM(
        input_dim=2048,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(models_dir, "best_bilstm.pt")
    history = []

    print("\nStarting training...")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Spearman Rho':<14} | {'Kendall Tau':<12} | {'Time'}")
    print("-" * 75)

    start_train_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = compute_masked_mse_loss(preds, targets, mask)
            loss.backward()

            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(1, num_batches)
        avg_val_loss, avg_rho, avg_tau = evaluate(model, val_loader, device)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - t0
        is_best = avg_val_loss < best_val_loss

        if is_best:
            best_val_loss = avg_val_loss
            # Save checkpoint dictionary
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "spearman_rho": avg_rho,
                "kendall_tau": avg_tau,
                "config": {
                    "input_dim": 2048,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout
                }
            }, best_checkpoint_path)

        star = " *" if is_best else ""
        print(f"{epoch:<8} | {avg_train_loss:<12.5f} | {avg_val_loss:<12.5f} | {avg_rho:<14.4f} | {avg_tau:<12.4f} | {epoch_time:.1f}s{star}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "spearman_rho": avg_rho,
            "kendall_tau": avg_tau
        })

    total_time = time.time() - start_train_time
    print("-" * 75)
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.2f} mins).")
    print(f"Best checkpoint saved to: {best_checkpoint_path} (Val Loss: {best_val_loss:.5f})")

    history_path = os.path.join(models_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TVSum VideoHighlightBiLSTM")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_bilstm(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
