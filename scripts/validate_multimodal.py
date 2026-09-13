"""
scripts/validate_multimodal.py
------------------------------
Compares Baseline (Visual-Only BiLSTM) vs Multimodal (Visual + Whisper BiLSTM)
on all 10 held-out TVSum validation videos across:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Spearman's Rank Correlation (Rho)
- Kendall's Rank Correlation (Tau)
"""

import os
import sys
import json
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.lstm_model import VideoHighlightBiLSTM
from src.multimodal_model import MultimodalHighlightBiLSTM
from src.dataset import get_normalized_ground_truth

VAL_ANNOTATIONS = os.path.join(BASE_DIR, "annotations", "tvsum_val_release.jsonl")
VISUAL_FEATS_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
TEXT_FEATS_DIR = os.path.join(BASE_DIR, "data", "processed", "text_features")
BASELINE_CKPT = os.path.join(BASE_DIR, "models", "best_bilstm.pt")
MULTIMODAL_CKPT = os.path.join(BASE_DIR, "models", "best_multimodal_bilstm.pt")


def evaluate_models():
    print("=" * 75)
    print("COMPARATIVE EVALUATION: BASELINE (VISUAL) VS MULTIMODAL (VISUAL + TEXT)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Baseline Model
    if not os.path.isfile(BASELINE_CKPT):
        raise FileNotFoundError(f"Baseline checkpoint missing: {BASELINE_CKPT}")
    base_data = torch.load(BASELINE_CKPT, map_location=device)
    base_cfg = base_data.get("config", {"input_dim": 2048, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3})
    baseline_model = VideoHighlightBiLSTM(
        input_dim=base_cfg["input_dim"],
        hidden_dim=base_cfg["hidden_dim"],
        num_layers=base_cfg["num_layers"],
        dropout=base_cfg["dropout"]
    ).to(device)
    baseline_model.load_state_dict(base_data["model_state_dict"])
    baseline_model.eval()

    # 2. Load Multimodal Model
    if not os.path.isfile(MULTIMODAL_CKPT):
        print(f"Multimodal checkpoint not found: {MULTIMODAL_CKPT}. Please run scripts/train_multimodal.py first.")
        return

    multi_data = torch.load(MULTIMODAL_CKPT, map_location=device)
    multi_cfg = multi_data.get("config", {
        "visual_dim": 2048, "text_dim": 384, "proj_visual_dim": 512, "proj_text_dim": 256,
        "hidden_dim": 256, "num_layers": 2, "dropout": 0.3
    })
    multimodal_model = MultimodalHighlightBiLSTM(
        visual_dim=multi_cfg["visual_dim"],
        text_dim=multi_cfg["text_dim"],
        proj_visual_dim=multi_cfg["proj_visual_dim"],
        proj_text_dim=multi_cfg["proj_text_dim"],
        hidden_dim=multi_cfg["hidden_dim"],
        num_layers=multi_cfg["num_layers"],
        dropout=multi_cfg["dropout"]
    ).to(device)
    multimodal_model.load_state_dict(multi_data["model_state_dict"])
    multimodal_model.eval()

    # 3. Read 10 validation videos
    val_records = []
    with open(VAL_ANNOTATIONS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    results = []

    print(f"\nEvaluating on {len(val_records)} validation videos...\n")
    print(f"{'Video ID':<13} | {'Segments':<8} | {'Baseline Rho':<12} | {'Multi Rho':<12} | {'Base Tau':<10} | {'Multi Tau':<10} | {'Winner':<8}")
    print("-" * 85)

    base_rhos, multi_rhos = [], []
    base_taus, multi_taus = [], []
    base_mses, multi_mses = [], []
    base_maes, multi_maes = [], []

    for rec in val_records:
        vid = rec["vid"]
        v_file = os.path.join(VISUAL_FEATS_DIR, f"{vid}.pt")
        t_file = os.path.join(TEXT_FEATS_DIR, f"{vid}.pt")

        v_feat = torch.load(v_file).float().to(device)
        t_feat = torch.load(t_file).float().to(device)
        gt = get_normalized_ground_truth(rec["label"]).numpy()

        T = min(v_feat.shape[0], t_feat.shape[0], len(gt))
        v_feat = v_feat[:T]
        t_feat = t_feat[:T]
        gt = gt[:T]

        with torch.no_grad():
            base_pred = baseline_model(v_feat.unsqueeze(0)).squeeze(0).cpu().numpy()
            multi_pred = multimodal_model(v_feat.unsqueeze(0), t_feat.unsqueeze(0)).squeeze(0).cpu().numpy()

        # Metrics
        b_mse = float(np.mean((base_pred - gt) ** 2))
        m_mse = float(np.mean((multi_pred - gt) ** 2))
        b_mae = float(np.mean(np.abs(base_pred - gt)))
        m_mae = float(np.mean(np.abs(multi_pred - gt)))

        b_rho, _ = spearmanr(base_pred, gt)
        m_rho, _ = spearmanr(multi_pred, gt)
        b_tau, _ = kendalltau(base_pred, gt)
        m_tau, _ = kendalltau(multi_pred, gt)

        b_rho = 0.0 if np.isnan(b_rho) else float(b_rho)
        m_rho = 0.0 if np.isnan(m_rho) else float(m_rho)
        b_tau = 0.0 if np.isnan(b_tau) else float(b_tau)
        m_tau = 0.0 if np.isnan(m_tau) else float(m_tau)

        base_mses.append(b_mse)
        multi_mses.append(m_mse)
        base_maes.append(b_mae)
        multi_maes.append(m_mae)
        base_rhos.append(b_rho)
        multi_rhos.append(m_rho)
        base_taus.append(b_tau)
        multi_taus.append(m_tau)

        winner = "Multi" if m_rho > b_rho else "Base"
        print(f"{vid:<13} | {T:<8} | {b_rho:12.4f} | {m_rho:12.4f} | {b_tau:10.4f} | {m_tau:10.4f} | {winner:<8}")

    print("-" * 85)
    print(f"{'OVERALL MEAN':<13} | {'--':<8} | {np.mean(base_rhos):12.4f} | {np.mean(multi_rhos):12.4f} | {np.mean(base_taus):10.4f} | {np.mean(multi_taus):10.4f} | {'--':<8}")
    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON TABLE:")
    print("=" * 75)
    print(f"Metric          | Baseline (Visual-Only) | Multimodal (Visual + Speech) | Delta")
    print("-" * 75)
    print(f"Spearman Rho    | {np.mean(base_rhos):22.4f} | {np.mean(multi_rhos):28.4f} | {np.mean(multi_rhos)-np.mean(base_rhos):+.4f}")
    print(f"Kendall Tau     | {np.mean(base_taus):22.4f} | {np.mean(multi_taus):28.4f} | {np.mean(multi_taus)-np.mean(base_taus):+.4f}")
    print(f"MSE             | {np.mean(base_mses):22.5f} | {np.mean(multi_mses):28.5f} | {np.mean(multi_mses)-np.mean(base_mses):+.5f}")
    print(f"MAE             | {np.mean(base_maes):22.5f} | {np.mean(multi_maes):28.5f} | {np.mean(multi_maes)-np.mean(base_maes):+.5f}")
    print(f"RMSE            | {np.sqrt(np.mean(base_mses)):22.5f} | {np.sqrt(np.mean(multi_mses)):28.5f} | {np.sqrt(np.mean(multi_mses))-np.sqrt(np.mean(base_mses)):+.5f}")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_models()
