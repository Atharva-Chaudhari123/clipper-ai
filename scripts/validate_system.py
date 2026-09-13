"""
scripts/validate_system.py
--------------------------
Comprehensive ML Validation Script for TVSum + ResNet-50 + BiLSTM Pipeline.

Executes:
Step 1: Checkpoint verification (models/best_bilstm.pt, params == 6,365,441, CUDA).
Step 2 & 3: Multi-video validation on all 10 TVSum validation videos (MSE, MAE, RMSE, Spearman rho, Kendall tau).
Step 4: Score plotting for at least 3 validation videos (outputs/plots/validation_<vid>_scores.png).
Step 5 & 6: Highlight selection (15% budget) and FFmpeg rendering (outputs/highlights/<vid>_highlight.mp4).
Step 7: Custom video inference and rendering on an external MP4 (C:/Users/mahin/Downloads/FS_demo.mp4).
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lstm_model import VideoHighlightBiLSTM
from src.dataset import get_normalized_ground_truth
from src.selector import select_highlight_segments
from src.renderer import render_highlight_video
from src.feature_extractor import ResNet50FeatureExtractor
import cv2


def run_complete_validation():
    print("=" * 70)
    print("FINAL MACHINE LEARNING VALIDATION PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 1: VERIFY CHECKPOINT
    # -------------------------------------------------------------
    checkpoint_path = r"C:\Users\mahin\OneDrive\Desktop\New folder\models\best_bilstm.pt"
    assert os.path.isfile(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg = checkpoint.get("config", {"input_dim": 2048, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3})
    model = VideoHighlightBiLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint loaded successfully: {checkpoint_path}")
    print(f"Checkpoint Epoch:       {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint Val Loss:    {checkpoint.get('val_loss', 'N/A'):.5f}")
    print(f"Total Parameters:       {total_params:,} (Expected: 6,365,441)")
    print(f"Execution Device:       {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    assert total_params == 6365441, f"Parameter mismatch! Expected 6,365,441, got {total_params}"

    # -------------------------------------------------------------
    # STEP 2 & 3: TEST ALL 10 TVSUM VALIDATION VIDEOS & METRICS
    # -------------------------------------------------------------
    val_jsonl = r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations\tvsum_val_release.jsonl"
    features_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features"
    video_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\video"

    val_records = []
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    print(f"\nEvaluating on all {len(val_records)} TVSum validation videos...")
    print(f"{'Video ID':<13} | {'Segments':<8} | {'MSE':<9} | {'MAE':<9} | {'RMSE':<9} | {'Spearman Rho':<13} | {'Kendall Tau'}")
    print("-" * 80)

    results_table = []
    all_mse = []
    all_mae = []
    all_rmse = []
    all_rho = []
    all_tau = []

    predictions_dict = {}

    for rec in val_records:
        vid = rec["vid"]
        feat_path = os.path.join(features_dir, f"{vid}.pt")
        feats = torch.load(feat_path).to(device)  # [T, 2048]
        ground_truth = get_normalized_ground_truth(rec["label"]).numpy()  # [T]

        with torch.no_grad():
            preds = model(feats.unsqueeze(0)).squeeze(0).cpu().numpy()  # [T]

        # Length alignment
        min_len = min(len(preds), len(ground_truth))
        p = preds[:min_len]
        gt = ground_truth[:min_len]

        predictions_dict[vid] = (p, gt, rec)

        mse = float(np.mean((p - gt) ** 2))
        mae = float(np.mean(np.abs(p - gt)))
        rmse = float(np.sqrt(mse))
        rho, _ = spearmanr(p, gt)
        tau, _ = kendalltau(p, gt)

        all_mse.append(mse)
        all_mae.append(mae)
        all_rmse.append(rmse)
        if not np.isnan(rho):
            all_rho.append(rho)
        if not np.isnan(tau):
            all_tau.append(tau)

        print(f"{vid:<13} | {min_len:<8} | {mse:<9.5f} | {mae:<9.5f} | {rmse:<9.5f} | {rho:<13.4f} | {tau:.4f}")
        results_table.append({
            "vid": vid,
            "segments": min_len,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "spearman": rho,
            "kendall": tau
        })

    overall_mse = float(np.mean(all_mse))
    overall_mae = float(np.mean(all_mae))
    overall_rmse = float(np.mean(all_rmse))
    overall_rho = float(np.mean(all_rho))
    overall_tau = float(np.mean(all_tau))

    print("-" * 80)
    print(f"{'OVERALL MEAN':<13} | {'--':<8} | {overall_mse:<9.5f} | {overall_mae:<9.5f} | {overall_rmse:<9.5f} | {overall_rho:<13.4f} | {overall_tau:.4f}")

    # -------------------------------------------------------------
    # STEP 4: VISUALIZATIONS FOR AT LEAST 3 VALIDATION VIDEOS
    # -------------------------------------------------------------
    plot_vids = ["sTEELN-vY30", "Se3oxnaPsz0", "JKpqYvAdIsw"]
    plots_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\outputs\plots"
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating importance score plots for validation videos...")
    for vid in plot_vids:
        p, gt, rec = predictions_dict[vid]
        intervals = select_highlight_segments(p, budget_ratio=0.15, min_segment_duration=4.0)

        plt.figure(figsize=(11, 4.5))
        time_axis = np.arange(len(p)) * 2.0  # 2s per segment

        plt.plot(time_axis, p, color="#007acc", linewidth=2.0, label="Predicted Importance (Bi-LSTM)")
        plt.plot(time_axis, gt, color="#ff8800", linewidth=1.5, linestyle="--", alpha=0.85, label="Human Ground Truth")

        for s, e in intervals:
            plt.axvspan(s, e, color="#28a745", alpha=0.25, label="Selected Highlight" if s == intervals[0][0] else None)

        query_title = rec.get("query", vid)
        plt.title(f"Validation Highlight Prediction: {vid} (\"{query_title}\")", fontsize=12, fontweight="bold")
        plt.xlabel("Time (seconds)", fontsize=10)
        plt.ylabel("Importance Score [0.0, 1.0]", fontsize=10)
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, f"validation_{vid}_scores.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  -> Saved plot: {plot_path}")

    # -------------------------------------------------------------
    # STEP 5 & 6: HIGHLIGHT SELECTION & VIDEO RENDERING (3 VIDEOS)
    # -------------------------------------------------------------
    highlights_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\outputs\highlights"
    os.makedirs(highlights_dir, exist_ok=True)

    print("\nRendering highlight videos for 3 validation videos (15% budget)...")
    render_summary = []

    for vid in plot_vids:
        p, gt, rec = predictions_dict[vid]
        src_mp4 = os.path.join(video_dir, f"{vid}.mp4")
        out_mp4 = os.path.join(highlights_dir, f"{vid}_highlight.mp4")

        # Select intervals with 15% budget
        intervals = select_highlight_segments(p, budget_ratio=0.15, min_segment_duration=4.0)

        total_video_dur = len(p) * 2.0
        selected_dur = sum(e - s for s, e in intervals)

        print(f"\n--- Video: {vid}.mp4 ---")
        print(f"  Video duration:    {total_video_dur:.1f}s")
        print(f"  Target 15% budget: {total_video_dur * 0.15:.1f}s")
        print(f"  Selected duration: {selected_dur:.1f}s")
        print(f"  Number of clips:   {len(intervals)}")
        for idx, (s, e) in enumerate(intervals, 1):
            print(f"    Clip {idx}: {s:6.1f}s -> {e:6.1f}s ({e-s:4.1f}s)")

        render_stats = render_highlight_video(src_mp4, intervals, out_mp4)
        print(f"  Rendered Output:   {out_mp4} ({render_stats['duration_seconds']:.1f}s, {render_stats['file_size_mb']} MB)")

        render_summary.append({
            "vid": vid,
            "duration": total_video_dur,
            "selected_duration": selected_dur,
            "clips": len(intervals),
            "output_path": out_mp4,
            "rendered_dur": render_stats["duration_seconds"],
            "size_mb": render_stats["file_size_mb"]
        })

    # -------------------------------------------------------------
    # STEP 7: CUSTOM VIDEO TEST (OUTSIDE TVSUM)
    # -------------------------------------------------------------
    custom_video_path = r"C:\Users\mahin\Downloads\FS_demo.mp4"
    print("\n" + "=" * 70)
    print("STEP 7: CUSTOM VIDEO INFERENCE TEST (NON-TVSUM)")
    print("=" * 70)
    print(f"Custom video path: {custom_video_path}")
    print("NOTE: This custom video has NO TVSum ground-truth annotations.")
    print("      Evaluation metrics (MSE, Spearman) cannot be calculated.")
    print("      Only inference and highlight video generation are demonstrated.")

    custom_out_mp4 = os.path.join(highlights_dir, "custom_video_highlight.mp4")
    custom_plot_png = os.path.join(plots_dir, "custom_video_scores.png")

    # 1. Probe custom video
    cap = cv2.VideoCapture(custom_video_path)
    c_fps = cap.get(cv2.CAP_PROP_FPS)
    c_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c_dur = c_frames / c_fps if c_fps > 0 else 0.0
    cap.release()
    c_segments = int(c_dur // 2.0)
    print(f"Custom video duration: {c_dur:.1f}s ({c_fps:.1f} FPS, {c_segments} 2-second segments)")

    # 2. Extract 0.5 FPS segment-center frames
    print("Extracting 0.5 FPS segment-center frames...")
    c_frames_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames\custom_video"
    os.makedirs(c_frames_dir, exist_ok=True)
    c_frame_paths = []
    cap = cv2.VideoCapture(custom_video_path)
    for i in range(c_segments):
        s_time = 2.0 * i + 1.0
        target_f = min(int(round(s_time * c_fps)), c_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if ret and frame is not None:
            fp = os.path.join(c_frames_dir, f"segment_{i:04d}.jpg")
            if not os.path.isfile(fp):
                cv2.imwrite(fp, frame)
            c_frame_paths.append(fp)
    cap.release()

    # 3. ResNet-50 feature extraction
    print(f"Extracting 2048-d visual features with frozen ResNet-50 on {device}...")
    extractor = ResNet50FeatureExtractor(device=device)
    c_features, _ = extractor.extract_features(c_frame_paths, initial_batch_size=8)
    print(f"Extracted feature matrix shape: {list(c_features.shape)}")

    # 4. BiLSTM prediction
    with torch.no_grad():
        c_preds = model(c_features.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    print(f"Predicted importance scores: min={c_preds.min():.4f}, max={c_preds.max():.4f}, mean={c_preds.mean():.4f}")

    # 5. Highlight selection (15% budget)
    c_intervals = select_highlight_segments(c_preds, budget_ratio=0.15, min_segment_duration=4.0)
    c_selected_dur = sum(e - s for s, e in c_intervals)
    print(f"Target 15% budget: {c_dur * 0.15:.1f}s, Selected duration: {c_selected_dur:.1f}s ({len(c_intervals)} clips)")
    for idx, (s, e) in enumerate(c_intervals, 1):
        print(f"  Clip {idx}: {s:6.1f}s -> {e:6.1f}s ({e-s:4.1f}s)")

    # 6. Render custom highlight
    print("Rendering custom highlight video with FFmpeg...")
    c_render_stats = render_highlight_video(custom_video_path, c_intervals, custom_out_mp4)
    print(f"Rendered custom highlight saved: {custom_out_mp4} ({c_render_stats['duration_seconds']:.1f}s, {c_render_stats['file_size_mb']} MB)")

    # 7. Plot custom predicted scores
    plt.figure(figsize=(11, 4.5))
    t_axis = np.arange(len(c_preds)) * 2.0
    plt.plot(t_axis, c_preds, color="#9b59b6", linewidth=2.0, label="Predicted Importance (Bi-LSTM)")
    for s, e in c_intervals:
        plt.axvspan(s, e, color="#2ecc71", alpha=0.25, label="Selected Highlight" if s == c_intervals[0][0] else None)
    plt.title("Custom Video Highlight Importance Scoring: FS_demo.mp4", fontsize=12, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Importance Score [0.0, 1.0]", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(custom_plot_png, dpi=150)
    plt.close()
    print(f"Saved custom score plot: {custom_plot_png}")

    print("\n" + "=" * 70)
    print("FINAL ML VALIDATION PIPELINE EXECUTION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_validation()
