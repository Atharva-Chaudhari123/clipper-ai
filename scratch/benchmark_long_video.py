"""
scratch/benchmark_long_video.py
-------------------------------
Benchmarking Phase 13 Multi-Reel Pipeline on the real 22-minute video:
"GPT 6 Astra - End of Developers?" (1314.0s / 21.9 min)
Measures individual timing of every stage and compares against the Old Pipeline.
"""

import os
import sys
import time
import json
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.fast_visual_screener import FastVisualScreener
from src.audio_pre_screener import extract_audio_fast, detect_speech_intervals
from src.candidate_generator import generate_candidate_windows
from src.fast_transcriber import transcribe_candidate_windows
from src.reel_selector import select_multiple_reels
from src.reel_renderer import render_multiple_reels

video_path = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\youtube_downloads\job_1789122850_2a2e77\Pa2G8RzbWZ4.mp4"
output_dir = os.path.join(BASE_DIR, "web_outputs", "benchmark_long_video")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("BENCHMARKING PHASE 13 TWO-STAGE MULTI-REEL PIPELINE")
print("Target Video: GPT 6 Astra - End of Developers?")
print(f"Path: {video_path}")
print("=" * 70)

t_start_total = time.time()

# 1. Visual Pre-Screening
print("\n[1/7] Running Fast Visual Pre-Screening (ResNet-50 + BiLSTM)...")
t0 = time.time()
screener = FastVisualScreener(baseline_model_path="models/best_bilstm.pt")
vis_res = screener.screen_video(video_path, sample_interval_sec=2.0, batch_size=32)
t_visual = time.time() - t0
video_dur = vis_res["duration_sec"]
vis_scores = vis_res["visual_scores"]
vis_feats = vis_res["visual_features"]
seg_times = vis_res["segment_times"]
print(f"  -> Visual screening completed in {t_visual:.2f}s ({len(seg_times)} segments, video: {video_dur:.1f}s)")

# 2. Audio Extraction & Fast VAD
print("\n[2/7] Running Fast Audio Extraction & RMS VAD Pre-Screening...")
t0 = time.time()
wav_path = os.path.join(output_dir, "audio_16k.wav")
has_audio = extract_audio_fast(video_path, wav_path)
t_extract = time.time() - t0

t0 = time.time()
speech_info = detect_speech_intervals(wav_path)
t_vad = time.time() - t0
speech_intervals = speech_info["speech_intervals"]
print(f"  -> Audio extraction: {t_extract:.2f}s | Audio VAD: {t_vad:.2f}s")
print(f"  -> Detected {len(speech_intervals)} speech intervals ({speech_info['speech_duration']:.1f}s speech)")

# 3. Candidate Window Generation
print("\n[3/7] Generating Candidate Windows (Overlap Merging & Duration Bounding)...")
t0 = time.time()
candidates = generate_candidate_windows(
    duration_sec=video_dur,
    visual_scores=vis_scores,
    segment_times=seg_times,
    speech_intervals=speech_intervals,
    target_window_duration=40.0,
    max_candidates=15
)
t_candidates = time.time() - t0
cand_audio_duration = sum(c["duration"] for c in candidates)
print(f"  -> Generated {len(candidates)} candidate windows in {t_candidates:.2f}s")
print(f"  -> Total Candidate Audio Duration to Transcribe: {cand_audio_duration:.1f}s (vs original {video_dur:.1f}s)")
print(f"  -> Audio Processing Reduction: {(1.0 - cand_audio_duration / video_dur) * 100:.1f}%")

# 4. Candidate-Only faster-whisper Transcription
print("\n[4/7] Running faster-whisper on Candidates ONLY...")
t0 = time.time()
transcribed_candidates = transcribe_candidate_windows(
    wav_path=wav_path,
    candidate_windows=candidates,
    model_size="base"
)
t_whisper = time.time() - t0
total_words = sum(len(c.get("full_text", "").split()) for c in transcribed_candidates)
print(f"  -> Candidate faster-whisper completed in {t_whisper:.2f}s ({total_words} words transcribed)")

# 5. Multimodal Scoring & Multi-Reel Selection
print("\n[5/7] Running Multimodal BiLSTM Scoring & Sentence-Boundary Snapping...")
t0 = time.time()
selected_reels = select_multiple_reels(
    candidates=transcribed_candidates,
    all_visual_features=vis_feats,
    video_duration=video_dur,
    target_reel_length="30-60",
    num_reels_option="auto",
    quality_threshold=0.20,
    multimodal_model_path="models/best_multimodal_bilstm.pt"
)
t_scoring_selection = time.time() - t0
print(f"  -> Selected {len(selected_reels)} Reels in {t_scoring_selection:.2f}s")

# 6. 9:16 Vertical Video Rendering with Captions
print("\n[6/7] Rendering 9:16 Vertical Reels with Burned-In Captions...")
t0 = time.time()
manifest = render_multiple_reels(
    source_video_path=video_path,
    selected_reels=selected_reels,
    output_dir=output_dir,
    aspect_ratio="9:16",
    captions=True
)
t_render = time.time() - t0
print(f"  -> Rendered {len(manifest['reels'])} 9:16 Reels in {t_render:.2f}s")

# Clean up temporary WAV
if os.path.isfile(wav_path):
    os.remove(wav_path)

t_total = time.time() - t_start_total

vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

print("\n" + "=" * 70)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"Original Video Duration:     {video_dur:.1f}s ({video_dur/60:.1f} min)")
print(f"Audio Duration to Whisper:   {cand_audio_duration:.1f}s ({cand_audio_duration/60:.1f} min)")
print(f"Whisper Audio Reduction:     {(1.0 - cand_audio_duration / video_dur) * 100:.1f}%")
print(f"Visual Screening Time:       {t_visual:.2f}s")
print(f"Audio Extraction + VAD Time: {t_extract + t_vad:.2f}s")
print(f"Candidate Gen Time:          {t_candidates:.2f}s")
print(f"faster-whisper Time:         {t_whisper:.2f}s ({t_whisper/60:.2f} min)")
print(f"Scoring & Selection Time:    {t_scoring_selection:.2f}s")
print(f"FFmpeg 9:16 Render Time:     {t_render:.2f}s ({t_render/60:.2f} min)")
print(f"Total Pipeline Time:         {t_total:.2f}s ({t_total/60:.2f} min)")
print(f"Number of Reels Generated:   {len(manifest['reels'])}")
print(f"GPU Device:                  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA Used:                   {torch.cuda.is_available()}")
print(f"VRAM Peak:                   {vram_peak_mb:.1f} MB")

# Print Reel Details
print("\n--- GENERATED REELS ---")
for r in manifest["reels"]:
    print(f"Reel {r['reel_num']}: {r['start_formatted']} -> {r['end_formatted']} ({r['duration_sec']}s) | Score: {r['score']} | File: {r['filename']} ({r['file_size_mb']} MB)")
    print(f"  Snippet: \"{r['transcript'][:80]}...\"")

