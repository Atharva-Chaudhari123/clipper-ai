"""
main.py
-------
Unified Pipeline CLI & Library for Automatic Video Highlight Detection.
Supports:
1. Local MP4 videos OR YouTube URLs (via yt-dlp)
2. Baseline Visual-Only BiLSTM OR Multimodal (ResNet-50 + Whisper Speech) BiLSTM
3. 0.5 FPS segment-center frame sampling + FFmpeg stream rendering
4. Rich metadata return with transcript excerpts and score plot generation
"""

import os
import sys
import uuid
import argparse

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from src.feature_extractor import ResNet50FeatureExtractor
from src.lstm_model import VideoHighlightBiLSTM
from src.multimodal_model import MultimodalHighlightBiLSTM
from src.selector import select_highlight_segments
from src.renderer import render_highlight_video
from src.extract_frames import find_annotation_record
from src.dataset import get_normalized_ground_truth
from src.youtube_downloader import download_youtube_video, is_valid_youtube_url
from src.audio_transcriber import extract_audio_from_video, transcribe_audio
from src.text_embedder import align_transcript_with_segments, get_sentence_embedder


def run_pipeline(
    video_path: str = None,
    youtube_url: str = None,
    model_mode: str = "baseline",  # "baseline" or "multimodal"
    model_checkpoint: str = None,
    budget_ratio: float = 0.15,
    output_video_path: str = None,
    plot_output_path: str = None,
    annotations_dir: str = "annotations",
    whisper_model_name: str = "base",
    stage_callback: callable = None,
    job_id: str = None
):
    print("=" * 65)
    print("AI AUTOMATIC VIDEO HIGHLIGHT GENERATOR")
    print(f"Mode: {model_mode.upper()}")
    print("=" * 65)

    youtube_metadata = None

    # Handle YouTube URL input
    if youtube_url:
        if not is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        if stage_callback:
            stage_callback("Validating and downloading YouTube video...")

        job_tag = job_id or f"yt_{uuid.uuid4().hex[:8]}"
        yt_dir = os.path.join("data", "youtube_downloads", job_tag)
        yt_info = download_youtube_video(youtube_url, output_dir=yt_dir)
        video_path = yt_info["video_path"]
        youtube_metadata = yt_info
        print(f"  -> Acquired YouTube video: '{yt_info['title']}' ({yt_info['duration']}s)")

    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]

    if output_video_path is None:
        output_video_path = os.path.join("outputs", "highlights", f"{video_id}_highlight.mp4")
    if plot_output_path is None:
        plot_output_path = os.path.join("outputs", "plots", f"{video_id}_scores.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(plot_output_path)), exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Probe video properties
    # -------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / fps if fps > 0 else 0.0
    cap.release()

    num_segments = max(int(video_dur // 2.0), 1)
    print(f"Video:             {video_filename}")
    print(f"Duration:          {video_dur:.2f}s ({fps:.2f} FPS, {total_frames} total frames)")
    print(f"Segments (2s):     {num_segments}")
    print(f"Target Highlight:  {budget_ratio * 100:.0f}% (~{video_dur * budget_ratio:.1f}s)")

    # -------------------------------------------------------------
    # Step 2: 0.5 FPS Segment-Center Frame Sampling
    # -------------------------------------------------------------
    if stage_callback:
        stage_callback("Extracting video segments at 0.5 FPS...")

    print("\n[1/4] Extracting segment-center frames (0.5 FPS)...")
    cap = cv2.VideoCapture(video_path)
    temp_frames_dir = os.path.join("data", "processed", "frames", video_id)
    os.makedirs(temp_frames_dir, exist_ok=True)
    frame_paths = []

    for i in range(num_segments):
        sample_time = 2.0 * i + 1.0
        target_frame_idx = min(int(round(sample_time * fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            f_name = f"segment_{i:04d}.jpg"
            f_path = os.path.join(temp_frames_dir, f_name)
            if not os.path.isfile(f_path):
                cv2.imwrite(f_path, frame)
            frame_paths.append(f_path)
    cap.release()
    print(f"  -> {len(frame_paths)} representative frames ready.")

    # -------------------------------------------------------------
    # Step 3: ResNet-50 Visual Feature Extraction
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if stage_callback:
        stage_callback("Extracting visual features with ResNet-50...")

    print(f"\n[2/4] Extracting 2048-d visual features with frozen ResNet-50 ({device})...")
    cached_feature_file = os.path.join("data", "processed", "features", f"{video_id}.pt")
    if os.path.isfile(cached_feature_file):
        print(f"  -> Found cached visual features: {cached_feature_file}")
        features = torch.load(cached_feature_file)
    else:
        extractor = ResNet50FeatureExtractor(device=device)
        features, _ = extractor.extract_features(frame_paths, initial_batch_size=8)
        os.makedirs(os.path.dirname(cached_feature_file), exist_ok=True)
        torch.save(features, cached_feature_file)
        print(f"  -> Saved visual features: {cached_feature_file}")

    print(f"  -> Feature matrix shape: {list(features.shape)}")

    # -------------------------------------------------------------
    # Step 4: Audio Extraction & Speech Transcription (if Multimodal)
    # -------------------------------------------------------------
    has_speech = False
    speech_lang = "none"
    transcript_data = {"segments": [], "full_text": ""}
    text_embeddings = None
    active_mode = model_mode

    if model_mode == "multimodal":
        multi_ckpt = model_checkpoint or os.path.join("models", "best_multimodal_bilstm.pt")
        if not os.path.isfile(multi_ckpt):
            print(f"  -> Notice: Multimodal checkpoint '{multi_ckpt}' not found. Falling back to baseline visual model.")
            active_mode = "baseline (visual fallback)"
        else:
            active_mode = "multimodal"

        if stage_callback:
            stage_callback("Extracting audio and transcribing speech (Whisper)...")

        temp_audio_dir = os.path.join("data", "processed", "temp_audio")
        os.makedirs(temp_audio_dir, exist_ok=True)
        temp_wav = os.path.join(temp_audio_dir, f"{video_id}_{uuid.uuid4().hex[:6]}.wav")

        print(f"\n[3/4] Extracting audio and transcribing speech with Whisper ({whisper_model_name})...")
        has_audio = extract_audio_from_video(video_path, temp_wav)

        if has_audio:
            transcript_data = transcribe_audio(temp_wav, model_size=whisper_model_name, device=str(device))
            has_speech = transcript_data.get("has_speech", False)
            speech_lang = transcript_data.get("language", "en")
            words_count = len(transcript_data.get("full_text", "").split())
            print(f"  -> Speech detected: {has_speech} ({words_count} words, lang={speech_lang})")
            if os.path.isfile(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
        else:
            print("  -> Video contains no audio stream. Using neutral text embeddings.")

        if stage_callback:
            stage_callback("Generating 384-D text embeddings and multimodal fusion...")

        aligned_texts = align_transcript_with_segments(
            transcript_data.get("segments", []),
            num_segments=len(features),
            segment_duration=2.0
        )
        embedder = get_sentence_embedder()
        text_embeddings = embedder.embed_texts(aligned_texts, batch_size=32)
        print(f"  -> Generated text embeddings shape: {list(text_embeddings.shape)}")

    # -------------------------------------------------------------
    # Step 5: Model Inference & Importance Scoring
    # -------------------------------------------------------------
    if active_mode == "multimodal":
        if stage_callback:
            stage_callback("Predicting multimodal importance scores (BiLSTM)...")
        print(f"\n[3/4] Predicting multimodal importance scores with Bi-LSTM...")
        multi_ckpt = model_checkpoint or os.path.join("models", "best_multimodal_bilstm.pt")
        checkpoint = torch.load(multi_ckpt, map_location=device)
        cfg = checkpoint.get("config", {
            "visual_dim": 2048, "text_dim": 384, "proj_visual_dim": 512, "proj_text_dim": 256,
            "hidden_dim": 256, "num_layers": 2, "dropout": 0.3
        })
        model = MultimodalHighlightBiLSTM(
            visual_dim=cfg["visual_dim"],
            text_dim=cfg["text_dim"],
            proj_visual_dim=cfg["proj_visual_dim"],
            proj_text_dim=cfg["proj_text_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            v_in = features.unsqueeze(0).to(device)
            t_in = text_embeddings.unsqueeze(0).to(device)
            preds = model(v_in, t_in).squeeze(0).cpu().numpy()

    else:
        if stage_callback:
            stage_callback("Predicting visual importance scores (BiLSTM)...")
        print(f"\n[3/4] Predicting visual importance scores with baseline Bi-LSTM...")
        base_ckpt = model_checkpoint or os.path.join("models", "best_bilstm.pt")
        checkpoint = torch.load(base_ckpt, map_location=device)
        cfg = checkpoint.get("config", {"input_dim": 2048, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3})
        model = VideoHighlightBiLSTM(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            x = features.unsqueeze(0).to(device)
            preds = model(x).squeeze(0).cpu().numpy()

    print(f"  -> Predicted {len(preds)} segment scores. Range: [{preds.min():.4f}, {preds.max():.4f}], Mean: {preds.mean():.4f}")

    # -------------------------------------------------------------
    # Step 6: Highlight Selection & Video Rendering
    # -------------------------------------------------------------
    if stage_callback:
        stage_callback("Selecting highlight clips and rendering video (FFmpeg)...")

    print(f"\n[4/4] Selecting highlight segments and rendering video...")
    intervals = select_highlight_segments(preds, budget_ratio=budget_ratio, min_segment_duration=4.0)
    print("  -> Selected highlight intervals:")
    total_highlight_sec = 0.0
    for idx, (s, e) in enumerate(intervals, 1):
        dur = e - s
        total_highlight_sec += dur
        print(f"     Clip {idx}: {s:6.1f}s -> {e:6.1f}s  ({dur:4.1f}s)")

    print(f"  -> Rendering final highlight video with FFmpeg...")
    render_stats = render_highlight_video(
        source_video_path=video_path,
        intervals=intervals,
        output_path=output_video_path
    )
    print(f"  -> SUCCESS! Highlight video saved to: {output_video_path}")
    print(f"     Rendered Duration: {render_stats['duration_seconds']:.1f}s, Size: {render_stats['file_size_mb']} MB")

    # -------------------------------------------------------------
    # Step 7: Plot Importance Scores
    # -------------------------------------------------------------
    ground_truth = None
    try:
        annot_rec = find_annotation_record(video_id, annotations_dir)
        ground_truth = get_normalized_ground_truth(annot_rec["label"]).numpy()
    except Exception:
        pass

    plt.figure(figsize=(12, 5))
    time_axis = np.arange(len(preds)) * 2.0  # 2 seconds per segment

    plt.plot(time_axis, preds, color="#007acc", linewidth=2.0, label=f"Predicted Importance ({active_mode.capitalize()})")

    if ground_truth is not None:
        gt_len = min(len(time_axis), len(ground_truth))
        plt.plot(time_axis[:gt_len], ground_truth[:gt_len], color="#ff8800", linewidth=1.5, linestyle="--", alpha=0.8, label="Human Ground Truth (TVSum)")

    for s, e in intervals:
        plt.axvspan(s, e, color="#28a745", alpha=0.25, label="Selected Highlight" if s == intervals[0][0] else None)

    title_name = youtube_metadata["title"] if youtube_metadata else video_filename
    plt.title(f"Video Highlight Importance Scoring: {title_name[:50]}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Importance Score [0.0, 1.0]", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150)
    plt.close()

    def format_ts(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    formatted_clips = []
    for idx, (s, e) in enumerate(intervals, 1):
        # Extract overlapping transcript excerpt for this clip
        clip_speech = []
        for seg in transcript_data.get("segments", []):
            if max(s, seg["start"]) < min(e, seg["end"]):
                t = seg["text"].strip()
                if t and t not in clip_speech:
                    clip_speech.append(t)
        clip_excerpt = " ".join(clip_speech).strip()

        formatted_clips.append({
            "clip_num": idx,
            "start_sec": round(s, 1),
            "end_sec": round(e, 1),
            "duration_sec": round(e - s, 1),
            "start_formatted": format_ts(s),
            "end_formatted": format_ts(e),
            "transcript": clip_excerpt
        })

    results = {
        "output_mode": "single",
        "video_filename": video_filename,
        "original_duration": round(video_dur, 1),
        "highlight_duration": round(render_stats["duration_seconds"], 1),
        "budget_ratio": budget_ratio,
        "num_clips": len(intervals),
        "clips": formatted_clips,
        "output_video_path": output_video_path,
        "plot_output_path": plot_output_path,
        "file_size_mb": render_stats["file_size_mb"],
        "model_mode": active_mode,
        "has_speech": has_speech,
        "speech_language": speech_lang,
        "youtube_meta": youtube_metadata,
        "full_transcript": transcript_data.get("full_text", "")
    }

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 65)
    return results


def run_multi_reel_pipeline(
    video_path: str = None,
    youtube_url: str = None,
    model_mode: str = "multimodal",
    num_reels: str = "auto",
    reel_length: str = "30-60",
    aspect_ratio: str = "9:16",
    captions: bool = True,
    use_face_tracking: bool = True,
    gemini_api_key: str = None,
    output_dir: str = None,
    whisper_model_name: str = "base",
    stage_callback: callable = None,
    job_id: str = None
):
    """
    Phase 13: High-throughput Two-Stage Multi-Reel Pipeline for Long Videos.
    Stages:
    1. Getting YouTube video
    2. Fast video analysis (coarse ResNet-50 visual screening)
    3. Detecting speech activity (fast RMS audio activity)
    4. Finding candidate moments (peak window generation & overlap merging)
    5. Transcribing selected moments (faster-whisper on candidates ONLY)
    6. Understanding candidate speech (MiniLM text embeddings)
    7. Ranking important moments (Multimodal BiLSTM scoring)
    8. Creating multiple reels (NMS, sentence-snapping, quality filtering)
    9. Converting to 9:16 (vertical center cropping)
    10. Adding captions (burned-in styled subtitles)
    11. Finalizing reels (manifest.json)
    """
    print("=" * 65)
    print("AI MULTI-REEL GENERATION PIPELINE")
    print(f"Mode: {model_mode.upper()} | Reels: {num_reels} | Length: {reel_length} | Aspect: {aspect_ratio}")
    print("=" * 65)

    youtube_metadata = None

    # Stage 1: Getting YouTube video
    if youtube_url:
        if not is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")

        if stage_callback:
            stage_callback("Getting YouTube video...")

        job_tag = job_id or f"yt_{uuid.uuid4().hex[:8]}"
        yt_dir = os.path.join("data", "youtube_downloads", job_tag)
        yt_info = download_youtube_video(youtube_url, output_dir=yt_dir)
        video_path = yt_info["video_path"]
        youtube_metadata = yt_info
        print(f"  -> Acquired YouTube video: '{yt_info['title']}' ({yt_info['duration']}s)")

    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    video_filename = os.path.basename(video_path)
    video_id = os.path.splitext(video_filename)[0]
    job_tag = job_id or f"reel_{uuid.uuid4().hex[:8]}"

    if output_dir is None:
        output_dir = os.path.join("web_outputs", job_tag)
    os.makedirs(output_dir, exist_ok=True)

    # Stage 2: Fast video analysis
    if stage_callback:
        stage_callback("Fast video analysis...")
    print("\n[Stage 2] Fast visual pre-screening with ResNet-50...")
    from src.fast_visual_screener import FastVisualScreener
    screener = FastVisualScreener(baseline_model_path="models/best_bilstm.pt")
    vis_res = screener.screen_video(video_path, sample_interval_sec=2.0, batch_size=16)
    video_dur = vis_res["duration_sec"]
    vis_scores = vis_res["visual_scores"]
    vis_features = vis_res["visual_features"]
    seg_times = vis_res["segment_times"]
    print(f"  -> Video duration: {video_dur:.1f}s | Visual segments: {len(seg_times)}")

    # Stage 3: Detecting speech activity
    if stage_callback:
        stage_callback("Detecting speech activity...")
    print("\n[Stage 3] Fast speech activity detection (RMS energy)...")
    from src.audio_pre_screener import extract_audio_fast, detect_speech_intervals
    temp_wav_path = os.path.join("data", "youtube_downloads", job_tag, "audio_16k.wav")
    has_audio = extract_audio_fast(video_path, temp_wav_path)

    speech_intervals = []
    has_speech = False
    if has_audio:
        speech_info = detect_speech_intervals(temp_wav_path)
        speech_intervals = speech_info["speech_intervals"]
        has_speech = speech_info["has_speech"]
        print(f"  -> Speech detected: {has_speech} ({len(speech_intervals)} intervals, {speech_info['speech_duration']}s)")
    else:
        print("  -> No audio stream detected. Proceeding with visual-only analysis.")

    # Stage 4: Finding candidate moments
    if stage_callback:
        stage_callback("Finding candidate moments...")
    print("\n[Stage 4] Generating candidate highlight windows...")
    from src.candidate_generator import generate_candidate_windows
    candidates = generate_candidate_windows(
        duration_sec=video_dur,
        visual_scores=vis_scores,
        segment_times=seg_times,
        speech_intervals=speech_intervals if has_speech else None,
        target_window_duration=40.0
    )
    print(f"  -> Generated {len(candidates)} candidate windows.")

    # Stage 5: Transcribing selected moments
    if stage_callback:
        stage_callback("Transcribing selected moments...")
    print("\n[Stage 5] Transcribing candidate windows with faster-whisper...")
    if has_speech and os.path.isfile(temp_wav_path) and model_mode == "multimodal":
        from src.fast_transcriber import transcribe_candidate_windows
        candidates = transcribe_candidate_windows(
            wav_path=temp_wav_path,
            candidate_windows=candidates,
            model_size=whisper_model_name,
            stage_callback=stage_callback
        )
    else:
        for cand in candidates:
            cand["segments"] = []
            cand["full_text"] = ""
            cand["has_speech"] = False

    # Stage 6: Understanding candidate speech & Stage 7: Ranking important moments & Stage 8: Creating multiple reels
    if stage_callback:
        stage_callback("Understanding candidate speech...")
    print("\n[Stage 6 & 7 & 8] Scoring candidates with Multimodal BiLSTM and creating reels...")
    from src.reel_selector import select_multiple_reels
    selected_reels = select_multiple_reels(
        candidates=candidates,
        all_visual_features=vis_features,
        video_duration=video_dur,
        target_reel_length=reel_length,
        num_reels_option=num_reels,
        quality_threshold=0.20,
        multimodal_model_path="models/best_multimodal_bilstm.pt",
        visual_sample_interval=2.0,
        gemini_api_key=gemini_api_key
    )
    print(f"  -> Selected {len(selected_reels)} diverse, high-scoring Reels.")

    # Stage 9: Converting to 9:16 & Stage 10: Adding captions & Stage 11: Finalizing reels
    if stage_callback:
        stage_callback("Converting to 9:16...")
    print("\n[Stage 9, 10 & 11] Rendering Reels in 9:16 with captions...")
    from src.reel_renderer import render_multiple_reels
    manifest = render_multiple_reels(
        source_video_path=video_path,
        selected_reels=selected_reels,
        output_dir=output_dir,
        aspect_ratio=aspect_ratio,
        captions=captions,
        use_face_tracking=use_face_tracking,
        stage_callback=stage_callback
    )

    # Clean up temporary WAV file
    if os.path.isfile(temp_wav_path):
        try:
            os.remove(temp_wav_path)
        except OSError:
            pass

    results = {
        "output_mode": "reels",
        "video_filename": video_filename,
        "original_duration": round(video_dur, 1),
        "num_reels": len(manifest["reels"]),
        "reels": manifest["reels"],
        "manifest_path": os.path.join(output_dir, "manifest.json"),
        "aspect_ratio": aspect_ratio,
        "captions": captions,
        "has_speech": has_speech,
        "model_mode": model_mode,
        "youtube_meta": youtube_metadata,
        "job_id": job_tag
    }

    print("\n" + "=" * 65)
    print(f"MULTI-REEL PIPELINE COMPLETED: {len(manifest['reels'])} REELS GENERATED")
    print("=" * 65)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Automatic Video Highlight Detection Pipeline")
    parser.add_argument("--video", type=str, default="video/akI8YFjEmUw.mp4", help="Path to input video")
    parser.add_argument("--youtube", type=str, default=None, help="YouTube video URL")
    parser.add_argument("--output-mode", type=str, default="reels", choices=["single", "reels"], help="Output mode")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["baseline", "multimodal"], help="Analysis mode")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--budget", type=float, default=0.15, help="Highlight duration budget ratio for single mode")
    parser.add_argument("--num-reels", type=str, default="auto", choices=["auto", "3", "5", "10"], help="Number of reels")
    parser.add_argument("--reel-length", type=str, default="30-60", choices=["15-30", "30-60", "60-90", "auto"], help="Target reel length")
    parser.add_argument("--aspect-ratio", type=str, default="9:16", choices=["9:16", "16:9"], help="Output aspect ratio")
    parser.add_argument("--no-captions", action="store_true", help="Disable burned-in captions")
    parser.add_argument("--output", type=str, default=None, help="Output path / directory")
    parser.add_argument("--plot", type=str, default=None, help="Output score plot path")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    args = parser.parse_args()

    if args.output_mode == "reels":
        run_multi_reel_pipeline(
            video_path=args.video if not args.youtube else None,
            youtube_url=args.youtube,
            model_mode=args.mode,
            num_reels=args.num_reels,
            reel_length=args.reel_length,
            aspect_ratio=args.aspect_ratio,
            captions=not args.no_captions,
            output_dir=args.output,
            whisper_model_name=args.whisper_model
        )
    else:
        run_pipeline(
            video_path=args.video if not args.youtube else None,
            youtube_url=args.youtube,
            model_mode=args.mode,
            model_checkpoint=args.model,
            budget_ratio=args.budget,
            output_video_path=args.output,
            plot_output_path=args.plot,
            whisper_model_name=args.whisper_model
        )

