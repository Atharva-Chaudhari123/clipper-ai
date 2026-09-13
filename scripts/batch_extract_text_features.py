"""
scripts/batch_extract_text_features.py
--------------------------------------
Batch extracts Whisper speech transcripts and 384-D sentence embeddings
for all TVSum dataset videos.

Outputs cached to:
- data/processed/transcripts/<video_id>.json
- data/processed/text_features/<video_id>.pt  [T, 384]
"""

import os
import sys
import json
import time
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.audio_transcriber import extract_audio_from_video, transcribe_audio
from src.text_embedder import align_transcript_with_segments, get_sentence_embedder

VIDEO_DIR = os.path.join(BASE_DIR, "video")
VISUAL_FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "data", "processed", "transcripts")
TEXT_FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "text_features")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "data", "processed", "temp_audio")

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(TEXT_FEATURES_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


def process_video_text(video_id: str, video_path: str, whisper_model: str = "base"):
    transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
    text_feat_file = os.path.join(TEXT_FEATURES_DIR, f"{video_id}.pt")
    visual_feat_file = os.path.join(VISUAL_FEATURES_DIR, f"{video_id}.pt")

    # Determine sequence length T from visual features
    if not os.path.isfile(visual_feat_file):
        print(f"[{video_id}] Warning: visual features not found at {visual_feat_file}. Skipping.")
        return False

    visual_feats = torch.load(visual_feat_file)
    num_segments = visual_feats.shape[0]  # T

    # Check if text features already cached
    if os.path.isfile(text_feat_file):
        existing = torch.load(text_feat_file)
        if existing.shape == (num_segments, 384):
            return True

    # Step 1: Transcribe if not already cached
    if os.path.isfile(transcript_file):
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
    else:
        temp_wav = os.path.join(TEMP_AUDIO_DIR, f"{video_id}.wav")
        has_audio = extract_audio_from_video(video_path, temp_wav)

        if has_audio:
            transcript_data = transcribe_audio(temp_wav, model_size=whisper_model)
        else:
            transcript_data = {
                "language": "none",
                "segments": [],
                "full_text": "",
                "has_speech": False
            }

        # Clean up temporary WAV
        if os.path.isfile(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2)

    # Step 2: Align transcript with 2-second segments
    aligned_texts = align_transcript_with_segments(
        transcript_data.get("segments", []),
        num_segments=num_segments,
        segment_duration=2.0
    )

    # Step 3: Generate 384-D sentence embeddings
    embedder = get_sentence_embedder()
    text_embeddings = embedder.embed_texts(aligned_texts, batch_size=32)

    # Step 4: Save text features
    torch.save(text_embeddings, text_feat_file)
    words_count = len(transcript_data.get("full_text", "").split())
    has_sp = transcript_data.get("has_speech", False)
    print(f"[{video_id}] Speech: {has_sp} ({words_count} words) | Text Feats: {list(text_embeddings.shape)}")
    return True


def main():
    print("=" * 65)
    print("BATCH EXTRACTING WHISPER TRANSCRIPTS & TEXT FEATURES FOR TVSUM")
    print("=" * 65)

    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    total = len(video_files)
    print(f"Found {total} TVSum videos in {VIDEO_DIR}\n")

    start_time = time.time()
    success_count = 0

    for idx, vf in enumerate(video_files, 1):
        vid = os.path.splitext(vf)[0]
        vpath = os.path.join(VIDEO_DIR, vf)
        print(f"({idx}/{total}) Processing {vid}...", end=" ", flush=True)
        t0 = time.time()
        ok = process_video_text(vid, vpath)
        if ok:
            success_count += 1
            print(f"Done in {time.time() - t0:.2f}s")
        else:
            print("Failed.")

    # Clean up temp audio dir
    if os.path.isdir(TEMP_AUDIO_DIR):
        try:
            os.rmdir(TEMP_AUDIO_DIR)
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"Extraction Completed: {success_count}/{total} videos processed in {time.time() - start_time:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
