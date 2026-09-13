"""
scripts/batch_extract_all.py
----------------------------
Batch processes all 50 TVSum videos:
1. Samples segment-center frames (0.5 FPS) -> data/processed/frames/<vid>/
2. Extracts 2048-d deep visual features with frozen ResNet-50 -> data/processed/features/<vid>.pt
3. Saves alignment metadata -> data/processed/features/<vid>_metadata.csv

Skips any video that is already processed.
"""

import os
import sys
import json
import glob
import time
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extract_frames import extract_segment_center_frames, find_annotation_record
from src.feature_extractor import extract_and_save_video_features


def process_all_tvsum_videos():
    video_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\video"
    annotations_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations"
    frames_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames"
    features_dir = r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features"

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    # Collect all video IDs from annotation files
    annot_files = sorted(glob.glob(os.path.join(annotations_dir, "*.jsonl")))
    all_vids = []
    for af in annot_files:
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_vids.append(item["vid"])

    print(f"Total TVSum videos to process: {len(all_vids)}")
    start_total_time = time.time()

    processed_count = 0
    skipped_count = 0

    for idx, vid in enumerate(all_vids, 1):
        feature_file = os.path.join(features_dir, f"{vid}.pt")
        video_path = os.path.join(video_dir, f"{vid}.mp4")

        if os.path.isfile(feature_file):
            print(f"[{idx}/{len(all_vids)}] Video '{vid}' already processed. Skipping.")
            skipped_count += 1
            continue

        if not os.path.isfile(video_path):
            print(f"[{idx}/{len(all_vids)}] [ERROR] MP4 not found for '{vid}' at {video_path}! Skipping.")
            continue

        t0 = time.time()
        print(f"[{idx}/{len(all_vids)}] Processing video '{vid}'...")

        # Step 1: Extract segment-center frames
        annot_record = find_annotation_record(vid, annotations_dir)
        frame_stats, v_frames_dir = extract_segment_center_frames(
            video_path=video_path,
            annotation_record=annot_record,
            output_base_dir=frames_dir
        )

        # Step 2: Extract 2048-d ResNet-50 features
        feat_stats = extract_and_save_video_features(
            video_id=vid,
            frames_base_dir=frames_dir,
            features_output_dir=features_dir,
            initial_batch_size=8
        )

        elapsed = time.time() - t0
        print(f"  -> Extracted {feat_stats['num_features']} features in {elapsed:.1f}s. Saved: {feat_stats['features_file']}")
        processed_count += 1

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 60)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total videos:      {len(all_vids)}")
    print(f"Newly processed:   {processed_count}")
    print(f"Previously cached: {skipped_count}")
    print(f"Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)")
    print("=" * 60)


if __name__ == "__main__":
    process_all_tvsum_videos()
