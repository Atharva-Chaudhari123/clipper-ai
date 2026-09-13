"""
src/extract_frames.py
---------------------
Phase 2: TVSum Segment-Center Frame Sampling (0.5 FPS)

This module performs segment-center video frame extraction for the TVSum dataset.
In TVSum, each ground-truth annotation corresponds to a 2-second segment [2*i, 2*(i+1)].
By sampling exactly one representative frame at the temporal center (2*i + 1.0 seconds),
we establish a strict 1-to-1 correspondence between:
    1 TVSum Annotation Segment <--> 1 Sampled Frame <--> 1 Feature Vector
"""

import os
import csv
import json
import glob
import cv2
import numpy as np


def find_annotation_record(video_id: str, annotations_dir: str) -> dict:
    """
    Searches both train and val TVSum JSONL files for the record matching video_id.

    Parameters:
        video_id (str): The TVSum YouTube video ID (e.g., 'akI8YFjEmUw').
        annotations_dir (str): Path to annotations directory.

    Returns:
        dict: The parsed JSON annotation record for this video.
    """
    jsonl_files = sorted(glob.glob(os.path.join(annotations_dir, "*.jsonl")))
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("vid") == video_id or item.get("qid") == video_id:
                        return item
    raise FileNotFoundError(f"Video ID '{video_id}' not found in annotation files at {annotations_dir}")


def extract_segment_center_frames(
    video_path: str,
    annotation_record: dict,
    output_base_dir: str
) -> tuple[dict, str]:
    """
    Extracts one frame at the center of each 2-second annotation segment.

    Parameters:
        video_path (str): Path to the source MP4 video file.
        annotation_record (dict): Parsed TVSum annotation entry.
        output_base_dir (str): Base directory where frames/<video_id>/ will be created.

    Returns:
        tuple[dict, str]: Summary statistics dict and the directory path where frames were saved.
    """
    video_id = annotation_record["vid"]
    video_frames_dir = os.path.join(output_base_dir, video_id)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0.0

    # Number of 2-second annotation segments provided by TVSum
    num_segments = len(annotation_record["label"])

    metadata_rows = []
    extracted_count = 0
    skipped_count = 0

    for i in range(num_segments):
        start_time = 2.0 * i
        end_time = 2.0 * (i + 1)
        sample_time = start_time + 1.0  # Center of the 2-second window

        # Video boundary check: verify if sample_time is within the video
        if sample_time >= video_duration:
            print(f"[Warning] Segment {i} sample_time ({sample_time:.2f}s) exceeds video duration ({video_duration:.2f}s). Skipping.")
            skipped_count += 1
            continue

        # Convert timestamp to frame index
        target_frame_idx = int(round(sample_time * fps))
        if target_frame_idx >= total_frames:
            target_frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            # Fallback attempt: seek by milliseconds if frame index seek fails
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            ret, frame = cap.read()

        if not ret or frame is None:
            print(f"[Warning] Failed to read frame for segment {i} at {sample_time:.2f}s. Skipping.")
            skipped_count += 1
            continue

        # Frame filename format: segment_0000.jpg
        frame_filename = f"segment_{i:04d}.jpg"
        frame_save_path = os.path.join(video_frames_dir, frame_filename)

        # Save frame to disk (JPEG format)
        cv2.imwrite(frame_save_path, frame)
        extracted_count += 1

        metadata_rows.append({
            "segment_index": i,
            "sample_time": round(sample_time, 2),
            "frame_path": frame_filename,
            "annotation_start": round(start_time, 2),
            "annotation_end": round(end_time, 2)
        })

    cap.release()

    # Save metadata.csv in data/processed/frames/<video_id>/metadata.csv
    metadata_csv_path = os.path.join(video_frames_dir, "metadata.csv")
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_index", "sample_time", "frame_path", "annotation_start", "annotation_end"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    stats = {
        "video_id": video_id,
        "video_duration": video_duration,
        "fps": fps,
        "total_video_frames": total_frames,
        "annotation_segments": num_segments,
        "frames_extracted": extracted_count,
        "metadata_rows": len(metadata_rows),
        "skipped_segments": skipped_count,
        "frames_dir": video_frames_dir,
        "metadata_csv": metadata_csv_path,
        "sample_rows": metadata_rows[:5]
    }
    return stats, video_frames_dir


def create_contact_sheet(
    frames_dir: str,
    metadata_csv_path: str,
    output_image_path: str,
    num_frames: int = 5
) -> str:
    """
    Creates a visual contact sheet combining extracted segment frames with annotations.

    Parameters:
        frames_dir (str): Directory where segment frames are stored.
        metadata_csv_path (str): Path to metadata.csv.
        output_image_path (str): Destination path for the contact sheet image.
        num_frames (int): Number of consecutive segment frames to include (default: 5).

    Returns:
        str: Absolute path of the saved contact sheet image.
    """
    # Read first num_frames entries from metadata.csv
    rows = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= num_frames:
                break

    card_width = 240
    card_height = 135
    header_height = 50
    padding = 10

    total_width = len(rows) * card_width + (len(rows) + 1) * padding
    total_height = card_height + header_height + 2 * padding

    # Create dark slate background canvas
    canvas = np.full((total_height, total_width, 3), (28, 28, 30), dtype=np.uint8)

    for idx, row in enumerate(rows):
        frame_filename = row["frame_path"]
        frame_path = os.path.join(frames_dir, frame_filename)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Resize image thumbnail to fit card
        thumb = cv2.resize(img, (card_width, card_height))

        # Position thumbnail
        x_offset = padding + idx * (card_width + padding)
        y_offset = padding

        canvas[y_offset:y_offset + card_height, x_offset:x_offset + card_width] = thumb

        # Add decorative border around thumbnail
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + card_width, y_offset + card_height), (80, 80, 90), 1)

        # Render metadata labels below the thumbnail
        seg_idx = row["segment_index"]
        s_time = row["sample_time"]
        r_start = row["annotation_start"]
        r_end = row["annotation_end"]

        line1 = f"Segment: {seg_idx}"
        line2 = f"Time: {s_time}s"
        line3 = f"Range: [{r_start}s - {r_end}s]"

        text_y = y_offset + card_height + 14
        cv2.putText(canvas, line1, (x_offset + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (x_offset + 5, text_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line3, (x_offset + 5, text_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    cv2.imwrite(output_image_path, canvas)
    return output_image_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVSum Segment-Center Frame Sampler (0.5 FPS)")
    parser.add_argument("--video_id", type=str, default="akI8YFjEmUw", help="TVSum video ID to process")
    parser.add_argument("--video_dir", type=str, default=r"C:\Users\mahin\OneDrive\Desktop\New folder\video")
    parser.add_argument("--annotations_dir", type=str, default=r"C:\Users\mahin\OneDrive\Desktop\New folder\annotations")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames")
    args = parser.parse_args()

    video_path = os.path.join(args.video_dir, f"{args.video_id}.mp4")
    annot = find_annotation_record(args.video_id, args.annotations_dir)

    print(f"Extracting segment-center frames for video: {args.video_id}")
    stats, f_dir = extract_segment_center_frames(video_path, annot, args.output_dir)

    contact_sheet_path = os.path.join(f_dir, "contact_sheet.jpg")
    create_contact_sheet(f_dir, stats["metadata_csv"], contact_sheet_path, num_frames=5)
    print(f"Contact sheet saved to: {contact_sheet_path}")
    print(f"Extraction finished: {stats['frames_extracted']} frames extracted for {stats['annotation_segments']} segments.")
