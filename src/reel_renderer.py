"""
src/reel_renderer.py
--------------------
Phase 13: 9:16 Vertical Video Renderer with Face-Aware Tracking & Styled Captions.

Extracts selected Reels from source video, reframes landscape video to 9:16 vertical
aspect ratio (1080x1920) using Smart Face Tracking or clean center-crop fallback,
aligns subtitle timestamps relative to reel start, burns in styled subtitles,
and writes reel_01.mp4 ... reel_N.mp4 and manifest.json.
"""

import os
import sys
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("ReelRenderer")

# Discover FFmpeg executable across environment or known locations
import shutil
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
for candidate in [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
    r"C:\pinokio\bin\miniconda\Library\bin\ffmpeg.exe"
]:
    if os.path.isfile(candidate):
        FFMPEG_BIN = candidate
        break

FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"
for candidate in [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"),
    r"C:\pinokio\bin\miniconda\Library\bin\ffprobe.exe"
]:
    if os.path.isfile(candidate):
        FFPROBE_BIN = candidate
        break


def probe_video_dimensions(video_path: str) -> Dict[str, int]:
    """Probes video width and height using ffprobe or OpenCV fallback."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        return {"width": int(stream["width"]), "height": int(stream["height"])}
    except Exception:
        # Fallback to OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return {"width": w, "height": h}
        except Exception:
            pass
        return {"width": 1920, "height": 1080}


def format_srt_timestamp(seconds: float) -> str:
    """Formats float seconds into SRT timestamp string HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def generate_reel_srt(
    speech_segments: List[Dict[str, Any]],
    reel_start_sec: float,
    reel_end_sec: float,
    output_srt_path: str
) -> bool:
    """
    Creates an SRT subtitle file with timestamps shifted relative to reel_start_sec.
    """
    if not speech_segments:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    count = 1

    with open(output_srt_path, "w", encoding="utf-8") as f:
        for seg in speech_segments:
            s = seg["start"]
            e = seg["end"]
            text = seg["text"].strip()

            # Check if segment overlaps reel time range
            if e <= reel_start_sec or s >= reel_end_sec or not text:
                continue

            rel_s = max(0.0, s - reel_start_sec)
            rel_e = min(reel_end_sec - reel_start_sec, e - reel_start_sec)

            if rel_e <= rel_s:
                continue

            f.write(f"{count}\n")
            f.write(f"{format_srt_timestamp(rel_s)} --> {format_srt_timestamp(rel_e)}\n")
            f.write(f"{text}\n\n")
            count += 1

    return count > 1


def render_single_reel_smart(
    source_video_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    aspect_ratio: str = "9:16",
    speech_segments: Optional[List[Dict[str, Any]]] = None,
    captions: bool = True,
    use_face_tracking: bool = True
) -> Dict[str, Any]:
    """
    Renders an individual reel using Smart Face Tracking with OpenCV or standard FFmpeg crop.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # If landscape and face tracking is enabled, attempt smart OpenCV camera tracking
    if aspect_ratio == "9:16" and use_face_tracking:
        try:
            from src.face_tracker import SmartFaceTracker
            import cv2

            cap = cv2.VideoCapture(source_video_path)
            if cap.isOpened():
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                if src_w > src_h:  # Landscape video
                    logger.info(f"Applying Smart Face Tracking for Reel [{start_sec:.1f}s - {start_sec + duration_sec:.1f}s]...")
                    start_frame = int(round(start_sec * fps))
                    total_frames = int(round(duration_sec * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    tracker = SmartFaceTracker(smoothing=0.15)
                    silent_temp = output_path + ".temp_silent.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    # Target 9:16 dimensions
                    crop_h = src_h
                    crop_w = int(round(src_h * (9.0 / 16.0)))
                    crop_w = max(2, crop_w - (crop_w % 2))
                    crop_h = max(2, crop_h - (crop_h % 2))

                    writer = cv2.VideoWriter(silent_temp, fourcc, fps, (1080, 1920))
                    frames_written = 0

                    while frames_written < total_frames:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        x0, y0, cw, ch = tracker.compute_crop_box(frame, target_aspect_ratio="9:16")
                        cropped = frame[y0:y0 + ch, x0:x0 + cw]
                        resized = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LINEAR)
                        writer.write(resized)
                        frames_written += 1

                    cap.release()
                    writer.release()

                    if frames_written > 0 and os.path.isfile(silent_temp):
                        # Mux audio and burn subtitles
                        srt_path = None
                        vf_filters = []
                        if captions and speech_segments:
                            srt_path = output_path.replace(".mp4", ".srt")
                            if generate_reel_srt(speech_segments, start_sec, start_sec + duration_sec, srt_path):
                                escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
                                # TikTok/Reels high-impact subtitle styling: Yellow font, bold black outline, centered at lower-third
                                sub_style = "FontName=Arial,FontSize=28,Bold=1,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=2,Alignment=2,MarginV=120"
                                vf_filters.append(f"subtitles='{escaped_srt}':force_style='{sub_style}'")

                        mux_cmd = [
                            FFMPEG_BIN, "-y",
                            "-i", silent_temp,
                            "-ss", str(start_sec),
                            "-i", source_video_path,
                            "-t", str(duration_sec),
                            "-map", "0:v:0",
                            "-map", "1:a:0?",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                            "-c:a", "aac", "-b:a", "128k",
                            "-shortest"
                        ]
                        if vf_filters:
                            mux_cmd.extend(["-vf", ",".join(vf_filters)])
                        mux_cmd.append(output_path)

                        res_mux = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                        if os.path.isfile(silent_temp):
                            try:
                                os.remove(silent_temp)
                            except OSError:
                                pass
                        if srt_path and os.path.isfile(srt_path):
                            try:
                                os.remove(srt_path)
                            except OSError:
                                pass

                        if res_mux.returncode == 0 and os.path.isfile(output_path):
                            file_size_bytes = os.path.getsize(output_path)
                            return {
                                "output_path": output_path,
                                "file_size_bytes": file_size_bytes,
                                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2)
                            }
        except Exception as e:
            logger.warning(f"Smart Face Tracking encountered an issue ({e}). Falling back to FFmpeg center crop.")

    # Standard FFmpeg fallback render
    return _render_ffmpeg_crop(
        source_video_path=source_video_path,
        start_sec=start_sec,
        duration_sec=duration_sec,
        output_path=output_path,
        aspect_ratio=aspect_ratio,
        speech_segments=speech_segments,
        captions=captions
    )


render_single_reel = render_single_reel_smart


def _render_ffmpeg_crop(
    source_video_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    aspect_ratio: str = "9:16",
    speech_segments: Optional[List[Dict[str, Any]]] = None,
    captions: bool = True
) -> Dict[str, Any]:
    """Fallback FFmpeg center-crop rendering."""
    dims = probe_video_dimensions(source_video_path)
    w, h = dims["width"], dims["height"]

    video_filters = []
    if aspect_ratio == "9:16":
        if w >= h:
            crop_w = "trunc(ih*9/16/2)*2"
            video_filters.append(f"crop={crop_w}:ih:(iw-{crop_w})/2:0")
            video_filters.append("scale=1080:1920:force_original_aspect_ratio=decrease")
            video_filters.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        else:
            video_filters.append("scale=1080:1920:force_original_aspect_ratio=decrease")
            video_filters.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

    srt_path = None
    if captions and speech_segments:
        srt_path = output_path.replace(".mp4", ".srt")
        has_subs = generate_reel_srt(
            speech_segments=speech_segments,
            reel_start_sec=start_sec,
            reel_end_sec=start_sec + duration_sec,
            output_srt_path=srt_path
        )
        if has_subs:
            escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
            sub_style = "FontName=Arial,FontSize=28,Bold=1,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=2,Alignment=2,MarginV=120"
            video_filters.append(f"subtitles='{escaped_srt}':force_style='{sub_style}'")

    vf_arg = ",".join(video_filters) if video_filters else "null"

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", str(start_sec),
        "-i", source_video_path,
        "-t", str(duration_sec),
        "-vf", vf_arg,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace")
            if "subtitles" in err and srt_path:
                no_sub_filters = [f for f in video_filters if not f.startswith("subtitles")]
                vf_fallback = ",".join(no_sub_filters) if no_sub_filters else "null"
                fallback_cmd = [
                    FFMPEG_BIN, "-y",
                    "-ss", str(start_sec),
                    "-i", source_video_path,
                    "-t", str(duration_sec),
                    "-vf", vf_fallback,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    output_path
                ]
                res_fb = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if res_fb.returncode != 0:
                    err_fb = res_fb.stderr.decode("utf-8", errors="replace")
                    raise RuntimeError(f"FFmpeg fallback failed: {err_fb[:300]}")
            else:
                raise RuntimeError(f"FFmpeg failed: {err[:300]}")
    finally:
        if srt_path and os.path.isfile(srt_path):
            try:
                os.remove(srt_path)
            except OSError:
                pass

    file_size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    return {
        "output_path": output_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2)
    }


def render_multiple_reels(
    source_video_path: str,
    selected_reels: List[Dict[str, Any]],
    output_dir: str,
    aspect_ratio: str = "9:16",
    captions: bool = True,
    use_face_tracking: bool = True,
    stage_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Renders all selected reels into output_dir and writes manifest.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_reels = []

    total_reels = len(selected_reels)
    logger.info(f"Rendering {total_reels} Reels ({aspect_ratio}, face_tracking={use_face_tracking}, captions={'ON' if captions else 'OFF'})...")

    for i, reel in enumerate(selected_reels):
        reel_num = reel["reel_num"]
        out_filename = f"reel_{reel_num:02d}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        if stage_callback:
            stage_callback(f"Rendering Reel {reel_num}/{total_reels} (Smart 9:16)...")

        render_info = render_single_reel_smart(
            source_video_path=source_video_path,
            start_sec=reel["start_sec"],
            duration_sec=reel["duration_sec"],
            output_path=out_path,
            aspect_ratio=aspect_ratio,
            speech_segments=reel.get("speech_segments"),
            captions=captions,
            use_face_tracking=use_face_tracking
        )

        manifest_item = {
            "reel_num": reel_num,
            "filename": out_filename,
            "output_path": out_path,
            "title": reel.get("title", f"Highlight #{reel_num}"),
            "viral_score": reel.get("viral_score", int(round(reel.get("score", 0.7) * 100))),
            "hook_sentence": reel.get("hook_sentence", ""),
            "virality_reason": reel.get("virality_reason", "High audience curiosity."),
            "start_sec": reel["start_sec"],
            "end_sec": reel["end_sec"],
            "start_formatted": reel["start_formatted"],
            "end_formatted": reel["end_formatted"],
            "duration_sec": reel["duration_sec"],
            "score": reel["score"],
            "transcript": reel["transcript"],
            "file_size_mb": render_info["file_size_mb"]
        }
        manifest_reels.append(manifest_item)

    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "source_video": os.path.basename(source_video_path),
        "total_reels": total_reels,
        "aspect_ratio": aspect_ratio,
        "captions": captions,
        "reels": manifest_reels
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully rendered all {total_reels} Reels! Manifest saved to {manifest_path}")
    return manifest_data
