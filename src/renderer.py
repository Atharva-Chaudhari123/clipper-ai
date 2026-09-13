"""
src/renderer.py
---------------
Phase 8: Highlight Video Rendering Engine using FFmpeg.

Extracts selected highlight segments from the original video and stitches them
into a polished, coherent short summary video (.mp4) with audio preserved.
"""

import os
import sys
import subprocess
import tempfile
import json

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()


def render_highlight_video(
    source_video_path: str,
    intervals: list[tuple[float, float]],
    output_path: str
) -> dict:
    """
    Renders a highlight video from a list of time intervals [(start, end), ...] using FFmpeg.

    Parameters:
        source_video_path (str): Path to original MP4 video.
        intervals (list[tuple[float, float]]): Highlight intervals in seconds.
        output_path (str): Destination path for the output MP4 highlight video.

    Returns:
        dict: Summary metadata of the rendered highlight video.
    """
    if not os.path.isfile(source_video_path):
        raise FileNotFoundError(f"Source video file not found: {source_video_path}")
    if not intervals:
        raise ValueError("No highlight intervals provided to render.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="highlight_render_")
    part_files = []

    try:
        # Step 1: Cut each highlight segment into a temporary clip
        for i, (start_sec, end_sec) in enumerate(intervals):
            part_path = os.path.join(temp_dir, f"part_{i:03d}.mp4")
            duration_sec = end_sec - start_sec

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-i", source_video_path,
                "-t", str(duration_sec),
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                part_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode != 0:
                err = res.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg segment extract failed: {err[:300]}")
            part_files.append(part_path)

        # Step 2: Create concat list file for FFmpeg
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for p in part_files:
                # Escape single quotes and backslashes for FFmpeg
                escaped_path = p.replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")

        # Step 3: Concatenate all parts into final highlight video
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path
        ]
        res_cat = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res_cat.returncode != 0:
            err_cat = res_cat.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg concat failed: {err_cat[:300]}")

        # Step 4: Probe rendered video duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-of", "json", output_path
        ]
        probe_res = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        probe_info = json.loads(probe_res.stdout)["format"]
        rendered_dur = float(probe_info["duration"])
        rendered_size = int(probe_info["size"])

    finally:
        # Clean up temporary part files
        for p in part_files:
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        concat_txt = os.path.join(temp_dir, "concat_list.txt")
        if os.path.isfile(concat_txt):
            try:
                os.remove(concat_txt)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    return {
        "output_path": output_path,
        "duration_seconds": rendered_dur,
        "file_size_bytes": rendered_size,
        "file_size_mb": round(rendered_size / (1024 * 1024), 2),
        "num_intervals": len(intervals),
        "intervals": intervals
    }
