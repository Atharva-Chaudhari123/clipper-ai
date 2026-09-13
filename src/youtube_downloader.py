"""
src/youtube_downloader.py
-------------------------
Safe acquisition of YouTube videos using yt-dlp.

Features:
- Regex validation for YouTube URLs (standard, shortened, and shorts).
- Safe download execution with yt-dlp into isolated directories.
- Metadata extraction (title, duration, uploader, channel).
- Size and duration constraints for safe inference.
- Rejection of non-YouTube or malicious URLs.
"""

import os
import sys
import re
import logging
from typing import Dict, Any, Optional
import yt_dlp

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("YouTubeDownloader")

# Valid YouTube URL patterns
YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|embed/|v/|shorts/)|youtu\.be/)([\w-]{11})([^\s]*)$"
)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validates whether the provided string is a legitimate YouTube URL.
    
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    """
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return bool(YOUTUBE_REGEX.match(url))


def extract_video_id(url: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID."""
    if not url:
        return None
    match = YOUTUBE_REGEX.match(url.strip())
    if match:
        return match.group(5)
    return None


def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    max_duration_seconds: int = 1800,  # 30 min max for highlight inference
    max_filesize_mb: int = 400
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the specified directory using yt-dlp.

    Args:
        youtube_url: Validated YouTube URL
        output_dir: Target directory for storing video
        max_duration_seconds: Maximum allowed video length
        max_filesize_mb: Maximum allowed file size

    Returns:
        Dict containing:
        - video_path: Local path to downloaded MP4
        - title: YouTube video title
        - duration: Video duration in seconds
        - uploader: Channel / uploader name
        - video_id: 11-char YouTube ID
        - webpage_url: Clean canonical URL
    """
    youtube_url = youtube_url.strip()
    if not is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}. Please provide a valid YouTube link.")

    video_id = extract_video_id(youtube_url)
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    ydl_opts = {
        # Prefer 720p or lower MP4 for rapid inference, fall back to best MP4
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": max_filesize_mb * 1024 * 1024,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        }
    }

    logger.info(f"Extracting YouTube metadata for URL: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(youtube_url, download=False)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch YouTube video metadata: {str(e)}")

        duration = info.get("duration", 0)
        if duration and duration > max_duration_seconds:
            raise ValueError(
                f"Video is too long ({duration/60:.1f} minutes). "
                f"Maximum supported duration is {max_duration_seconds/60:.0f} minutes."
            )

        title = info.get("title", f"YouTube Video {video_id}")
        uploader = info.get("uploader") or info.get("channel") or "Unknown Creator"

        logger.info(f"Downloading YouTube video: '{title}' ({duration}s)...")
        try:
            ydl.download([youtube_url])
        except Exception as e:
            raise RuntimeError(f"Failed to download YouTube video: {str(e)}")

    # Locate the downloaded video file
    expected_mp4 = os.path.join(output_dir, f"{video_id}.mp4")
    if not os.path.isfile(expected_mp4):
        # Scan dir in case extension was different
        candidates = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(video_id)]
        if candidates:
            expected_mp4 = candidates[0]
        else:
            raise FileNotFoundError(f"Downloaded video file not found in {output_dir}")

    return {
        "video_path": os.path.abspath(expected_mp4),
        "title": title,
        "duration": float(duration) if duration else 0.0,
        "uploader": uploader,
        "video_id": video_id,
        "webpage_url": f"https://www.youtube.com/watch?v=VIDEO_ID"
    }
