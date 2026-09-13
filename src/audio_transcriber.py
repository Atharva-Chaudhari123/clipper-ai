"""
src/audio_transcriber.py
------------------------
Audio extraction via FFmpeg and speech transcription using Whisper.

Features:
- FFmpeg 16 kHz mono PCM WAV extraction.
- Automatic detection of silent/audio-less video files.
- Whisper speech transcription with timestamp preservation.
- Structured segment output: [{"start": 0.0, "end": 4.2, "text": "..."}].
- Configurable model size (default: "base" for 4GB VRAM GTX 1650).
- CUDA acceleration with seamless CPU fallback.
- Temporary audio file cleanup.
"""

import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
import torch

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("AudioTranscriber")

# Global cache for loaded Whisper model to prevent reloading across jobs
_LOADED_WHISPER_MODEL = None
_LOADED_MODEL_SIZE = None


def extract_audio_from_video(video_path: str, output_wav_path: str) -> bool:
    """
    Extracts audio from video into a 16 kHz mono PCM WAV file using FFmpeg.

    Returns:
        True if audio was successfully extracted, False if video has no audio stream.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            # If video has no audio stream, FFmpeg returns error like "Output file does not contain any stream"
            if "does not contain any stream" in err or "does not contain audio" in err:
                logger.warning(f"Video {video_path} does not contain an audio stream.")
                return False
            logger.warning(f"FFmpeg audio extraction warning: {err[:200]}")
            return False

        # Verify output WAV exists and has non-zero size
        if os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 1024:
            return True
        return False

    except Exception as e:
        logger.error(f"Error executing FFmpeg audio extraction: {str(e)}")
        return False


def get_whisper_model(model_size: str = "base", device: str = None):
    """Loads and caches Whisper model on specified device."""
    global _LOADED_WHISPER_MODEL, _LOADED_MODEL_SIZE
    import whisper

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _LOADED_WHISPER_MODEL is not None and _LOADED_MODEL_SIZE == model_size:
        return _LOADED_WHISPER_MODEL

    logger.info(f"Loading Whisper model '{model_size}' on device: {device}...")
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        if device == "cuda":
            logger.warning(f"Failed to load Whisper on CUDA ({str(e)}). Falling back to CPU...")
            model = whisper.load_model(model_size, device="cpu")
        else:
            raise e

    _LOADED_WHISPER_MODEL = model
    _LOADED_MODEL_SIZE = model_size
    return model


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribes a WAV audio file using Whisper.

    Args:
        audio_path: Path to mono 16 kHz WAV file
        model_size: Whisper model size ("base", "small", "tiny")
        device: "cuda" or "cpu"
        language: Language code (e.g. "en"), or None for auto-detection

    Returns:
        Dict with:
        - language: Detected language
        - segments: List of [{"start": float, "end": float, "text": str}]
        - full_text: Full transcribed text string
        - has_speech: Boolean indicating whether speech was detected
    """
    if not os.path.isfile(audio_path):
        return {
            "language": "unknown",
            "segments": [],
            "full_text": "",
            "has_speech": False
        }

    model = get_whisper_model(model_size=model_size, device=device)
    fp16 = torch.cuda.is_available() and (next(model.parameters()).device.type == "cuda")

    transcribe_options = {
        "fp16": fp16,
        "verbose": False
    }
    if language:
        transcribe_options["language"] = language

    try:
        result = model.transcribe(audio_path, **transcribe_options)
    except Exception as e:
        logger.error(f"Whisper transcription failed: {str(e)}")
        return {
            "language": "unknown",
            "segments": [],
            "full_text": "",
            "has_speech": False
        }

    raw_segments = result.get("segments", [])
    parsed_segments = []
    for s in raw_segments:
        text = s.get("text", "").strip()
        if text:
            parsed_segments.append({
                "start": round(float(s.get("start", 0.0)), 2),
                "end": round(float(s.get("end", 0.0)), 2),
                "text": text
            })

    full_text = result.get("text", "").strip()
    detected_lang = result.get("language", "en")
    has_speech = len(parsed_segments) > 0 and len(full_text) > 0

    return {
        "language": detected_lang,
        "segments": parsed_segments,
        "full_text": full_text,
        "has_speech": has_speech
    }
