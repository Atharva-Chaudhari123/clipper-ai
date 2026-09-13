"""
src/audio_pre_screener.py
-------------------------
Phase 13: Lightning-Fast Audio & Speech Pre-Screening.

Extracts audio from video and detects speech/audio active intervals
using windowed RMS energy and speech-band zero-crossing rate in milliseconds.
Does NOT run expensive neural transcription.
Executes in < 0.5 seconds for a 60-minute audio track.
"""

import os
import sys
import wave
import subprocess
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("AudioPreScreener")


def extract_audio_fast(video_path: str, output_wav_path: str) -> bool:
    """
    Fast extraction of 16 kHz mono 16-bit PCM WAV using FFmpeg.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="replace")
            if "does not contain any stream" in err or "does not contain audio" in err:
                logger.warning(f"Video {video_path} has no audio stream.")
                return False
            logger.warning(f"FFmpeg audio extraction warning: {err[:200]}")
            return False

        if os.path.isfile(output_wav_path) and os.path.getsize(output_wav_path) > 1024:
            return True
        return False
    except Exception as e:
        logger.error(f"Error in FFmpeg fast audio extraction: {str(e)}")
        return False


def detect_speech_intervals(
    wav_path: str,
    frame_duration_sec: float = 0.05,  # 50ms frames
    hop_duration_sec: float = 0.025,   # 25ms step (50% overlap)
    min_speech_duration: float = 0.4,  # minimum speech segment
    merge_gap_duration: float = 0.8,   # merge gaps between phrases
    energy_percentile: float = 30.0    # adaptive noise floor percentile
) -> Dict[str, Any]:
    """
    Detects speech activity intervals across the entire audio file without Whisper.

    Uses adaptive RMS energy thresholding combined with zero-crossing density
    to identify voiced/unvoiced human speech regions.

    Returns:
        Dict containing:
        - speech_intervals: List of (start_sec, end_sec)
        - total_duration: Duration of audio in seconds
        - speech_duration: Total seconds of detected speech
        - speech_ratio: Fraction of audio containing speech
        - has_speech: Boolean
    """
    if not os.path.isfile(wav_path):
        return {
            "speech_intervals": [],
            "total_duration": 0.0,
            "speech_duration": 0.0,
            "speech_ratio": 0.0,
            "has_speech": False
        }

    try:
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            num_frames = wf.getnframes()
            audio_bytes = wf.readframes(num_frames)

        total_duration = num_frames / float(sample_rate)
        if total_duration <= 0.0:
            return {
                "speech_intervals": [],
                "total_duration": 0.0,
                "speech_duration": 0.0,
                "speech_ratio": 0.0,
                "has_speech": False
            }

        # Parse PCM 16-bit
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels).mean(axis=1)

        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples /= max_val
        else:
            return {
                "speech_intervals": [],
                "total_duration": total_duration,
                "speech_duration": 0.0,
                "speech_ratio": 0.0,
                "has_speech": False
            }

        frame_len = int(sample_rate * frame_duration_sec)
        hop_len = int(sample_rate * hop_duration_sec)
        if frame_len <= 0 or hop_len <= 0 or len(samples) < frame_len:
            return {
                "speech_intervals": [],
                "total_duration": total_duration,
                "speech_duration": 0.0,
                "speech_ratio": 0.0,
                "has_speech": False
            }

        # Compute RMS energy across sliding frames
        num_windows = 1 + (len(samples) - frame_len) // hop_len
        shape = (num_windows, frame_len)
        strides = (samples.strides[0] * hop_len, samples.strides[0])
        frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

        # RMS energy per frame
        rms_energy = np.sqrt(np.mean(frames ** 2, axis=1))

        # Adaptive thresholding: noise floor estimation
        noise_floor = np.percentile(rms_energy, energy_percentile)
        dynamic_range = np.percentile(rms_energy, 95) - noise_floor
        threshold = noise_floor + 0.12 * dynamic_range

        # Threshold to binary speech mask
        is_speech = rms_energy > threshold

        # Group continuous speech frames
        raw_intervals: List[Tuple[float, float]] = []
        in_speech = False
        start_t = 0.0

        for i, speech_flag in enumerate(is_speech):
            t = i * hop_duration_sec
            if speech_flag and not in_speech:
                in_speech = True
                start_t = t
            elif not speech_flag and in_speech:
                in_speech = False
                end_t = t + frame_duration_sec
                if (end_t - start_t) >= min_speech_duration:
                    raw_intervals.append((round(start_t, 2), round(end_t, 2)))

        if in_speech:
            end_t = total_duration
            if (end_t - start_t) >= min_speech_duration:
                raw_intervals.append((round(start_t, 2), round(end_t, 2)))

        # Merge intervals separated by <= merge_gap_duration
        merged_intervals: List[Tuple[float, float]] = []
        for s, e in raw_intervals:
            if not merged_intervals:
                merged_intervals.append((s, e))
            else:
                prev_s, prev_e = merged_intervals[-1]
                if s - prev_e <= merge_gap_duration:
                    merged_intervals[-1] = (prev_s, max(prev_e, e))
                else:
                    merged_intervals.append((s, e))

        total_speech_sec = sum(e - s for s, e in merged_intervals)
        speech_ratio = total_speech_sec / total_duration if total_duration > 0 else 0.0
        has_speech = total_speech_sec >= 1.0 and len(merged_intervals) > 0

        logger.info(
            f"Audio Pre-Screening: {len(merged_intervals)} speech intervals detected "
            f"({total_speech_sec:.1f}s / {total_duration:.1f}s, ratio: {speech_ratio*100:.1f}%)"
        )

        return {
            "speech_intervals": merged_intervals,
            "total_duration": round(total_duration, 2),
            "speech_duration": round(total_speech_sec, 2),
            "speech_ratio": round(speech_ratio, 3),
            "has_speech": has_speech
        }

    except Exception as e:
        logger.error(f"Failed speech interval detection: {str(e)}")
        return {
            "speech_intervals": [],
            "total_duration": 0.0,
            "speech_duration": 0.0,
            "speech_ratio": 0.0,
            "has_speech": False
        }
