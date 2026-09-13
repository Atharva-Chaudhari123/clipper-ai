"""
src/fast_transcriber.py
-----------------------
Phase 13: High-Throughput Candidate-Only Speech Transcriber using faster-whisper.

Replaces slow openai-whisper with faster-whisper (CTranslate2) on GPU.
Transcribes ONLY candidate highlight windows by slicing in-memory PCM audio,
reducing transcription time by 60–90% on long videos.
Maps relative window timestamps to absolute video timestamps.
"""

import os
import wave
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

logger = logging.getLogger("FastTranscriber")

_LOADED_MODEL = None
_LOADED_KEY = None


def get_faster_whisper_model(
    model_size: str = "base",
    device: Optional[str] = None,
    compute_type: Optional[str] = None
):
    """
    Loads and caches a faster-whisper model on GPU with memory-efficient compute type.
    """
    global _LOADED_MODEL, _LOADED_KEY
    import sys

    # On Windows, register NVIDIA CUDA 12 DLLs for CTranslate2 if present
    if sys.platform == "win32":
        venv_base = sys.prefix
        for sub in [
            os.path.join(venv_base, "lib", "site-packages", "nvidia", "cublas", "bin"),
            os.path.join(venv_base, "lib", "site-packages", "nvidia", "cudnn", "bin")
        ]:
            if os.path.isdir(sub):
                try:
                    os.add_dll_directory(sub)
                except (AttributeError, OSError):
                    pass
                if sub not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = sub + os.pathsep + os.environ.get("PATH", "")

    from faster_whisper import WhisperModel
    import ctranslate2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type is None:
        if device == "cuda":
            supported = ctranslate2.get_supported_compute_types("cuda")
            if "int8_float16" in supported:
                compute_type = "int8_float16"
            elif "float16" in supported:
                compute_type = "float16"
            else:
                compute_type = "int8"
        else:
            compute_type = "int8"

    cache_key = (model_size, device, compute_type)
    if _LOADED_MODEL is not None and _LOADED_KEY == cache_key:
        return _LOADED_MODEL

    logger.info(f"Loading faster-whisper '{model_size}' on {device} (compute_type: {compute_type})...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda" and compute_type == "int8_float16":
            logger.warning(f"Failed int8_float16 ({str(e)}). Retrying float16...")
            model = WhisperModel(model_size, device=device, compute_type="float16")
            compute_type = "float16"
            cache_key = (model_size, device, compute_type)
        elif device == "cuda":
            logger.warning(f"CUDA faster-whisper failed ({str(e)}). Falling back to CPU int8...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            cache_key = (model_size, "cpu", "int8")
        else:
            raise e

    _LOADED_MODEL = model
    _LOADED_KEY = cache_key
    return model


def transcribe_candidate_windows(
    wav_path: str,
    candidate_windows: List[Dict[str, Any]],
    model_size: str = "base",
    device: Optional[str] = None,
    language: Optional[str] = "en",
    stage_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Transcribes ONLY candidate windows from the 16 kHz mono WAV file.

    Slices PCM audio samples directly in memory without disk I/O, runs faster-whisper,
    and maps relative segment timestamps to absolute video timestamps.

    Args:
        wav_path: Path to 16 kHz mono PCM WAV file
        candidate_windows: List of candidate window dicts [{"start": float, "end": float, ...}]
        model_size: Whisper model size (default: "base")
        device: "cuda" or "cpu"
        language: Language code (default: "en", or None for auto-detect)
        stage_callback: Progress callback

    Returns:
        Updated candidate_windows with "segments", "full_text", and "has_speech".
    """
    if not os.path.isfile(wav_path) or not candidate_windows:
        for cand in candidate_windows:
            cand["segments"] = []
            cand["full_text"] = ""
            cand["has_speech"] = False
        return candidate_windows

    # Load entire 16 kHz mono WAV into memory as normalized float32 array
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples /= max_val

    model = get_faster_whisper_model(model_size=model_size, device=device)

    total_candidates = len(candidate_windows)
    logger.info(f"Transcribing {total_candidates} candidate audio windows with faster-whisper...")

    for idx, cand in enumerate(candidate_windows):
        start_sec = cand["start"]
        end_sec = cand["end"]

        start_sample = max(0, int(start_sec * sr))
        end_sample = min(len(samples), int(end_sec * sr))

        if stage_callback and (idx % max(1, total_candidates // 5) == 0 or idx == total_candidates - 1):
            stage_callback(f"Transcribing candidate speech ({idx + 1}/{total_candidates})...")

        if end_sample <= start_sample:
            cand["segments"] = []
            cand["full_text"] = ""
            cand["has_speech"] = False
            continue

        cand_audio = samples[start_sample:end_sample]

        # Transcribe in-memory audio chunk
        try:
            segments_generator, info = model.transcribe(
                cand_audio,
                language=language,
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )

            parsed_segments = []
            text_pieces = []

            for seg in segments_generator:
                txt = seg.text.strip()
                if txt:
                    # Map relative timestamps to absolute video timestamps
                    abs_start = round(start_sec + seg.start, 2)
                    abs_end = round(start_sec + seg.end, 2)
                    parsed_segments.append({
                        "start": abs_start,
                        "end": abs_end,
                        "text": txt
                    })
                    text_pieces.append(txt)

            cand["segments"] = parsed_segments
            cand["full_text"] = " ".join(text_pieces)
            cand["has_speech"] = len(parsed_segments) > 0 and len(cand["full_text"]) > 0

        except Exception as e:
            logger.warning(f"Failed to transcribe candidate {cand.get('candidate_id', idx)}: {str(e)}")
            cand["segments"] = []
            cand["full_text"] = ""
            cand["has_speech"] = False

    return candidate_windows


def transcribe_full_audio_fast(
    wav_path: str,
    model_size: str = "base",
    device: Optional[str] = None,
    language: Optional[str] = "en"
) -> Dict[str, Any]:
    """
    Fast transcription of an entire audio file (useful for benchmarks and short videos).
    """
    if not os.path.isfile(wav_path):
        return {"language": "unknown", "segments": [], "full_text": "", "has_speech": False}

    model = get_faster_whisper_model(model_size=model_size, device=device)

    try:
        segments_generator, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=3,
            vad_filter=True
        )

        parsed_segments = []
        text_pieces = []
        for s in segments_generator:
            txt = s.text.strip()
            if txt:
                parsed_segments.append({
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": txt
                })
                text_pieces.append(txt)

        full_text = " ".join(text_pieces)
        return {
            "language": info.language if info else "en",
            "segments": parsed_segments,
            "full_text": full_text,
            "has_speech": len(parsed_segments) > 0
        }
    except Exception as e:
        logger.error(f"Full audio faster-whisper transcription error: {str(e)}")
        return {"language": "unknown", "segments": [], "full_text": "", "has_speech": False}
