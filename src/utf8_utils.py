"""
src/utf8_utils.py
-----------------
Windows-compatible UTF-8 and Unicode safety utilities.
Guarantees that logging, standard output/error, subprocess output,
and file operations handle non-ASCII / emoji characters (e.g. 😰 🎬 ✅)
safely without raising 'charmap' UnicodeEncodeError on Windows.
"""

import os
import sys
import logging

def setup_utf8_environment():
    """
    Configures Python stdout, stderr, and standard logging to handle UTF-8 safely
    with character replacement for any unsupported console environments on Windows.
    """
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Add known FFmpeg and FFprobe binary locations to PATH on Windows
    ffmpeg_dirs = [
        r"C:\pinokio\bin\miniconda\Library\bin",
        r"C:\pinokio\bin\miniconda\pkgs\ffmpeg-8.0.1-gpl_h74fd8f1_909\Library\bin"
    ]
    for fdir in ffmpeg_dirs:
        if os.path.isdir(fdir) and fdir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = fdir + os.pathsep + os.environ.get("PATH", "")

    for stream_name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def sanitize_for_logging(text: str) -> str:
    """
    Sanitizes string for safe logging/printing under any codepage.
    If text can be encoded in UTF-8, it returns it as-is.
    """
    if not isinstance(text, str):
        text = str(text)
    return text


class SafeStreamHandler(logging.StreamHandler):
    """
    A robust logging StreamHandler that never crashes on Windows charmap encoding errors.
    If emitting a log record fails due to a UnicodeEncodeError, it falls back to
    encoding with errors='replace'.
    """
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                stream = self.stream
                safe_msg = msg.encode("utf-8", errors="replace").decode(
                    stream.encoding or "utf-8", errors="replace"
                )
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)
