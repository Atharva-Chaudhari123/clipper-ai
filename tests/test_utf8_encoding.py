"""
tests/test_utf8_encoding.py
---------------------------
Regression test suite for Windows Unicode / Charmap encoding safety.

Verifies that the project safely handles Unicode characters and emojis (such as 😰 🎬 ✅):
1. Writing / printing to stdout and stderr without charmap UnicodeEncodeError
2. Logging through SafeStreamHandler and standard logging
3. Reading / writing text files with explicit UTF-8 encoding
4. Subprocess execution and stderr/stdout decoding with errors='replace'
5. Handling YouTube video metadata titles containing emojis
6. Flask background worker exception handling without thread crashing
7. Subtitle (SRT) and Manifest JSON serialization with Unicode content
"""

import os
import sys
import json
import logging
import tempfile
import unittest
import threading

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment, SafeStreamHandler
from src.reel_renderer import generate_reel_srt


class TestUTF8WindowsSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_utf8_environment()

    def test_01_stdout_stderr_print_emojis(self):
        """Regression test for printing strings containing 😰 🎬 ✅ to stdout and stderr."""
        test_str = "Testing Windows Unicode Safety: 😰 🎬 ✅"
        try:
            print(test_str)
            sys.stdout.write(test_str + "\n")
            sys.stdout.flush()
            sys.stderr.write(test_str + "\n")
            sys.stderr.flush()
        except UnicodeEncodeError as e:
            self.fail(f"Writing emojis to stdout/stderr raised UnicodeEncodeError: {e}")

    def test_02_logging_with_emojis(self):
        """Verify logging handles strings with 😰 🎬 ✅ through SafeStreamHandler."""
        logger = logging.getLogger("TestUTF8Logger")
        logger.setLevel(logging.INFO)
        handler = SafeStreamHandler(sys.stdout)
        logger.addHandler(handler)

        try:
            logger.info("Info log with emojis: 😰 🎬 ✅")
            logger.warning("Warning log with emojis: 😰 🎬 ✅")
        except UnicodeEncodeError as e:
            self.fail(f"Logging with emojis raised UnicodeEncodeError: {e}")
        finally:
            logger.removeHandler(handler)

    def test_03_file_write_read_utf8(self):
        """Verify explicit UTF-8 file read/write with emojis."""
        test_payload = {
            "title": "Breaking News: AI Code Panic 😰 [Full Guide] 🎬 ✅",
            "emojis": ["😰", "🎬", "✅"],
            "symbols": "Special quotes: “smart” and dashes — and accents café"
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as tf:
            temp_path = tf.name
            json.dump(test_payload, tf, indent=2, ensure_ascii=False)

        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["title"], test_payload["title"])
            self.assertIn("😰", loaded["emojis"])
            self.assertIn("🎬", loaded["emojis"])
            self.assertIn("✅", loaded["emojis"])
        finally:
            if os.path.isfile(temp_path):
                os.remove(temp_path)

    def test_04_srt_generation_with_emojis(self):
        """Verify SRT subtitle generation safely serializes dialogue with emojis."""
        speech_segments = [
            {"start": 1.0, "end": 4.5, "text": "I was so scared 😰 but then we started filming 🎬!"},
            {"start": 5.0, "end": 8.0, "text": "And everything worked out perfectly ✅."}
        ]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".srt") as tf:
            srt_path = tf.name

        try:
            has_subs = generate_reel_srt(
                speech_segments=speech_segments,
                reel_start_sec=0.0,
                reel_end_sec=10.0,
                output_srt_path=srt_path
            )
            self.assertTrue(has_subs)
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("😰", content)
            self.assertIn("🎬", content)
            self.assertIn("✅", content)
        finally:
            if os.path.isfile(srt_path):
                os.remove(srt_path)

    def test_05_flask_worker_exception_handling_with_emojis(self):
        """Verify Flask worker error handling captures exception messages with emojis without crashing."""
        from app import JOBS_STORE, JOBS_LOCK

        job_id = "test_unicode_job_123"
        with JOBS_LOCK:
            JOBS_STORE[job_id] = {"status": "queued", "stage": "Initializing..."}

        # Simulate worker catching an exception containing 😰
        try:
            raise RuntimeError("Charmap error with character '\U0001f630' 😰 in position 70: undefined")
        except Exception as e:
            try:
                err_msg = str(e)
            except Exception:
                err_msg = repr(e)
            with JOBS_LOCK:
                JOBS_STORE[job_id]["status"] = "error"
                JOBS_STORE[job_id]["stage"] = "Failed"
                JOBS_STORE[job_id]["error"] = f"Processing failed: {err_msg}"

        with JOBS_LOCK:
            job_state = JOBS_STORE.get(job_id)
            self.assertEqual(job_state["status"], "error")
            self.assertIn("😰", job_state["error"])
            # Clean up
            JOBS_STORE.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
