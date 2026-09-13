"""
tests/test_flask_app.py
-----------------------
Unit and integration tests for the Flask Web Application.

Tests:
1. Flask application starts and initializes properly.
2. Home page returns HTTP 200 and contains required UI elements.
3. Upload route accepts valid MP4 and redirects to processing.
4. Invalid file extension is rejected without crashing.
5. Missing or empty file upload is handled gracefully.
6. Result page correctly displays metadata for a completed job.
7. Download route allows downloading highlight.mp4 with safe attachment headers.
8. Generated media route serves video and plot files, blocking directory traversal.
"""

import os
import sys
import io
import json
import shutil
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

from app import app, is_safe_job_id, allowed_file, JOBS_STORE, JOBS_LOCK


class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        """Set up Flask test client and temporary test directories."""
        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

        self.test_outputs_dir = os.path.join(BASE_DIR, "web_outputs", "test_mock_job_999")
        os.makedirs(self.test_outputs_dir, exist_ok=True)

        # Create mock segments.json and highlight.mp4 for result / download tests
        self.mock_meta = {
            "video_filename": "sample_video.mp4",
            "original_display_name": "sample_video.mp4",
            "original_duration": 120.0,
            "highlight_duration": 18.0,
            "budget_ratio": 0.15,
            "num_clips": 2,
            "clips": [
                {
                    "clip_num": 1,
                    "start_sec": 10.0,
                    "end_sec": 20.0,
                    "duration_sec": 10.0,
                    "start_formatted": "00:10",
                    "end_formatted": "00:20"
                },
                {
                    "clip_num": 2,
                    "start_sec": 45.0,
                    "end_sec": 53.0,
                    "duration_sec": 8.0,
                    "start_formatted": "00:45",
                    "end_formatted": "00:53"
                }
            ],
            "output_video_path": os.path.join(self.test_outputs_dir, "highlight.mp4"),
            "plot_output_path": os.path.join(self.test_outputs_dir, "scores.png"),
            "file_size_mb": 4.5
        }

        with open(os.path.join(self.test_outputs_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(self.mock_meta, f)

        # Create dummy mp4 and png files
        with open(os.path.join(self.test_outputs_dir, "highlight.mp4"), "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42")

        with open(os.path.join(self.test_outputs_dir, "scores.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    def tearDown(self):
        """Clean up test job directory and memory store."""
        if os.path.isdir(self.test_outputs_dir):
            shutil.rmtree(self.test_outputs_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS_STORE.pop("test_mock_job_999", None)

    # 1. Flask application starts
    def test_01_app_starts(self):
        self.assertIsNotNone(self.app)
        self.assertEqual(self.app.name, "app")
        self.assertTrue(callable(self.client.get))

    # 2. Home page returns HTTP 200
    def test_02_home_page_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Automatic Video Highlight Detection", html)
        self.assertIn("ResNet-50", html)
        self.assertIn("BiLSTM", html)
        self.assertIn("Generate", html)
        self.assertIn("15%", html)

    # 3. Upload page works (valid file and redirection)
    @patch("threading.Thread.start")
    def test_03_valid_upload_redirects_to_processing(self, mock_thread_start):
        data = {
            "video_file": (io.BytesIO(b"fake_mp4_video_bytes"), "test_video.mp4"),
            "budget": "15"
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/processing/job_", response.headers["Location"])

    # 4. Invalid extension is rejected
    def test_04_invalid_extension_rejected(self):
        data = {
            "video_file": (io.BytesIO(b"not_a_video"), "document.txt"),
            "budget": "15"
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Invalid file format", html)

    # 5. Missing upload is handled
    def test_05_missing_upload_handled(self):
        # Empty POST
        response = self.client.post("/upload", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("No video file selected", html)

        # Empty filename
        data = {
            "video_file": (io.BytesIO(b""), ""),
            "budget": "15"
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("No file was chosen", html)

    # 6. Result page works for a completed job
    def test_06_result_page_for_completed_job(self):
        response = self.client.get("/result/test_mock_job_999")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Your Highlights Are Ready!", html)
        self.assertIn("sample_video.mp4", html)
        self.assertIn("120.0s", html)
        self.assertIn("18.0s", html)
        self.assertIn("Clip 1", html)
        self.assertIn("00:10", html)
        self.assertIn("00:20", html)
        self.assertIn("Clip 2", html)
        self.assertIn("00:45", html)
        self.assertIn("00:53", html)

    # 7. Download route works
    def test_07_download_route_works(self):
        response = self.client.get("/download/test_mock_job_999")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "video/mp4")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
        self.assertIn("test_mock_job_999_highlight.mp4", response.headers.get("Content-Disposition", ""))
        self.assertTrue(len(response.data) > 0)

    # 8. Generated MP4 and media exists & security checks
    def test_08_media_routes_and_security(self):
        # Test video playback stream endpoint
        response_video = self.client.get("/media/test_mock_job_999/highlight.mp4")
        self.assertEqual(response_video.status_code, 200)
        self.assertEqual(response_video.mimetype, "video/mp4")

        # Test score plot endpoint
        response_plot = self.client.get("/media/test_mock_job_999/scores.png")
        self.assertEqual(response_plot.status_code, 200)
        self.assertEqual(response_plot.mimetype, "image/png")

        # Test disallowed file (e.g. segments.json or sensitive models)
        response_disallowed = self.client.get("/media/test_mock_job_999/segments.json")
        self.assertEqual(response_disallowed.status_code, 403)

        # Test directory traversal attack prevention
        self.assertFalse(is_safe_job_id("../models"))
        self.assertFalse(is_safe_job_id("job_123/../../etc"))
        self.assertTrue(is_safe_job_id("job_1726000000_a1b2c3"))

        response_traversal = self.client.get("/media/..%2f..%2fmodels/highlight.mp4")
        self.assertIn(response_traversal.status_code, [400, 404])


if __name__ == "__main__":
    unittest.main(verbosity=2)
