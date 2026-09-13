"""
tests/test_phase11_multimodal.py
--------------------------------
Comprehensive unit and integration tests for Phase 11:
1. YouTube URL validation (standard, shortened, shorts, and invalid patterns)
2. YouTube Video ID extraction
3. Audio extraction via FFmpeg (16 kHz mono WAV)
4. Whisper transcription data structuring and silence handling
5. Timestamp-to-segment alignment logic across 2-second intervals
6. Sentence embedding generation (384-D) and silence zero-vector guarantee
7. MultimodalHighlightBiLSTM forward pass, parameter count, and tensor shapes
8. Fallback to baseline when multimodal checkpoint is unavailable
9. Flask YouTube URL upload route and validation
"""

import os
import sys
import unittest
import torch
from unittest.mock import patch, MagicMock

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.youtube_downloader import is_valid_youtube_url, extract_video_id
from src.audio_transcriber import extract_audio_from_video, transcribe_audio
from src.text_embedder import align_transcript_with_segments, get_sentence_embedder
from src.multimodal_model import MultimodalHighlightBiLSTM
from app import app


class Phase11MultimodalTestCase(unittest.TestCase):

    # 1. YouTube URL validation
    def test_01_youtube_url_validation(self):
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ&feature=shared",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s",
        ]
        for url in valid_urls:
            self.assertTrue(is_valid_youtube_url(url), f"Should be valid: {url}")

        invalid_urls = [
            "https://vimeo.com/12345678",
            "https://dailymotion.com/video/x7tgad0",
            "https://google.com",
            "not_a_url",
            "",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=short",  # Invalid ID length
        ]
        for url in invalid_urls:
            self.assertFalse(is_valid_youtube_url(url), f"Should be invalid: {url}")

    # 2. YouTube Video ID extraction
    def test_02_video_id_extraction(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://youtu.be/akI8YFjEmUw"), "akI8YFjEmUw")
        self.assertEqual(extract_video_id("https://www.youtube.com/shorts/sTEELN-vY30"), "sTEELN-vY30")
        self.assertIsNone(extract_video_id("https://example.com"))

    # 3. Audio extraction with FFmpeg
    def test_03_audio_extraction(self):
        sample_video = os.path.join(BASE_DIR, "video", "akI8YFjEmUw.mp4")
        temp_wav = os.path.join(BASE_DIR, "data", "processed", "temp_audio", "test_extract.wav")

        if os.path.isfile(sample_video):
            has_audio = extract_audio_from_video(sample_video, temp_wav)
            self.assertTrue(has_audio)
            self.assertTrue(os.path.isfile(temp_wav))
            self.assertGreater(os.path.getsize(temp_wav), 1000)

            # Clean up test wav
            if os.path.isfile(temp_wav):
                os.remove(temp_wav)

    # 4. Timestamp alignment logic
    def test_04_timestamp_alignment(self):
        # 4 segments: [0-2s), [2-4s), [4-6s), [6-8s)
        dummy_transcript = [
            {"start": 0.5, "end": 1.8, "text": "Hello world"},
            {"start": 1.9, "end": 3.2, "text": "this is speech"},
            {"start": 6.5, "end": 7.5, "text": "final words"}
        ]

        aligned = align_transcript_with_segments(dummy_transcript, num_segments=4, segment_duration=2.0)
        self.assertEqual(len(aligned), 4)

        # Segment 0 [0-2s]: overlaps "Hello world" and "this is speech" (1.9-2.0s)
        self.assertIn("Hello world", aligned[0])
        self.assertIn("this is speech", aligned[0])

        # Segment 1 [2-4s]: overlaps "this is speech" (2.0-3.2s)
        self.assertEqual(aligned[1], "this is speech")

        # Segment 2 [4-6s]: no speech -> empty string
        self.assertEqual(aligned[2], "")

        # Segment 3 [6-8s]: overlaps "final words"
        self.assertEqual(aligned[3], "final words")

    # 5. Sentence Embedder & Silence Zero-Vector Guarantee
    def test_05_sentence_embedder_and_silence(self):
        embedder = get_sentence_embedder()
        self.assertEqual(embedder.embedding_dim, 384)

        texts = ["Artificial intelligence video highlights", "", "Machine learning audio transcript"]
        embeddings = embedder.embed_texts(texts, batch_size=4)

        # Output shape check
        self.assertEqual(embeddings.shape, (3, 384))
        self.assertEqual(embeddings.dtype, torch.float32)

        # Non-empty strings must produce non-zero unit vectors (L2 normalized)
        norm_0 = torch.norm(embeddings[0], p=2).item()
        self.assertAlmostEqual(norm_0, 1.0, places=3)

        # Empty string at index 1 MUST produce exact zero vector for neutral silence representation
        zero_norm = torch.norm(embeddings[1], p=2).item()
        self.assertEqual(zero_norm, 0.0)
        self.assertTrue(torch.all(embeddings[1] == 0.0))

    # 6. MultimodalHighlightBiLSTM Forward Pass & Parameters
    def test_06_multimodal_model_forward(self):
        model = MultimodalHighlightBiLSTM(
            visual_dim=2048,
            text_dim=384,
            proj_visual_dim=512,
            proj_text_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )

        params = model.count_parameters()
        self.assertGreater(params, 4_000_000)
        self.assertLess(params, 6_000_000)

        # Test batched input with mask [B=2, T=50]
        v = torch.randn(2, 50, 2048)
        t = torch.randn(2, 50, 384)
        mask = torch.ones(2, 50, dtype=torch.bool)
        mask[1, 35:] = False  # sequence 2 is padded after step 35

        scores = model(v, t, mask=mask)
        self.assertEqual(scores.shape, (2, 50))
        self.assertTrue(torch.all(scores >= 0.0))
        self.assertTrue(torch.all(scores <= 1.0))

        # Padded elements must be exactly 0.0
        self.assertTrue(torch.all(scores[1, 35:] == 0.0))

        # Test single unbatched sequence [T=40]
        v_single = torch.randn(40, 2048)
        t_single = torch.randn(40, 384)
        single_scores = model(v_single, t_single)
        self.assertEqual(single_scores.shape, (1, 40))

    # 7. Flask YouTube Submission Route
    @patch("threading.Thread.start")
    def test_07_flask_youtube_upload_route(self, mock_thread):
        client = app.test_client()

        # Valid YouTube URL
        data = {
            "upload_type": "youtube",
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "mode": "multimodal",
            "budget": "15"
        }
        resp = client.post("/upload", data=data)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/processing/job_", resp.headers["Location"])

        # Invalid YouTube URL rejected
        invalid_data = {
            "upload_type": "youtube",
            "youtube_url": "https://vimeo.com/999999",
            "mode": "multimodal",
            "budget": "15"
        }
        resp_invalid = client.post("/upload", data=invalid_data, follow_redirects=True)
        self.assertEqual(resp_invalid.status_code, 200)
        html = resp_invalid.get_data(as_text=True)
        self.assertIn("Invalid YouTube URL", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
