"""
src/face_tracker.py
-------------------
Smart Face Tracking & Dynamic 9:16 Video Reframing Engine.

Features:
- High-accuracy neural face detection via OpenCV YuNet (ONNX) with Haar Cascade fallback
- Exponential moving average (EMA) camera smoothing to prevent jerky panning
- Speaker bounding box centering with safety boundaries
- Native FFmpeg crop command generation or frame-by-frame reframing
"""

import os
import logging
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger("FaceTracker")

YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet.onnx")
HAAR_XML_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")


class SmartFaceTracker:
    def __init__(self, smoothing: float = 0.15, score_threshold: float = 0.55):
        """
        Initializes the face tracker.
        :param smoothing: Weight of new face center (0.1 - 0.2 produces smooth cinema-like panning).
        :param score_threshold: Confidence threshold for face detection.
        """
        self.smoothing = smoothing
        self.score_threshold = score_threshold
        self.detector = None
        self.use_yunet = False
        self.last_center: Optional[Tuple[int, int]] = None

        self._init_detector()

    def _init_detector(self):
        """Attempts to load YuNet deep learning detector, falling back to Haar if available."""
        if os.path.isfile(YUNET_MODEL_PATH) and hasattr(cv2, "FaceDetectorYN"):
            try:
                # Initialize with placeholder input shape
                self.detector = cv2.FaceDetectorYN.create(
                    YUNET_MODEL_PATH,
                    "",
                    (320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.use_yunet = True
                logger.info("SmartFaceTracker initialized with high-precision YuNet neural detector.")
                return
            except Exception as e:
                logger.warning(f"Could not initialize YuNet ({e}).")

        # Fallback to Haar Cascade if supported by cv2 build
        if hasattr(cv2, "CascadeClassifier") and os.path.isfile(HAAR_XML_PATH):
            try:
                self.detector = cv2.CascadeClassifier(HAAR_XML_PATH)
                self.use_yunet = False
                logger.info("SmartFaceTracker initialized with Haar cascade detector.")
                return
            except Exception as e:
                logger.warning(f"Could not load Haar cascade ({e}).")

        logger.info("SmartFaceTracker: No neural face detector available. Will fallback to center cropping.")

    def detect_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the most prominent face in the frame.
        Returns bounding box (x, y, w, h) or None if no face detected.
        """
        if self.detector is None:
            return None

        h, w = frame_bgr.shape[:2]

        if self.use_yunet:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(frame_bgr)
                if faces is not None and len(faces) > 0:
                    # Sort by face bounding box area (w * h) to pick the primary subject
                    # YuNet output layout: [x, y, w, h, x_re, y_re, ...]
                    best_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                    fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                    return (max(0, fx), max(0, fy), max(1, fw), max(1, fh))
            except Exception as e:
                logger.debug(f"YuNet detection error: {e}")
                return None
        else:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    return (int(x), int(y), int(fw), int(fh))
            except Exception as e:
                logger.debug(f"Haar detection error: {e}")
                return None

        return None

    def update_camera_center(self, frame_bgr: np.ndarray) -> Tuple[int, int]:
        """
        Updates camera center using exponential moving average smoothing.
        Returns smoothed (cx, cy).
        """
        h, w = frame_bgr.shape[:2]
        default_cx = w // 2
        default_cy = h // 2

        face = self.detect_face(frame_bgr)
        if face is not None:
            x, y, fw, fh = face
            target_cx = x + fw // 2
            target_cy = y + fh // 2

            if self.last_center is None:
                self.last_center = (target_cx, target_cy)
            else:
                lx, ly = self.last_center
                self.last_center = (
                    int(round(lx + (target_cx - lx) * self.smoothing)),
                    int(round(ly + (target_cy - ly) * self.smoothing))
                )
        else:
            if self.last_center is None:
                self.last_center = (default_cx, default_cy)
            else:
                # Slowly drift back toward center if no face detected
                lx, ly = self.last_center
                self.last_center = (
                    int(round(lx + (default_cx - lx) * 0.05)),
                    int(round(ly + (default_cy - ly) * 0.05))
                )

        return self.last_center

    def compute_crop_box(self, frame_bgr: np.ndarray, target_aspect_ratio: str = "9:16") -> Tuple[int, int, int, int]:
        """
        Computes (x0, y0, crop_w, crop_h) for reframing.
        """
        src_h, src_w = frame_bgr.shape[:2]
        if target_aspect_ratio == "9:16":
            target_ratio = 9.0 / 16.0
        elif target_aspect_ratio == "1:1":
            target_ratio = 1.0
        elif target_aspect_ratio == "4:5":
            target_ratio = 4.0 / 5.0
        else:
            target_ratio = 9.0 / 16.0

        if target_ratio < (src_w / src_h):
            crop_h = src_h
            crop_w = int(round(crop_h * target_ratio))
        else:
            crop_w = src_w
            crop_h = int(round(crop_w / target_ratio))

        crop_w = max(2, crop_w - (crop_w % 2))
        crop_h = max(2, crop_h - (crop_h % 2))

        cx, cy = self.update_camera_center(frame_bgr)
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))

        return (x0, y0, crop_w, crop_h)
