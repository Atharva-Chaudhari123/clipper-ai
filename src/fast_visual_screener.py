"""
src/fast_visual_screener.py
---------------------------
Phase 13: Fast Visual Pre-Screening for Long Videos.

Samples video frames directly in memory at coarse intervals (default: 1 frame per 2s segment)
and runs batched ResNet-50 feature extraction on GPU without writing thousands of
individual JPG files to disk. Evaluates baseline BiLSTM temporal scores and visual
dynamics to produce a continuous visual saliency curve.
"""

import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from torchvision import models, transforms
from typing import Dict, Any, Optional

logger = logging.getLogger("FastVisualScreener")


class FastVisualScreener:
    def __init__(self, baseline_model_path: str = "models/best_bilstm.pt", device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize ResNet-50 for fast in-memory inference
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.to(self.device)
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load baseline BiLSTM if available
        self.bilstm = None
        if os.path.isfile(baseline_model_path):
            try:
                from src.lstm_model import VideoHighlightBiLSTM
                model = VideoHighlightBiLSTM(input_dim=2048, hidden_dim=256, num_layers=2)
                state = torch.load(baseline_model_path, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.bilstm = model
                logger.info("Loaded baseline BiLSTM for fast visual screening.")
            except Exception as e:
                logger.warning(f"Could not load baseline BiLSTM: {str(e)}")

    @torch.no_grad()
    def screen_video(
        self,
        video_path: str,
        sample_interval_sec: float = 2.0,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Extracts visual features and computes visual activity scores.

        Args:
            video_path: Path to MP4 video
            sample_interval_sec: Sampling interval in seconds (default: 2.0s)
            batch_size: Batch size for GPU inference (default: 16)

        Returns:
            Dict containing:
            - segment_times: list of timestamps (seconds)
            - visual_features: torch.Tensor of shape (T, 2048)
            - visual_scores: np.ndarray of shape (T,) normalized to [0, 1]
            - duration_sec: total duration
            - num_segments: T
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        if duration_sec <= 0.0:
            cap.release()
            return {
                "segment_times": [],
                "visual_features": torch.empty((0, 2048)),
                "visual_scores": np.array([]),
                "duration_sec": 0.0,
                "num_segments": 0
            }

        num_segments = max(1, int(np.ceil(duration_sec / sample_interval_sec)))
        sample_times = [min(i * sample_interval_sec + (sample_interval_sec / 2.0), max(0.0, duration_sec - 0.1)) for i in range(num_segments)]

        logger.info(f"Screening video: {duration_sec:.1f}s, {num_segments} segments at {sample_interval_sec}s interval...")

        # Extract frames in batches
        feature_list = []
        batch_tensors = []

        for i, t in enumerate(sample_times):
            frame_idx = min(int(round(t * fps)), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                tensor = self.transform(pil_img)
            else:
                # Fallback: zero tensor
                tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

            batch_tensors.append(tensor)

            if len(batch_tensors) == batch_size or i == num_segments - 1:
                batch = torch.stack(batch_tensors).to(self.device)
                feats = self.resnet(batch)  # shape (B, 2048)
                feature_list.append(feats.cpu())
                batch_tensors = []

        cap.release()

        if not feature_list:
            return {
                "segment_times": sample_times,
                "visual_features": torch.empty((0, 2048)),
                "visual_scores": np.array([]),
                "duration_sec": duration_sec,
                "num_segments": 0
            }

        visual_features = torch.cat(feature_list, dim=0)  # shape (T, 2048)

        # Compute visual dynamic change: cosine distance between adjacent frames
        norm_feats = visual_features / (visual_features.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        cos_sim = (norm_feats[:-1] * norm_feats[1:]).sum(dim=-1)
        cos_dist = 1.0 - cos_sim
        # Pad first element
        dynamics = torch.cat([cos_dist[:1], cos_dist]).numpy()
        dyn_min, dyn_max = np.min(dynamics), np.max(dynamics)
        if dyn_max > dyn_min:
            dynamics = (dynamics - dyn_min) / (dyn_max - dyn_min)
        else:
            dynamics = np.zeros_like(dynamics)

        # Run baseline BiLSTM if loaded
        if self.bilstm is not None:
            inp = visual_features.unsqueeze(0).to(self.device)  # shape (1, T, 2048)
            raw_scores = self.bilstm(inp).squeeze().cpu().numpy()  # shape (T,)
            if raw_scores.ndim == 0:
                raw_scores = np.array([float(raw_scores)])
            # Fuse BiLSTM predictions with scene dynamics
            visual_scores = 0.7 * raw_scores + 0.3 * dynamics
        else:
            visual_scores = dynamics

        # Temporal smoothing with 3-window moving average
        if len(visual_scores) >= 3:
            k = np.ones(3) / 3.0
            visual_scores = np.convolve(visual_scores, k, mode="same")

        return {
            "segment_times": sample_times,
            "visual_features": visual_features,
            "visual_scores": visual_scores,
            "duration_sec": round(duration_sec, 2),
            "num_segments": num_segments
        }
