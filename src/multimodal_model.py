"""
src/multimodal_model.py
-----------------------
Multimodal Deep Learning Architecture for Video Highlight Detection.

Fuses:
1. Pretrained ResNet-50 visual features (2048-D)
2. Whisper speech transcript embeddings (384-D)

Architecture:
Visual Features [B, T, 2048]  --> Linear(2048, 512) + LayerNorm + ReLU + Dropout
Text Features   [B, T, 384]   --> Linear(384, 256)  + LayerNorm + ReLU + Dropout
                                              |
                                      Concatenation [B, T, 768]
                                              |
                               2-layer Bidirectional LSTM (hidden_size=256)
                                              |
                                   Temporal State [B, T, 512]
                                              |
                                        Scoring Head
                           Linear(512, 128) + ReLU + Dropout + Linear(128, 1)
                                              |
                                           Sigmoid
                                              |
                                 Importance Scores [B, T] in [0, 1]
"""

import torch
import torch.nn as nn
from typing import Optional


class MultimodalHighlightBiLSTM(nn.Module):
    """
    Multimodal sequence scoring model combining visual and text modalities.
    """
    def __init__(
        self,
        visual_dim: int = 2048,
        text_dim: int = 384,
        proj_visual_dim: int = 512,
        proj_text_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MultimodalHighlightBiLSTM, self).__init__()

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.fused_dim = proj_visual_dim + proj_text_dim  # 768

        # Modality projection branches
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, proj_visual_dim),
            nn.LayerNorm(proj_visual_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_text_dim),
            nn.LayerNorm(proj_text_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=self.fused_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Importance Scoring Head
        lstm_out_dim = hidden_dim * 2  # Bidirectional = 512
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            visual_features: [B, T, 2048]
            text_features:   [B, T, 384]
            mask:            [B, T] optional boolean mask (True for valid, False for pad)

        Returns:
            scores: [B, T] tensor with predicted segment scores in [0.0, 1.0]
        """
        # Ensure 3D batch dimension
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)

        # 1. Project individual modalities
        v_proj = self.visual_proj(visual_features)  # [B, T, 512]
        t_proj = self.text_proj(text_features)      # [B, T, 256]

        # 2. Concatenate into fused multimodal features
        fused = torch.cat([v_proj, t_proj], dim=-1)  # [B, T, 768]

        # 3. Temporal BiLSTM
        lstm_out, _ = self.lstm(fused)               # [B, T, 512]

        # 4. Importance score prediction
        scores = self.fc(lstm_out).squeeze(-1)       # [B, T]

        # 5. Apply sequence mask if provided
        if mask is not None:
            scores = scores * mask.float()

        return scores

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
