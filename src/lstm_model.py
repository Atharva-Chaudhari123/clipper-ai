"""
src/lstm_model.py
-----------------
Phase 5: Bidirectional LSTM Model Architecture for Video Highlight Scoring.

A PyTorch neural network that takes a temporal sequence of 2048-d visual feature vectors
and models forward and backward temporal dependencies across video segments.
Outputs an importance score in [0.0, 1.0] for every 2-second segment.
"""

import torch
import torch.nn as nn


class VideoHighlightBiLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal video highlight prediction.

    Parameters:
        input_dim (int): Input feature dimension per segment (default: 2048 from ResNet-50).
        hidden_dim (int): Hidden units per directional LSTM layer (default: 256 -> 512 bidirectional).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        dropout (float): Dropout rate for regularization (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(VideoHighlightBiLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        # Input shape: [Batch, Seq_Len, 2048]
        # Output shape: [Batch, Seq_Len, hidden_dim * 2] (512 features per step)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Non-linear projection scoring head
        # Maps 512-d bidirectional hidden state to 1 importance score in [0.0, 1.0]
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Guarantees scores are strictly bounded in [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x (torch.Tensor): Input feature tensor of shape [B, T, 2048].

        Returns:
            torch.Tensor: Predicted segment importance scores of shape [B, T].
        """
        # lstm_out shape: [B, T, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)

        # scores shape: [B, T, 1]
        scores = self.scoring_head(lstm_out)

        # Squeeze to [B, T]
        return scores.squeeze(-1)


def compute_masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes Mean Squared Error loss strictly over valid non-padded segment positions.

    Parameters:
        predictions (torch.Tensor): Predicted scores of shape [B, T].
        targets (torch.Tensor): Ground-truth scores of shape [B, T].
        mask (torch.Tensor): Boolean mask of shape [B, T] (True = valid frame, False = padding).

    Returns:
        torch.Tensor: Scalar MSE loss averaged over all valid frames.
    """
    diff_sq = (predictions - targets) ** 2
    masked_diff = diff_sq * mask.float()
    total_valid = mask.float().sum()
    if total_valid > 0:
        return masked_diff.sum() / total_valid
    return masked_diff.sum()
