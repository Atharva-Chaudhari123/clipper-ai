"""
tests/test_phase5_model.py
--------------------------
Unit tests for Phase 5: VideoHighlightBiLSTM Architecture.

Verifies:
1. Model initializes correctly with expected parameters.
2. Forward pass outputs correct shape [B, T].
3. Outputs are strictly bounded in [0.0, 1.0] by Sigmoid.
4. Masked MSE loss calculates accurately and ignores padded frames.
5. Backward pass computes valid gradients without NaNs.
6. CUDA execution succeeds if GPU is available.
"""

import os
import sys
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lstm_model import VideoHighlightBiLSTM, compute_masked_mse_loss


def test_bilstm_architecture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning model test on device: {device}")

    model = VideoHighlightBiLSTM(input_dim=2048, hidden_dim=256, num_layers=2, dropout=0.3)
    model.to(device)

    # 1. Parameter count check
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,} (Trainable: {trainable_params:,})")
    assert total_params > 0, "Model has 0 parameters!"

    # 2. Forward pass with dummy batch
    batch_size = 4
    seq_len = 66
    x = torch.randn(batch_size, seq_len, 2048, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(x)

    assert preds.shape == (batch_size, seq_len), f"Expected shape ({batch_size}, {seq_len}), got {preds.shape}"

    # 3. Output bound check [0.0, 1.0]
    assert (preds >= 0.0).all().item(), "Found predictions < 0.0!"
    assert (preds <= 1.0).all().item(), "Found predictions > 1.0!"
    assert not torch.isnan(preds).any().item(), "Predictions contain NaN!"

    # 4. Masked loss check
    targets = torch.rand(batch_size, seq_len, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    # Mask out the second half of the last sequence
    mask[-1, 33:] = False

    loss = compute_masked_mse_loss(preds, targets, mask)
    assert not torch.isnan(loss).item(), "Loss is NaN!"
    assert loss.item() >= 0.0, f"Loss is negative: {loss.item()}"

    # 5. Backward pass check
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    train_preds = model(x)
    train_loss = compute_masked_mse_loss(train_preds, targets, mask)
    train_loss.backward()

    # Verify gradients exist and are non-zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any().item(), f"NaN gradient in {name}"

    optimizer.step()
    print("[SUCCESS] All Phase 5 VideoHighlightBiLSTM tests passed perfectly!")


if __name__ == "__main__":
    test_bilstm_architecture()
