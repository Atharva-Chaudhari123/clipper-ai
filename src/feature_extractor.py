"""
src/feature_extractor.py
------------------------
Phase 3: Pretrained ResNet-50 Feature Extraction for TVSum Video Frames.

Converts sampled video frames into 2048-dimensional deep visual feature vectors.
ResNet-50 is frozen and used solely as a fixed feature extractor.
The final 1000-class classification head is replaced with nn.Identity(),
yielding the 2048-dimensional feature representation from the global average pooling layer.
"""

import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


def get_device_info() -> tuple[torch.device, str]:
    """
    Detects hardware, verifies CUDA availability, and prints system diagnostic info.
    
    Returns:
        tuple[torch.device, str]: The PyTorch device and the device name string.
    """
    cuda_available = torch.cuda.is_available()
    py_torch_ver = torch.__version__
    cuda_ver = torch.version.cuda if cuda_available else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    device_str = "cuda" if cuda_available else "cpu"
    device = torch.device(device_str)

    print("=" * 60)
    print("HARDWARE & PYTORCH DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch version: {py_torch_ver}")
    print(f"CUDA available:  {cuda_available}")
    print(f"CUDA version:    {cuda_ver}")
    print(f"GPU:             {gpu_name}")
    print(f"Device:          {device_str}")
    print("=" * 60)

    if not cuda_available:
        print("[WARNING] CUDA is NOT available! Running on CPU will be slower.")
    
    return device, gpu_name


class ResNet50FeatureExtractor(nn.Module):
    """
    Pretrained ResNet-50 visual feature extractor.
    
    The classification head (fc) is replaced by nn.Identity(), so forward() outputs
    the 2048-dimensional feature representation from the adaptive average pooling layer.
    All model parameters are frozen (requires_grad = False).
    """

    def __init__(self, device: torch.device):
        super(ResNet50FeatureExtractor, self).__init__()
        self.device = device

        # Load pretrained ResNet-50 using official torchvision weights
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        except AttributeError:
            self.model = models.resnet50(pretrained=True)

        # Replace final 1000-class classification layer with Identity
        # Output before fc is [B, 2048]
        self.model.fc = nn.Identity()

        # Freeze all parameters - we do NOT train ResNet-50
        for param in self.model.parameters():
            param.requires_grad = False

        # Set model strictly to evaluation mode (disables dropout & batchnorm updates)
        self.model.eval()
        self.model.to(self.device)

        # Standard ImageNet preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Loads and preprocesses a single image file."""
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            tensor_img = self.transform(rgb_img)
        return tensor_img

    def extract_features(
        self,
        frame_paths: list[str],
        initial_batch_size: int = 8
    ) -> tuple[torch.Tensor, int]:
        """
        Extracts 2048-d features for a list of image paths with automatic OOM recovery.

        Parameters:
            frame_paths (list[str]): Ordered list of image file paths.
            initial_batch_size (int): Starting batch size (default: 8).

        Returns:
            tuple[torch.Tensor, int]: Feature tensor of shape [N, 2048] on CPU and final batch size used.
        """
        batch_size = initial_batch_size
        num_frames = len(frame_paths)
        
        while batch_size >= 1:
            try:
                features_list = []
                for start_idx in range(0, num_frames, batch_size):
                    end_idx = min(start_idx + batch_size, num_frames)
                    batch_paths = frame_paths[start_idx:end_idx]

                    # Load and preprocess batch images
                    batch_tensors = [self.preprocess_image(p) for p in batch_paths]
                    batch_tensor = torch.stack(batch_tensors, dim=0).to(self.device)

                    # Forward pass without gradients
                    with torch.no_grad():
                        batch_features = self.model(batch_tensor)  # Shape: [B, 2048]

                    # Transfer to CPU immediately to keep GPU VRAM footprint minimal
                    features_list.append(batch_features.cpu())

                    # Release GPU batch tensor reference
                    del batch_tensor, batch_features

                all_features = torch.cat(features_list, dim=0)  # Shape: [N, 2048]
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                return all_features, batch_size

            except torch.cuda.OutOfMemoryError:
                print(f"[CUDA OOM] Batch size {batch_size} exceeded VRAM. Reducing batch size...")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                batch_size //= 2
                if batch_size < 1:
                    raise RuntimeError("CUDA Out Of Memory even with batch_size=1!")

        raise RuntimeError("Failed to extract features with any batch size.")


def extract_and_save_video_features(
    video_id: str,
    frames_base_dir: str,
    features_output_dir: str,
    initial_batch_size: int = 8
) -> dict:
    """
    Coordinates feature extraction for one video, preserving temporal order and saving metadata.

    Parameters:
        video_id (str): TVSum video ID.
        frames_base_dir (str): Directory containing frames/<video_id>/ and metadata.csv.
        features_output_dir (str): Directory where features/ will be stored.
        initial_batch_size (int): Starting batch size.

    Returns:
        dict: Summary statistics and paths.
    """
    device, gpu_name = get_device_info()

    video_frames_dir = os.path.join(frames_base_dir, video_id)
    metadata_csv_path = os.path.join(video_frames_dir, "metadata.csv")

    if not os.path.isfile(metadata_csv_path):
        raise FileNotFoundError(f"Missing metadata.csv at: {metadata_csv_path}")

    # Read metadata to guarantee exact sequential ordering: segment 0 to 65
    metadata_rows = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_rows.append(row)

    # Sort rows by segment_index integer to be 100% sure ordering is preserved
    metadata_rows.sort(key=lambda r: int(r["segment_index"]))

    frame_paths = [os.path.join(video_frames_dir, r["frame_path"]) for r in metadata_rows]
    for fp in frame_paths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Frame file not found: {fp}")

    print(f"Initializing frozen ResNet-50 feature extractor on {device}...")
    extractor = ResNet50FeatureExtractor(device=device)

    print(f"Extracting 2048-d visual features for {len(frame_paths)} frames of '{video_id}'...")
    features, final_batch_size = extractor.extract_features(
        frame_paths,
        initial_batch_size=initial_batch_size
    )

    os.makedirs(features_output_dir, exist_ok=True)

    # Save feature tensor to data/processed/features/<video_id>.pt
    features_file_path = os.path.join(features_output_dir, f"{video_id}.pt")
    torch.save(features, features_file_path)

    # Save feature metadata to data/processed/features/<video_id>_metadata.csv
    feature_meta_csv_path = os.path.join(features_output_dir, f"{video_id}_metadata.csv")
    with open(feature_meta_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_index", "sample_time", "annotation_start", "annotation_end", "frame_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({
                "segment_index": row["segment_index"],
                "sample_time": row["sample_time"],
                "annotation_start": row["annotation_start"],
                "annotation_end": row["annotation_end"],
                "frame_path": row["frame_path"]
            })

    # Statistical and numerical checks
    has_nan = bool(torch.isnan(features).any().item())
    has_inf = bool(torch.isinf(features).any().item())

    # Reload check
    reloaded_tensor = torch.load(features_file_path)
    reload_passed = bool((reloaded_tensor.shape == features.shape) and torch.equal(reloaded_tensor, features))

    # First vector stats
    first_vec = features[0]
    first_10_vals = first_vec[:10].tolist()

    summary = {
        "video_id": video_id,
        "num_frames": len(frame_paths),
        "num_features": features.shape[0],
        "feature_dim": features.shape[1],
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "batch_size_used": final_batch_size,
        "features_file": features_file_path,
        "metadata_file": feature_meta_csv_path,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "reload_passed": reload_passed,
        "first_vec_10": [round(x, 4) for x in first_10_vals],
        "first_vec_min": float(first_vec.min().item()),
        "first_vec_max": float(first_vec.max().item()),
        "first_vec_mean": float(first_vec.mean().item()),
        "first_vec_std": float(first_vec.std().item()),
        "matrix_min": float(features.min().item()),
        "matrix_max": float(features.max().item()),
        "matrix_mean": float(features.mean().item()),
        "matrix_std": float(features.std().item()),
    }
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: Pretrained ResNet-50 Feature Extraction")
    parser.add_argument("--video_id", type=str, default="akI8YFjEmUw")
    parser.add_argument("--frames_dir", type=str, default=r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\frames")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\mahin\OneDrive\Desktop\New folder\data\processed\features")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    results = extract_and_save_video_features(
        video_id=args.video_id,
        frames_base_dir=args.frames_dir,
        features_output_dir=args.output_dir,
        initial_batch_size=args.batch_size
    )

    print("\nFeature extraction completed successfully!")
    print(f"Features saved: {results['features_file']}")
    print(f"Metadata saved: {results['metadata_file']}")
    print(f"Shape: {results['shape']}, Batch size: {results['batch_size_used']}")
