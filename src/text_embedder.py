"""
src/text_embedder.py
--------------------
Aligns Whisper speech transcripts with 2-second video segments and
generates 384-dimensional text embeddings using all-MiniLM-L6-v2.

Features:
- Precise timestamp intersection alignment between speech intervals and video segments.
- Handling of silence/no-speech segments using zero vectors (torch.zeros(384)).
- Batched inference with attention-masked mean pooling and L2 normalization.
- CUDA acceleration with CPU fallback.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("TextEmbedder")

# Global singleton to avoid reloading transformer weights
_LOADED_EMBEDDER = None


def align_transcript_with_segments(
    transcript_segments: List[Dict[str, Any]],
    num_segments: int,
    segment_duration: float = 2.0
) -> List[str]:
    """
    Aligns Whisper timestamped transcript segments with discrete video segments.

    Args:
        transcript_segments: List of [{"start": float, "end": float, "text": str}]
        num_segments: Total number of 2-second video segments (T)
        segment_duration: Duration of each video segment (default: 2.0 seconds)

    Returns:
        List of strings of length `num_segments`, where each element contains the
        spoken text overlapping that specific 2-second segment, or "" if silent.
    """
    aligned_texts = ["" for _ in range(num_segments)]
    if not transcript_segments:
        return aligned_texts

    for i in range(num_segments):
        seg_start = i * segment_duration
        seg_end = (i + 1) * segment_duration

        overlapping_texts = []
        for ts in transcript_segments:
            w_start = ts.get("start", 0.0)
            w_end = ts.get("end", 0.0)
            text = ts.get("text", "").strip()

            # Check for temporal overlap between [seg_start, seg_end] and [w_start, w_end]
            if max(seg_start, w_start) < min(seg_end, w_end):
                if text and text not in overlapping_texts:
                    overlapping_texts.append(text)

        aligned_texts[i] = " ".join(overlapping_texts).strip()

    return aligned_texts


class SentenceEmbedder:
    """
    Produces 384-dimensional dense sentence embeddings using all-MiniLM-L6-v2.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing SentenceEmbedder '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size  # 384

    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Embeds a list of text strings into [N, 384] tensor.
        Empty strings ("") are assigned a neutral zero-vector torch.zeros(384).
        """
        num_texts = len(texts)
        if num_texts == 0:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        # Output tensor initialized to zeros for silence handling
        all_embeddings = torch.zeros((num_texts, self.embedding_dim), dtype=torch.float32)

        # Separate non-empty texts to embed efficiently
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_indices:
            return all_embeddings

        non_empty_texts = [texts[i] for i in non_empty_indices]

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(non_empty_texts), batch_size):
                batch_slice = slice(i, i + batch_size)
                batch_texts = non_empty_texts[batch_slice]
                target_indices = non_empty_indices[batch_slice]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                model_output = self.model(**encoded)
                sentence_embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                # L2 normalize
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

                all_embeddings[target_indices] = sentence_embeddings.cpu()

        return all_embeddings

    def encode(self, text_or_texts):
        """Convenience method for embedding single string or list of strings."""
        if isinstance(text_or_texts, str):
            res = self.embed_texts([text_or_texts])
            return res[0]
        return self.embed_texts(list(text_or_texts))



def get_sentence_embedder(device: Optional[str] = None) -> SentenceEmbedder:
    """Returns singleton SentenceEmbedder instance."""
    global _LOADED_EMBEDDER
    if _LOADED_EMBEDDER is None:
        _LOADED_EMBEDDER = SentenceEmbedder(device=device)
    return _LOADED_EMBEDDER
