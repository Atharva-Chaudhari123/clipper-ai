"""
src/reel_selector.py
--------------------
Phase 13: Multi-Reel Selector with Temporal Non-Maximum Suppression (NMS),
Sentence-Boundary Snapping, and Multimodal BiLSTM Scoring.

Features:
- Multimodal importance score evaluation using models/best_multimodal_bilstm.pt
- Sentence-boundary alignment from Whisper segment timestamps (avoids mid-sentence cuts)
- Configurable reel durations (15–30s, 30–60s, 60–90s, Auto)
- Configurable reel count (Auto, 3, 5, 10)
- Temporal overlap suppression (NMS) to guarantee content diversity
- Minimum quality thresholding
"""

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("ReelSelector")


def snap_boundaries_to_sentences(
    raw_start: float,
    raw_end: float,
    speech_segments: List[Dict[str, Any]],
    target_min: float = 30.0,
    target_max: float = 60.0
) -> Tuple[float, float, str]:
    """
    Adjusts start and end timestamps so they snap cleanly to sentence / phrase boundaries.
    Avoids cutting speech in the middle of a sentence.
    """
    if not speech_segments:
        return raw_start, raw_end, ""

    # Filter segments that fall within or near the raw window
    relevant_segs = [
        s for s in speech_segments
        if s["end"] >= raw_start - 3.0 and s["start"] <= raw_end + 3.0
    ]

    if not relevant_segs:
        return raw_start, raw_end, ""

    # Find closest sentence start near raw_start
    best_start = raw_start
    min_start_diff = float("inf")
    for s in relevant_segs:
        diff = abs(s["start"] - raw_start)
        if diff < min_start_diff and diff <= 4.0:
            min_start_diff = diff
            best_start = s["start"]

    # Find closest sentence end that produces duration within [target_min, target_max]
    best_end = raw_end
    candidates_end = []
    for s in relevant_segs:
        cand_end = s["end"]
        cand_dur = cand_end - best_start
        if target_min <= cand_dur <= target_max:
            candidates_end.append((cand_end, abs(cand_end - raw_end)))

    if candidates_end:
        candidates_end.sort(key=lambda x: x[1])
        best_end = candidates_end[0][0]
    else:
        # Fallback: keep duration within limits
        dur = raw_end - best_start
        if dur > target_max:
            best_end = best_start + target_max
        elif dur < target_min:
            best_end = best_start + target_min

    # Collect transcript text for snapped range
    included_text = []
    for s in relevant_segs:
        if s["start"] >= best_start - 0.5 and s["end"] <= best_end + 0.5:
            included_text.append(s["text"])

    transcript_snippet = " ".join(included_text)
    return round(best_start, 2), round(best_end, 2), transcript_snippet


def score_candidate_multimodal(
    candidate: Dict[str, Any],
    all_visual_features: torch.Tensor,
    visual_sample_interval: float = 2.0,
    multimodal_model_path: str = "models/best_multimodal_bilstm.pt",
    device: Optional[str] = None
) -> float:
    """
    Computes multimodal BiLSTM score for a single candidate window.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    start_sec = candidate["start"]
    end_sec = candidate["end"]

    # Slice visual features corresponding to candidate window
    start_idx = max(0, int(start_sec // visual_sample_interval))
    end_idx = min(all_visual_features.shape[0], int(np.ceil(end_sec / visual_sample_interval)))

    if end_idx <= start_idx:
        return candidate.get("pre_score", 0.5)

    vis_slice = all_visual_features[start_idx:end_idx].to(device)  # shape (T_c, 2048)
    T_c = vis_slice.shape[0]

    # Compute text embeddings for candidate segments
    from src.text_embedder import get_sentence_embedder
    embedder = get_sentence_embedder(device=device)

    # Align speech segments within candidate to 2s intervals
    cand_segs = candidate.get("segments", [])
    text_embeddings = torch.zeros((T_c, 384), dtype=torch.float32, device=device)

    for i in range(T_c):
        seg_start = start_sec + i * visual_sample_interval
        seg_end = seg_start + visual_sample_interval
        seg_texts = [
            s["text"] for s in cand_segs
            if not (s["end"] <= seg_start or s["start"] >= seg_end)
        ]
        if seg_texts:
            comb = " ".join(seg_texts).strip()
            emb = embedder.encode(comb)  # shape (384,)
            text_embeddings[i] = emb.to(device)

    # Load multimodal BiLSTM
    try:
        from src.multimodal_model import MultimodalHighlightBiLSTM
        model = MultimodalHighlightBiLSTM(visual_dim=2048, text_dim=384, hidden_dim=256, num_layers=2)
        state = torch.load(multimodal_model_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        with torch.no_grad():
            v_inp = vis_slice.unsqueeze(0)   # (1, T_c, 2048)
            t_inp = text_embeddings.unsqueeze(0)  # (1, T_c, 384)
            scores = model(v_inp, t_inp).squeeze().cpu().numpy()

        if scores.ndim == 0:
            final_score = float(scores)
        else:
            final_score = float(np.mean(np.sort(scores)[-max(1, len(scores)//2):]))
        return round(final_score, 4)

    except Exception as e:
        logger.warning(f"Multimodal scoring fallback to pre_score ({str(e)})")
        return candidate.get("pre_score", 0.5)


def select_multiple_reels(
    candidates: List[Dict[str, Any]],
    all_visual_features: torch.Tensor,
    video_duration: float,
    target_reel_length: str = "30-60",  # "15-30", "30-60", "60-90", "auto"
    num_reels_option: str = "auto",     # "auto", "3", "5", "10"
    quality_threshold: float = 0.20,
    multimodal_model_path: str = "models/best_multimodal_bilstm.pt",
    visual_sample_interval: float = 2.0,
    gemini_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ranks candidates, applies NMS, snaps to sentence boundaries, and outputs curated Reels.
    """
    if not candidates:
        return []

    # Map target reel duration range
    if target_reel_length == "15-30":
        t_min, t_max = 15.0, 30.0
    elif target_reel_length == "60-90":
        t_min, t_max = 60.0, 90.0
    elif target_reel_length == "auto":
        t_min, t_max = 25.0, 60.0
    else:  # default "30-60"
        t_min, t_max = 30.0, 60.0

    # Determine desired count
    if num_reels_option in ["3", "5", "10"]:
        max_desired = int(num_reels_option)
    else:
        # Auto heuristic based on video duration
        if video_duration <= 300:      # <= 5 min
            max_desired = 3
        elif video_duration <= 900:    # <= 15 min
            max_desired = 5
        elif video_duration <= 2400:   # <= 40 min
            max_desired = 8
        else:                          # 40+ min
            max_desired = 12

    logger.info(f"Scoring {len(candidates)} candidates with Multimodal BiLSTM...")

    # Score each candidate
    scored_candidates = []
    for cand in candidates:
        mm_score = score_candidate_multimodal(
            candidate=cand,
            all_visual_features=all_visual_features,
            visual_sample_interval=visual_sample_interval,
            multimodal_model_path=multimodal_model_path
        )
        cand["final_score"] = mm_score
        scored_candidates.append(cand)

    # Filter below quality threshold
    passing = [c for c in scored_candidates if c["final_score"] >= quality_threshold]
    if len(passing) < 2 and scored_candidates:
        # If very few pass, take top 2 to avoid empty results
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        passing = scored_candidates[:max(1, min(len(scored_candidates), 2))]

    # Sort descending by score
    passing.sort(key=lambda x: x["final_score"], reverse=True)

    # Non-Maximum Suppression (NMS) over time overlap
    selected: List[Dict[str, Any]] = []

    def compute_overlap_sec(s1, e1, s2, e2):
        inter_start = max(s1, s2)
        inter_end = min(e1, e2)
        return max(0.0, inter_end - inter_start)

    for cand in passing:
        c_start, c_end = cand["start"], cand["end"]

        # Check overlap against already selected reels
        too_close = False
        for s in selected:
            inter = compute_overlap_sec(c_start, c_end, s["start_sec"], s["end_sec"])
            # If overlap > 12s or IoU > 0.3, suppress
            cand_dur = c_end - c_start
            if inter > 12.0 or (inter / cand_dur) > 0.30:
                too_close = True
                break

        if not too_close:
            # Snap start & end to sentence boundaries
            snapped_s, snapped_e, snippet = snap_boundaries_to_sentences(
                raw_start=c_start,
                raw_end=c_end,
                speech_segments=cand.get("segments", []),
                target_min=t_min,
                target_max=t_max
            )
            # Ensure within video bounds
            snapped_s = max(0.0, snapped_s)
            snapped_e = min(video_duration, snapped_e)
            reel_dur = round(snapped_e - snapped_s, 2)

            if reel_dur >= max(10.0, t_min * 0.7):
                selected.append({
                    "start_sec": snapped_s,
                    "end_sec": snapped_e,
                    "duration_sec": reel_dur,
                    "score": cand["final_score"],
                    "transcript": snippet or cand.get("full_text", ""),
                    "speech_segments": cand.get("segments", [])
                })

        if len(selected) >= max_desired:
            break

    # Sort selected reels chronologically for clean presentation
    selected.sort(key=lambda x: x["start_sec"])

    # Format timestamps and enrich with LLM/NLP virality scoring
    from src.virality_llm import analyze_clip_virality

    for i, reel in enumerate(selected):
        reel["reel_num"] = i + 1
        s_min, s_sec = int(reel["start_sec"] // 60), int(reel["start_sec"] % 60)
        e_min, e_sec = int(reel["end_sec"] // 60), int(reel["end_sec"] % 60)
        reel["start_formatted"] = f"{s_min:02d}:{s_sec:02d}"
        reel["end_formatted"] = f"{e_min:02d}:{e_sec:02d}"

        # Analyze virality, catchy title, and hook sentence
        virality_data = analyze_clip_virality(
            transcript_text=reel.get("transcript", ""),
            duration_sec=reel["duration_sec"],
            api_key=gemini_api_key
        )
        reel["title"] = virality_data["title"]
        reel["viral_score"] = virality_data["viral_score"]
        reel["hook_sentence"] = virality_data["hook_sentence"]
        reel["virality_reason"] = virality_data["virality_reason"]

    logger.info(f"Selected {len(selected)} high-quality Reels (target was {max_desired}).")
    return selected
