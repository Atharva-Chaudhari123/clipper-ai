"""
src/virality_llm.py
-------------------
AI Semantic Virality Analysis & Hook Detection Engine (Opus Clip Style).

Uses Google Gemini (or local fallback heuristic) to analyze candidate transcripts for:
1. Viral Score (0-100)
2. Attention-grabbing Catchy Title
3. Opening 3-second Hook Sentence
4. Virality Rationale (why viewers stop scrolling)
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ViralityLLM")

VIRALITY_PROMPT_TEMPLATE = """You are an elite short-form video editor who creates viral TikToks, Instagram Reels, and YouTube Shorts.
Analyze this video speech transcript excerpt from a candidate highlight clip.

Evaluate the viral potential based on these criteria:
1. HOOK MOMENTS — statement creating immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong...")
2. EMOTIONAL PEAKS — surprise, humor, excitement, raw reactions
3. OPINION BOMBS / CONTRARIAN VIEWS — strong polarizing statements
4. PRACTICAL VALUE / REVELATION — surprising facts or actionable advice
5. STANDALONE COHERENCE — does it make sense as a 30-60s clip?

Clip Duration: {duration:.1f} seconds
Transcript Excerpt:
\"\"\"{transcript}\"\"\"

Respond with STRICT JSON format only (no markdown quotes, no backticks, no comments):
{{
  "title": "Short punchy title under 6 words",
  "viral_score": 85,
  "hook_sentence": "The single best opening sentence to hook viewers",
  "virality_reason": "One clear sentence explaining why this clip will engage audiences on social media"
}}
"""


def analyze_clip_virality(
    transcript_text: str,
    duration_sec: float = 45.0,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes a candidate transcript using Gemini or heuristic fallback.
    Returns:
        {
            "title": str,
            "viral_score": int (0-100),
            "hook_sentence": str,
            "virality_reason": str
        }
    """
    clean_text = transcript_text.strip()
    if not clean_text:
        return _heuristic_fallback(clean_text, duration_sec)

    # 1. Try Gemini API if key is available in args or environment
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            prompt = VIRALITY_PROMPT_TEMPLATE.format(
                duration=duration_sec,
                transcript=clean_text
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                }
            )
            raw_text = response.text.strip()
            # Clean markdown codeblocks if returned
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            data = json.loads(raw_text)

            return {
                "title": str(data.get("title", "Must-Watch Moment")).strip()[:60],
                "viral_score": max(10, min(100, int(data.get("viral_score", 75)))),
                "hook_sentence": str(data.get("hook_sentence", clean_text[:100])).strip(),
                "virality_reason": str(data.get("virality_reason", "High audience curiosity and strong narrative flow.")).strip()
            }
        except Exception as e:
            logger.warning(f"Gemini virality analysis failed ({e}). Using local NLP heuristic.")

    # 2. Local heuristic analysis fallback (works 100% offline without API key)
    return _heuristic_fallback(clean_text, duration_sec)


def _heuristic_fallback(transcript_text: str, duration_sec: float) -> Dict[str, Any]:
    """
    Offline NLP heuristics scoring based on hook words, sentiment intensity, and sentence length.
    """
    if not transcript_text:
        return {
            "title": "Highlight Clip",
            "viral_score": 60,
            "hook_sentence": "",
            "virality_reason": "Visual highlight moment."
        }

    words = transcript_text.split()
    lower_text = transcript_text.lower()

    # Hook trigger keywords
    high_impact_keywords = [
        "secret", "never", "always", "truth", "crazy", "insane", "mistake",
        "why", "how to", "best", "worst", "stop", "nobody", "everyone",
        "future", "ai", "money", "failed", "success", "shocking"
    ]
    keyword_hits = sum(1 for kw in high_impact_keywords if kw in lower_text)

    # Word count and pacing
    wpm = (len(words) / max(duration_sec, 5.0)) * 60.0  # target ~130-180 WPM
    pacing_score = 25 if 120 <= wpm <= 190 else 15

    # Compute base score
    base_score = 55 + min(25, keyword_hits * 6) + pacing_score
    viral_score = max(50, min(98, int(base_score)))

    # Extract opening hook sentence
    sentences = re.split(r"(?<=[.!?])\s+", transcript_text)
    hook_sentence = sentences[0].strip() if sentences else transcript_text[:80]

    # Generate catchy title from first strong words
    candidate_title_words = words[:6]
    title = " ".join(candidate_title_words).strip().capitalize()
    if len(title) > 40:
        title = title[:37] + "..."

    reason = "Strong opening question and engaging speech delivery." if "?" in hook_sentence else "High topic relevance with curiosity-driven narrative hook."

    return {
        "title": title or "Engaging Highlight",
        "viral_score": viral_score,
        "hook_sentence": hook_sentence,
        "virality_reason": reason
    }
