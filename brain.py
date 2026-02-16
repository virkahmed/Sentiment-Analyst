"""
LLM reasoning engine: Analyst prompt and structured output for implied probability
and recommendation (BUY_YES / BUY_NO / HOLD).
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT_TEMPLATE = """You are a Lead Quantitative Researcher at a prediction market hedge fund. Your task is to analyze scraped text from niche internet forums and determine the probability of a specific event occurring on Kalshi.

Input Data:
- Market Description: {market_description}
- Current Kalshi Price: {current_kalshi_price}
- Scraped Content: {scraped_content}

Analysis Guidelines:
- Source Reliability: Penalize "hype" or "doomerism." Weight information higher if the user cites specific data, legislative trackers, or historical precedents.
- Information Asymmetry: Look for "niche" insights that the general public might be missing (e.g., a specific court filing mentioned in a legal subreddit that hasn't hit mainstream news).
- Counter-Signaling: Note if the consensus in the thread is overwhelmingly emotional without evidence; this often suggests a "crowded trade" that might be wrong.

Respond with the required JSON only: implied_probability (0-1), confidence_score (0-1), key_signals (list of strings), contrarian_risks (list of strings), recommendation (one of BUY_YES, BUY_NO, HOLD)."""


ANALYST_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyst_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "implied_probability": {
                    "type": "number",
                    "description": "Implied probability of the event (0.0 to 1.0)",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "Confidence in the estimate (0.0 to 1.0)",
                },
                "key_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key signals from the content",
                },
                "contrarian_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of contrarian risks",
                },
                "recommendation": {
                    "type": "string",
                    "enum": ["BUY_YES", "BUY_NO", "HOLD"],
                    "description": "Suggested direction (BUY_YES / BUY_NO / HOLD)",
                },
            },
            "required": [
                "implied_probability",
                "confidence_score",
                "key_signals",
                "contrarian_risks",
                "recommendation",
            ],
            "additionalProperties": False,
        },
    },
}

# Approximate chars per token for truncation (conservative)
CHARS_PER_TOKEN = 3
MAX_INPUT_TOKENS = 8000
MAX_CONTENT_CHARS = MAX_INPUT_TOKENS * CHARS_PER_TOKEN


def format_threads_for_prompt(threads: list[dict[str, Any]]) -> str:
    """Format thread list into a single string for the prompt; truncate if over cap."""
    parts: list[str] = []
    total = 0
    for t in threads:
        block = (
            f"[Thread: {t.get('title', '')}]\n"
            f"Body: {t.get('body', '')}\n"
        )
        for c in t.get("comments", [])[:20]:
            block += f"  - {c.get('author', '')}: {c.get('body', '')[:500]}\n"
        if total + len(block) > MAX_CONTENT_CHARS:
            block = block[: max(0, MAX_CONTENT_CHARS - total - 100)] + "\n[... truncated]"
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts) if parts else "(No content)"


def estimate_probability(
    market_description: str,
    current_kalshi_price: float,
    scraped_threads: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Call the LLM with the Analyst prompt and return parsed dict:
    implied_probability, confidence_score, key_signals, contrarian_risks, recommendation.
    """
    content = format_threads_for_prompt(scraped_threads)
    system = SYSTEM_PROMPT_TEMPLATE.format(
        market_description=market_description[:2000],
        current_kalshi_price=current_kalshi_price,
        scraped_content=content,
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Analyze the above and respond with the required JSON only."},
        ],
        response_format=ANALYST_RESPONSE_SCHEMA,
    )
    msg = resp.choices[0].message
    raw = getattr(msg, "content", None) or ""
    try:
        if hasattr(msg, "refusal") and getattr(msg, "refusal", None):
            return {
                "implied_probability": 0.5,
                "confidence_score": 0.0,
                "key_signals": [],
                "contrarian_risks": ["LLM refused"],
                "recommendation": "HOLD",
            }
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    return {
        "implied_probability": float(parsed.get("implied_probability", 0.5)),
        "confidence_score": float(parsed.get("confidence_score", 0.0)),
        "key_signals": list(parsed.get("key_signals", []) or []),
        "contrarian_risks": list(parsed.get("contrarian_risks", []) or []),
        "recommendation": str(parsed.get("recommendation", "HOLD")).strip().upper(),
    }
