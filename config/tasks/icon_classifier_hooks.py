"""
Hooks for the icon_classifier task: SFT/pairs export builders.

These stay in Python (not YAML) because candidate-list assembly and
confidence synthesis are real logic, not schema. Everything is
deterministic per record id so exports are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def _rng(record_id: str, salt: str) -> random.Random:
    seed = int.from_bytes(
        hashlib.sha256(f"{record_id}:{salt}".encode()).digest()[:8], "big"
    )
    return random.Random(seed)


def _build_candidates(record: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    """positive + hard negatives + random vocab distractors, shuffled."""
    k = ctx["config"].get("exports", {}).get("sft_candidates", 5)
    rng = _rng(record["id"], "candidates")

    candidates = [record["positive_icon"]] + record["hard_negatives"][: k - 1]
    if len(candidates) < k:
        excluded = set(candidates) | set(record["acceptable_icons"])
        pool = sorted(ctx["vocab"]["icons"] - excluded)
        candidates += rng.sample(pool, min(k - len(candidates), len(pool)))

    rng.shuffle(candidates)
    return candidates


def build_sft(record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Chat-format example for fine-tuning a small LLM."""
    candidates = _build_candidates(record, ctx)
    rng = _rng(record["id"], "confidence")
    # Ambiguous interests (multiple acceptable icons) get lower confidence
    if len(record["acceptable_icons"]) > 1:
        confidence = round(rng.uniform(0.55, 0.80), 2)
    else:
        confidence = round(rng.uniform(0.72, 0.95), 2)

    prompt = (
        f"Interest: {record['interest']}\n"
        f"Candidates: {', '.join(candidates)}\n"
        f"Return the best icon."
    )
    completion = json.dumps({"icon": record["positive_icon"], "confidence": confidence})
    return {
        "id": record["id"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
    }


def build_pairs(record: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Flat (text, label) pair for a classical classifier."""
    return {
        "id": record["id"],
        "text": record["interest"],
        "label": record["positive_icon"],
        "acceptable_labels": record["acceptable_icons"],
        "category": record["category"],
    }


EXPORTERS = {"sft": build_sft, "pairs": build_pairs}
