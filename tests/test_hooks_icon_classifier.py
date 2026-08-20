"""
The icon_classifier hooks: candidate assembly and confidence synthesis.

Everything here must be deterministic per record id — exports are
regenerated routinely and must not churn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

HOOKS_REL = Path("config/tasks/icon_classifier_hooks.py")

VOCAB = {f"icon{i}" for i in range(20)} | {"gym", "book", "music", "art"}


@pytest.fixture(scope="module")
def hooks(repo_root, load_hooks_module):
    return load_hooks_module(repo_root / HOOKS_REL)


@pytest.fixture
def ctx():
    return {"vocab": {"icons": set(VOCAB)}, "config": {"exports": {"sft_candidates": 5}}}


def _rec(**kw):
    base = {
        "id": "r1",
        "interest": "lifting weights",
        "positive_icon": "gym",
        "hard_negatives": ["book", "music"],
        "acceptable_icons": ["gym"],
        "category": "fitness",
    }
    base.update(kw)
    return base


# ── determinism ─────────────────────────────────────────────────────────


def test_rng_is_deterministic_per_id_and_salt(hooks):
    assert hooks._rng("r1", "a").random() == hooks._rng("r1", "a").random()


def test_rng_differs_across_salts_and_ids(hooks):
    assert hooks._rng("r1", "a").random() != hooks._rng("r1", "b").random()
    assert hooks._rng("r1", "a").random() != hooks._rng("r2", "a").random()


def test_candidates_are_identical_across_calls(hooks, ctx):
    assert hooks._build_candidates(_rec(), ctx) == hooks._build_candidates(_rec(), ctx)


# ── _build_candidates ───────────────────────────────────────────────────


def test_candidates_always_contain_the_positive(hooks, ctx):
    assert "gym" in hooks._build_candidates(_rec(), ctx)


def test_candidates_reach_the_requested_size(hooks, ctx):
    assert len(hooks._build_candidates(_rec(), ctx)) == 5


def test_distractors_avoid_acceptable_icons(hooks, ctx):
    """
    An acceptable-but-not-positive icon must never appear as a distractor:
    it would be a candidate the model is punished for choosing correctly.
    """
    rec = _rec(acceptable_icons=["gym", "art"], hard_negatives=[])
    candidates = hooks._build_candidates(rec, ctx)
    assert "art" not in candidates


def test_distractors_come_only_from_the_vocabulary(hooks, ctx):
    candidates = hooks._build_candidates(_rec(), ctx)
    assert set(candidates) <= VOCAB


def test_small_vocab_pool_does_not_raise(hooks):
    """rng.sample with min(...) — a pool smaller than k must not blow up."""
    ctx = {"vocab": {"icons": {"gym", "book"}}, "config": {"exports": {"sft_candidates": 5}}}
    candidates = hooks._build_candidates(_rec(hard_negatives=[]), ctx)
    assert "gym" in candidates
    assert len(candidates) <= 5


# ── build_sft ───────────────────────────────────────────────────────────


def test_build_sft_shape(hooks, ctx):
    out = hooks.build_sft(_rec(), ctx)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant"]
    assert out["id"] == "r1"


def test_build_sft_completion_is_valid_json_naming_the_positive(hooks, ctx):
    out = hooks.build_sft(_rec(), ctx)
    payload = json.loads(out["messages"][1]["content"])
    assert payload["icon"] == "gym"
    assert set(payload) == {"icon", "confidence"}


def test_confidence_is_lower_when_the_interest_is_ambiguous(hooks, ctx):
    """Multiple acceptable icons -> the lower band, by design."""
    ambiguous = json.loads(
        hooks.build_sft(_rec(acceptable_icons=["gym", "art"]), ctx)["messages"][1]["content"]
    )["confidence"]
    assert 0.55 <= ambiguous <= 0.80

    unambiguous = json.loads(
        hooks.build_sft(_rec(acceptable_icons=["gym"]), ctx)["messages"][1]["content"]
    )["confidence"]
    assert 0.72 <= unambiguous <= 0.95


def test_build_sft_prompt_carries_the_interest_and_candidates(hooks, ctx):
    prompt = hooks.build_sft(_rec(), ctx)["messages"][0]["content"]
    assert "lifting weights" in prompt
    assert "gym" in prompt


# ── build_pairs ─────────────────────────────────────────────────────────


def test_build_pairs_has_an_exact_key_set(hooks, ctx):
    out = hooks.build_pairs(_rec(), ctx)
    assert set(out) == {"id", "text", "label", "acceptable_labels", "category"}
    assert out["text"] == "lifting weights"
    assert out["label"] == "gym"
