"""
Hash dedup and quality heuristics.

Every test runs with use_embeddings=False, so the suite never imports
sentence-transformers and never downloads a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from seedmill.dedup import DedupFilter


def _filter(**kw):
    kw.setdefault("use_embeddings", False)
    return DedupFilter(**kw)


def test_exact_duplicate_is_rejected_on_the_second_sighting():
    f = _filter()
    rec = {"a": "a sufficiently long value"}
    accepted, rejected = f.filter_batch([rec, dict(rec)])
    assert len(accepted) == 1
    assert rejected[0]["_reject_reason"] == "exact_duplicate"


def test_hash_is_independent_of_key_order():
    """json.dumps(sort_keys=True) — reordered keys are the same record."""
    f = _filter()
    accepted, rejected = f.filter_batch(
        [
            {"a": "long enough value", "b": "another one"},
            {"b": "another one", "a": "long enough value"},
        ]
    )
    assert len(accepted) == 1 and len(rejected) == 1


def test_check_quality_false_bypasses_the_heuristics():
    """This is the path the engine uses — short enum values must survive."""
    f = _filter(min_field_length=10)
    accepted, rejected = f.filter_batch([{"label": "gym"}], check_quality=False)
    assert len(accepted) == 1 and not rejected


def test_short_field_is_rejected_when_quality_checking():
    f = _filter(min_field_length=10)
    accepted, rejected = f.filter_batch([{"label": "gym"}], check_quality=True)
    assert not accepted
    assert "too short" in rejected[0]["_reject_reason"]


def test_skip_length_check_fields_exempts_named_fields():
    """How the engine avoids rejecting every categorical field."""
    f = _filter(min_field_length=10, skip_length_check_fields={"label"})
    accepted, _ = f.filter_batch([{"label": "gym"}], check_quality=True)
    assert len(accepted) == 1


def test_repetitive_content_is_rejected():
    phrase = "the quick brown fox jumps "
    f = _filter(min_field_length=1)
    accepted, rejected = f.filter_batch([{"text": phrase * 4}], check_quality=True)
    assert not accepted
    assert "repetitive" in rejected[0]["_reject_reason"]


def test_load_existing_seeds_the_hash_index(tmp_path: Path):
    rec = {"a": "a sufficiently long value"}
    p = tmp_path / "zz_existing.jsonl"
    p.write_text(json.dumps(rec) + "\n")

    f = _filter()
    assert f.load_existing(str(p)) == 1

    accepted, rejected = f.filter_batch([dict(rec)])
    assert not accepted
    assert rejected[0]["_reject_reason"] == "exact_duplicate"


def test_load_existing_on_a_missing_path_returns_zero(tmp_path: Path):
    assert _filter().load_existing(str(tmp_path / "nope.jsonl")) == 0


def test_stats_reports_hashes_and_no_embeddings():
    f = _filter()
    f.filter_batch([{"a": "a sufficiently long value"}], check_quality=False)
    assert f.stats == {"unique_hashes": 1, "stored_embeddings": 0}


def test_embeddings_extra_is_genuinely_optional(monkeypatch):
    """
    With sentence-transformers unavailable, _get_encoder must swallow the
    ImportError and turn embeddings off rather than raising. This is what
    makes shipping it as an optional extra safe.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    f = DedupFilter(use_embeddings=True)

    accepted, rejected = f.filter_batch([{"a": "a sufficiently long value"}], check_quality=False)

    assert f.use_embeddings is False
    assert len(accepted) == 1 and not rejected
