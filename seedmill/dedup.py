"""
Embedding-based deduplication and quality filtering.

Uses sentence-transformers to compute embeddings for generated records,
then filters out near-duplicates via cosine similarity. Also provides
basic quality heuristics (length checks, repetition detection).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


class DedupFilter:
    """
    Deduplication and quality filter for generated records.

    Uses two strategies:
    1. Exact hash dedup — fast, catches identical records.
    2. Embedding similarity — catches paraphrased/near-duplicate content.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        min_field_length: int = 10,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        skip_length_check_fields: set[str] | None = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_field_length = min_field_length
        self.use_embeddings = use_embeddings
        self.skip_length_check_fields = skip_length_check_fields or set()
        self._hashes: set[str] = set()
        self._embeddings: list[np.ndarray] = []
        self._encoder = None
        self._embedding_model = embedding_model

    def _get_encoder(self) -> Any:
        """Lazy-load the sentence transformer model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self._embedding_model)
            except ImportError:
                self.use_embeddings = False
                return None
        return self._encoder

    def _record_to_text(self, record: dict[str, Any]) -> str:
        """Concatenate all string values of a record for embedding."""
        parts = []
        for v in record.values():
            if isinstance(v, str):
                parts.append(v)
            else:
                parts.append(str(v))
        return " ".join(parts)

    def _record_hash(self, record: dict[str, Any]) -> str:
        """Deterministic hash of a record for exact dedup."""
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _is_near_duplicate(self, embedding: np.ndarray) -> bool:
        """Check if an embedding is too similar to any stored embedding."""
        for stored in self._embeddings:
            if self._cosine_similarity(embedding, stored) > self.similarity_threshold:
                return True
        return False

    # ── quality heuristics ──────────────────────────────────────────────

    def _check_quality(self, record: dict[str, Any]) -> list[str]:
        """
        Run basic quality checks on a record.

        Skips length checks on fields in skip_length_check_fields (enum
        fields, short categorical values, nested config objects, etc.).

        Returns list of issue descriptions (empty = passed).
        """
        issues = []

        for key, value in record.items():
            if isinstance(value, str):
                # Skip length check for enum/categorical/short-by-nature fields
                if key not in self.skip_length_check_fields:
                    if len(value.strip()) < self.min_field_length:
                        issues.append(f"Field '{key}' too short ({len(value.strip())} chars)")

                # Repetition detection — if any 5-word phrase repeats 3+ times
                words = value.split()
                if len(words) >= 15:
                    for i in range(len(words) - 4):
                        phrase = " ".join(words[i : i + 5])
                        if value.count(phrase) >= 3:
                            issues.append(f"Field '{key}' contains repetitive content")
                            break

        return issues

    # ── public API ──────────────────────────────────────────────────────

    def filter_batch(
        self,
        records: list[dict[str, Any]],
        *,
        check_quality: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Filter a batch of records for duplicates and quality.

        Args:
            records: List of record dicts to filter.
            check_quality: Whether to run quality heuristics.

        Returns:
            Tuple of (accepted_records, rejected_records).
        """
        accepted = []
        rejected = []

        for record in records:
            # 1. Exact hash check
            h = self._record_hash(record)
            if h in self._hashes:
                rejected.append({**record, "_reject_reason": "exact_duplicate"})
                continue

            # 2. Quality check
            if check_quality:
                issues = self._check_quality(record)
                if issues:
                    rejected.append({**record, "_reject_reason": f"quality: {'; '.join(issues)}"})
                    continue

            # 3. Embedding similarity check
            if self.use_embeddings:
                encoder = self._get_encoder()
                if encoder is not None:
                    text = self._record_to_text(record)
                    emb = encoder.encode(text, normalize_embeddings=True)
                    if self._is_near_duplicate(emb):
                        rejected.append({**record, "_reject_reason": "near_duplicate"})
                        continue
                    self._embeddings.append(emb)

            # Passed all checks
            self._hashes.add(h)
            accepted.append(record)

        return accepted, rejected

    def load_existing(self, filepath: str) -> int:
        """
        Load existing records from a JSONL file into the dedup index.

        Call this before generation to avoid duplicating records that
        already exist in a partially-generated dataset.

        Returns:
            Count of records loaded.
        """
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            return 0

        count = 0
        encoder = self._get_encoder() if self.use_embeddings else None

        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            self._hashes.add(self._record_hash(record))

            if encoder is not None:
                text = self._record_to_text(record)
                emb = encoder.encode(text, normalize_embeddings=True)
                self._embeddings.append(emb)

            count += 1

        return count

    @property
    def stats(self) -> dict[str, int]:
        """Return current dedup index stats."""
        return {
            "unique_hashes": len(self._hashes),
            "stored_embeddings": len(self._embeddings),
        }
