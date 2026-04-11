"""Caching helpers for computed representations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def stable_namespace_hash(payload: dict[str, Any]) -> str:
    """Build a stable hash for encoder cache namespaces."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def sequence_cache_key(sequence: str) -> str:
    """Build a stable cache key for a canonical sequence string."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


class RepresentationCache:
    """Persist encoder outputs on disk using a per-sequence hash layout."""

    def __init__(
        self,
        cache_root: Path,
        encoder_name: str,
        namespace_payload: dict[str, Any],
    ) -> None:
        namespace_hash = stable_namespace_hash(namespace_payload)
        self.cache_dir = cache_root / encoder_name / namespace_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }

    def _vector_path(self, sequence: str) -> Path:
        return self.cache_dir / f"{sequence_cache_key(sequence)}.npy"

    def get(self, sequence: str) -> np.ndarray | None:
        """Load a cached representation vector if it exists."""
        path = self._vector_path(sequence)
        if not path.exists():
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return np.load(path)

    def set(self, sequence: str, vector: np.ndarray) -> None:
        """Persist a representation vector for a sequence."""
        path = self._vector_path(sequence)
        np.save(path, vector.astype(np.float32, copy=False))
        self.stats["writes"] += 1

    def reset_stats(self) -> None:
        """Reset per-call cache statistics."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }
