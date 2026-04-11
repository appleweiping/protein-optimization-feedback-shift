"""Shared representation interface definitions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from representation.cache import RepresentationCache


def canonicalize_sequence(sequence: str) -> str:
    """Normalize representation inputs into canonical uppercase sequences."""
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Encoder inputs must be non-empty sequences.")
    return normalized


class Encoder(ABC):
    """Unified feature-space encoder interface used across the whole project."""

    def __init__(
        self,
        name: str,
        cache_dir: str | Path,
        batch_size: int = 64,
        cache_enabled: bool = True,
        logger: Any | None = None,
    ) -> None:
        self.name = name
        self.batch_size = max(1, int(batch_size))
        self.logger = logger
        self._dim: int | None = None
        self._cache_root = Path(cache_dir)
        self._cache_enabled = cache_enabled
        self._call_stats = {
            "encode_calls": 0,
            "encoded_sequences": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "total_seconds": 0.0,
            "last_seconds": 0.0,
        }
        self.cache: RepresentationCache | None = None
        if self._cache_enabled and not self.defer_cache_initialization():
            self._initialize_cache()

    def defer_cache_initialization(self) -> bool:
        """Allow subclasses to delay cache creation until shape-dependent config is known."""
        return False

    def _prepare_sequences(self, sequences: tuple[str, ...]) -> None:
        """Optional hook for subclasses to resolve sequence-dependent state before caching."""

    def _initialize_cache(self) -> None:
        """Create the cache namespace if caching is enabled and not initialized yet."""
        if not self._cache_enabled or self.cache is not None:
            return
        self.cache = RepresentationCache(
            cache_root=self._cache_root,
            encoder_name=self.name,
            namespace_payload=self.cache_signature(),
        )

    def cache_signature(self) -> dict[str, Any]:
        """Return the stable cache namespace payload for this encoder."""
        return {
            "name": self.name,
            **self.config_summary(),
        }

    @abstractmethod
    def config_summary(self) -> dict[str, Any]:
        """Return encoder-specific config used for reporting and caching."""

    @abstractmethod
    def _encode_uncached_batch(self, sequences: tuple[str, ...]) -> np.ndarray:
        """Encode a batch of canonical sequences without using the cache."""

    @abstractmethod
    def get_dim(self) -> int:
        """Return the fixed feature dimension of this encoder."""

    def encode(self, sequence_list: list[str] | tuple[str, ...]) -> np.ndarray:
        """Encode sequences into a shared feature matrix with transparent caching."""
        sequences = tuple(canonicalize_sequence(sequence) for sequence in sequence_list)
        if not sequences:
            dim = self._dim or self.get_dim()
            return np.zeros((0, dim), dtype=np.float32)

        self._prepare_sequences(sequences)
        self._initialize_cache()
        start = time.perf_counter()
        if self.cache is not None:
            self.cache.reset_stats()

        results: list[np.ndarray | None] = [None] * len(sequences)
        missing_pairs: list[tuple[int, str]] = []
        for index, sequence in enumerate(sequences):
            cached = self.cache.get(sequence) if self.cache is not None else None
            if cached is None:
                missing_pairs.append((index, sequence))
            else:
                results[index] = cached.astype(np.float32, copy=False)

        for start_index in range(0, len(missing_pairs), self.batch_size):
            batch_pairs = missing_pairs[start_index : start_index + self.batch_size]
            batch_sequences = tuple(sequence for _, sequence in batch_pairs)
            batch_features = self._encode_uncached_batch(batch_sequences).astype(np.float32, copy=False)
            if batch_features.shape[0] != len(batch_sequences):
                raise ValueError(
                    f"Encoder '{self.name}' returned {batch_features.shape[0]} rows for {len(batch_sequences)} sequences."
                )
            for row_index, (original_index, sequence) in enumerate(batch_pairs):
                vector = batch_features[row_index]
                results[original_index] = vector
                if self.cache is not None:
                    self.cache.set(sequence, vector)

        feature_matrix = np.vstack([result for result in results if result is not None]).astype(np.float32, copy=False)
        self._dim = int(feature_matrix.shape[1])
        elapsed = time.perf_counter() - start
        self._call_stats["encode_calls"] += 1
        self._call_stats["encoded_sequences"] += len(sequences)
        self._call_stats["total_seconds"] += elapsed
        self._call_stats["last_seconds"] = elapsed
        if self.cache is not None:
            self._call_stats["cache_hits"] += self.cache.stats["hits"]
            self._call_stats["cache_misses"] += self.cache.stats["misses"]
            self._call_stats["cache_writes"] += self.cache.stats["writes"]

        if self.logger is not None:
            self.logger.info(
                "Encoder '%s' encoded %s sequences in %.3fs (hits=%s misses=%s writes=%s).",
                self.name,
                len(sequences),
                elapsed,
                self.cache.stats["hits"] if self.cache is not None else 0,
                self.cache.stats["misses"] if self.cache is not None else len(sequences),
                self.cache.stats["writes"] if self.cache is not None else 0,
            )
        return feature_matrix

    def batch_encode(self, sequence_list: list[str] | tuple[str, ...]) -> np.ndarray:
        """Alias for encode() to keep a stable interface across modules."""
        return self.encode(sequence_list)

    def get_stats(self) -> dict[str, Any]:
        """Return encoder usage statistics for logging and diagnostics."""
        stats = dict(self._call_stats)
        stats["feature_dim"] = self._dim
        stats["cache_enabled"] = self.cache is not None
        return stats

    def to_tensor(self, features: np.ndarray) -> Any:
        """Convert a numpy feature matrix into a torch tensor when available."""
        try:
            import torch  # type: ignore

            return torch.from_numpy(features)
        except Exception:
            return features


def build_encoder(config: dict[str, Any], logger: Any | None = None) -> Encoder:
    """Construct an encoder from representation config."""
    name = str(config.get("name", "onehot")).strip().lower()
    if name == "onehot":
        from representation.onehot_encoder import OneHotEncoder

        return OneHotEncoder(config=config, logger=logger)
    if name == "esm":
        from representation.esm_encoder import ESMEncoder

        return ESMEncoder(config=config, logger=logger)
    raise ValueError(f"Unsupported encoder name '{name}'.")
