"""One-hot protein sequence encoder."""

from __future__ import annotations

from typing import Any

import numpy as np

from representation.interface import Encoder


class OneHotEncoder(Encoder):
    """Stable one-hot or bag-of-amino-acids protein encoder."""

    CANONICAL_VOCAB = tuple("ACDEFGHIKLMNPQRSTVWY") + ("X",)

    def __init__(self, config: dict[str, Any], logger: Any | None = None) -> None:
        self.mode = str(config.get("onehot_mode", "flattened")).strip().lower()
        if self.mode not in {"flattened", "bag"}:
            raise ValueError("onehot_mode must be 'flattened' or 'bag'.")
        self.fixed_length = config.get("fixed_length")
        self.normalize_bag = bool(config.get("bag_normalize", True))
        self._resolved_length: int | None = int(self.fixed_length) if self.fixed_length else None
        self._char_index = {char: index for index, char in enumerate(self.CANONICAL_VOCAB)}
        super().__init__(
            name="onehot",
            cache_dir=config.get("cache_dir", "data/processed/caches"),
            batch_size=int(config.get("batch_size", 64)),
            cache_enabled=bool(config.get("cache_enabled", True)),
            logger=logger,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fixed_length": self._resolved_length if self.mode == "flattened" else self.fixed_length,
            "normalize_bag": self.normalize_bag,
        }

    def defer_cache_initialization(self) -> bool:
        return self.mode == "flattened" and self._resolved_length is None

    def _prepare_sequences(self, sequences: tuple[str, ...]) -> None:
        if self.mode == "flattened":
            self._resolve_length(sequences)

    def _validate_sequences(self, sequences: tuple[str, ...]) -> None:
        illegal = sorted(
            {
                char
                for sequence in sequences
                for char in sequence
                if char not in self._char_index
            }
        )
        if illegal:
            raise ValueError(f"OneHotEncoder encountered unsupported residue(s): {illegal}")

    def _resolve_length(self, sequences: tuple[str, ...]) -> int:
        batch_max = max(len(sequence) for sequence in sequences)
        if self._resolved_length is None:
            self._resolved_length = batch_max
            return self._resolved_length
        if batch_max > self._resolved_length:
            raise ValueError(
                f"OneHotEncoder received sequence length {batch_max} beyond fixed feature length {self._resolved_length}."
            )
        return self._resolved_length

    def _encode_uncached_batch(self, sequences: tuple[str, ...]) -> np.ndarray:
        self._validate_sequences(sequences)
        vocab_size = len(self.CANONICAL_VOCAB)

        if self.mode == "bag":
            features = np.zeros((len(sequences), vocab_size), dtype=np.float32)
            for row_index, sequence in enumerate(sequences):
                for char in sequence:
                    features[row_index, self._char_index[char]] += 1.0
                if self.normalize_bag and sequence:
                    features[row_index] /= float(len(sequence))
            return features

        resolved_length = self._resolve_length(sequences)
        features = np.zeros((len(sequences), resolved_length * vocab_size), dtype=np.float32)
        for row_index, sequence in enumerate(sequences):
            for position, char in enumerate(sequence):
                column = position * vocab_size + self._char_index[char]
                features[row_index, column] = 1.0
        return features

    def get_dim(self) -> int:
        if self.mode == "bag":
            return len(self.CANONICAL_VOCAB)
        if self._resolved_length is None:
            raise ValueError("OneHotEncoder dimension is unresolved before the first encode call.")
        return self._resolved_length * len(self.CANONICAL_VOCAB)
