"""Random acquisition baseline."""

from __future__ import annotations

import random as pyrandom
from typing import Any

import numpy as np

from acquisition.registry import BaseAcquisition


class RandomAcquisition(BaseAcquisition):
    """Uniform random batch selection from the candidate pool."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="random", config=config)
        self.seed = int(config.get("seed", 7))

    def select(
        self,
        candidates: list[Any] | tuple[Any, ...],
        predictions: dict[str, Any],
        batch_size: int,
    ):
        candidate_count = self._candidate_count(candidates, predictions)
        rng = pyrandom.Random(self.seed)
        indices = list(range(candidate_count))
        rng.shuffle(indices)
        selected = np.asarray(indices[: min(max(1, int(batch_size)), candidate_count)], dtype=np.int64)
        scores = np.ones((candidate_count,), dtype=np.float32)
        details = [
            {
                "index": int(index),
                "score": 1.0,
                "mean": float(self._mean_vector(predictions)[index]),
                "sigma": float(self._sigma_vector(predictions)[index]),
            }
            for index in selected
        ]
        return self._build_selection(scores, selected, details)
