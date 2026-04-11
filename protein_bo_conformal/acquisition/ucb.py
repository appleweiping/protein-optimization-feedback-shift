"""Upper confidence bound acquisition."""

from __future__ import annotations

from typing import Any

import numpy as np

from acquisition.registry import BaseAcquisition


class UCBAcquisition(BaseAcquisition):
    """Classic upper confidence bound acquisition."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="ucb", config=config)
        self.beta = float(config.get("beta", 1.0))

    def select(
        self,
        candidates: list[Any] | tuple[Any, ...],
        predictions: dict[str, Any],
        batch_size: int,
    ):
        self._candidate_count(candidates, predictions)
        mean = self._mean_vector(predictions)
        sigma = self._sigma_vector(predictions)
        scores = (mean + self.beta * sigma).astype(np.float32, copy=False)
        selected = self._stable_topk(scores, batch_size=batch_size)
        details = [
            {
                "index": int(index),
                "score": float(scores[index]),
                "mean": float(mean[index]),
                "sigma": float(sigma[index]),
                "beta": self.beta,
                "score_breakdown": {
                    "mean_term": float(mean[index]),
                    "uncertainty_term": float(self.beta * sigma[index]),
                },
            }
            for index in selected
        ]
        return self._build_selection(scores, selected, details)
