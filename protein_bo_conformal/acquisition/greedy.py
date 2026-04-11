"""Greedy acquisition baseline."""

from __future__ import annotations

from typing import Any

from acquisition.registry import BaseAcquisition


class GreedyAcquisition(BaseAcquisition):
    """Select top-k candidates by predicted mean fitness."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="greedy", config=config)

    def select(
        self,
        candidates: list[Any] | tuple[Any, ...],
        predictions: dict[str, Any],
        batch_size: int,
    ):
        self._candidate_count(candidates, predictions)
        mean = self._mean_vector(predictions)
        sigma = self._sigma_vector(predictions)
        selected = self._stable_topk(mean, batch_size=batch_size)
        details = [
            {
                "index": int(index),
                "score": float(mean[index]),
                "mean": float(mean[index]),
                "sigma": float(sigma[index]),
            }
            for index in selected
        ]
        return self._build_selection(mean, selected, details)
