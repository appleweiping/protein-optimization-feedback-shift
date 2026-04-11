"""Expected improvement acquisition baseline."""

from __future__ import annotations

from typing import Any

from acquisition.registry import BaseAcquisition, gaussian_ei


class ExpectedImprovementAcquisition(BaseAcquisition):
    """Expected improvement baseline built on mean and sigma predictions."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(name="ei", config=config)
        self.xi = float(config.get("xi", 0.0))

    def select(
        self,
        candidates: list[Any] | tuple[Any, ...],
        predictions: dict[str, Any],
        batch_size: int,
    ):
        self._candidate_count(candidates, predictions)
        mean = self._mean_vector(predictions)
        sigma = self._sigma_vector(predictions)
        best_observed = self._best_observed(predictions)
        scores = gaussian_ei(mean, sigma, best_observed=best_observed, xi=self.xi)
        selected = self._stable_topk(scores, batch_size=batch_size)
        details = [
            {
                "index": int(index),
                "score": float(scores[index]),
                "mean": float(mean[index]),
                "sigma": float(sigma[index]),
                "best_observed": best_observed,
                "xi": self.xi,
            }
            for index in selected
        ]
        return self._build_selection(scores, selected, details)
