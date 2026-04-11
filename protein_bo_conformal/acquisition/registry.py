"""Registry and shared interfaces for acquisition strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AcquisitionSelection:
    """Structured selection result returned by every acquisition strategy."""

    name: str
    batch_size: int
    selected_indices: tuple[int, ...]
    selected_scores: tuple[float, ...]
    selected_details: tuple[dict[str, Any], ...]
    score_summary: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "selected_indices": list(self.selected_indices),
            "selected_scores": list(self.selected_scores),
            "selected_details": list(self.selected_details),
            "score_summary": dict(self.score_summary),
        }


class BaseAcquisition:
    """Common acquisition protocol for all decision rules."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = dict(config)

    def select(
        self,
        candidates: list[Any] | tuple[Any, ...],
        predictions: dict[str, Any],
        batch_size: int,
    ) -> AcquisitionSelection:
        raise NotImplementedError

    def _mean_vector(self, predictions: dict[str, Any]) -> np.ndarray:
        return _vector_from_predictions(predictions, "mean")

    def _sigma_vector(self, predictions: dict[str, Any]) -> np.ndarray:
        if "sigma" not in predictions:
            return np.zeros_like(self._mean_vector(predictions))
        return _vector_from_predictions(predictions, "sigma")

    def _best_observed(self, predictions: dict[str, Any]) -> float:
        if "best_observed" in predictions:
            return float(predictions["best_observed"])
        means = self._mean_vector(predictions)
        return float(means.max()) if means.size else 0.0

    def _candidate_count(self, candidates: list[Any] | tuple[Any, ...], predictions: dict[str, Any]) -> int:
        means = self._mean_vector(predictions)
        if len(candidates) != int(means.shape[0]):
            raise ValueError(
                f"Candidate count {len(candidates)} does not match prediction count {int(means.shape[0])}."
            )
        return len(candidates)

    def _stable_topk(self, scores: np.ndarray, batch_size: int) -> np.ndarray:
        if scores.ndim != 1:
            raise ValueError("Acquisition scores must be rank-1.")
        if scores.size == 0:
            return np.zeros((0,), dtype=np.int64)
        limited = min(max(1, int(batch_size)), int(scores.shape[0]))
        return np.argsort(-scores, kind="mergesort")[:limited]

    def _build_selection(
        self,
        scores: np.ndarray,
        selected_indices: np.ndarray,
        extra_details: list[dict[str, Any]],
    ) -> AcquisitionSelection:
        selected_scores = tuple(float(scores[index]) for index in selected_indices)
        return AcquisitionSelection(
            name=self.name,
            batch_size=int(len(selected_indices)),
            selected_indices=tuple(int(index) for index in selected_indices.tolist()),
            selected_scores=selected_scores,
            selected_details=tuple(extra_details),
            score_summary={
                "max_score": float(scores.max()) if scores.size else 0.0,
                "min_score": float(scores.min()) if scores.size else 0.0,
                "mean_score": float(scores.mean()) if scores.size else 0.0,
                "std_score": float(scores.std()) if scores.size else 0.0,
            },
        )


def _vector_from_predictions(predictions: dict[str, Any], key: str) -> np.ndarray:
    if key not in predictions:
        raise ValueError(f"Predictions are missing required key '{key}'.")
    vector = np.asarray(predictions[key], dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Prediction field '{key}' must be rank-1, received shape {tuple(vector.shape)}.")
    return vector


def gaussian_ei(mean: np.ndarray, sigma: np.ndarray, best_observed: float, xi: float) -> np.ndarray:
    """Compute expected improvement under a Gaussian surrogate approximation."""
    sigma = np.maximum(sigma.astype(np.float32, copy=False), 1e-8)
    improvement = mean - float(best_observed) - float(xi)
    z = improvement / sigma
    normal_pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return (improvement * normal_cdf + sigma * normal_pdf).astype(np.float32, copy=False)


_ACQUISITION_REGISTRY: dict[str, Callable[[dict[str, Any]], BaseAcquisition]] = {}


def register_acquisition(name: str, builder: Callable[[dict[str, Any]], BaseAcquisition]) -> None:
    """Register a named acquisition builder."""
    normalized = name.strip().lower()
    _ACQUISITION_REGISTRY[normalized] = builder


def build_acquisition(config: dict[str, Any]) -> BaseAcquisition:
    """Construct an acquisition strategy from config."""
    from acquisition.conformal_ucb import ConformalUCBAcquisition
    from acquisition.ei import ExpectedImprovementAcquisition
    from acquisition.greedy import GreedyAcquisition
    from acquisition.random import RandomAcquisition
    from acquisition.ucb import UCBAcquisition

    if not _ACQUISITION_REGISTRY:
        register_acquisition("random", lambda cfg: RandomAcquisition(cfg))
        register_acquisition("greedy", lambda cfg: GreedyAcquisition(cfg))
        register_acquisition("ucb", lambda cfg: UCBAcquisition(cfg))
        register_acquisition("ei", lambda cfg: ExpectedImprovementAcquisition(cfg))
        register_acquisition("conformal_ucb", lambda cfg: ConformalUCBAcquisition(cfg))

    name = str(config.get("name") or config.get("acquisition_type") or "greedy").strip().lower()
    if name not in _ACQUISITION_REGISTRY:
        raise ValueError(f"Unsupported acquisition strategy '{name}'.")
    return _ACQUISITION_REGISTRY[name](config)
