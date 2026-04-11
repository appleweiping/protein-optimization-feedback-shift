"""Shift diagnostics for Day 10 failure analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_matrix(embeddings: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be a rank-2 matrix, received shape {tuple(matrix.shape)}.")
    return matrix


def _pairwise_min_distances(reference_embeddings: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray:
    reference = _as_matrix(reference_embeddings)
    query = _as_matrix(query_embeddings)
    if reference.shape[0] == 0 or query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    return distances.min(axis=1).astype(np.float32, copy=False)


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }
    return {
        "count": float(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
    }


def compute_embedding_distance(
    reference_embeddings: np.ndarray | list[list[float]],
    query_embeddings: np.ndarray | list[list[float]],
) -> dict[str, float]:
    """Measure how far a query set sits from a reference set in embedding space."""
    reference = _as_matrix(reference_embeddings)
    query = _as_matrix(query_embeddings)
    if reference.shape[0] == 0 or query.shape[0] == 0:
        return {
            "reference_count": float(reference.shape[0]),
            "query_count": float(query.shape[0]),
            "mean_centroid_distance": 0.0,
            "mean_nearest_neighbor_distance": 0.0,
            "median_nearest_neighbor_distance": 0.0,
            "max_nearest_neighbor_distance": 0.0,
        }

    reference_centroid = reference.mean(axis=0, keepdims=True)
    centroid_distances = np.linalg.norm(query - reference_centroid, axis=1)
    nn_distances = _pairwise_min_distances(reference, query)
    return {
        "reference_count": float(reference.shape[0]),
        "query_count": float(query.shape[0]),
        "mean_centroid_distance": float(centroid_distances.mean()),
        "mean_nearest_neighbor_distance": float(nn_distances.mean()) if nn_distances.size else 0.0,
        "median_nearest_neighbor_distance": float(np.median(nn_distances)) if nn_distances.size else 0.0,
        "max_nearest_neighbor_distance": float(nn_distances.max()) if nn_distances.size else 0.0,
    }


def compute_support_overlap_proxy(
    reference_embeddings: np.ndarray | list[list[float]],
    query_embeddings: np.ndarray | list[list[float]],
) -> dict[str, float]:
    """Approximate support overlap using nearest-neighbor distances."""
    nn_distances = _pairwise_min_distances(reference_embeddings, query_embeddings)
    if nn_distances.size == 0:
        return {
            "support_overlap_proxy": 0.0,
            "distance_mean": 0.0,
            "distance_std": 0.0,
            "distance_p25": 0.0,
            "distance_p75": 0.0,
        }
    scale = float(nn_distances.mean()) if float(nn_distances.mean()) > 1e-8 else 1.0
    normalized = np.exp(-nn_distances / scale)
    return {
        "support_overlap_proxy": float(normalized.mean()),
        "distance_mean": float(nn_distances.mean()),
        "distance_std": float(nn_distances.std()),
        "distance_p25": float(np.percentile(nn_distances, 25)),
        "distance_p75": float(np.percentile(nn_distances, 75)),
    }


def compute_selection_shift(
    train_embeddings: np.ndarray | list[list[float]],
    candidate_embeddings: np.ndarray | list[list[float]],
    selected_embeddings: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    """Compare selected samples against the full candidate pool under the same train reference."""
    candidate_distance = compute_embedding_distance(train_embeddings, candidate_embeddings)
    selected_distance = compute_embedding_distance(train_embeddings, selected_embeddings)
    candidate_overlap = compute_support_overlap_proxy(train_embeddings, candidate_embeddings)
    selected_overlap = compute_support_overlap_proxy(train_embeddings, selected_embeddings)
    return {
        "candidate_distance": candidate_distance,
        "selected_distance": selected_distance,
        "candidate_support_overlap": candidate_overlap,
        "selected_support_overlap": selected_overlap,
        "distance_gap": {
            "centroid_mean_gap": float(
                selected_distance["mean_centroid_distance"] - candidate_distance["mean_centroid_distance"]
            ),
            "nearest_neighbor_mean_gap": float(
                selected_distance["mean_nearest_neighbor_distance"]
                - candidate_distance["mean_nearest_neighbor_distance"]
            ),
        },
        "support_overlap_gap": float(
            selected_overlap["support_overlap_proxy"] - candidate_overlap["support_overlap_proxy"]
        ),
    }


def summarize_shift_series(records: list[dict[str, float]], key: str) -> dict[str, float]:
    """Summarize one scalar field over a series of shift diagnostic records."""
    values = np.asarray([float(record[key]) for record in records if key in record], dtype=np.float32)
    return _summary(values)
