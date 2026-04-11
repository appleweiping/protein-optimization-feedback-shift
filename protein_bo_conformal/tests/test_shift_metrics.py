"""Tests for Day 10 shift diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.metrics import compute_mu_sigma_correlation, compute_uncertainty_behavior
from evaluation.shift_metrics import (
    compute_embedding_distance,
    compute_selection_shift,
    compute_support_overlap_proxy,
)


class ShiftMetricsTest(unittest.TestCase):
    def test_compute_embedding_distance_reports_positive_distances(self) -> None:
        train = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        query = np.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        summary = compute_embedding_distance(train, query)
        self.assertGreater(summary["mean_centroid_distance"], 0.0)
        self.assertGreater(summary["mean_nearest_neighbor_distance"], 0.0)

    def test_compute_selection_shift_detects_farther_selected_points(self) -> None:
        train = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        candidate = np.asarray([[0.1, 0.0], [0.0, 0.2], [0.2, 0.1]], dtype=np.float32)
        selected = np.asarray([[2.0, 2.0], [2.5, 2.5]], dtype=np.float32)
        summary = compute_selection_shift(train, candidate, selected)
        self.assertGreater(summary["distance_gap"]["centroid_mean_gap"], 0.0)
        self.assertLess(summary["support_overlap_gap"], 0.0)

    def test_uncertainty_behavior_reports_correlations(self) -> None:
        round_payloads = [
            {
                "round_index": 0,
                "selected": [
                    {"predicted_mean": 0.0, "predicted_sigma": 0.1, "true_fitness": 0.0},
                    {"predicted_mean": 0.5, "predicted_sigma": 0.4, "true_fitness": -0.5},
                ],
                "candidate_prediction_summary": {"mean_sigma": 0.2},
            }
        ]
        summary = compute_uncertainty_behavior(round_payloads)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["candidate_sigma_mean"], 0.2)
        self.assertNotEqual(compute_mu_sigma_correlation(round_payloads[0]["selected"]), 0.0)

    def test_support_overlap_proxy_is_bounded(self) -> None:
        train = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        query = np.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        summary = compute_support_overlap_proxy(train, query)
        self.assertGreaterEqual(summary["support_overlap_proxy"], 0.0)
        self.assertLessEqual(summary["support_overlap_proxy"], 1.0)


if __name__ == "__main__":
    unittest.main()
