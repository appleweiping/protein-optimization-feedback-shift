"""Unit tests for Day 6 acquisition strategies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition.registry import build_acquisition


class AcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = ["A", "B", "C", "D"]
        self.predictions = {
            "mean": np.asarray([0.5, 0.9, 0.8, 0.1], dtype=np.float32),
            "sigma": np.asarray([0.1, 0.05, 0.3, 0.01], dtype=np.float32),
            "best_observed": 0.75,
        }

    def test_greedy_selects_top_mean(self) -> None:
        acquisition = build_acquisition({"name": "greedy"})
        selection = acquisition.select(self.candidates, self.predictions, batch_size=2)
        self.assertEqual(selection.selected_indices, (1, 2))

    def test_ucb_prefers_high_sigma_candidate(self) -> None:
        acquisition = build_acquisition({"name": "ucb", "beta": 2.0})
        selection = acquisition.select(self.candidates, self.predictions, batch_size=1)
        self.assertEqual(selection.selected_indices[0], 2)

    def test_random_is_seeded(self) -> None:
        acquisition_a = build_acquisition({"name": "random", "seed": 7})
        acquisition_b = build_acquisition({"name": "random", "seed": 7})
        selection_a = acquisition_a.select(self.candidates, self.predictions, batch_size=3)
        selection_b = acquisition_b.select(self.candidates, self.predictions, batch_size=3)
        self.assertEqual(selection_a.selected_indices, selection_b.selected_indices)

    def test_ei_returns_stable_batch(self) -> None:
        acquisition = build_acquisition({"name": "ei", "xi": 0.0})
        selection = acquisition.select(self.candidates, self.predictions, batch_size=2)
        self.assertEqual(len(selection.selected_indices), 2)
        self.assertGreaterEqual(selection.selected_scores[0], selection.selected_scores[1])


if __name__ == "__main__":
    unittest.main()
