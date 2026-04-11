"""Tests for Day 8 evaluation helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import (
    aggregate_final_metrics,
    aggregate_metric_curves,
    aggregate_threshold_hit_times,
    build_run_metrics,
    compute_seed_stability,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_build_run_metrics_reads_recorder_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            loop_summary_path = temp_root / "loop_summary.json"
            rounds_path = temp_root / "rounds.jsonl"
            loop_summary_path.write_text(
                json.dumps(
                    {
                        "label": "greedy",
                        "initial_best_so_far": 0.5,
                        "final_best_so_far": 1.5,
                        "best_improvement": 1.0,
                        "round_count": 2,
                        "total_selected": 4,
                        "stopping_reason": "max_rounds_reached",
                        "trajectory": [
                            {"step": 0.0, "best_so_far": 0.5},
                            {"step": 1.0, "best_so_far": 1.0},
                            {"step": 2.0, "best_so_far": 1.5},
                        ],
                        "paths": {"rounds_jsonl": str(rounds_path)},
                    }
                ),
                encoding="utf-8",
            )
            rounds_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "round_index": 0,
                                "selected_count": 2,
                                "selected": [
                                    {"predicted_mean": 0.2, "predicted_sigma": 0.1, "true_fitness": 0.9},
                                    {"predicted_mean": 0.3, "predicted_sigma": 0.2, "true_fitness": 1.0},
                                ],
                                "candidate_prediction_summary": {"mean_mu": 0.15, "mean_sigma": 0.05},
                            }
                        ),
                        json.dumps(
                            {
                                "round_index": 1,
                                "selected_count": 2,
                                "selected": [
                                    {"predicted_mean": 0.5, "predicted_sigma": 0.3, "true_fitness": 1.4},
                                    {"predicted_mean": 0.6, "predicted_sigma": 0.4, "true_fitness": 1.5},
                                ],
                                "candidate_prediction_summary": {"mean_mu": 0.4, "mean_sigma": 0.25},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            split_statistics = {
                "train_summary": {"fitness": {"max": 0.5}},
                "candidate_summary": {"fitness": {"max": 2.0}},
            }

            metrics = build_run_metrics(loop_summary_path, rounds_path, split_statistics)
            self.assertEqual(metrics["label"], "greedy")
            self.assertAlmostEqual(metrics["final_simple_regret"], 0.5)
            self.assertAlmostEqual(metrics["average_round_improvement"], 0.5)
            self.assertEqual(len(metrics["round_selection_stats"]), 2)
            self.assertEqual(len(metrics["sample_efficiency_curve"]), 3)

    def test_aggregate_helpers_average_across_seeds(self) -> None:
        run_metrics = {
            "greedy": [
                {
                    "final_best_so_far": 1.2,
                    "final_simple_regret": 0.4,
                    "best_improvement": 0.7,
                    "threshold_hit_times": {"0.50": 1.0},
                    "stage_metrics": {
                        "early_stage_sample_efficiency": 0.7,
                        "late_stage_sample_efficiency": 0.0,
                    },
                    "best_so_far_curve": [
                        {"step": 0.0, "best_so_far": 0.5},
                        {"step": 1.0, "best_so_far": 1.2},
                    ],
                    "sample_efficiency_curve": [
                        {"step": 0.0, "sample_efficiency": 0.0},
                        {"step": 1.0, "sample_efficiency": 0.7},
                    ],
                    "round_selection_stats": [
                        {"step": 0.0, "mean_predicted_sigma": 0.2, "mean_true_fitness": 0.8},
                    ],
                },
                {
                    "final_best_so_far": 1.4,
                    "final_simple_regret": 0.2,
                    "best_improvement": 0.9,
                    "threshold_hit_times": {"0.50": 1.0},
                    "stage_metrics": {
                        "early_stage_sample_efficiency": 0.9,
                        "late_stage_sample_efficiency": 0.0,
                    },
                    "best_so_far_curve": [
                        {"step": 0.0, "best_so_far": 0.6},
                        {"step": 1.0, "best_so_far": 1.4},
                    ],
                    "sample_efficiency_curve": [
                        {"step": 0.0, "sample_efficiency": 0.0},
                        {"step": 1.0, "sample_efficiency": 0.9},
                    ],
                    "round_selection_stats": [
                        {"step": 0.0, "mean_predicted_sigma": 0.3, "mean_true_fitness": 0.9},
                    ],
                },
            ]
        }
        curve = aggregate_metric_curves(run_metrics, "best_so_far_curve", "best_so_far")
        final_summary = aggregate_final_metrics(run_metrics)
        threshold_summary = aggregate_threshold_hit_times(run_metrics)
        stability = compute_seed_stability(run_metrics)
        self.assertAlmostEqual(curve["greedy"][1]["mean"], 1.3)
        self.assertAlmostEqual(final_summary["greedy"]["final_best_so_far_mean"], 1.3)
        self.assertAlmostEqual(final_summary["greedy"]["selected_sigma_mean"], 0.25)
        self.assertAlmostEqual(threshold_summary["greedy"]["0.50"]["hit_rate"], 1.0)
        self.assertAlmostEqual(stability["greedy"]["seed_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
