"""Evaluation metrics built directly from closed-loop recorder outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_best_so_far_curve(loop_summary: dict[str, Any]) -> list[dict[str, float]]:
    """Read the best-so-far trajectory emitted by the recorder."""
    trajectory = loop_summary.get("trajectory", [])
    return [
        {
            "step": float(point["step"]),
            "best_so_far": float(point["best_so_far"]),
        }
        for point in trajectory
    ]


def compute_simple_regret_curve(
    loop_summary: dict[str, Any],
    global_best_fitness: float,
) -> list[dict[str, float]]:
    """Compute simple regret against the best fitness available in the split."""
    curve = compute_best_so_far_curve(loop_summary)
    optimum = float(global_best_fitness)
    return [
        {
            "step": point["step"],
            "simple_regret": float(optimum - point["best_so_far"]),
        }
        for point in curve
    ]


def compute_average_round_improvement(loop_summary: dict[str, Any]) -> float:
    """Average best-so-far gain across rounds."""
    curve = compute_best_so_far_curve(loop_summary)
    if len(curve) <= 1:
        return 0.0
    improvements = [
        curve[index]["best_so_far"] - curve[index - 1]["best_so_far"]
        for index in range(1, len(curve))
    ]
    return float(sum(improvements) / len(improvements))


def summarize_round_selection_stats(round_payloads: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Summarize selected-point statistics for every loop round."""
    summaries: list[dict[str, float]] = []
    for payload in round_payloads:
        selected = payload.get("selected", [])
        if selected:
            mean_predicted_mu = sum(float(item["predicted_mean"]) for item in selected) / len(selected)
            mean_predicted_sigma = sum(float(item["predicted_sigma"]) for item in selected) / len(selected)
            mean_true_fitness = sum(float(item["true_fitness"]) for item in selected) / len(selected)
            max_true_fitness = max(float(item["true_fitness"]) for item in selected)
        else:
            mean_predicted_mu = 0.0
            mean_predicted_sigma = 0.0
            mean_true_fitness = 0.0
            max_true_fitness = 0.0

        candidate_summary = payload.get("candidate_prediction_summary", {})
        summaries.append(
            {
                "step": float(payload.get("round_index", 0)) + 1.0,
                "round_index": float(payload.get("round_index", 0)),
                "selected_count": float(payload.get("selected_count", 0)),
                "mean_predicted_mu": float(mean_predicted_mu),
                "mean_predicted_sigma": float(mean_predicted_sigma),
                "mean_true_fitness": float(mean_true_fitness),
                "max_true_fitness": float(max_true_fitness),
                "candidate_mean_mu": float(candidate_summary.get("mean_mu", 0.0)),
                "candidate_mean_sigma": float(candidate_summary.get("mean_sigma", 0.0)),
            }
        )
    return summaries


def build_run_metrics(
    loop_summary_path: Path,
    rounds_jsonl_path: Path,
    split_statistics: dict[str, Any],
) -> dict[str, Any]:
    """Build the full metric bundle for a single recorder output."""
    loop_summary = load_json(loop_summary_path)
    round_payloads = load_jsonl(rounds_jsonl_path)
    train_best = float(split_statistics["train_summary"]["fitness"]["max"])
    candidate_best = float(split_statistics["candidate_summary"]["fitness"]["max"])
    global_best_fitness = max(train_best, candidate_best)

    best_so_far_curve = compute_best_so_far_curve(loop_summary)
    simple_regret_curve = compute_simple_regret_curve(loop_summary, global_best_fitness)
    round_stats = summarize_round_selection_stats(round_payloads)

    return {
        "label": loop_summary.get("label"),
        "initial_best_so_far": float(loop_summary.get("initial_best_so_far", 0.0)),
        "final_best_so_far": float(loop_summary.get("final_best_so_far", 0.0)),
        "best_improvement": float(loop_summary.get("best_improvement", 0.0)),
        "average_round_improvement": compute_average_round_improvement(loop_summary),
        "global_best_fitness": float(global_best_fitness),
        "final_simple_regret": float(simple_regret_curve[-1]["simple_regret"]) if simple_regret_curve else 0.0,
        "best_so_far_curve": best_so_far_curve,
        "simple_regret_curve": simple_regret_curve,
        "round_selection_stats": round_stats,
        "round_count": int(loop_summary.get("round_count", 0)),
        "total_selected": int(loop_summary.get("total_selected", 0)),
        "stopping_reason": loop_summary.get("stopping_reason", "unknown"),
        "paths": dict(loop_summary.get("paths", {})),
    }


def aggregate_metric_curves(
    run_metrics_by_method: dict[str, list[dict[str, Any]]],
    curve_key: str,
    value_key: str,
) -> dict[str, list[dict[str, float]]]:
    """Aggregate aligned metric curves across random seeds."""
    aggregated: dict[str, list[dict[str, float]]] = {}
    for method, run_metrics_list in run_metrics_by_method.items():
        if not run_metrics_list:
            aggregated[method] = []
            continue

        step_count = min(len(run_metrics[curve_key]) for run_metrics in run_metrics_list)
        method_points: list[dict[str, float]] = []
        for index in range(step_count):
            step = float(run_metrics_list[0][curve_key][index]["step"])
            values = [float(run_metrics[curve_key][index][value_key]) for run_metrics in run_metrics_list]
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            method_points.append(
                {
                    "step": float(step),
                    "mean": float(mean_value),
                    "std": float(variance**0.5),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
            )
        aggregated[method] = method_points
    return aggregated


def aggregate_final_metrics(run_metrics_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Aggregate scalar final metrics across random seeds."""
    aggregates: dict[str, dict[str, float]] = {}
    for method, run_metrics_list in run_metrics_by_method.items():
        final_best_values = [float(run_metrics["final_best_so_far"]) for run_metrics in run_metrics_list]
        final_regret_values = [float(run_metrics["final_simple_regret"]) for run_metrics in run_metrics_list]
        improvement_values = [float(run_metrics["best_improvement"]) for run_metrics in run_metrics_list]
        sigma_values = [
            float(stat["mean_predicted_sigma"])
            for run_metrics in run_metrics_list
            for stat in run_metrics["round_selection_stats"]
        ]
        true_fitness_values = [
            float(stat["mean_true_fitness"])
            for run_metrics in run_metrics_list
            for stat in run_metrics["round_selection_stats"]
        ]

        def summarize(values: list[float], prefix: str) -> dict[str, float]:
            if not values:
                return {
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_min": 0.0,
                    f"{prefix}_max": 0.0,
                }
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            return {
                f"{prefix}_mean": float(mean_value),
                f"{prefix}_std": float(variance**0.5),
                f"{prefix}_min": float(min(values)),
                f"{prefix}_max": float(max(values)),
            }

        aggregate = {}
        aggregate.update(summarize(final_best_values, "final_best_so_far"))
        aggregate.update(summarize(final_regret_values, "final_simple_regret"))
        aggregate.update(summarize(improvement_values, "best_improvement"))
        aggregate.update(summarize(sigma_values, "selected_sigma"))
        aggregate.update(summarize(true_fitness_values, "selected_true_fitness"))
        aggregate["seed_count"] = float(len(run_metrics_list))
        aggregates[method] = aggregate
    return aggregates
