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


def compute_threshold_hit_time(
    best_so_far_curve: list[dict[str, float]],
    threshold: float,
) -> float | None:
    """Return the first step where best-so-far reaches a target threshold."""
    target = float(threshold)
    for point in best_so_far_curve:
        if float(point["best_so_far"]) >= target:
            return float(point["step"])
    return None


def compute_sample_efficiency_curve(
    best_so_far_curve: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Compute cumulative gain per consumed budget step."""
    if not best_so_far_curve:
        return []
    initial_best = float(best_so_far_curve[0]["best_so_far"])
    curve: list[dict[str, float]] = []
    for point in best_so_far_curve:
        step = float(point["step"])
        gain = float(point["best_so_far"]) - initial_best
        efficiency = gain / step if step > 0 else 0.0
        curve.append(
            {
                "step": step,
                "sample_efficiency": float(efficiency),
            }
        )
    return curve


def compute_stage_metrics(best_so_far_curve: list[dict[str, float]]) -> dict[str, float]:
    """Split the trajectory into early and late stages for budget-dependent analysis."""
    if len(best_so_far_curve) <= 1:
        return {
            "early_stage_gain": 0.0,
            "late_stage_gain": 0.0,
            "early_stage_sample_efficiency": 0.0,
            "late_stage_sample_efficiency": 0.0,
        }

    midpoint_index = max(1, (len(best_so_far_curve) - 1) // 2)
    initial_best = float(best_so_far_curve[0]["best_so_far"])
    midpoint_best = float(best_so_far_curve[midpoint_index]["best_so_far"])
    final_best = float(best_so_far_curve[-1]["best_so_far"])
    midpoint_step = float(best_so_far_curve[midpoint_index]["step"])
    final_step = float(best_so_far_curve[-1]["step"])
    late_budget = max(final_step - midpoint_step, 1e-6)

    early_gain = midpoint_best - initial_best
    late_gain = final_best - midpoint_best
    return {
        "early_stage_gain": float(early_gain),
        "late_stage_gain": float(late_gain),
        "early_stage_sample_efficiency": float(early_gain / midpoint_step) if midpoint_step > 0 else 0.0,
        "late_stage_sample_efficiency": float(late_gain / late_budget),
    }


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
            min_true_fitness = min(float(item["true_fitness"]) for item in selected)
        else:
            mean_predicted_mu = 0.0
            mean_predicted_sigma = 0.0
            mean_true_fitness = 0.0
            max_true_fitness = 0.0
            min_true_fitness = 0.0

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
                "min_true_fitness": float(min_true_fitness),
                "candidate_mean_mu": float(candidate_summary.get("mean_mu", 0.0)),
                "candidate_mean_sigma": float(candidate_summary.get("mean_sigma", 0.0)),
            }
        )
    return summaries


def compute_selection_statistics(round_selection_stats: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate round-level selection behavior into a compact summary."""
    if not round_selection_stats:
        return {
            "selected_mu_mean": 0.0,
            "selected_sigma_mean": 0.0,
            "selected_true_fitness_mean": 0.0,
            "candidate_mu_mean": 0.0,
            "candidate_sigma_mean": 0.0,
        }

    def mean_of(key: str) -> float:
        values = [float(item[key]) for item in round_selection_stats]
        return float(sum(values) / len(values))

    return {
        "selected_mu_mean": mean_of("mean_predicted_mu"),
        "selected_sigma_mean": mean_of("mean_predicted_sigma"),
        "selected_true_fitness_mean": mean_of("mean_true_fitness"),
        "candidate_mu_mean": mean_of("candidate_mean_mu"),
        "candidate_sigma_mean": mean_of("candidate_mean_sigma"),
    }


def build_run_metrics(
    loop_summary_path: Path,
    rounds_jsonl_path: Path,
    split_statistics: dict[str, Any],
    threshold_fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Build the full metric bundle for a single recorder output."""
    loop_summary = load_json(loop_summary_path)
    round_payloads = load_jsonl(rounds_jsonl_path)
    train_best = float(split_statistics["train_summary"]["fitness"]["max"])
    candidate_best = float(split_statistics["candidate_summary"]["fitness"]["max"])
    global_best_fitness = max(train_best, candidate_best)

    best_so_far_curve = compute_best_so_far_curve(loop_summary)
    simple_regret_curve = compute_simple_regret_curve(loop_summary, global_best_fitness)
    sample_efficiency_curve = compute_sample_efficiency_curve(best_so_far_curve)
    round_stats = summarize_round_selection_stats(round_payloads)
    selection_statistics = compute_selection_statistics(round_stats)
    stage_metrics = compute_stage_metrics(best_so_far_curve)
    threshold_hits: dict[str, float | None] = {}
    for fraction in threshold_fractions or []:
        threshold_value = float(global_best_fitness) * float(fraction)
        threshold_hits[f"{float(fraction):.2f}"] = compute_threshold_hit_time(best_so_far_curve, threshold_value)

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
        "sample_efficiency_curve": sample_efficiency_curve,
        "round_selection_stats": round_stats,
        "selection_statistics": selection_statistics,
        "stage_metrics": stage_metrics,
        "threshold_hit_times": threshold_hits,
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
        early_efficiency_values = [
            float(run_metrics["stage_metrics"]["early_stage_sample_efficiency"])
            for run_metrics in run_metrics_list
        ]
        late_efficiency_values = [
            float(run_metrics["stage_metrics"]["late_stage_sample_efficiency"])
            for run_metrics in run_metrics_list
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
        aggregate.update(summarize(early_efficiency_values, "early_stage_sample_efficiency"))
        aggregate.update(summarize(late_efficiency_values, "late_stage_sample_efficiency"))
        aggregate["seed_count"] = float(len(run_metrics_list))
        aggregates[method] = aggregate
    return aggregates


def aggregate_threshold_hit_times(
    run_metrics_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate threshold hit times and hit rates across random seeds."""
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for method, run_metrics_list in run_metrics_by_method.items():
        threshold_keys = sorted(
            {
                key
                for run_metrics in run_metrics_list
                for key in run_metrics.get("threshold_hit_times", {}).keys()
            }
        )
        method_summary: dict[str, dict[str, float]] = {}
        for threshold_key in threshold_keys:
            observed = [
                value
                for run_metrics in run_metrics_list
                for value in [run_metrics.get("threshold_hit_times", {}).get(threshold_key)]
                if value is not None
            ]
            hit_rate = (
                sum(1 for run_metrics in run_metrics_list if run_metrics.get("threshold_hit_times", {}).get(threshold_key) is not None)
                / max(len(run_metrics_list), 1)
            )
            if observed:
                mean_value = sum(float(value) for value in observed) / len(observed)
                variance = sum((float(value) - mean_value) ** 2 for value in observed) / len(observed)
                method_summary[threshold_key] = {
                    "hit_rate": float(hit_rate),
                    "mean_hit_time": float(mean_value),
                    "std_hit_time": float(variance**0.5),
                    "min_hit_time": float(min(observed)),
                    "max_hit_time": float(max(observed)),
                }
            else:
                method_summary[threshold_key] = {
                    "hit_rate": float(hit_rate),
                    "mean_hit_time": -1.0,
                    "std_hit_time": 0.0,
                    "min_hit_time": -1.0,
                    "max_hit_time": -1.0,
                }
        aggregates[method] = method_summary
    return aggregates


def compute_seed_stability(run_metrics_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Compute cross-seed stability diagnostics for final performance."""
    stability: dict[str, dict[str, float]] = {}
    for method, run_metrics_list in run_metrics_by_method.items():
        values = [float(run_metrics["final_best_so_far"]) for run_metrics in run_metrics_list]
        if not values:
            stability[method] = {
                "seed_variance": 0.0,
                "seed_std": 0.0,
                "coefficient_of_variation": 0.0,
                "range": 0.0,
            }
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std = variance**0.5
        stability[method] = {
            "seed_variance": float(variance),
            "seed_std": float(std),
            "coefficient_of_variation": float(std / abs(mean_value)) if abs(mean_value) > 1e-8 else 0.0,
            "range": float(max(values) - min(values)),
        }
    return stability
