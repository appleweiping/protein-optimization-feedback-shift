"""Text report helpers for baseline experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_analysis_note(
    path: Path,
    experiment_name: str,
    dataset_name: str,
    split_type: str,
    methods_in_order: list[str],
    final_metric_summary: dict[str, dict[str, float]],
    checks: dict[str, Any],
) -> None:
    """Write a short markdown note summarizing the Day 8 baseline outcome."""
    lines = [
        f"# {experiment_name} analysis note",
        "",
        "## Setup",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Split: `{split_type}`",
        f"- Methods: {', '.join(f'`{name}`' for name in methods_in_order)}",
        "",
        "## Main observations",
        "",
    ]

    for method in methods_in_order:
        summary = final_metric_summary.get(method, {})
        lines.append(
            "- "
            + f"`{method}` final best-so-far mean = {summary.get('final_best_so_far_mean', 0.0):.4f}, "
            + f"final simple regret mean = {summary.get('final_simple_regret_mean', 0.0):.4f}, "
            + f"selected sigma mean = {summary.get('selected_sigma_mean', 0.0):.4f}"
        )

    lines.extend(["", "## Sanity checks", ""])
    for key, value in checks.items():
        lines.append(f"- `{key}` = {str(value).lower()}")

    greedy = final_metric_summary.get("greedy", {})
    random_summary = final_metric_summary.get("random", {})
    ucb = final_metric_summary.get("ucb", {})
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Greedy serving as a stable improvement over random indicates the closed-loop system "
                "is genuinely converting surrogate predictions into optimization gain."
            ),
            (
                "UCB differing from greedy while not universally dominating it is the key Day 8 signal: "
                "uncertainty-aware selection is changing behavior, but its decision value is not automatically guaranteed."
            ),
            (
                f"In this run family, greedy mean final best-so-far = {greedy.get('final_best_so_far_mean', 0.0):.4f}, "
                f"random = {random_summary.get('final_best_so_far_mean', 0.0):.4f}, "
                f"ucb = {ucb.get('final_best_so_far_mean', 0.0):.4f}."
            ),
            "",
            "This note is intended as the first paper-facing evidence that the offline decision loop is both runnable and behaviorally meaningful.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(
    path: Path,
    methods_in_order: list[str],
    final_metric_summary: dict[str, dict[str, float]],
    threshold_hit_summary: dict[str, dict[str, dict[str, float]]],
    seed_stability: dict[str, dict[str, float]],
) -> None:
    """Write the baseline summary table in CSV format."""
    fieldnames = [
        "method",
        "final_best_so_far_mean",
        "final_best_so_far_std",
        "final_simple_regret_mean",
        "best_improvement_mean",
        "selected_sigma_mean",
        "early_stage_sample_efficiency_mean",
        "late_stage_sample_efficiency_mean",
        "seed_std",
        "coefficient_of_variation",
    ]
    dynamic_threshold_fields = []
    for method in methods_in_order:
        dynamic_threshold_fields.extend(
            [
                f"threshold_{threshold_key}_hit_rate"
                for threshold_key in threshold_hit_summary.get(method, {}).keys()
            ]
        )
    fieldnames.extend(sorted(set(dynamic_threshold_fields)))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods_in_order:
            summary = final_metric_summary.get(method, {})
            stability = seed_stability.get(method, {})
            row = {
                "method": method,
                "final_best_so_far_mean": summary.get("final_best_so_far_mean", 0.0),
                "final_best_so_far_std": summary.get("final_best_so_far_std", 0.0),
                "final_simple_regret_mean": summary.get("final_simple_regret_mean", 0.0),
                "best_improvement_mean": summary.get("best_improvement_mean", 0.0),
                "selected_sigma_mean": summary.get("selected_sigma_mean", 0.0),
                "early_stage_sample_efficiency_mean": summary.get("early_stage_sample_efficiency_mean", 0.0),
                "late_stage_sample_efficiency_mean": summary.get("late_stage_sample_efficiency_mean", 0.0),
                "seed_std": stability.get("seed_std", 0.0),
                "coefficient_of_variation": stability.get("coefficient_of_variation", 0.0),
            }
            for threshold_key, stats in threshold_hit_summary.get(method, {}).items():
                row[f"threshold_{threshold_key}_hit_rate"] = stats.get("hit_rate", 0.0)
            writer.writerow(row)


def write_summary_latex(
    path: Path,
    methods_in_order: list[str],
    final_metric_summary: dict[str, dict[str, float]],
    seed_stability: dict[str, dict[str, float]],
) -> None:
    """Write a lightweight LaTeX table for baseline comparisons."""
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Final best $\\uparrow$ & Regret $\\downarrow$ & Selected $\\sigma$ & Seed std \\\\",
        "\\midrule",
    ]
    for method in methods_in_order:
        summary = final_metric_summary.get(method, {})
        stability = seed_stability.get(method, {})
        lines.append(
            f"{method} & "
            f"{summary.get('final_best_so_far_mean', 0.0):.3f} $\\pm$ {summary.get('final_best_so_far_std', 0.0):.3f} & "
            f"{summary.get('final_simple_regret_mean', 0.0):.3f} & "
            f"{summary.get('selected_sigma_mean', 0.0):.3f} & "
            f"{stability.get('seed_std', 0.0):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
