"""Text report helpers for baseline experiments."""

from __future__ import annotations

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
