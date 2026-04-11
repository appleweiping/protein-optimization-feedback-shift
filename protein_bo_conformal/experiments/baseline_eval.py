"""Standardized Day 9 evaluation package entrypoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    aggregate_final_metrics,
    aggregate_metric_curves,
    aggregate_threshold_hit_times,
    build_run_metrics,
    compute_seed_stability,
)
from evaluation.plotting import write_bar_svg, write_curve_svg, write_grouped_bar_svg
from evaluation.report import write_analysis_note, write_summary_csv, write_summary_latex
from loop.runner import ClosedLoopRunner
from utils.config import ConfigNode, dump_yaml, load_config, stable_config_hash
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Day 9 evaluation package over baseline methods."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/day8_baseline_greedy_ucb.yaml",
        help="Experiment config merged on top of the default config stack.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional experiment name override.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "run"


def create_layout(root_dir: Path) -> dict[str, Path]:
    layout = {
        "run_dir": root_dir,
        "logs_dir": root_dir / "logs",
        "checkpoints_dir": root_dir / "checkpoints",
        "plots_dir": root_dir / "plots",
        "tables_dir": root_dir / "tables",
        "artifacts_dir": root_dir / "artifacts",
        "report_dir": root_dir / "report",
        "runs_dir": root_dir / "runs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def build_subrun_layout(run_dir: Path) -> dict[str, Path]:
    layout = {
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "checkpoints_dir": run_dir / "checkpoints",
        "plots_dir": run_dir / "plots",
        "tables_dir": run_dir / "tables",
        "artifacts_dir": run_dir / "artifacts",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_experiment_config(project_root: Path, override_path: Path, name_override: str | None) -> ConfigNode:
    base_config = project_root / "config" / "base.yaml"
    default_layers = [
        project_root / "config" / "dataset.yaml",
        project_root / "config" / "representation.yaml",
        project_root / "config" / "model.yaml",
        project_root / "config" / "acquisition.yaml",
    ]
    config = load_config(base_config, override_path, default_layer_paths=default_layers)
    config_dict = config.to_dict()
    if name_override:
        config_dict.setdefault("experiment", {})
        config_dict["experiment"]["name"] = name_override
    return ConfigNode(config_dict)


def derive_run_config(base_config: ConfigNode, acquisition_name: str, seed: int, run_name: str) -> ConfigNode:
    config_dict = base_config.to_dict()
    config_dict.setdefault("experiment", {})
    config_dict["experiment"]["name"] = run_name
    config_dict["experiment"]["description"] = (
        f"Day 9 evaluation run for acquisition={acquisition_name}, seed={seed}."
    )
    config_dict.setdefault("runtime", {})
    config_dict["runtime"]["seed"] = int(seed)
    config_dict.setdefault("dataset", {})
    config_dict["dataset"]["split_seed"] = int(seed)
    config_dict.setdefault("model", {})
    config_dict["model"]["seed"] = int(seed)
    config_dict.setdefault("acquisition", {})
    config_dict["acquisition"]["name"] = acquisition_name
    config_dict["acquisition"]["acquisition_type"] = acquisition_name
    config_dict["acquisition"]["seed"] = int(seed)
    config_dict.setdefault("loop", {})
    config_dict["loop"]["comparison_acquisition_names"] = []
    config_hash = stable_config_hash(config_dict)
    config_dict.setdefault("runtime", {})
    config_dict["runtime"]["config_hash"] = config_hash
    return ConfigNode(config_dict)


def run_single_baseline(
    project_root: Path,
    config: ConfigNode,
    layout: dict[str, Path],
    experiment_logger: Any,
) -> dict[str, Any]:
    config_hash = str(config.runtime.config_hash)
    logger = build_logger(
        name=f"protein_bo_conformal.baseline.{config.experiment.name}",
        log_dir=layout["logs_dir"],
        experiment_id=config.experiment.name,
        level=str(config.runtime.log_level),
    )
    seed_report = set_global_seed(
        seed=int(config.runtime.seed),
        deterministic=bool(config.runtime.deterministic),
        logger=logger,
    )
    device_info = resolve_device(config.runtime.device, logger=logger)
    context = {
        "project_root": project_root,
        "run_id": config.experiment.name,
        "run_dir": layout["run_dir"],
        "paths": layout,
        "seed_report": seed_report,
        "device_info": device_info.to_dict(),
        "config_hash": config_hash,
    }
    (layout["run_dir"] / "config_snapshot.yaml").write_text(
        dump_yaml(config.to_dict()),
        encoding="utf-8",
    )
    runner = ClosedLoopRunner(config=config, logger=logger, context=context)
    summary = runner.run()
    experiment_logger.info(
        "Finished subrun %s final summary written to %s",
        config.experiment.name,
        layout["artifacts_dir"] / "runner_summary.json",
    )
    return summary


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    override_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_experiment_config(project_root, override_path, args.name)
    experiment_hash = stable_config_hash(config.to_dict())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = sanitize_name(config.experiment.name)
    experiment_dir = project_root / "outputs" / "results" / "baseline" / f"{timestamp}-{experiment_name}-{experiment_hash[:8]}"
    layout = create_layout(experiment_dir)
    logger = build_logger(
        name=f"protein_bo_conformal.baseline_suite.{experiment_name}",
        log_dir=layout["logs_dir"],
        experiment_id=experiment_name,
        level=str(config.runtime.log_level),
    )

    experiment_section = config.experiment.to_dict()
    seeds = [int(value) for value in experiment_section.get("random_seeds", [7])]
    methods = [str(value).strip().lower() for value in experiment_section.get("baseline_methods", ["random", "greedy", "ucb"])]
    threshold_fractions = [float(value) for value in experiment_section.get("threshold_fractions_of_global_best", [0.5, 0.8, 0.95])]
    logger.info("Starting Day 9 evaluation package: methods=%s seeds=%s", methods, seeds)

    run_metrics_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subrun_index: list[dict[str, Any]] = []
    split_statistics_reference: dict[str, Any] | None = None
    for seed in seeds:
        for method in methods:
            run_name = f"{experiment_name}-{method}-seed{seed}"
            derived_config = derive_run_config(config, method, seed, run_name)
            subrun_hash = str(derived_config.runtime.config_hash)[:8]
            subrun_dir = layout["runs_dir"] / f"{method}-seed{seed}-{subrun_hash}"
            subrun_layout = build_subrun_layout(subrun_dir)
            logger.info("Running baseline subrun method=%s seed=%s", method, seed)
            summary = run_single_baseline(
                project_root=project_root,
                config=derived_config,
                layout=subrun_layout,
                experiment_logger=logger,
            )
            split_statistics = dict(summary["split_statistics"])
            split_statistics_reference = split_statistics
            method_summary = summary["loop_suite_summary"]["methods"][method]
            loop_summary_path = Path(method_summary["paths"]["rounds_jsonl"]).with_name(f"{method}_loop_summary.json")
            rounds_jsonl_path = Path(method_summary["paths"]["rounds_jsonl"])
            run_metrics = build_run_metrics(
                loop_summary_path,
                rounds_jsonl_path,
                split_statistics,
                threshold_fractions=threshold_fractions,
            )
            run_metrics["seed"] = int(seed)
            run_metrics["run_dir"] = str(subrun_dir)
            run_metrics["method"] = method
            metrics_path = subrun_layout["artifacts_dir"] / f"{method}_metrics.json"
            write_json(metrics_path, run_metrics)
            run_metrics_by_method[method].append(run_metrics)
            subrun_index.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "run_dir": str(subrun_dir),
                    "metrics_path": str(metrics_path),
                    "runner_summary_path": str(subrun_layout["artifacts_dir"] / "runner_summary.json"),
                }
            )

    final_metric_summary = aggregate_final_metrics(run_metrics_by_method)
    best_so_far_curves = aggregate_metric_curves(run_metrics_by_method, "best_so_far_curve", "best_so_far")
    simple_regret_curves = aggregate_metric_curves(run_metrics_by_method, "simple_regret_curve", "simple_regret")
    sample_efficiency_curves = aggregate_metric_curves(run_metrics_by_method, "sample_efficiency_curve", "sample_efficiency")
    selected_sigma_curves = aggregate_metric_curves(run_metrics_by_method, "round_selection_stats", "mean_predicted_sigma")
    selected_fitness_curves = aggregate_metric_curves(run_metrics_by_method, "round_selection_stats", "mean_true_fitness")
    threshold_hit_summary = aggregate_threshold_hit_times(run_metrics_by_method)
    seed_stability = compute_seed_stability(run_metrics_by_method)

    best_curve_path = layout["plots_dir"] / "baseline_best_so_far.svg"
    regret_curve_path = layout["plots_dir"] / "baseline_simple_regret.svg"
    sample_efficiency_curve_path = layout["plots_dir"] / "baseline_sample_efficiency.svg"
    sigma_curve_path = layout["plots_dir"] / "baseline_selected_sigma.svg"
    fitness_curve_path = layout["plots_dir"] / "baseline_selected_true_fitness.svg"
    final_best_bar_path = layout["plots_dir"] / "baseline_final_best_so_far_bar.svg"
    stage_efficiency_bar_path = layout["plots_dir"] / "baseline_stage_efficiency.svg"
    stability_bar_path = layout["plots_dir"] / "baseline_seed_stability.svg"
    write_curve_svg(best_curve_path, "Day 9 evaluation: best-so-far vs budget", best_so_far_curves, "best-so-far")
    write_curve_svg(regret_curve_path, "Day 9 evaluation: simple regret vs budget", simple_regret_curves, "simple regret")
    write_curve_svg(
        sample_efficiency_curve_path,
        "Day 9 evaluation: sample efficiency vs budget",
        sample_efficiency_curves,
        "sample efficiency",
    )
    write_curve_svg(sigma_curve_path, "Day 9 evaluation: selected sigma by round", selected_sigma_curves, "selected sigma")
    write_curve_svg(fitness_curve_path, "Day 9 evaluation: selected true fitness by round", selected_fitness_curves, "selected true fitness")
    write_bar_svg(
        final_best_bar_path,
        "Day 9 evaluation: final best-so-far mean",
        {method: summary["final_best_so_far_mean"] for method, summary in final_metric_summary.items()},
        "final best-so-far mean",
    )
    write_grouped_bar_svg(
        stage_efficiency_bar_path,
        "Day 9 evaluation: early vs late stage sample efficiency",
        {
            method: {
                "early": summary["early_stage_sample_efficiency_mean"],
                "late": summary["late_stage_sample_efficiency_mean"],
            }
            for method, summary in final_metric_summary.items()
        },
        "sample efficiency",
    )
    write_bar_svg(
        stability_bar_path,
        "Day 9 evaluation: cross-seed std of final best-so-far",
        {method: summary["seed_std"] for method, summary in seed_stability.items()},
        "seed std",
    )

    checks: dict[str, Any] = {}
    if "greedy" in final_metric_summary and "random" in final_metric_summary:
        checks["greedy_beats_random_mean"] = (
            final_metric_summary["greedy"]["final_best_so_far_mean"]
            > final_metric_summary["random"]["final_best_so_far_mean"]
        )
    if "ucb" in final_metric_summary and "greedy" in final_metric_summary:
        checks["ucb_differs_from_greedy_mean"] = (
            abs(
                final_metric_summary["ucb"]["final_best_so_far_mean"]
                - final_metric_summary["greedy"]["final_best_so_far_mean"]
            )
            > 1e-6
        )
    if "ucb" in final_metric_summary and "random" in final_metric_summary:
        checks["ucb_not_worse_than_random_mean"] = (
            final_metric_summary["ucb"]["final_best_so_far_mean"]
            >= final_metric_summary["random"]["final_best_so_far_mean"]
        )
    if "ucb" in final_metric_summary and "greedy" in final_metric_summary:
        checks["ucb_prefers_higher_sigma_than_greedy"] = (
            final_metric_summary["ucb"]["selected_sigma_mean"]
            > final_metric_summary["greedy"]["selected_sigma_mean"]
        )

    analysis_note_path = layout["report_dir"] / "baseline_analysis_note.md"
    summary_csv_path = layout["tables_dir"] / "baseline_summary_table.csv"
    summary_latex_path = layout["tables_dir"] / "baseline_summary_table.tex"
    write_analysis_note(
        path=analysis_note_path,
        experiment_name=config.experiment.name,
        dataset_name=str(config.dataset.registry_name),
        split_type=str(config.dataset.split_type),
        methods_in_order=methods,
        final_metric_summary=final_metric_summary,
        checks=checks,
    )
    write_summary_csv(
        summary_csv_path,
        methods,
        final_metric_summary,
        threshold_hit_summary,
        seed_stability,
    )
    write_summary_latex(
        summary_latex_path,
        methods,
        final_metric_summary,
        seed_stability,
    )

    summary = {
        "experiment_name": str(config.experiment.name),
        "experiment_dir": str(experiment_dir),
        "methods": methods,
        "random_seeds": seeds,
        "threshold_fractions_of_global_best": threshold_fractions,
        "dataset": str(config.dataset.registry_name),
        "split_type": str(config.dataset.split_type),
        "final_metric_summary": final_metric_summary,
        "threshold_hit_summary": threshold_hit_summary,
        "seed_stability": seed_stability,
        "checks": checks,
        "plots": {
            "best_so_far_curve": str(best_curve_path),
            "simple_regret_curve": str(regret_curve_path),
            "sample_efficiency_curve": str(sample_efficiency_curve_path),
            "selected_sigma_curve": str(sigma_curve_path),
            "selected_true_fitness_curve": str(fitness_curve_path),
            "final_best_bar": str(final_best_bar_path),
            "stage_efficiency_bar": str(stage_efficiency_bar_path),
            "seed_stability_bar": str(stability_bar_path),
        },
        "tables": {
            "summary_csv": str(summary_csv_path),
            "summary_latex": str(summary_latex_path),
        },
        "analysis_note_path": str(analysis_note_path),
        "subruns": subrun_index,
        "split_statistics_reference": split_statistics_reference or {},
        "successful": True,
    }
    write_json(layout["artifacts_dir"] / "baseline_summary.json", summary)
    logger.info("Day 9 evaluation package completed successfully. Summary at %s", layout["artifacts_dir"] / "baseline_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
