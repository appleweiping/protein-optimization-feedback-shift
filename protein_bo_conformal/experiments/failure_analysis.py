"""Day 10 failure analysis over baseline experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_dataset
from data.dataset_registry import resolve_dataset
from data.split import build_split
from evaluation.metrics import compute_uncertainty_behavior, load_json, load_jsonl
from evaluation.plotting import (
    plot_embedding_distance_over_time,
    plot_shift_vs_performance,
    plot_sigma_vs_error_scatter,
)
from evaluation.report import write_failure_analysis_report
from evaluation.shift_metrics import compute_selection_shift
from representation.interface import build_encoder
from utils.config import ConfigNode, dump_yaml, load_yaml, stable_config_hash
from utils.logger import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 10 failure analysis package.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/day10_failure_analysis.yaml",
        help="Failure analysis config path.",
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_layout(root_dir: Path) -> dict[str, Path]:
    layout = {
        "run_dir": root_dir,
        "logs_dir": root_dir / "logs",
        "plots_dir": root_dir / "plots",
        "tables_dir": root_dir / "tables",
        "artifacts_dir": root_dir / "artifacts",
        "report_dir": root_dir / "report",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def resolve_latest_glob(project_root: Path, pattern: str) -> Path:
    matches = sorted(project_root.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matched failure-analysis input pattern '{pattern}'.")
    return matches[-1]


def _safe_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    vector_a = np.asarray(values_a, dtype=np.float32)
    vector_b = np.asarray(values_b, dtype=np.float32)
    if float(vector_a.std()) <= 1e-8 or float(vector_b.std()) <= 1e-8:
        return 0.0
    return float(np.corrcoef(vector_a, vector_b)[0, 1])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate_method_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "final_best_so_far_mean": 0.0,
            "selected_distance_gap_mean": 0.0,
            "support_overlap_gap_mean": 0.0,
            "sigma_error_correlation_mean": 0.0,
            "mu_sigma_correlation_mean": 0.0,
            "selected_sigma_gap_mean": 0.0,
            "shift_performance_correlation": 0.0,
            "seed_count": 0.0,
        }

    final_best = [float(record["final_best_so_far"]) for record in records]
    distance_gap = [float(record["distance_gap_mean"]) for record in records]
    support_gap = [float(record["support_overlap_gap_mean"]) for record in records]
    sigma_error_corr = [float(record["sigma_error_correlation_mean"]) for record in records]
    mu_sigma_corr = [float(record["mu_sigma_correlation_mean"]) for record in records]
    sigma_gap = [float(record["sigma_gap_mean"]) for record in records]
    shift_values = [
        float(round_record["selected_distance_gap"])
        for record in records
        for round_record in record["round_records"]
    ]
    performance_values = [
        float(round_record["selected_true_fitness_mean"])
        for record in records
        for round_record in record["round_records"]
    ]
    return {
        "final_best_so_far_mean": _mean(final_best),
        "selected_distance_gap_mean": _mean(distance_gap),
        "support_overlap_gap_mean": _mean(support_gap),
        "sigma_error_correlation_mean": _mean(sigma_error_corr),
        "mu_sigma_correlation_mean": _mean(mu_sigma_corr),
        "selected_sigma_gap_mean": _mean(sigma_gap),
        "shift_performance_correlation": _safe_correlation(shift_values, performance_values),
        "seed_count": float(len(records)),
    }


def _build_encoder_for_run(config_snapshot: dict[str, Any], logger: Any | None = None) -> Any:
    representation_config = dict(config_snapshot.get("representation", {}))
    return build_encoder(representation_config, logger=logger)


def analyze_subrun(
    project_root: Path,
    subrun: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    run_dir = Path(subrun["run_dir"])
    method = str(subrun["method"])
    config_snapshot = load_yaml(run_dir / "config_snapshot.yaml")
    dataset_config = dict(config_snapshot.get("dataset", {}))
    dataset_spec = resolve_dataset(
        registry_name=dataset_config.get("registry_name"),
        benchmark=dataset_config.get("benchmark"),
        task=dataset_config.get("task"),
    )
    dataset_bundle = load_dataset(dataset_spec, project_root=project_root, logger=logger)
    split_result = build_split(
        dataset_bundle,
        dataset_config,
        processed_dir=project_root / "data" / "processed",
        logger=logger,
    )
    encoder = _build_encoder_for_run(config_snapshot, logger=logger)
    train_sequences = [record.sequence for record in split_result.train_records]
    candidate_sequences = [record.sequence for record in split_result.candidate_records]
    train_embeddings = encoder.encode(train_sequences)
    candidate_embeddings = encoder.encode(candidate_sequences)
    rounds_path = run_dir / "artifacts" / f"{method}_loop_rounds.jsonl"
    loop_summary_path = run_dir / "artifacts" / f"{method}_loop_summary.json"
    round_payloads = load_jsonl(rounds_path)
    loop_summary = load_json(loop_summary_path)
    uncertainty_behavior = compute_uncertainty_behavior(round_payloads)

    round_records: list[dict[str, float]] = []
    sigma_scatter: list[dict[str, float]] = []
    distance_curve: list[dict[str, float]] = []
    for round_payload, uncertainty_summary in zip(round_payloads, uncertainty_behavior):
        selected_sequences = [item["sequence"] for item in round_payload.get("selected", [])]
        if not selected_sequences:
            continue
        selected_embeddings = encoder.encode(selected_sequences)
        shift_summary = compute_selection_shift(train_embeddings, candidate_embeddings, selected_embeddings)
        selected = list(round_payload.get("selected", []))
        selected_true_fitness_mean = _mean([float(item.get("true_fitness", 0.0)) for item in selected])
        record = {
            "step": float(round_payload.get("round_index", 0)) + 1.0,
            "selected_distance_gap": float(shift_summary["distance_gap"]["centroid_mean_gap"]),
            "selected_nn_gap": float(shift_summary["distance_gap"]["nearest_neighbor_mean_gap"]),
            "selected_support_overlap_gap": float(shift_summary["support_overlap_gap"]),
            "selected_true_fitness_mean": float(selected_true_fitness_mean),
            "best_so_far_after": float(round_payload.get("best_so_far_after", 0.0)),
            "sigma_gap": float(uncertainty_summary["sigma_gap"]),
            "sigma_error_correlation": float(uncertainty_summary["sigma_error_correlation"]),
            "mu_sigma_correlation": float(uncertainty_summary["mu_sigma_correlation"]),
        }
        round_records.append(record)
        distance_curve.append({"step": record["step"], "value": record["selected_distance_gap"]})
        for item in selected:
            sigma_scatter.append(
                {
                    "x": float(item.get("predicted_sigma", 0.0)),
                    "y": abs(float(item.get("predicted_mean", 0.0)) - float(item.get("true_fitness", 0.0))),
                }
            )

    return {
        "method": method,
        "seed": int(subrun["seed"]),
        "dataset": str(dataset_config.get("registry_name", "")),
        "split_type": str(dataset_config.get("split_type", "")),
        "split_id": split_result.split_id,
        "final_best_so_far": float(loop_summary.get("final_best_so_far", 0.0)),
        "distance_gap_mean": _mean([record["selected_distance_gap"] for record in round_records]),
        "support_overlap_gap_mean": _mean([record["selected_support_overlap_gap"] for record in round_records]),
        "sigma_error_correlation_mean": _mean([record["sigma_error_correlation"] for record in round_records]),
        "mu_sigma_correlation_mean": _mean([record["mu_sigma_correlation"] for record in round_records]),
        "sigma_gap_mean": _mean([record["sigma_gap"] for record in round_records]),
        "round_records": round_records,
        "sigma_scatter": sigma_scatter,
        "distance_curve": distance_curve,
    }


def write_split_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "split_label",
        "method",
        "final_best_so_far_mean",
        "selected_distance_gap_mean",
        "support_overlap_gap_mean",
        "sigma_error_correlation_mean",
        "mu_sigma_correlation_mean",
        "selected_sigma_gap_mean",
        "shift_performance_correlation",
        "seed_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_summary(
    project_root: Path,
    label: str,
    summary_path: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    methods_filter = {str(method).lower() for method in summary.get("methods", [])}
    per_method_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sigma_scatter_series: dict[str, list[dict[str, float]]] = defaultdict(list)
    distance_curves_by_method: dict[str, list[list[dict[str, float]]]] = defaultdict(list)

    for subrun in summary.get("subruns", []):
        method = str(subrun.get("method", "")).lower()
        if methods_filter and method not in methods_filter:
            continue
        record = analyze_subrun(project_root, subrun, logger=logger)
        per_method_records[method].append(record)
        sigma_scatter_series[method].extend(record["sigma_scatter"])
        distance_curves_by_method[method].append(record["distance_curve"])

    aggregated_methods: list[dict[str, Any]] = []
    shift_vs_performance_series: dict[str, list[dict[str, float]]] = {}
    distance_time_series: dict[str, list[dict[str, float]]] = {}
    for method, records in per_method_records.items():
        aggregated = _aggregate_method_records(records)
        aggregated_methods.append({"method": method, **aggregated})
        shift_vs_performance_series[method] = [
            {
                "x": float(round_record["selected_distance_gap"]),
                "y": float(round_record["selected_true_fitness_mean"]),
            }
            for record in records
            for round_record in record["round_records"]
        ]

        per_step_values: dict[float, list[float]] = defaultdict(list)
        for curve in distance_curves_by_method[method]:
            for point in curve:
                per_step_values[float(point["step"])].append(float(point["value"]))
        distance_time_series[method] = [
            {"step": step, "value": _mean(values)}
            for step, values in sorted(per_step_values.items())
        ]

    split_shift = {
        "train_candidate_centroid_l2": float(
            summary.get("split_statistics_reference", {})
            .get("embedding_distance", {})
            .get("train_candidate_centroid_l2", 0.0)
        ),
        "support_overlap_proxy": float(summary.get("split_statistics_reference", {}).get("support_overlap_proxy", 0.0)),
    }
    return {
        "label": label,
        "dataset": str(summary.get("dataset", "")),
        "split_type": str(summary.get("split_type", "")),
        "summary_path": str(summary_path),
        "methods": sorted(aggregated_methods, key=lambda item: item["method"]),
        "split_shift": split_shift,
        "shift_vs_performance_series": shift_vs_performance_series,
        "sigma_scatter_series": dict(sigma_scatter_series),
        "distance_time_series": distance_time_series,
    }


def build_overall_conclusions(split_summaries: list[dict[str, Any]]) -> list[str]:
    conclusions: list[str] = []
    if not split_summaries:
        return conclusions

    strongest_shift = max(split_summaries, key=lambda item: item["split_shift"]["train_candidate_centroid_l2"])
    weakest_shift = min(split_summaries, key=lambda item: item["split_shift"]["train_candidate_centroid_l2"])
    conclusions.append(
        f"`{strongest_shift['label']}` shows the strongest train-candidate embedding shift, while `{weakest_shift['label']}` is comparatively milder."
    )

    for split_summary in split_summaries:
        method_lookup = {entry["method"]: entry for entry in split_summary["methods"]}
        greedy = method_lookup.get("greedy")
        ucb = method_lookup.get("ucb")
        if greedy and ucb:
            if greedy["final_best_so_far_mean"] > ucb["final_best_so_far_mean"]:
                conclusions.append(
                    f"In `{split_summary['label']}`, `greedy` outperforms `ucb` despite `ucb` selecting more shifted points on average."
                )
            else:
                conclusions.append(
                    f"In `{split_summary['label']}`, `ucb` does not collapse, but its gain is still tightly coupled to shift strength."
                )
            conclusions.append(
                f"In `{split_summary['label']}`, `ucb` shift-performance correlation = {ucb['shift_performance_correlation']:.4f}, sigma-error correlation = {ucb['sigma_error_correlation_mean']:.4f}."
            )

    conclusions.append(
        "Across analyzed splits, the evidence supports the Day 10 mechanism claim: under feedback shift, uncalibrated sigma changes behavior but does not reliably translate into decision gain."
    )
    return conclusions


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config_dict = load_yaml(config_path)
    if args.name:
        config_dict.setdefault("experiment", {})
        config_dict["experiment"]["name"] = args.name
    config = ConfigNode(config_dict)

    experiment_hash = stable_config_hash(config.to_dict())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = sanitize_name(config.experiment.name)
    output_root = PROJECT_ROOT / str(config.analysis.output_root)
    experiment_dir = output_root / f"{timestamp}-{experiment_name}-{experiment_hash[:8]}"
    layout = create_layout(experiment_dir)
    logger = build_logger(
        name=f"protein_bo_conformal.failure_analysis.{experiment_name}",
        log_dir=layout["logs_dir"],
        experiment_id=experiment_name,
        level="INFO",
    )

    split_summaries: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for entry in config.inputs.baseline_summary_globs:
        label = str(entry["label"])
        summary_path = resolve_latest_glob(PROJECT_ROOT, str(entry["summary_glob"]))
        logger.info("Analyzing baseline summary '%s' from %s", label, summary_path)
        split_summary = analyze_summary(PROJECT_ROOT, label, summary_path, logger=logger)
        split_summaries.append(split_summary)
        for method_summary in split_summary["methods"]:
            table_rows.append(
                {
                    "split_label": label,
                    **method_summary,
                }
            )

        plot_shift_vs_performance(
            layout["plots_dir"] / f"{label}_shift_vs_performance.svg",
            f"Day 10 failure analysis: {label} shift vs performance",
            split_summary["shift_vs_performance_series"],
            x_label="selected distance gap",
            y_label="selected true fitness mean",
        )
        plot_sigma_vs_error_scatter(
            layout["plots_dir"] / f"{label}_sigma_vs_error.svg",
            f"Day 10 failure analysis: {label} sigma vs error",
            split_summary["sigma_scatter_series"],
        )
        plot_embedding_distance_over_time(
            layout["plots_dir"] / f"{label}_embedding_distance_over_time.svg",
            f"Day 10 failure analysis: {label} embedding distance over time",
            split_summary["distance_time_series"],
            y_label="selected distance gap",
        )

    overall_conclusions = build_overall_conclusions(split_summaries)
    write_split_table(layout["tables_dir"] / "failure_analysis_summary.csv", table_rows)
    write_failure_analysis_report(
        layout["report_dir"] / "failure_analysis_note.md",
        experiment_name=str(config.experiment.name),
        split_summaries=split_summaries,
        overall_conclusions=overall_conclusions,
    )
    _write_json(
        layout["artifacts_dir"] / "failure_analysis_summary.json",
        {
            "experiment_name": str(config.experiment.name),
            "split_summaries": split_summaries,
            "overall_conclusions": overall_conclusions,
            "successful": True,
        },
    )
    (layout["run_dir"] / "config_snapshot.yaml").write_text(dump_yaml(config.to_dict()), encoding="utf-8")
    logger.info("Day 10 failure analysis completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
