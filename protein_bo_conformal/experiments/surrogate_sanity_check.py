"""Run Day 5 surrogate sanity checks across a real split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import DatasetRecord, load_dataset
from data.dataset_registry import resolve_dataset
from data.split import build_split
from main import build_run_config, create_run_layout, write_json, write_run_metadata
from models.checkpoint import CheckpointManager
from models.ensemble import DeepEnsemble
from models.trainer import EnsembleTrainer
from representation.interface import build_encoder
from utils.config import load_config
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 5 surrogate sanity checks.")
    parser.add_argument("--config", type=str, default=None, help="Optional experiment override config path.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name override.")
    return parser.parse_args()


def _sequence_distance(seq_a: str, seq_b: str) -> int:
    shared = sum(left != right for left, right in zip(seq_a, seq_b))
    return shared + abs(len(seq_a) - len(seq_b))


def _encode_records(records: tuple[DatasetRecord, ...], encoder: Any) -> tuple[np.ndarray, np.ndarray]:
    sequences = [record.sequence for record in records]
    targets = np.asarray([record.fitness for record in records], dtype=np.float32)
    features = encoder.encode(sequences).astype(np.float32, copy=False)
    return features, targets


def _regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    variance = float(np.var(targets))
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(math.sqrt(mse)),
        "r2": float(1.0 - mse / variance) if variance > 0.0 else 0.0,
        "bias": float(np.mean(residuals)),
    }


def _member_diversity(member_predictions: np.ndarray) -> dict[str, float]:
    if member_predictions.size == 0:
        return {
            "mean_member_std": 0.0,
            "max_member_std": 0.0,
            "mean_pairwise_l2": 0.0,
        }
    member_std = member_predictions.std(axis=0)
    pairwise: list[float] = []
    for left in range(member_predictions.shape[0]):
        for right in range(left + 1, member_predictions.shape[0]):
            pairwise.append(
                float(np.linalg.norm(member_predictions[left] - member_predictions[right]) / np.sqrt(member_predictions.shape[1]))
            )
    return {
        "mean_member_std": float(member_std.mean()),
        "max_member_std": float(member_std.max()),
        "mean_pairwise_l2": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _nearest_train_distances(
    evaluation_records: tuple[DatasetRecord, ...],
    train_records: tuple[DatasetRecord, ...],
) -> np.ndarray:
    return np.asarray(
        [
            min(_sequence_distance(record.sequence, train_record.sequence) for train_record in train_records)
            for record in evaluation_records
        ],
        dtype=np.float32,
    )


def _distance_uncertainty_summary(distances: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    if distances.size == 0 or sigma.size == 0:
        return {
            "pearson_distance_sigma": 0.0,
            "high_distance_mean_sigma": 0.0,
            "low_distance_mean_sigma": 0.0,
        }
    if float(distances.std()) == 0.0 or float(sigma.std()) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(distances, sigma)[0, 1])
    threshold = float(np.median(distances))
    high_mask = distances >= threshold
    low_mask = distances < threshold
    return {
        "pearson_distance_sigma": correlation,
        "high_distance_mean_sigma": float(sigma[high_mask].mean()) if bool(high_mask.any()) else float(sigma.mean()),
        "low_distance_mean_sigma": float(sigma[low_mask].mean()) if bool(low_mask.any()) else float(sigma.mean()),
    }


def _write_predictions_csv(
    path: Path,
    evaluation_records: tuple[DatasetRecord, ...],
    small_predictions: dict[str, np.ndarray],
    large_predictions: dict[str, np.ndarray],
    small_distances: np.ndarray,
    large_distances: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "true_fitness",
                "mutation_count",
                "mu_small",
                "sigma_small",
                "mu_large",
                "sigma_large",
                "nearest_distance_small",
                "nearest_distance_large",
            ],
        )
        writer.writeheader()
        for index, record in enumerate(evaluation_records):
            writer.writerow(
                {
                    "sequence": record.sequence,
                    "true_fitness": record.fitness,
                    "mutation_count": record.mutation_count,
                    "mu_small": float(small_predictions["mean"][index]),
                    "sigma_small": float(small_predictions["sigma"][index]),
                    "mu_large": float(large_predictions["mean"][index]),
                    "sigma_large": float(large_predictions["sigma"][index]),
                    "nearest_distance_small": float(small_distances[index]),
                    "nearest_distance_large": float(large_distances[index]),
                }
            )


def _scale(values: list[float], min_value: float, max_value: float) -> list[float]:
    if not values:
        return []
    span = max(max_value - min_value, 1e-6)
    return [(value - min_value) / span for value in values]


def _write_scatter_svg(
    path: Path,
    title: str,
    targets: np.ndarray,
    small_predictions: np.ndarray,
    large_predictions: np.ndarray,
) -> None:
    width, height = 760, 420
    margin = 50
    values = list(targets) + list(small_predictions) + list(large_predictions)
    min_value = min(values)
    max_value = max(values)
    scaled_targets = _scale(list(targets), min_value, max_value)
    scaled_small = _scale(list(small_predictions), min_value, max_value)
    scaled_large = _scale(list(large_predictions), min_value, max_value)
    points: list[str] = []
    for target_value, prediction_value in zip(scaled_targets, scaled_small):
        x = margin + target_value * (width - 2 * margin)
        y = height - margin - prediction_value * (height - 2 * margin)
        points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='#1d4ed8' opacity='0.7' />")
    for target_value, prediction_value in zip(scaled_targets, scaled_large):
        x = margin + target_value * (width - 2 * margin)
        y = height - margin - prediction_value * (height - 2 * margin)
        points.append(f"<rect x='{x-3.5:.1f}' y='{y-3.5:.1f}' width='7' height='7' fill='#dc2626' opacity='0.65' />")
    diag = f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{margin}' stroke='#6b7280' stroke-dasharray='4 4' />"
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{diag}
{''.join(points)}
<rect x='{width - 220}' y='42' width='14' height='14' fill='#1d4ed8' />
<text x='{width - 198}' y='54' font-size='12' font-family='Arial'>small-train</text>
<rect x='{width - 120}' y='42' width='14' height='14' fill='#dc2626' />
<text x='{width - 98}' y='54' font-size='12' font-family='Arial'>large-train</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_uncertainty_histogram_svg(
    path: Path,
    title: str,
    small_sigma: np.ndarray,
    large_sigma: np.ndarray,
) -> None:
    width, height = 760, 420
    margin = 50
    values = list(small_sigma) + list(large_sigma)
    if not values:
        return
    bins = 8
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-6)
    step = span / bins

    def bucketize(series: np.ndarray) -> list[int]:
        counts = [0 for _ in range(bins)]
        for value in series:
            index = min(bins - 1, int((float(value) - min_value) / step))
            counts[index] += 1
        return counts

    small_counts = bucketize(small_sigma)
    large_counts = bucketize(large_sigma)
    max_count = max(small_counts + large_counts + [1])
    band_width = (width - 2 * margin) / bins
    bar_width = band_width * 0.35
    bars: list[str] = []
    labels: list[str] = []
    for index in range(bins):
        x_base = margin + index * band_width
        small_height = (small_counts[index] / max_count) * (height - 2 * margin)
        large_height = (large_counts[index] / max_count) * (height - 2 * margin)
        bars.append(
            f"<rect x='{x_base:.1f}' y='{height - margin - small_height:.1f}' width='{bar_width:.1f}' height='{small_height:.1f}' fill='#1d4ed8' />"
        )
        bars.append(
            f"<rect x='{x_base + bar_width + 6:.1f}' y='{height - margin - large_height:.1f}' width='{bar_width:.1f}' height='{large_height:.1f}' fill='#dc2626' />"
        )
        lower = min_value + index * step
        upper = lower + step
        labels.append(
            f"<text x='{x_base + band_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='10'>{lower:.3f}-{upper:.3f}</text>"
        )

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{''.join(bars)}
{''.join(labels)}
<rect x='{width - 220}' y='42' width='14' height='14' fill='#1d4ed8' />
<text x='{width - 198}' y='54' font-size='12' font-family='Arial'>small-train</text>
<rect x='{width - 120}' y='42' width='14' height='14' fill='#dc2626' />
<text x='{width - 98}' y='54' font-size='12' font-family='Arial'>large-train</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_loss_curve_svg(path: Path, title: str, member_reports: list[dict[str, Any]]) -> None:
    width, height = 760, 420
    margin = 50
    max_epoch = max(len(report["history"]) for report in member_reports)
    if max_epoch <= 0:
        return
    mean_train: list[float] = []
    mean_val: list[float] = []
    for epoch_index in range(max_epoch):
        train_values: list[float] = []
        val_values: list[float] = []
        for report in member_reports:
            history = report["history"]
            item = history[min(epoch_index, len(history) - 1)]
            train_values.append(float(item["train_loss"]))
            val_values.append(float(item["validation_loss"]))
        mean_train.append(float(np.mean(train_values)))
        mean_val.append(float(np.mean(val_values)))
    all_values = mean_train + mean_val
    min_value = min(all_values)
    max_value = max(all_values)
    span = max(max_value - min_value, 1e-6)

    def line(series: list[float], color: str) -> str:
        points: list[str] = []
        for idx, value in enumerate(series):
            x = margin + (idx / max(max_epoch - 1, 1)) * (width - 2 * margin)
            y = height - margin - ((value - min_value) / span) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}' />"

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{line(mean_train, '#0f766e')}
{line(mean_val, '#b45309')}
<rect x='{width - 220}' y='42' width='14' height='14' fill='#0f766e' />
<text x='{width - 198}' y='54' font-size='12' font-family='Arial'>train loss</text>
<rect x='{width - 120}' y='42' width='14' height='14' fill='#b45309' />
<text x='{width - 98}' y='54' font-size='12' font-family='Arial'>val loss</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _train_regime(
    regime_name: str,
    train_records: tuple[DatasetRecord, ...],
    evaluation_records: tuple[DatasetRecord, ...],
    encoder: Any,
    model_config: dict[str, Any],
    checkpoint_manager: CheckpointManager,
    device: str,
    logger: Any,
    split_id: str,
    round_index: int,
) -> dict[str, Any]:
    train_features, train_targets = _encode_records(train_records, encoder)
    evaluation_features, evaluation_targets = _encode_records(evaluation_records, encoder)
    ensemble = DeepEnsemble.from_config(model_config, input_dim=int(train_features.shape[1]), device=device, logger=logger)
    trainer = EnsembleTrainer(
        config=model_config,
        device=device,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
    )
    training_summary = trainer.fit(
        ensemble=ensemble,
        train_features=train_features,
        train_targets=train_targets,
        round_index=round_index,
        split_id=split_id,
        round_metadata={
            "regime": regime_name,
            "train_record_count": len(train_records),
            "evaluation_record_count": len(evaluation_records),
            "train_target_mean": float(train_targets.mean()) if train_targets.size else 0.0,
            "train_target_std": float(train_targets.std()) if train_targets.size else 0.0,
        },
    )
    predictions = ensemble.predict_with_uncertainty(evaluation_features, batch_size=int(model_config.get("batch_size", 32)))
    train_predictions = ensemble.predict_with_uncertainty(train_features, batch_size=int(model_config.get("batch_size", 32)))
    diversity = _member_diversity(predictions["member_predictions"])
    nearest_distances = _nearest_train_distances(evaluation_records, train_records)
    distance_summary = _distance_uncertainty_summary(nearest_distances, predictions["sigma"])
    result = {
        "regime": regime_name,
        "train_record_count": len(train_records),
        "evaluation_record_count": len(evaluation_records),
        "training_summary": training_summary,
        "prediction_summary": _regression_metrics(predictions["mean"], evaluation_targets),
        "uncertainty_summary": {
            "mean_sigma": float(predictions["sigma"].mean()),
            "max_sigma": float(predictions["sigma"].max()),
            "min_sigma": float(predictions["sigma"].min()),
            "nonzero_fraction": float(np.mean(predictions["sigma"] > 1e-8)),
            "train_mean_sigma": float(train_predictions["sigma"].mean()),
            "candidate_minus_train_mean_sigma": float(predictions["sigma"].mean() - train_predictions["sigma"].mean()),
        },
        "member_diversity": diversity,
        "distance_uncertainty_summary": distance_summary,
        "predictions": predictions,
        "train_predictions": train_predictions,
        "targets": evaluation_targets,
        "nearest_distances": nearest_distances,
    }
    return result


def main() -> int:
    args = parse_args()
    base_config = PROJECT_ROOT / "config" / "base.yaml"
    default_layers = [
        PROJECT_ROOT / "config" / "dataset.yaml",
        PROJECT_ROOT / "config" / "representation.yaml",
        PROJECT_ROOT / "config" / "model.yaml",
    ]
    default_override = PROJECT_ROOT / "config" / "experiment" / "day5_surrogate_sanity.yaml"
    override_config = Path(args.config).resolve() if args.config else default_override

    config = load_config(base_config, override_config, default_layer_paths=default_layers)
    config, config_hash = build_run_config(config, argparse.Namespace(name=args.name or config.experiment.name))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config.runtime.run_name}-{config_hash[:8]}"
    layout = create_run_layout(PROJECT_ROOT, run_id)
    write_run_metadata(config, layout, run_id, config_hash, override_config)

    logger = build_logger(
        name=f"protein_bo_conformal.{run_id}",
        log_dir=layout["logs_dir"],
        experiment_id=run_id,
        level=config.runtime.log_level,
    )
    seed_report = set_global_seed(
        seed=int(config.runtime.seed),
        deterministic=bool(config.runtime.deterministic),
        logger=logger,
    )
    device_info = resolve_device(config.runtime.device, logger=logger)

    dataset_spec = resolve_dataset(
        registry_name=config.dataset.registry_name,
        benchmark=config.dataset.benchmark,
        task=config.dataset.task,
    )
    dataset_bundle = load_dataset(dataset_spec, PROJECT_ROOT, logger=logger)
    split_result = build_split(
        dataset_bundle,
        config.dataset.to_dict(),
        processed_dir=PROJECT_ROOT / "data" / "processed",
        logger=logger,
    )

    model_config = config.model.to_dict()
    large_train_records = split_result.train_records
    candidate_records = split_result.candidate_records
    small_train_size = max(2, int(model_config.get("sanity_small_train_size", 16)))
    if len(large_train_records) < small_train_size:
        raise ValueError(
            f"Day 5 sanity check expects at least {small_train_size} train records in the large-train regime."
        )
    small_train_records = large_train_records[:small_train_size]
    evaluation_records = candidate_records
    if len(evaluation_records) < 8:
        raise ValueError("Day 5 sanity check expects at least 8 candidate records for evaluation.")

    encoder = build_encoder(config.representation.to_dict(), logger=logger)
    checkpoint_manager = CheckpointManager(layout["checkpoints_dir"], logger=logger)

    small_result = _train_regime(
        regime_name="small_train",
        train_records=small_train_records,
        evaluation_records=evaluation_records,
        encoder=encoder,
        model_config=model_config,
        checkpoint_manager=checkpoint_manager,
        device=device_info.resolved,
        logger=logger,
        split_id=split_result.split_id,
        round_index=0,
    )
    large_result = _train_regime(
        regime_name="large_train",
        train_records=large_train_records,
        evaluation_records=evaluation_records,
        encoder=encoder,
        model_config=model_config,
        checkpoint_manager=checkpoint_manager,
        device=device_info.resolved,
        logger=logger,
        split_id=split_result.split_id,
        round_index=1,
    )

    if small_result["uncertainty_summary"]["nonzero_fraction"] <= 0.0:
        raise ValueError("Small-train ensemble produced zero uncertainty for all evaluation points.")
    if large_result["uncertainty_summary"]["nonzero_fraction"] <= 0.0:
        raise ValueError("Large-train ensemble produced zero uncertainty for all evaluation points.")
    if small_result["member_diversity"]["mean_member_std"] <= 1e-8:
        raise ValueError("Small-train ensemble collapsed: member diversity is effectively zero.")
    if large_result["member_diversity"]["mean_member_std"] <= 1e-8:
        raise ValueError("Large-train ensemble collapsed: member diversity is effectively zero.")

    uncertainty_trend = {
        "mean_sigma_small": small_result["uncertainty_summary"]["mean_sigma"],
        "mean_sigma_large": large_result["uncertainty_summary"]["mean_sigma"],
    }
    uncertainty_trend["small_greater_than_large"] = bool(
        uncertainty_trend["mean_sigma_small"] >= uncertainty_trend["mean_sigma_large"]
    )
    behavior_checks = {
        "small_sigma_nonzero": bool(small_result["uncertainty_summary"]["nonzero_fraction"] > 0.0),
        "large_sigma_nonzero": bool(large_result["uncertainty_summary"]["nonzero_fraction"] > 0.0),
        "small_member_diverse": bool(small_result["member_diversity"]["mean_member_std"] > 1e-8),
        "large_member_diverse": bool(large_result["member_diversity"]["mean_member_std"] > 1e-8),
        "small_data_scale_sigma_higher_than_large": bool(uncertainty_trend["small_greater_than_large"]),
    }
    behavior_diagnostics = {
        "small_candidate_more_uncertain_than_train": bool(
            small_result["uncertainty_summary"]["candidate_minus_train_mean_sigma"] >= 0.0
        ),
        "large_candidate_more_uncertain_than_train": bool(
            large_result["uncertainty_summary"]["candidate_minus_train_mean_sigma"] >= 0.0
        ),
    }

    predictions_path = layout["tables_dir"] / "surrogate_predictions.csv"
    _write_predictions_csv(
        predictions_path,
        evaluation_records=evaluation_records,
        small_predictions=small_result["predictions"],
        large_predictions=large_result["predictions"],
        small_distances=small_result["nearest_distances"],
        large_distances=large_result["nearest_distances"],
    )

    scatter_path = layout["plots_dir"] / "surrogate_scatter.svg"
    uncertainty_hist_path = layout["plots_dir"] / "surrogate_uncertainty_histogram.svg"
    loss_curve_path = layout["plots_dir"] / "surrogate_loss_curve.svg"
    _write_scatter_svg(
        scatter_path,
        title="Prediction Mean vs True Fitness",
        targets=small_result["targets"],
        small_predictions=small_result["predictions"]["mean"],
        large_predictions=large_result["predictions"]["mean"],
    )
    _write_uncertainty_histogram_svg(
        uncertainty_hist_path,
        title="Ensemble Uncertainty Histogram",
        small_sigma=small_result["predictions"]["sigma"],
        large_sigma=large_result["predictions"]["sigma"],
    )
    _write_loss_curve_svg(
        loss_curve_path,
        title="Average Ensemble Training Curves",
        member_reports=small_result["training_summary"]["member_reports"] + large_result["training_summary"]["member_reports"],
    )

    report = {
        "run_id": run_id,
        "config_hash": config_hash,
        "seed_report": seed_report,
        "device_info": device_info.to_dict(),
        "dataset_summary": dataset_bundle.to_summary_dict(),
        "split_id": split_result.split_id,
        "representation_summary": {
            "encoder": config.representation.name,
            "encoder_stats": encoder.get_stats(),
        },
        "small_train": {
            key: value
            for key, value in small_result.items()
            if key not in {"predictions", "train_predictions", "targets", "nearest_distances"}
        },
        "large_train": {
            key: value
            for key, value in large_result.items()
            if key not in {"predictions", "train_predictions", "targets", "nearest_distances"}
        },
        "uncertainty_trend": uncertainty_trend,
        "behavior_checks": behavior_checks,
        "behavior_diagnostics": behavior_diagnostics,
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "scatter_plot": str(scatter_path),
            "uncertainty_histogram": str(uncertainty_hist_path),
            "loss_curve_plot": str(loss_curve_path),
        },
    }
    report_path = layout["artifacts_dir"] / "surrogate_sanity_summary.json"
    write_json(report_path, report)
    logger.info("Surrogate sanity report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
