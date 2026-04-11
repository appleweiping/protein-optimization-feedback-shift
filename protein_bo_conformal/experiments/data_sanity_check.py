"""Run Day 3 dataset and oracle sanity checks across selected benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_dataset
from data.dataset_registry import resolve_dataset
from data.oracle import Oracle
from data.split import build_split
from data.validation import validate_oracle_consistency, validate_split_against_oracle
from main import build_run_config, create_run_layout, write_json, write_run_metadata
from utils.config import load_config
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 3 data sanity checks.")
    parser.add_argument("--config", type=str, default=None, help="Optional override config path.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="flip.aav,flip2.alpha_amylase.one_to_many,proteingym.blat_ecolx_firnberg_2014",
        help="Comma-separated registry names to validate.",
    )
    return parser.parse_args()


def _dataset_override(registry_name: str) -> dict[str, object]:
    spec = resolve_dataset(registry_name=registry_name)
    split_type = "low_resource"
    if spec.name == "flip.gb1":
        split_type = "mutation_extrapolation"
    elif spec.benchmark == "flip2":
        split_type = "predefined"
    elif spec.benchmark == "proteingym":
        split_type = "low_resource"
    return {
        "registry_name": spec.name,
        "benchmark": spec.benchmark,
        "task": spec.task,
        "split_type": split_type,
    }


def main() -> int:
    args = parse_args()
    base_config = PROJECT_ROOT / "config" / "base.yaml"
    default_layers = [PROJECT_ROOT / "config" / "dataset.yaml"]
    override_config = Path(args.config).resolve() if args.config else None

    config = load_config(base_config, override_config, default_layer_paths=default_layers)
    config, config_hash = build_run_config(config, argparse.Namespace(name="day3-data-sanity"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-day3-data-sanity-{config_hash[:8]}"
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

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    report: dict[str, object] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "seed_report": seed_report,
        "device_info": device_info.to_dict(),
        "datasets": {},
    }

    for dataset_name in dataset_names:
        dataset_config = config.dataset.to_dict()
        dataset_config.update(_dataset_override(dataset_name))
        spec = resolve_dataset(registry_name=dataset_name)
        bundle = load_dataset(spec, PROJECT_ROOT, logger=logger)
        split_result = build_split(
            bundle,
            dataset_config,
            processed_dir=PROJECT_ROOT / "data" / "processed",
            logger=logger,
        )
        oracle = Oracle(
            bundle.records,
            logger=logger,
            enable_query_logging=bool(dataset_config.get("validation", {}).get("enable_query_logging", False)),
        )
        report["datasets"][dataset_name] = {
            "dataset_summary": bundle.to_summary_dict(),
            "split_id": split_result.split_id,
            "split_statistics": split_result.statistics,
            "oracle_consistency": validate_oracle_consistency(bundle, oracle, dataset_config, logger=logger),
            "split_queryability": validate_split_against_oracle(split_result, oracle, dataset_config, logger=logger),
        }

    report_path = layout["artifacts_dir"] / "data_sanity_check_report.json"
    write_json(report_path, report)
    logger.info("Data sanity check report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
