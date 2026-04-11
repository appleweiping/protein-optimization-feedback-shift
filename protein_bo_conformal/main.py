"""Project entrypoint for the reproducible protein optimization shell."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loop.runner import ClosedLoopRunner
from utils.config import ConfigNode, dump_yaml, load_config, stable_config_hash
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the execution shell."""
    parser = argparse.ArgumentParser(
        description="Run the protein_bo_conformal research execution shell."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional experiment config path merged on top of config/base.yaml.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    """Convert a run name into a filesystem-friendly identifier."""
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    compact = "-".join(part for part in normalized.split("-") if part)
    return compact or "run"


def create_run_layout(project_root: Path, run_id: str) -> dict[str, Path]:
    """Create the canonical output layout for a single run."""
    run_dir = project_root / "outputs" / "results" / run_id
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
    """Write JSON with stable formatting for auditability."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_run_metadata(
    config: ConfigNode,
    layout: dict[str, Path],
    run_id: str,
    config_hash: str,
    override_path: Path | None,
) -> None:
    """Persist run metadata and the config snapshot before execution starts."""
    snapshot_path = layout["run_dir"] / "config_snapshot.yaml"
    snapshot_path.write_text(dump_yaml(config.to_dict()), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "config_hash": config_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": "config/base.yaml",
        "override_config": str(override_path) if override_path else None,
        "run_dir": str(layout["run_dir"]),
        "status": "initialized",
    }
    write_json(layout["run_dir"] / "run_manifest.json", manifest)


def build_run_config(config: ConfigNode, args: argparse.Namespace) -> tuple[ConfigNode, str]:
    """Apply CLI-level run name overrides and derive the stable config hash."""
    config_dict = config.to_dict()
    if args.name:
        config_dict.setdefault("experiment", {})
        config_dict["experiment"]["name"] = args.name

    config_hash = stable_config_hash(config_dict)
    run_name = sanitize_name(config_dict["experiment"]["name"])
    config_dict.setdefault("runtime", {})
    config_dict["runtime"]["config_hash"] = config_hash
    config_dict["runtime"]["run_name"] = run_name
    return ConfigNode(config_dict), config_hash


def main() -> int:
    """Execute the research-grade shell end-to-end."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    base_config = project_root / "config" / "base.yaml"
    default_layers = [project_root / "config" / "dataset.yaml"]
    override_config = Path(args.config).resolve() if args.config else None

    try:
        config = load_config(
            base_config,
            override_config,
            default_layer_paths=default_layers,
        )
        config, config_hash = build_run_config(config, args)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = config.runtime.run_name
        run_id = f"{timestamp}-{run_name}-{config_hash[:8]}"
        layout = create_run_layout(project_root, run_id)
        write_run_metadata(config, layout, run_id, config_hash, override_config)

        logger = build_logger(
            name=f"protein_bo_conformal.{run_id}",
            log_dir=layout["logs_dir"],
            experiment_id=run_id,
            level=config.runtime.log_level,
        )

        logger.info("Starting execution shell.")
        logger.info("Project root: %s", project_root)
        logger.info("Run directory: %s", layout["run_dir"])
        logger.info("Config hash: %s", config_hash)

        seed_report = set_global_seed(
            seed=int(config.runtime.seed),
            deterministic=bool(config.runtime.deterministic),
            logger=logger,
        )
        device_info = resolve_device(config.runtime.device, logger=logger)

        context = {
            "project_root": project_root,
            "run_id": run_id,
            "run_dir": layout["run_dir"],
            "paths": layout,
            "seed_report": seed_report,
            "device_info": device_info.to_dict(),
            "config_hash": config_hash,
        }

        runner = ClosedLoopRunner(config=config, logger=logger, context=context)
        summary = runner.run()

        manifest_path = layout["run_dir"] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["summary_path"] = str(layout["artifacts_dir"] / "runner_summary.json")
        write_json(manifest_path, manifest)

        logger.info("Execution shell finished successfully.")
        logger.info("Summary: %s", summary["message"])
        return 0
    except Exception as exc:  # pragma: no cover - final safety net
        failure_message = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failed_at": datetime.now().isoformat(timespec="seconds"),
        }

        if "layout" in locals():
            write_json(layout["run_dir"] / "run_failure.json", failure_message)
        if "logger" in locals():
            logger.exception("Execution shell failed.")
        else:
            print(json.dumps(failure_message, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
