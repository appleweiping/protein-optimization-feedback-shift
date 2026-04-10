"""Closed-loop experiment runner shell for Day 1 reproducibility work."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import ConfigNode


class ClosedLoopRunner:
    """Minimal runner that validates the execution shell and records artifacts."""

    def __init__(self, config: ConfigNode, logger: Any, context: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger
        self.context = context
        self.artifacts_dir = Path(context["paths"]["artifacts_dir"])

    def run(self) -> dict[str, Any]:
        """Run a no-op shell pass and emit auditable metadata artifacts."""
        self.logger.info("Runner initialized for experiment '%s'.", self.config.experiment.name)
        self.logger.info("Validating config sections and output layout.")

        required_sections = (
            "experiment",
            "runtime",
            "dataset",
            "representation",
            "model",
            "uq",
            "acquisition",
            "proposal",
            "loop",
            "evaluation",
        )
        missing = [section for section in required_sections if section not in self.config.to_dict()]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

        trace_payload = {
            "run_id": self.context["run_id"],
            "stage": "day1_execution_shell",
            "status": "ok",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": "Execution shell validated. Loop runner interface is ready for Day 2 integration.",
        }
        trace_path = self.artifacts_dir / "runner_trace.jsonl"
        trace_path.write_text(json.dumps(trace_payload) + "\n", encoding="utf-8")

        summary = {
            "run_id": self.context["run_id"],
            "status": "completed",
            "message": "Day 1 shell completed successfully.",
            "project_root": str(self.context["project_root"]),
            "config_hash": self.context["config_hash"],
            "seed_report": self.context["seed_report"],
            "device_info": self.context["device_info"],
            "outputs": {
                key: str(value)
                for key, value in self.context["paths"].items()
                if key != "run_dir"
            },
            "validated_sections": list(required_sections),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        summary_path = self.artifacts_dir / "runner_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.logger.info("Runner summary written to %s", summary_path)
        return summary
