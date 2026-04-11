"""Checkpoint save and load helpers for surrogate models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Persist ensemble members and training summaries with stable naming."""

    def __init__(self, root_dir: str | Path, logger: Any | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def round_dir(self, round_index: int) -> Path:
        path = self.root_dir / f"round_{int(round_index):03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def member_path(self, round_index: int, member_index: int) -> Path:
        return self.round_dir(round_index) / f"member_{int(member_index):03d}.pt"

    def summary_path(self, round_index: int) -> Path:
        return self.round_dir(round_index) / "training_summary.json"

    def save_member(
        self,
        model: torch.nn.Module,
        round_index: int,
        member_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.member_path(round_index, member_index)
        payload = {
            "round_index": int(round_index),
            "member_index": int(member_index),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        if self.logger is not None:
            self.logger.info("Saved checkpoint for round=%s member=%s to %s", round_index, member_index, path)
        return path

    def load_member(
        self,
        model: torch.nn.Module,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location)
        model.load_state_dict(payload["state_dict"])
        return dict(payload.get("metadata", {}))

    def save_training_summary(self, round_index: int, summary: dict[str, Any]) -> Path:
        path = self.summary_path(round_index)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if self.logger is not None:
            self.logger.info("Saved training summary for round=%s to %s", round_index, path)
        return path
