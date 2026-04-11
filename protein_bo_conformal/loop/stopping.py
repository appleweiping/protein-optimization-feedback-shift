"""Stopping criteria for closed-loop runs."""

from __future__ import annotations

from dataclasses import dataclass

from loop.state import LoopState


@dataclass(frozen=True)
class StopDecision:
    """Structured stopping or continuation decision for one loop iteration."""

    stop: bool
    reason: str
    next_batch_size: int
    remaining_budget: int


class LoopStopping:
    """Standardized stopping logic for closed-loop execution."""

    def __init__(
        self,
        total_rounds: int,
        total_budget: int,
        query_batch_size: int,
    ) -> None:
        self.total_rounds = max(1, int(total_rounds))
        self.total_budget = max(1, int(total_budget))
        self.query_batch_size = max(1, int(query_batch_size))

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "LoopStopping":
        return cls(
            total_rounds=int(config.get("total_rounds", 1)),
            total_budget=int(config.get("total_budget", 1)),
            query_batch_size=int(config.get("query_batch_size", 1)),
        )

    def remaining_budget(self, state: LoopState) -> int:
        return max(0, self.total_budget - state.observed_count)

    def next_batch_size(self, state: LoopState) -> int:
        return min(
            self.query_batch_size,
            self.remaining_budget(state),
            state.candidate_count,
        )

    def decide(self, state: LoopState) -> StopDecision:
        remaining_budget = self.remaining_budget(state)
        if state.candidate_count <= 0:
            return StopDecision(True, "candidate_pool_exhausted", 0, remaining_budget)
        if state.round_index >= self.total_rounds:
            return StopDecision(True, "max_rounds_reached", 0, remaining_budget)
        if remaining_budget <= 0:
            return StopDecision(True, "total_budget_reached", 0, remaining_budget)

        next_batch_size = self.next_batch_size(state)
        if next_batch_size <= 0:
            return StopDecision(True, "batch_size_zero", 0, remaining_budget)
        return StopDecision(False, "continue", next_batch_size, remaining_budget)
