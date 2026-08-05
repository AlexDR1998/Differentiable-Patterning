"""Stable PyTree-compatible contracts at the compiled-step boundary."""

from typing import Any, NamedTuple


class TrainState(NamedTuple):
    model: Any
    states: Any
    targets: Any
    optimizer_state: Any
    key: Any


class StepOutput(NamedTuple):
    state: TrainState
    loss: Any
    metrics: dict[str, Any]


class RolloutResult(NamedTuple):
    key: Any
    states: Any
    regulariser_totals: dict[str, Any]

