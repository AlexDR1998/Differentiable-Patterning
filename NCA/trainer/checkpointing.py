"""Checkpoint policy, independent from the numerical training step."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BestCheckpoint:
    path: Path
    warmup: int
    best_loss: float | None = None
    best_iteration: int | None = None

    def should_save(self, iteration: int, loss: float) -> bool:
        return iteration > self.warmup and (
            self.best_loss is None or loss < self.best_loss
        )

    def record(self, iteration: int, loss: float) -> None:
        self.best_iteration = iteration
        self.best_loss = loss

