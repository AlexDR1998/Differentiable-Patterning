"""Structured result returned by training loops."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainingResult:
    """Small, serialisable summary needed to publish a trained model."""

    checkpoint_path: Optional[Path]
    best_iteration: Optional[int]
    best_loss: Optional[float]
    completed: bool
    error_code: int = 0
    wandb_run_id: Optional[str] = None

