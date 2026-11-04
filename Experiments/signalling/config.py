"""Configuration owned by the signalling task workflow."""

from dataclasses import dataclass, field
from typing import Any

from Common.config import ConfigValue
from Experiments.signalling.tasks import TASKS, task_parameters


@dataclass(frozen=True)
class SignalTaskDataConfig(ConfigValue):
    # A task from Experiments/signalling/tasks.py
    task: str = "remote_switch"
    size: int = 64
    # Settings of the task function, e.g. {length: [10, 56], front_step: 8, frames: 3}
    params: dict[str, Any] = field(default_factory=dict)
    # The last target is repeated this many times, to train persistence
    target_repeats: int = 2
    # Fraction of the pool replaced by fresh random instances after every step
    resample_fraction: float = 0.5
    noise_strength: float = 0.0
    # Seed of the first pool, separate from the model seed
    seed: int = 0
    # Held-out instances (and rollouts of each) measured at the end of training
    evaluation_samples: int = 16
    evaluation_repeats: int = 4
    # Evaluation rollout lengths, in multiples of the trained rollout
    evaluation_horizons: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"data.signal_task.task must be one of {tuple(TASKS)}")
        unknown = set(self.params) - set(task_parameters(self.task))
        if unknown:
            raise ValueError(
                f"data.signal_task.params for {self.task} must be drawn from "
                f"{task_parameters(self.task)}, got {sorted(unknown)}"
            )
        if self.size < 16:
            raise ValueError("data.signal_task.size must be at least 16")
        if self.target_repeats < 1:
            raise ValueError("data.signal_task.target_repeats must be at least 1")
        if not 0 <= self.resample_fraction <= 1:
            raise ValueError("data.signal_task.resample_fraction must be in [0, 1]")
        if self.evaluation_samples < 1 or self.evaluation_repeats < 1:
            raise ValueError("data.signal_task.evaluation_samples and evaluation_repeats must be positive")
        if not self.evaluation_horizons or min(self.evaluation_horizons) < 1:
            raise ValueError("data.signal_task.evaluation_horizons must be positive")
