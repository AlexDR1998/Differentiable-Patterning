"""Runtime inputs that are derived while assembling an NCA experiment."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainerContext:
    """Non-serialisable values needed by a configured training run.

    User choices belong in the experiment config.  This object only carries
    values derived from loaded data or the runtime environment.
    """

    run_name: str
    model_directory: str
    data_augmenter: type
    storage_id: str | None = None
    boundary_mask: Any = None
    channel_schema: Any = None
    channel_names: tuple[str, ...] | list[str] | None = None
    timepoint_names: tuple[str, ...] | list[str] | None = None
    loss_time_channel_mask: Any = None
    observed_channels: int | None = None
    data_channels: int | None = None
    evaluation_input: Any = None
    validation_data: Any = None
    validation_boundary_mask: Any = None
    validation_loss_time_channel_mask: Any = None
    training_intervention_times: Any = None
    validation_intervention_times: Any = None
