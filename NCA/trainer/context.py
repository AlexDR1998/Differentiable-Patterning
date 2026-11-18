"""Runtime values derived from loaded data, passed to the NCA trainer."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainerContext:
    """Non-serialisable values needed by a configured training run.

    User choices belong in the experiment config; this only holds values
    derived from loaded data or the runtime environment.
    """

    run_name: str
    model_directory: str
    # Built by the entrypoint from the loaded data (see NCA/trainer/data_augmenter)
    data_augmenter: Any
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
    # One label per batch item naming its experimental condition (e.g. a
    # curriculum entry). Matching of predictions to targets in the
    # multi-target loss stays within one condition. None falls back to the
    # intervention times.
    training_condition_ids: Any = None
    validation_condition_ids: Any = None
    # Training loss weight of each batch item (e.g. per condition). None
    # weighs every item equally.
    training_batch_weights: Any = None
    # Observation time of each target image (any unit), used by non-uniform
    # training.loop.interval_mode settings to size each transition.
    observation_times: tuple[float, ...] | None = None
    # W&B tags for the run, built from the experiment config by the caller.
    wandb_tags: tuple[str, ...] = ()
