"""Configuration owned by the impulse-optimisation workflow."""

from dataclasses import dataclass

from Common.config import ConfigValue
from Common.trainer.config import LossConfig


@dataclass(frozen=True)
class CheckpointLoadConfig(ConfigValue):
    """Trained model to perturb, looked up by ID (or alias) in the model registry."""

    model_id: str
    # Defaults to MODEL_STORE_ROOT, then ./models, as for the registry CLI.
    store_root: str | None = None


@dataclass(frozen=True)
class ImpulsePairSourceConfig(ConfigValue):
    type: str
    # Indices into data.emoji.pairs: the run switches the source pattern to
    # the target pattern. Every pair source only uses these two patterns.
    source_index: int = 0
    target_index: int = 1
    stabilisation_steps: tuple[int, ...] = (128, 256)
    target_steps: int = 192
    # Time indices into the source trajectory, for trajectory_state
    initial_time: int = 0
    target_time: int = -1


@dataclass(frozen=True)
class ImpulseRolloutConfig(ConfigValue):
    steps: int = 64
    # Number of final rollout states the target loss is averaged over.
    loss_window: int = 1
    evaluation_steps: int = 256
    # Save the observed channels of every n-th evaluation state (must divide
    # evaluation_steps); keeping every full state can run out of GPU memory.
    evaluation_frame_every: int = 8
    scan_kind: str = "lax"
    # Train at 1/4, 1/2, 3/4 and then all of steps and loss_window, for a
    # quarter of the iterations each
    curriculum: bool = False
    # Saved states of a checkpointed rollout. null: equinox chooses a few and
    # the backward pass reruns most steps more than once; set it to steps to
    # save every state and rerun each step once (faster, more memory).
    checkpoints: int | None = None


@dataclass(frozen=True)
class ImpulseInterventionConfig(ConfigValue):
    channels: str = "hidden"
    spatial: str = "local"
    width: float = 0.2
    # Side length in pixels of the square impulse (spatial: patch only)
    patch_size: int = 3
    # patch only: number of separate squares, each with its own centre and values
    patches: int = 1
    # sparse only: one impulse pixel every stride pixels in each direction
    # (1: every pixel, as global)
    stride: int = 1
    # Number of NCA steps the impulse is applied for, at the same centre
    duration: int = 1
    # Not used by the run: a label for grouping runs in W&B, e.g. the
    # patch_size**2 * duration budget of a sweep
    budget: int | None = None
    # Std of the random initial values (0: start at zero, where the centre
    # has no gradient until the values have grown)
    init_scale: float = 0.0
    # random (uniform over the lattice interior) or centre
    initial_location: str = "random"
    # Number of starts (different random centres and values). Each is
    # optimised for scout_fraction of the iterations, then only the start
    # with the lowest validation objective continues.
    restarts: int = 1
    scout_fraction: float = 0.1


@dataclass(frozen=True)
class ImpulseObjectiveConfig(ConfigValue):
    type: str = "targeted"
    target_weight: float = 1.0
    tolerance: float = 0.01
    constraint_weight: float = 100.0
    magnitude: str = "l2"
    reward_weight: float = 1.0
    # attractor_contrast only: weight of the cross-entropy over attractors,
    # and its temperature (null: the smallest distance between two attractors)
    contrast_weight: float = 0.1
    temperature: float | None = None


@dataclass(frozen=True)
class ImpulseOptimiserConfig(ConfigValue):
    type: str = "adam"
    learn_rate: float = 1e-3
    gradient_clip_norm: float | None = 1.0


@dataclass(frozen=True)
class ImpulseConfig(ConfigValue):
    iterations: int
    batch_size: int
    resample_every: int
    log_every: int
    pair_source: ImpulsePairSourceConfig
    rollout: ImpulseRolloutConfig
    intervention: ImpulseInterventionConfig
    objective: ImpulseObjectiveConfig
    loss: LossConfig
    optimiser: ImpulseOptimiserConfig
    # Not used by the run; sweep over it to repeat otherwise identical configs
    # with different seeds.
    repeat: int = 0
