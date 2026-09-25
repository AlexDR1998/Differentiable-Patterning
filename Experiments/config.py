"""Typed configuration for experiment workflows.

OmegaConf is only used to compose YAML files. Everything else receives the
immutable dataclasses defined here, whose section and field names match the
YAML files (``system``, ``data``, ``model``, ``run``, ``trainer``,
``optimiser``, ``loss``, ...).

Older configs (schema version 1: earlier sweep files, manifests and every model
bundle saved before the change) are translated by ``upgrade_legacy_config`` as
they are read. That function is the only place old spellings are handled.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from Common.config import ConfigValue
from Common.trainer.config import (
    GroupedPointwiseLossConfig,
    LossConfig,
    LossTermConfig,
    LossWeightScheduleConfig,
    MultiTargetLossConfig,
    OptimizerConfig,
    OttLossConfig,
    PointwiseLossConfig,
    ScheduleConfig,
    SummaryLossConfig,
    VggLossConfig,
    WassersteinLossConfig,
)
from Experiments.emoji.config import (
    EmojiDataConfig,
    EmojiPairConfig,
    ProbabilityScheduleConfig,
)
from Experiments.impulse.config import (
    CheckpointLoadConfig,
    ImpulseConfig,
    ImpulseInterventionConfig,
    ImpulseObjectiveConfig,
    ImpulseOptimizerConfig,
    ImpulsePairSourceConfig,
    ImpulseRolloutConfig,
    OutputConfig,
)
from Experiments.snowmelt.config import SnowmeltDataConfig
from Experiments.micropatterns.config import (
    KnockoutConfig,
    MicropatternDataConfig,
    ModelInitializationConfig,
)
from NCA.model.config import (
    HNCAModelConfig,
    KANConfig,
    KANModelConfig,
    ModelConfig,
)
from NCA.trainer.config import (
    ArchivedSyclTrainerBackendConfig,
    NvidiaTrainerBackendConfig,
    PoolAdmissionConfig,
    TrainerBackendConfig,
    TrainerConfig,
)
from NCA.trainer.interval_schedule import INTERVAL_MODES


CONFIG_SCHEMA_VERSION = 2

# data.dataset -> the data section that holds its settings
DATA_SECTIONS = {
    "emojis": "emoji",
    "micropatterns": "micropattern",
    "micropatterns_260726": "micropattern",
    "snowmelt": "snowmelt",
}


@dataclass(frozen=True)
class ExperimentMetadataConfig(ConfigValue):
    name: str
    stability_mode: str | None = None


@dataclass(frozen=True)
class SystemConfig(ConfigValue):
    precision: str = "highest"
    gpu: str | None = None
    xla_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig(ConfigValue):
    mode: str = "train"
    t: int = 32
    iterations: int = 2000
    # The best checkpoint is only saved after this many iterations.
    checkpoint_warmup: int = 64
    write_images: bool = True
    write_videos: bool = True
    # Emoji only: set t from the fire rate instead of using run.t.
    derive_t_from_fire_rate: bool = False
    fire_rate_step_numerator: int | None = None
    repeat: int | None = None
    # Per-transition timing (NCA/trainer/interval_schedule.py). With a non-uniform
    # mode, t is the step count of the reference interval (median by default).
    interval_mode: str = "uniform"
    reference_interval: float | None = None
    interval_steps: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.iterations <= 0 or self.t <= 0:
            raise ValueError("run.iterations and run.t must be positive")
        if self.checkpoint_warmup < 0:
            raise ValueError("run.checkpoint_warmup cannot be negative")
        if self.interval_mode not in INTERVAL_MODES:
            raise ValueError(f"run.interval_mode must be one of {INTERVAL_MODES}")
        if self.reference_interval is not None and self.reference_interval <= 0:
            raise ValueError("run.reference_interval must be positive")
        if self.interval_steps is not None:
            if self.interval_mode != "steps":
                raise ValueError("run.interval_steps requires interval_mode='steps'")
            if any(int(step) <= 0 for step in self.interval_steps):
                raise ValueError("run.interval_steps must be positive")


@dataclass(frozen=True)
class DataConfig(ConfigValue):
    dataset: str
    batches: int
    downsample: int = 1
    # Exactly one of these is set, chosen by DATA_SECTIONS[dataset].
    emoji: EmojiDataConfig | None = None
    micropattern: MicropatternDataConfig | None = None
    snowmelt: SnowmeltDataConfig | None = None
    knockout: KnockoutConfig = field(default_factory=KnockoutConfig)

    def __post_init__(self):
        if self.dataset not in DATA_SECTIONS:
            raise ValueError(f"Unsupported data.dataset {self.dataset!r}")
        if self.downsample <= 0:
            raise ValueError("data.downsample must be positive")
        expected = DATA_SECTIONS[self.dataset]
        for section in ("emoji", "micropattern", "snowmelt"):
            if (getattr(self, section) is not None) != (section == expected):
                raise ValueError(
                    f"data.dataset={self.dataset!r} needs data.{expected} settings "
                    "and no other data section"
                )


@dataclass(frozen=True)
class WandbConfig(ConfigValue):
    project: str
    group: str
    tags: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SingularValueLoggingConfig(ConfigValue):
    enabled: bool = False
    plot_spectra: bool = True
    epsilon: float = 1e-8


@dataclass(frozen=True)
class LoggingConfig(ConfigValue):
    backend: str
    wandb: WandbConfig
    singular_values: SingularValueLoggingConfig = field(
        default_factory=SingularValueLoggingConfig
    )

    def __post_init__(self):
        if self.backend not in {"none", "wandb", "tensorboard"}:
            raise ValueError(
                "logging.backend must be 'none', 'wandb' or 'tensorboard'"
            )


@dataclass(frozen=True)
class ModelStoreConfig(ConfigValue):
    enabled: bool = True
    root: str = "models"
    collection: str | None = None
    model_factory: str = "NCA.model.factory:build_model"


@dataclass(frozen=True)
class LabelsConfig(ConfigValue):
    scaling: str | None = None
    gpu: str | None = None


@dataclass(frozen=True)
class ExperimentConfig(ConfigValue):
    schema_version: int
    seed: int
    experiment: ExperimentMetadataConfig
    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    run: RunConfig
    trainer: TrainerConfig
    optimiser: OptimizerConfig
    loss: LossConfig
    logging: LoggingConfig
    model_store: ModelStoreConfig
    initialization: ModelInitializationConfig = field(
        default_factory=ModelInitializationConfig
    )
    labels: LabelsConfig = field(default_factory=LabelsConfig)


@dataclass(frozen=True)
class ImpulseExperimentConfig(ConfigValue):
    schema_version: int
    seed: int
    experiment: ExperimentMetadataConfig
    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    checkpoint: CheckpointLoadConfig
    impulse: ImpulseConfig


# ---------------------------------------------------------------------------
# Older configs
# ---------------------------------------------------------------------------

def upgrade_legacy_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a schema-version-1 config into the current layout.

    Version 1 was written in two spellings, and both are still found in files:

    * sweep YAML files and manifests used top-level ``knockout``,
      ``run.warmup``, flat ``trainer.pool_admission_*`` keys,
      ``run.filename_mode`` and ``data.emoji.regenerate``;
    * saved model bundles used ``runtime``,
      ``training.{loop, trainer, optimizer, loss, checkpoint}``,
      ``data.preprocessing`` and ``data.{augmentation, intervention}``.

    Saved bundles are never edited; they are upgraded each time they are read.
    Keys that are already in the current layout pass through unchanged.
    """
    root = copy.deepcopy(dict(value))
    root["schema_version"] = CONFIG_SCHEMA_VERSION

    def move(source, old, target, new):
        if old in source:
            if new in target:
                raise ValueError(f"Config sets both {old!r} and {new!r}")
            target[new] = source.pop(old)

    move(root, "runtime", root, "system")

    # Saved bundles nested the training sections under "training".
    training = dict(root.pop("training", None) or {})
    for old, new in (
        ("loop", "run"), ("trainer", "trainer"), ("optimizer", "optimiser"), ("loss", "loss"),
    ):
        move(training, old, root, new)
    checkpoint = dict(training.pop("checkpoint", None) or {})
    if training:
        raise ValueError(f"Unknown configuration fields under training: {sorted(training)}")

    if "run" in root or checkpoint:
        run = dict(root.get("run") or {})
        # run.warmup only ever controlled the checkpoint warmup (and, through
        # the null default below, the pool admission warmup).
        move(run, "warmup", run, "checkpoint_warmup")
        if "warmup" in checkpoint:
            run.setdefault("checkpoint_warmup", checkpoint["warmup"])
        # Only the default run-naming scheme remains.
        run.pop("filename_mode", None)
        root["run"] = run

    if "trainer" in root:
        trainer = dict(root["trainer"] or {})
        pool = dict(trainer.pop("pool_admission", None) or {})
        for key in [key for key in trainer if key.startswith("pool_admission_")]:
            pool[key.removeprefix("pool_admission_")] = trainer.pop(key)
        if pool:
            trainer["pool_admission"] = pool
        root["trainer"] = trainer

    if "data" in root:
        data = dict(root["data"] or {})
        if "preprocessing" in data:
            preprocessing = dict(data.pop("preprocessing") or {})
            if preprocessing.get("steps"):
                raise ValueError("data.preprocessing.steps is no longer supported")
            data.setdefault("downsample", preprocessing.get("downsample", 1))
        section = DATA_SECTIONS.get(data.get("dataset"))
        if "augmentation" in data and section is not None:
            move(data, "augmentation", data, section)
            if section == "emoji":
                # Bundles record the value training actually used in
                # regeneration.enabled; "regenerate" was a redundant copy.
                data["emoji"] = dict(data["emoji"])
                data["emoji"].pop("regenerate", None)
        elif section == "emoji" and "regenerate" in (data.get("emoji") or {}):
            # Old sweeps set data.emoji.regenerate, which only took effect when
            # regeneration.enabled was absent. In generated manifests the base
            # config had already filled it in, so regenerate was ignored.
            # Reproduce that here; current sweep files set
            # regeneration.enabled directly.
            emoji = dict(data["emoji"])
            regeneration = dict(emoji.get("regeneration") or {})
            regeneration.setdefault("enabled", emoji.pop("regenerate"))
            emoji["regeneration"] = regeneration
            data["emoji"] = emoji
        move(data, "intervention", data, "knockout")
        move(root, "knockout", data, "knockout")
        root["data"] = data
    return root


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _strict(cls, value: Any, path: str, **converters: Any):
    node = _mapping(value, path)
    allowed = {item.name for item in fields(cls)}
    unknown = set(node) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields under {path}: {sorted(unknown)}")
    for key, converter in converters.items():
        if key in node:
            node[key] = converter(node[key])
    return cls(**node)


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _optional_tuple(value: Any) -> tuple[Any, ...] | None:
    return None if value is None else _tuple(value)


_POINTWISE_LOSSES = {
    "l1", "l2", "euclidean", "cosine", "spectral",
    "spectral_no_phase", "spectral_phase", "bhattacharyya", "kl_divergence",
    "hellinger", "average_amplitude",
}
_VGG_LOSSES = {"vgg", "vgg_grouped", "vgg_grouped_and_l2"}
_OTT_LOSSES = {"ott", "ott_chstack", "ott_grouped", "ott_grouped_and_l2"}
_WASSERSTEIN_LOSSES = {
    "sliced_wasserstein_spatial", "sliced_wasserstein_channel",
    "sliced_wasserstein_full", "sliced_wasserstein_rotational",
    "spectral_wasserstein_full", "emd_loss",
}
_SUMMARY_LOSSES = {
    "radial_profile", "radial_profile_grouped", "channel_correlation",
    "channel_correlation_grouped",
}


def _loss_term(value: Any, path: str) -> LossTermConfig:
    if isinstance(value, str):
        value = {"type": value}
    node = _mapping(value, path)
    loss_type = str(node.get("type", "l2"))
    if loss_type in _POINTWISE_LOSSES:
        cls = PointwiseLossConfig
    elif loss_type == "l2_grouped":
        cls = GroupedPointwiseLossConfig
    elif loss_type in _VGG_LOSSES:
        cls = VggLossConfig
    elif loss_type in _OTT_LOSSES:
        cls = OttLossConfig
    elif loss_type in _WASSERSTEIN_LOSSES:
        cls = WassersteinLossConfig
    elif loss_type in _SUMMARY_LOSSES:
        cls = SummaryLossConfig
    elif loss_type == "multi_target":
        cls = MultiTargetLossConfig
    else:
        raise ValueError(f"Unsupported loss type at {path}: {loss_type!r}")
    converters = {
        "channels": lambda x: x if x is None or isinstance(x, str) else _tuple(x),
        "experiment_groups": _optional_tuple,
        "channel_importance": _optional_tuple,
        "schedule": lambda x: None if x is None else _strict(
            LossWeightScheduleConfig, x, f"{path}.schedule"
        ),
        "multi_target_schedules": lambda schedules: None
        if schedules is None
        else {
            str(name): _strict(
                LossWeightScheduleConfig,
                schedule,
                f"{path}.multi_target_schedules.{name}",
            )
            for name, schedule in _mapping(
                schedules, f"{path}.multi_target_schedules"
            ).items()
        },
    }
    return _strict(cls, node, path, **converters)


def _loss_config(value: Any, path: str) -> LossConfig:
    node = _mapping(value, path)
    terms = node.get("terms", ({"type": "l2"},))
    if isinstance(terms, (str, bytes)) or not isinstance(terms, (list, tuple)):
        raise TypeError(f"{path}.terms must be a sequence of mappings")
    node["terms"] = tuple(
        _loss_term(term, f"{path}.terms[{index}]")
        for index, term in enumerate(terms)
    )
    return _strict(LossConfig, node, path)


def _emoji_config(value: Any) -> EmojiDataConfig:
    raw = _mapping(value, "data.emoji")
    raw["sequence"] = _tuple(raw.get("sequence"))
    raw["pairs"] = tuple(
        _strict(EmojiPairConfig, pair, "data.emoji.pairs[]") for pair in raw.get("pairs", ())
    )
    if raw.get("timesteps") is not None:
        raw["timesteps"] = _tuple(raw["timesteps"])
    raw["pad"] = tuple(raw.get("pad", (10, 10, 10, 10)))
    raw["terminal_carry"] = _strict(
        ProbabilityScheduleConfig, raw.get("terminal_carry"), "data.emoji.terminal_carry"
    )
    # Regeneration defaults to damaging every batch from the start.
    regeneration = _mapping(raw.get("regeneration"), "data.emoji.regeneration")
    regeneration.setdefault("enabled", True)
    regeneration.setdefault("initial_probability", 1.0)
    regeneration.setdefault("final_probability", regeneration["initial_probability"])
    raw["regeneration"] = _strict(
        ProbabilityScheduleConfig, regeneration, "data.emoji.regeneration"
    )
    return _strict(EmojiDataConfig, raw, "data.emoji")


def _micropattern_config(value: Any) -> MicropatternDataConfig:
    raw = _mapping(value, "data.micropattern")
    raw["experiment_groups"] = _optional_tuple(raw.get("experiment_groups"))
    for split_name in ("train_replicates", "validation_replicates"):
        if raw.get(split_name) is not None:
            raw[split_name] = tuple(int(value) for value in raw[split_name])
    raw["timesteps"] = _tuple(raw.get("timesteps", (0, 12, 24, 36, 48)))
    return _strict(MicropatternDataConfig, raw, "data.micropattern")


def _data_config(value: Any) -> DataConfig:
    node = _mapping(value, "data")
    dataset = str(node.get("dataset"))
    if dataset not in DATA_SECTIONS:
        raise ValueError(f"Unsupported data.dataset {dataset!r}")
    section = DATA_SECTIONS[dataset]
    parsers = {
        "emoji": _emoji_config,
        "micropattern": _micropattern_config,
        "snowmelt": lambda raw: _strict(
            SnowmeltDataConfig, raw, "data.snowmelt",
            target_channels=_tuple, static_channels=_tuple,
        ),
    }
    return _strict(
        DataConfig,
        node,
        "data",
        batches=int,
        downsample=int,
        knockout=lambda raw: _strict(
            KnockoutConfig, raw, "data.knockout", curriculum=_optional_tuple
        ),
        **{section: parsers[section]},
    )


def _model_config(value: Any) -> ModelConfig:
    node = _mapping(value, "model")
    node["kernel_str"] = _tuple(node.get("kernel_str"))
    if node.get("family") == "FastKaNCA":
        node["kan"] = _strict(KANConfig, node.get("kan"), "model.kan")
        return _strict(KANModelConfig, node, "model")
    if node.get("family") == "HNCA":
        return _strict(HNCAModelConfig, node, "model")
    return _strict(ModelConfig, node, "model")


def _trainer_backend(value: Any) -> TrainerBackendConfig:
    node = _mapping(value, "trainer.backend")
    backend_type = str(node.get("type", "none"))
    classes = {
        "none": TrainerBackendConfig,
        "nvidia": NvidiaTrainerBackendConfig,
        "sycl": ArchivedSyclTrainerBackendConfig,
    }
    if backend_type not in classes:
        raise ValueError(
            f"trainer.backend.type must be one of {sorted(classes)}, got {backend_type!r}"
        )
    return _strict(classes[backend_type], node, "trainer.backend")


def _trainer_config(value: Any, checkpoint_warmup: int) -> TrainerConfig:
    node = _mapping(value, "trainer")
    pool = _mapping(node.get("pool_admission"), "trainer.pool_admission")
    # A null pool admission warmup means "the same as run.checkpoint_warmup".
    if pool.get("warmup") is None:
        pool["warmup"] = checkpoint_warmup
    node["pool_admission"] = _strict(PoolAdmissionConfig, pool, "trainer.pool_admission")
    node["backend"] = _trainer_backend(node.get("backend"))
    return _strict(TrainerConfig, node, "trainer")


def _optimiser_config(value: Any) -> OptimizerConfig:
    node = _mapping(value, "optimiser")
    schedule = _mapping(node.get("schedule"), "optimiser.schedule")
    for key in ("warmup_init_lr", "final_factor", "transition_fraction", "decay_rate"):
        if schedule.get(key) is not None:
            schedule[key] = float(schedule[key])
    node["schedule"] = _strict(ScheduleConfig, schedule, "optimiser.schedule")
    for key in ("learn_rate", "decay_rate", "sam_rho", "gradient_clip_norm"):
        if node.get(key) is not None:
            node[key] = float(node[key])
    return _strict(OptimizerConfig, node, "optimiser")


def _logging_config(value: Any) -> LoggingConfig:
    node = _mapping(value, "logging")
    node["wandb"] = _strict(
        WandbConfig, node.get("wandb"), "logging.wandb", tags=_optional_tuple
    )
    node["singular_values"] = _strict(
        SingularValueLoggingConfig, node.get("singular_values"), "logging.singular_values"
    )
    return _strict(LoggingConfig, node, "logging")


def _current_layout(value: Mapping[str, Any]) -> dict[str, Any]:
    root = dict(value)
    schema_version = int(root.get("schema_version", 1))
    if schema_version == 1:
        root = upgrade_legacy_config(root)
    elif schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment config schema version {schema_version}")
    return root


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def experiment_config_from_mapping(value: Mapping[str, Any]) -> ExperimentConfig:
    """Convert one resolved config mapping into an ExperimentConfig."""
    root = _current_layout(value)
    allowed = {field_.name for field_ in fields(ExperimentConfig)}
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration fields: {sorted(unknown)}")

    run = _strict(
        RunConfig,
        root.get("run"),
        "run",
        interval_steps=lambda x: None if x is None else tuple(int(v) for v in x),
        reference_interval=lambda x: None if x is None else float(x),
    )
    return ExperimentConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        seed=int(root.get("seed", 0)),
        experiment=_strict(ExperimentMetadataConfig, root["experiment"], "experiment"),
        system=_strict(SystemConfig, root.get("system"), "system", xla_flags=_tuple),
        data=_data_config(root["data"]),
        model=_model_config(root["model"]),
        run=run,
        trainer=_trainer_config(root.get("trainer"), run.checkpoint_warmup),
        optimiser=_optimiser_config(root.get("optimiser")),
        loss=_loss_config(root.get("loss"), "loss"),
        logging=_logging_config(root["logging"]),
        model_store=_strict(ModelStoreConfig, root.get("model_store"), "model_store"),
        initialization=_strict(
            ModelInitializationConfig, root.get("initialization"), "initialization"
        ),
        labels=_strict(LabelsConfig, root.get("labels"), "labels"),
    )


def impulse_experiment_config_from_mapping(
    value: Mapping[str, Any],
) -> ImpulseExperimentConfig:
    root = _current_layout(value)
    # The sweep tools add logging.wandb.group to every config, but impulse
    # runs do not log to W&B.
    root.pop("logging", None)
    allowed = {field_.name for field_ in fields(ImpulseExperimentConfig)}
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level impulse configuration fields: {sorted(unknown)}")

    impulse_node = _mapping(root["impulse"], "impulse")
    impulse_node["pair_source"] = _strict(
        ImpulsePairSourceConfig,
        impulse_node.get("pair_source"),
        "impulse.pair_source",
        stabilisation_steps=_tuple,
    )
    impulse_node["rollout"] = _strict(ImpulseRolloutConfig, impulse_node.get("rollout"), "impulse.rollout")
    impulse_node["intervention"] = _strict(ImpulseInterventionConfig, impulse_node.get("intervention"), "impulse.intervention")
    impulse_node["objective"] = _strict(ImpulseObjectiveConfig, impulse_node.get("objective"), "impulse.objective")
    impulse_node["loss"] = _loss_config(impulse_node.get("loss"), "impulse.loss")
    impulse_node["optimiser"] = _strict(ImpulseOptimizerConfig, impulse_node.get("optimiser"), "impulse.optimiser")
    impulse_node["output"] = _strict(OutputConfig, impulse_node.get("output"), "impulse.output")
    return ImpulseExperimentConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        seed=int(root.get("seed", 0)),
        experiment=_strict(ExperimentMetadataConfig, root["experiment"], "experiment"),
        system=_strict(SystemConfig, root.get("system"), "system", xla_flags=_tuple),
        data=_data_config(root["data"]),
        model=_model_config(root["model"]),
        checkpoint=_strict(CheckpointLoadConfig, root.get("checkpoint"), "checkpoint"),
        impulse=_strict(ImpulseConfig, impulse_node, "impulse"),
    )


def config_to_dict(config: ConfigValue) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a configuration dataclass")

    def serialise(value: Any) -> Any:
        if is_dataclass(value):
            return {
                item.name: serialise(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): serialise(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serialise(item) for item in value]
        return value

    return serialise(config)


def load_experiment_config(cfg: Any) -> ExperimentConfig | ImpulseExperimentConfig:
    """Resolve OmegaConf at the sole framework boundary, then discard it."""
    from omegaconf import OmegaConf

    value = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    if not isinstance(value, Mapping):
        raise TypeError("Experiment configuration must resolve to a mapping")
    if "impulse" in value:
        return impulse_experiment_config_from_mapping(value)
    return experiment_config_from_mapping(value)


__all__ = [name for name in globals() if name.endswith("Config")] + [
    "CONFIG_SCHEMA_VERSION", "DATA_SECTIONS", "config_to_dict",
    "experiment_config_from_mapping", "impulse_experiment_config_from_mapping",
    "load_experiment_config", "upgrade_legacy_config",
]
