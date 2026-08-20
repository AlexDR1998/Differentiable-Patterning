"""Typed configuration boundary for supported experiment workflows.

OmegaConf is deliberately confined to experiment composition.  The rest of
the codebase receives these immutable, serialisable values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from Common.config import ConfigValue
from Common.dataloader.preprocessing import PreprocessingConfig, ProcessingStep
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
from Experiments.micropatterns.config import KnockoutConfig, MicropatternDataConfig
from NCA.model.config import (
    KANConfig,
    KANModelConfig,
    ModelConfig,
)
from NCA.trainer.config import (
    NvidiaTrainerBackendConfig,
    PoolAdmissionConfig,
    SyclTrainerBackendConfig,
    TrainerBackendConfig,
    TrainerConfig,
)


CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentMetadataConfig(ConfigValue):
    name: str
    stability_mode: str | None = None


@dataclass(frozen=True)
class RuntimeConfig(ConfigValue):
    precision: str = "highest"
    gpu: str | None = None
    xla_flags: tuple[str, ...] = ()


def _trainer_backend(value: Any, path: str) -> TrainerBackendConfig:
    node = _mapping(value, path)
    backend_type = str(node.get("type", "none"))
    classes = {
        "none": TrainerBackendConfig,
        "nvidia": NvidiaTrainerBackendConfig,
        "sycl": SyclTrainerBackendConfig,
    }
    if backend_type not in classes:
        raise ValueError(
            f"{path}.type must be one of {sorted(classes)}, got {backend_type!r}"
        )
    return _strict(classes[backend_type], node, path)


@dataclass(frozen=True)
class CheckpointConfig(ConfigValue):
    warmup: int = 64


@dataclass(frozen=True)
class TrainingLoopConfig(ConfigValue):
    mode: str = "train"
    t: int = 32
    iterations: int = 2000
    write_images: bool = True
    derive_t_from_fire_rate: bool = False
    fire_rate_step_numerator: int | None = None
    filename_mode: str = "typed"
    repeat: int | None = None

    def __post_init__(self):
        if self.iterations <= 0 or self.t <= 0:
            raise ValueError("training loop iterations and t must be positive")


@dataclass(frozen=True)
class TrainingConfig(ConfigValue):
    loop: TrainingLoopConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True)
class DataConfig(ConfigValue):
    dataset: str
    batches: int
    preprocessing: PreprocessingConfig
    augmentation: EmojiDataConfig | MicropatternDataConfig
    intervention: KnockoutConfig = field(default_factory=KnockoutConfig)

    @property
    def downsample(self) -> int:
        return self.preprocessing.downsample

    @property
    def emoji(self) -> EmojiDataConfig:
        if not isinstance(self.augmentation, EmojiDataConfig):
            raise AttributeError("emoji configuration requested for non-emoji data")
        return self.augmentation

    @property
    def micropattern(self) -> MicropatternDataConfig:
        if not isinstance(self.augmentation, MicropatternDataConfig):
            raise AttributeError("micropattern configuration requested for non-micropattern data")
        return self.augmentation


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
    singular_values: SingularValueLoggingConfig | None = None

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
    model_factory: str = "Experiments.config_helpers:build_model"


@dataclass(frozen=True)
class LabelsConfig(ConfigValue):
    scaling: str | None = None
    gpu: str | None = None


@dataclass(frozen=True)
class ExperimentConfig(ConfigValue):
    schema_version: int
    seed: int
    experiment: ExperimentMetadataConfig
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    model_store: ModelStoreConfig
    labels: LabelsConfig = field(default_factory=LabelsConfig)

    # Compatibility names at the Experiments boundary. Common/ and NCA receive
    # focused leaf configs, not this root object.
    @property
    def system(self): return self.runtime
    @property
    def run(self): return self.training.loop
    @property
    def trainer(self): return self.training.trainer
    @property
    def optimiser(self): return self.training.optimizer
    @property
    def loss(self): return self.training.loss
    @property
    def knockout(self): return self.data.intervention


@dataclass(frozen=True)
class ImpulseExperimentConfig(ConfigValue):
    schema_version: int
    seed: int
    experiment: ExperimentMetadataConfig
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    checkpoint: CheckpointLoadConfig
    impulse: ImpulseConfig

    @property
    def system(self): return self.runtime


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


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
        "experiment_groups": lambda x: None if x is None else _tuple(x),
        "channel_importance": lambda x: None if x is None else _tuple(x),
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
    term_nodes = [({"type": term} if isinstance(term, str) else dict(term)) for term in terms]
    node["terms"] = tuple(
        _loss_term(term, f"{path}.terms[{index}]")
        for index, term in enumerate(term_nodes)
    )
    return _strict(LossConfig, node, path)


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


def experiment_config_from_mapping(value: Mapping[str, Any]) -> ExperimentConfig:
    """Convert one resolved Hydra/OmegaConf mapping into the stable schema."""
    root = dict(value)
    schema_version = int(root.get("schema_version", CONFIG_SCHEMA_VERSION))
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment config schema version {schema_version}")
    allowed = {"schema_version", "seed", "experiment", "system", "runtime", "data", "model", "training", "trainer", "optimiser", "loss", "run", "logging", "model_store", "knockout", "labels"}
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration fields: {sorted(unknown)}")

    data_node = _mapping(root["data"], "data")
    dataset = str(data_node.pop("dataset"))
    batches = int(data_node.pop("batches"))
    preprocessing_node = _mapping(
        data_node.pop("preprocessing", None), "data.preprocessing"
    )
    if "downsample" not in preprocessing_node:
        preprocessing_node["downsample"] = int(data_node.pop("downsample", 1))
    if "steps" in preprocessing_node:
        preprocessing_node["steps"] = tuple(
            ProcessingStep(step) for step in preprocessing_node["steps"]
        )
    if "histogram_percentiles" in preprocessing_node:
        preprocessing_node["histogram_percentiles"] = tuple(
            preprocessing_node["histogram_percentiles"]
        )
    preprocessing = _strict(
        PreprocessingConfig, preprocessing_node, "data.preprocessing"
    )
    stable_augmentation = data_node.pop("augmentation", None)
    stable_intervention = data_node.pop("intervention", None)
    emoji_node = data_node.pop("emoji", stable_augmentation if dataset == "emojis" else None)
    micropattern_node = data_node.pop("micropattern", stable_augmentation if dataset != "emojis" else None)
    if data_node:
        raise ValueError(f"Unknown configuration fields under data: {sorted(data_node)}")
    if dataset == "emojis":
        raw = _mapping(emoji_node, "data.emoji")
        raw["sequence"] = _tuple(raw.get("sequence"))
        raw["pairs"] = tuple(_strict(EmojiPairConfig, pair, "data.emoji.pairs[]") for pair in raw.get("pairs", ()))
        if raw.get("timesteps") is not None: raw["timesteps"] = _tuple(raw["timesteps"])
        raw["pad"] = tuple(raw.get("pad", (10, 10, 10, 10)))
        raw["terminal_carry"] = _strict(ProbabilityScheduleConfig, raw.get("terminal_carry"), "data.emoji.terminal_carry")
        regeneration = _mapping(raw.get("regeneration"), "data.emoji.regeneration")
        regeneration.setdefault("enabled", raw.get("regenerate", True))
        regeneration.setdefault("initial_probability", 1.0)
        regeneration.setdefault("final_probability", regeneration["initial_probability"])
        raw["regeneration"] = _strict(ProbabilityScheduleConfig, regeneration, "data.emoji.regeneration")
        augmentation = _strict(EmojiDataConfig, raw, "data.emoji")
    elif dataset in {"micropatterns", "micropatterns_260726"}:
        raw = _mapping(micropattern_node, "data.micropattern")
        if raw.get("experiment_groups") is not None: raw["experiment_groups"] = _tuple(raw["experiment_groups"])
        raw["timesteps"] = _tuple(raw.get("timesteps", (0, 12, 24, 36, 48)))
        augmentation = _strict(MicropatternDataConfig, raw, "data.micropattern")
    else:
        raise ValueError(f"Unsupported data.dataset {dataset!r}")
    intervention = _strict(
        KnockoutConfig,
        stable_intervention if stable_intervention is not None else root.get("knockout"),
        "data.intervention",
    )
    data = DataConfig(dataset, batches, preprocessing, augmentation, intervention)

    model_node = _mapping(root["model"], "model")
    for obsolete in ("normalization", "normalization_mean", "normalization_std", "normalization_eps", "normalization_channels"):
        if obsolete in model_node:
            raise ValueError(f"model.{obsolete} was removed with NormalizedNCA")
    model_node["kernel_str"] = _tuple(model_node.get("kernel_str"))
    model_class: type[ModelConfig] = ModelConfig
    if model_node.get("family") == "FastKaNCA":
        model_node["kan"] = _strict(KANConfig, model_node.get("kan"), "model.kan")
        model_class = KANModelConfig
    model = _strict(model_class, model_node, "model")

    training_node = _mapping(root.get("training"), "training")
    run_node = _mapping(training_node.pop("loop", root.get("run", {})), "training.loop")
    trainer_node = _mapping(training_node.pop("trainer", root.get("trainer", {})), "training.trainer")
    optimizer_node = training_node.pop("optimizer", root.get("optimiser", {}))
    loss_node = _mapping(training_node.pop("loss", root.get("loss", {})), "training.loss")
    legacy_checkpoint_warmup = run_node.pop("warmup", 64)
    checkpoint_node = training_node.pop("checkpoint", {"warmup": legacy_checkpoint_warmup})
    if training_node:
        raise ValueError(f"Unknown configuration fields under training: {sorted(training_node)}")
    pool_keys = {key for key in trainer_node if key.startswith("pool_admission_")}
    pool = _mapping(trainer_node.pop("pool_admission", None), "training.trainer.pool_admission")
    pool.update({key.removeprefix("pool_admission_"): trainer_node.pop(key) for key in pool_keys})
    trainer_node["pool_admission"] = _strict(PoolAdmissionConfig, pool, "training.trainer.pool_admission")
    trainer_node["backend"] = _trainer_backend(
        trainer_node.get("backend"), "training.trainer.backend"
    )
    optimizer_raw = _mapping(optimizer_node, "training.optimizer")
    schedule_raw = _mapping(optimizer_raw.get("schedule"), "training.optimizer.schedule")
    for key in ("warmup_init_lr", "final_factor", "transition_fraction", "decay_rate"):
        if schedule_raw.get(key) is not None:
            schedule_raw[key] = float(schedule_raw[key])
    optimizer_raw["schedule"] = _strict(ScheduleConfig, schedule_raw, "training.optimizer.schedule")
    for key in ("learn_rate", "decay_rate", "sam_rho", "gradient_clip_norm"):
        if optimizer_raw.get(key) is not None:
            optimizer_raw[key] = float(optimizer_raw[key])
    training = TrainingConfig(
        loop=_strict(TrainingLoopConfig, run_node, "training.loop"),
        trainer=_strict(TrainerConfig, trainer_node, "training.trainer"),
        optimizer=_strict(OptimizerConfig, optimizer_raw, "training.optimizer"),
        loss=_loss_config(loss_node, "training.loss"),
        checkpoint=_strict(CheckpointConfig, checkpoint_node, "training.checkpoint"),
    )
    logging_node = _mapping(root["logging"], "logging")
    logging_node["wandb"] = _strict(WandbConfig, logging_node.get("wandb"), "logging.wandb", tags=lambda x: None if x is None else _tuple(x))
    if logging_node.get("singular_values") is not None:
        logging_node["singular_values"] = _strict(SingularValueLoggingConfig, logging_node["singular_values"], "logging.singular_values")
    logging = _strict(LoggingConfig, logging_node, "logging")
    uses_sycl_model = model.family in {"NCA_sycl", "gNCA_sycl"}
    uses_sycl_trainer = training.trainer.backend.type == "sycl"
    if uses_sycl_model != uses_sycl_trainer:
        raise ValueError(
            "model.family in {'NCA_sycl', 'gNCA_sycl'} and training.trainer.backend.type='sycl' "
            "must be configured together"
        )
    return ExperimentConfig(
        schema_version=schema_version,
        seed=int(root.get("seed", 0)),
        experiment=_strict(ExperimentMetadataConfig, root["experiment"], "experiment"),
        runtime=_strict(RuntimeConfig, root.get("runtime", root.get("system")), "runtime", xla_flags=_tuple),
        data=data,
        model=model,
        training=training,
        logging=logging,
        model_store=_strict(ModelStoreConfig, root.get("model_store"), "model_store"),
        labels=_strict(LabelsConfig, root.get("labels"), "labels"),
    )


def impulse_experiment_config_from_mapping(
    value: Mapping[str, Any],
) -> ImpulseExperimentConfig:
    root = dict(value)
    allowed = {"schema_version", "seed", "experiment", "system", "runtime", "data", "model", "checkpoint", "impulse"}
    unknown = set(root) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level impulse configuration fields: {sorted(unknown)}")

    # Reuse the exact data/model parser so reconstruction has one contract.
    synthetic = {
        "schema_version": root.get("schema_version", CONFIG_SCHEMA_VERSION),
        "seed": root.get("seed", 0),
        "experiment": root["experiment"],
        "runtime": root.get("runtime", root.get("system", {})),
        "data": root["data"],
        "model": root["model"],
        "training": {
            "loop": {"t": 1, "iterations": 1},
            "trainer": {},
            "optimizer": {"warmup_steps": 0},
            "loss": {},
            "checkpoint": {"warmup": 0},
        },
        "logging": {"backend": "none", "wandb": {"project": "none", "group": "none"}},
        "model_store": {"enabled": False},
    }
    common = experiment_config_from_mapping(synthetic)
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
    impulse_loss = _mapping(impulse_node.get("loss"), "impulse.loss")
    impulse_node["loss"] = _loss_config(impulse_loss, "impulse.loss")
    impulse_node["optimiser"] = _strict(ImpulseOptimizerConfig, impulse_node.get("optimiser"), "impulse.optimiser")
    impulse_node["output"] = _strict(OutputConfig, impulse_node.get("output"), "impulse.output")
    return ImpulseExperimentConfig(
        schema_version=common.schema_version,
        seed=common.seed,
        experiment=common.experiment,
        runtime=common.runtime,
        data=common.data,
        model=common.model,
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
    "CONFIG_SCHEMA_VERSION", "config_to_dict", "experiment_config_from_mapping",
    "impulse_experiment_config_from_mapping", "load_experiment_config"
]
