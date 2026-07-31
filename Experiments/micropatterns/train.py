"""Config-driven micropattern NCA training and benchmarking entrypoint."""

import os

from Experiments.config_helpers import _cfg_get
from Experiments.micropatterns.config_helpers import build_loss_filename


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def build_run_name(cfg, model_name, optimiser_name):
    mode = cfg.run.get("mode", "train")
    if mode == "benchmark":
        loss_name = build_loss_filename(cfg)
        details = (
            f"runtime_t{cfg.run.t}"
            f"_ds{cfg.data.downsample}"
            f"_batches{cfg.data.batches}"
            f"_{cfg.system.precision}"
            f"_loop{cfg.trainer.loop_autodiff}"
            f"_gpu{cfg.system.gpu}"
        )
        if cfg.trainer.get("sharding", None) is not None:
            details += f"_shard{cfg.trainer.sharding}"
        if cfg.trainer.get("pool_admission_enabled", True):
            details += (
                f"_pool_ema{cfg.trainer.get('pool_admission_relative_threshold', 1.25)}"
                f"_prev{cfg.trainer.get('pool_admission_previous_relative_threshold', 1.10)}"
            )
        if cfg.system.xla_flags:
            details += "_xla_flags_" + "".join(list(cfg.system.xla_flags))
        repeat = _cfg_get(cfg.run, "repeat", None)
        if repeat is not None:
            details += f"_rep{repeat}"
    elif mode == "train":
        loss_name = build_loss_filename(
            cfg,
            include_layers=cfg.model.family in {"uNCA", "isouNCA"},
            include_loss_args=any(
                "ott" in name for name in _as_list(cfg.loss.primary)
            ),
        )
        details = (
            f"train_{cfg.run.scaling}"
            f"_t{cfg.run.t}"
            f"_lr{cfg.optimiser.learn_rate}"
            f"_dr{cfg.optimiser.decay_rate}"
            f"_dup{int(cfg.data.get('duplicate_final_timestep', False))}"
            f"_irp{cfg.data.get('intermediate_reinjection_probability', 0.5)}"
            f"_irpend{cfg.data.get('intermediate_reinjection_probability_end', cfg.data.get('intermediate_reinjection_probability', 0.5))}"
            f"_irpstart{cfg.data.get('intermediate_reinjection_decay_start_fraction', 0.25)}"
        )
        repeat = _cfg_get(cfg.run, "repeat", None)
        if repeat is not None:
            details += f"_rep{repeat}"
    else:
        raise ValueError(f"Unknown run.mode {mode!r}; expected 'train' or 'benchmark'")

    if cfg.model.family == "NCA_sycl":
        details += f"_fuse{cfg.trainer.get('sycl_fused_steps', 2)}"
        details += f"_sync{int(cfg.trainer.get('sycl_synchronize_custom_calls', False))}"
        details += f"_stagesync{int(cfg.trainer.get('sycl_strict_stage_synchronization', False))}"
        details += f"_regreduce{cfg.trainer.get('sycl_regulariser_reduction', 'atomic')}"
        details += f"_mklserial{int(cfg.trainer.get('sycl_serialize_onemkl', False))}"
        details += f"_bwdserial{int(cfg.trainer.get('sycl_serialize_backward_custom_calls', False))}"

    return f"{model_name}_{loss_name}_{details}_{optimiser_name}"


def run(cfg):
    import jax
    from dotenv import load_dotenv

    from Experiments.config_helpers import (
        build_loss_args,
        build_model,
        compute_model_channel_statistics,
    )
    from Experiments.micropatterns.config_helpers import (
        build_data_augmenter,
        expand_channel_timestep_mask_for_loss,
        load_data,
    )
    from Experiments.nca_training import build_trainer, train_model
    from NCA.trainer.optimizer import build_optimizer

    load_dotenv()
    model_root = _cfg_get(_cfg_get(cfg, "model_store", None), "root", None)
    if not model_root:
        raise ValueError("model_store.root must be set for micropattern training.")

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    data, aux, channel_names, boundary, mask, _ = load_data(cfg)
    if cfg.model.family == "NormalizedNCA":
        schema = aux.get("channel_schema")
        cfg.model.normalization_channels = (
            schema.n_state_channels if schema is not None else data.shape[2]
        )
    if (
        cfg.model.family == "NormalizedNCA"
        and cfg.model.get("normalization", "none") == "fixed"
        and (
            cfg.model.get("normalization_mean", None) is None
            or cfg.model.get("normalization_std", None) is None
        )
    ):
        statistics_mask = aux.get("measurement_mask", mask)
        mean, std = compute_model_channel_statistics(
            data,
            model_channels=cfg.model.channels,
            channel_schema=schema,
            measurement_mask=statistics_mask,
            epsilon=cfg.model.get("normalization_eps", 1e-6),
        )
        cfg.model.normalization_mean = mean.tolist()
        cfg.model.normalization_std = std.tolist()
    model, model_name = build_model(cfg, key=model_key)
    optimiser, optimiser_name, schedule = build_optimizer(cfg, return_schedule=True)
    schema = aux.get("channel_schema")
    augmenter, _ = build_data_augmenter(
        cfg, mask, schema, cfg.data.get("batch_multiplier", 1)
    )
    target_timepoints = [f"t{time}h" for time in list(cfg.data.timesteps)[1:]]
    if cfg.data.get("duplicate_final_timestep", False):
        target_timepoints.append(f"{target_timepoints[-1]}_steady")

    run_name = build_run_name(cfg, model_name, optimiser_name)
    trainer = build_trainer(
        cfg,
        model=model,
        data=data,
        run_name=run_name,
        data_augmenter=augmenter,
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        BOUNDARY_MASK=boundary,
        CHANNEL_NAMES=channel_names,
        CHANNEL_SCHEMA=schema,
        TIMEPOINT_NAMES=target_timepoints,
        LOSS_TIME_CHANNEL_MASK=expand_channel_timestep_mask_for_loss(cfg, mask, schema),
    )
    loss_overrides = {"D": 3} if cfg.run.get("mode", "train") == "benchmark" else None
    return train_model(
        cfg,
        trainer=trainer,
        optimiser=optimiser,
        learning_rate_schedule=schedule,
        loss_args=build_loss_args(cfg, overrides=loss_overrides),
        run_name=run_name,
        key=train_key,
    )
