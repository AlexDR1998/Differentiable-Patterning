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
    mode = cfg.training.loop.mode
    if mode == "benchmark":
        loss_name = build_loss_filename(cfg.training.loss)
        details = (
            f"runtime_t{cfg.training.loop.t}"
            f"_ds{cfg.data.downsample}"
            f"_batches{cfg.data.batches}"
            f"_{cfg.runtime.precision}"
            f"_loop{cfg.training.trainer.loop_autodiff}"
            f"_gpu{cfg.labels.gpu}"
        )
        if cfg.training.trainer.sharding is not None:
            details += f"_shard{cfg.training.trainer.sharding}"
        if cfg.training.trainer.pool_admission.enabled:
            details += (
                f"_pool_ema{cfg.training.trainer.pool_admission.relative_threshold}"
                f"_prev{cfg.training.trainer.pool_admission.previous_relative_threshold}"
            )
        if cfg.runtime.xla_flags:
            details += "_xla_flags_" + "".join(list(cfg.runtime.xla_flags))
        repeat = cfg.training.loop.repeat
        if repeat is not None:
            details += f"_rep{repeat}"
    elif mode == "train":
        loss_name = build_loss_filename(
            cfg.training.loss,
            include_loss_args=any(
                "ott" in term.type for term in cfg.training.loss.terms
            ),
        )
        details = (
            f"train_{cfg.labels.scaling}"
            f"_t{cfg.training.loop.t}"
            f"_lr{cfg.training.optimizer.learn_rate}"
            f"_dr{cfg.training.optimizer.decay_rate}"
            f"_dup{int(cfg.data.micropattern.get('duplicate_final_timestep', False))}"
            f"_irp{cfg.data.micropattern.get('intermediate_reinjection_probability', 0.5)}"
            f"_irpend{cfg.data.micropattern.get('intermediate_reinjection_probability_end', cfg.data.micropattern.get('intermediate_reinjection_probability', 0.5))}"
            f"_irpstart{cfg.data.micropattern.get('intermediate_reinjection_decay_start_fraction', 0.25)}"
        )
        experiment_groups = _as_list(cfg.data.micropattern.get("experiment_groups"))
        if experiment_groups:
            details += "_eg" + "-".join(str(group) for group in experiment_groups)
        repeat = cfg.training.loop.repeat
        if repeat is not None:
            details += f"_rep{repeat}"
    else:
        raise ValueError(f"Unknown run.mode {mode!r}; expected 'train' or 'benchmark'")

    if cfg.model.family == "NCA_sycl":
        backend = cfg.training.trainer.backend
        details += f"_fuse{backend.fused_steps}"
        details += f"_sync{int(backend.synchronize_custom_calls)}"
        details += f"_stagesync{int(backend.strict_stage_synchronization)}"
        details += f"_regreduce{backend.regulariser_reduction}"
        details += f"_mklserial{int(backend.serialize_onemkl)}"
        details += f"_bwdserial{int(backend.serialize_backward_custom_calls)}"

    return f"{model_name}_{loss_name}_{details}_{optimiser_name}"


def run(cfg):
    import jax
    from dotenv import load_dotenv

    from Experiments.config_helpers import build_model
    from NCA.registry import evaluation_input_provenance
    from Experiments.micropatterns.config_helpers import (
        build_data_augmenter,
        expand_channel_timestep_mask_for_loss,
        load_data,
    )
    from Experiments.nca_training import run_training
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.optimizer import build_optimizer

    load_dotenv()
    model_root = _cfg_get(_cfg_get(cfg, "model_store", None), "root", None)
    if not model_root:
        raise ValueError("model_store.root must be set for micropattern training.")

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    data, aux, channel_names, boundary, mask, _ = load_data(cfg.data)
    model, model_name = build_model(cfg.model, key=model_key)
    _, optimiser_name, _ = build_optimizer(
        cfg.training.optimizer,
        cfg.training.loop.iterations,
        return_schedule=True,
    )
    schema = aux.get("channel_schema")
    augmenter, _ = build_data_augmenter(
        cfg.data,
        cfg.training.loop.iterations,
        mask,
        schema,
    )
    target_timepoints = [f"t{time}h" for time in list(cfg.data.micropattern.timesteps)[1:]]
    if cfg.data.micropattern.get("duplicate_final_timestep", False):
        target_timepoints.append(f"{target_timepoints[-1]}_steady")

    run_name = build_run_name(cfg, model_name, optimiser_name)
    context = TrainerContext(
        run_name=run_name,
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        data_augmenter=augmenter,
        boundary_mask=boundary,
        channel_names=channel_names,
        channel_schema=schema,
        timepoint_names=target_timepoints,
        loss_time_channel_mask=expand_channel_timestep_mask_for_loss(
            cfg.data, mask, schema
        ),
        evaluation_input=evaluation_input_provenance(
            data, boundary_mask=boundary
        ),
    )
    loss_overrides = {"D": 3} if cfg.training.loop.mode == "benchmark" else None
    return run_training(
        cfg,
        model=model,
        data=data,
        context=context,
        key=train_key,
        loss_overrides=loss_overrides,
    )
