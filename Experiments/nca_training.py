"""Shared assembly for config-driven NCA training entrypoints."""

from __future__ import annotations

from Experiments.config_helpers import build_pool_admission_config, build_tags


def build_trainer(cfg, *, model, data, run_name, data_augmenter, model_directory, **domain_kwargs):
    """Construct the configured JAX or SYCL trainer."""

    is_sycl = cfg.model.family == "NCA_sycl"
    if is_sycl:
        from NCA.trainer.NCA_sycl_trainer import NCA_sycl_Trainer

        trainer_class = NCA_sycl_Trainer
    else:
        from NCA.trainer.NCA_trainer import NCA_Trainer

        trainer_class = NCA_Trainer
    sycl_kwargs = {}
    if is_sycl:
        sycl_kwargs = {
            "SYCL_FUSED_STEPS": cfg.trainer.get("sycl_fused_steps", 2),
            "SYCL_SYNCHRONIZE_CUSTOM_CALLS": cfg.trainer.get(
                "sycl_synchronize_custom_calls", False
            ),
            "SYCL_STRICT_STAGE_SYNCHRONIZATION": cfg.trainer.get(
                "sycl_strict_stage_synchronization", False
            ),
            "SYCL_REGULARISER_REDUCTION": cfg.trainer.get(
                "sycl_regulariser_reduction", "atomic"
            ),
            "SYCL_PMEAN_LOSS": cfg.trainer.get("sycl_pmean_loss", True),
            "SYCL_PMEAN_REGULARISERS": cfg.trainer.get(
                "sycl_pmean_regularisers", True
            ),
            "SYCL_SERIALIZE_CUSTOM_CALLS": cfg.trainer.get(
                "sycl_serialize_custom_calls", False
            ),
            "SYCL_SERIALIZE_ONEMKL": cfg.trainer.get(
                "sycl_serialize_onemkl", False
            ),
            "SYCL_SERIALIZE_BACKWARD_CUSTOM_CALLS": cfg.trainer.get(
                "sycl_serialize_backward_custom_calls", False
            ),
        }

    return trainer_class(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        DATA_AUGMENTER=data_augmenter,
        BOUNDARY_MODE=cfg.trainer.get("boundary_mode", "soft"),
        SHARDING=cfg.trainer.get("sharding", None),
        GRAD_LOSS=cfg.trainer.get("grad_loss", False),
        MODEL_DIRECTORY=model_directory,
        **domain_kwargs,
        **sycl_kwargs,
    )


def train_model(
    cfg,
    *,
    trainer,
    optimiser,
    learning_rate_schedule,
    loss_args,
    run_name,
    key,
    timesteps=None,
):
    """Run the common training contract, leaving data assembly to each domain."""

    trainer.train(
        t=cfg.run.t if timesteps is None else timesteps,
        iters=cfg.run.iterations,
        optimiser=optimiser,
        LEARNING_RATE_SCHEDULE=learning_rate_schedule,
        REGULARISER_COEFFS=dict(cfg.loss.regulariser_coeffs),
        WARMUP=cfg.run.warmup,
        WRITE_IMAGES=cfg.run.write_images,
        LOSS_FUNC_STR=cfg.loss.primary,
        LOSS_ARGS=loss_args,
        KNOCKOUT_ARGS={
            "channel": cfg.knockout.get("channel", None),
            "time": cfg.knockout.get("time", None),
        },
        wandb_args={
            "project": cfg.logging.wandb.project,
            "group": cfg.logging.wandb.group,
            "tags": build_tags(cfg),
            "name": run_name,
        },
        LOG_EVERY=cfg.trainer.log_every,
        CLEAR_CACHE_EVERY=cfg.trainer.clear_cache_every,
        LOOP_AUTODIFF=cfg.trainer.loop_autodiff,
        POOL_ADMISSION_CONFIG=build_pool_admission_config(cfg),
        SINGULAR_VALUE_LOGGING_CONFIG=cfg.logging.get("singular_values", None),
        SPARSE_PRUNING=cfg.run.get("sparse_pruning", False),
        TARGET_SPARSITY=cfg.run.get("target_sparsity", 0.5),
        JAX_TRACE=cfg.trainer.get("jax_trace", False),
        key=key,
    )
