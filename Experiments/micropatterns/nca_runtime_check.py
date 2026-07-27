import sys
import os
from dotenv import load_dotenv

load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
sys.path.append(CODE_PATH)  # type: ignore
os.chdir(CODE_PATH) # type: ignore

from Experiments.micropatterns.config_helpers import build_loss_filename

def build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str):
    loss_str = build_loss_filename(cfg)
    if cfg.system.xla_flags is not None:
        _xla_str = "xla_flags_"+"".join(list(cfg.system.xla_flags))
    else:
        _xla_str = ""
    pool_enabled = cfg.trainer.get("pool_admission_enabled", True)
    runtime_str = (
        f"runtime_t{cfg.run.t}"
        f"_ds{cfg.data.downsample}"
        f"_batches{cfg.data.batches}"
        f"_{cfg.system.precision}"
        f"_loop{cfg.trainer.loop_autodiff}"
        f"_batch{cfg.trainer.get('batch_mode', 'tree')}"
        f"_gpu{cfg.system.gpu}"
    )
    if cfg.trainer.get("sharding", None) is not None:
        runtime_str += f"_shard{cfg.trainer.sharding}"
    if cfg.model.family == "NCA_sycl":
        runtime_str += f"_fuse{cfg.trainer.get('sycl_fused_steps', 2)}"
        runtime_str += (
            f"_sync{int(cfg.trainer.get('sycl_synchronize_custom_calls', False))}"
        )
        runtime_str += (
            f"_stagesync{int(cfg.trainer.get('sycl_strict_stage_synchronization', False))}"
        )
        runtime_str += (
            f"_regreduce{cfg.trainer.get('sycl_regulariser_reduction', 'atomic')}"
        )
    if pool_enabled:
        pool_rel = cfg.trainer.get("pool_admission_relative_threshold", 1.25)
        pool_prev_rel = cfg.trainer.get("pool_admission_previous_relative_threshold", 1.10)
        runtime_str += f"_pool_ema{pool_rel}_prev{pool_prev_rel}"
    if _xla_str:
        runtime_str += f"_{_xla_str}"
    return "_".join([
        model_cfg_str,
        # data_cfg_str,
        # data_augmenter_cfg_str,
        loss_str,
        runtime_str,
    ])

def run(cfg):
    import jax
    from NCA.trainer.NCA_trainer import NCA_Trainer
    from NCA.trainer.NCA_sycl_trainer import NCA_sycl_Trainer
    from NCA.trainer.optimizer import build_optimizer
    from Experiments.config_helpers import build_loss_args, build_pool_admission_config, build_tags
    from Experiments.micropatterns.config_helpers import (
        build_data_augmenter,
        build_model,
        expand_channel_timestep_mask_for_loss,
        load_data,
    )
    
    
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model,_model_cfg_str = build_model(cfg, key=model_key)
    # optimiser,_ = build_optimizer(cfg)
    # model = build_model(cfg)
    optimiser,opt_name,learning_rate_schedule = build_optimizer(
        cfg,
        return_schedule=True,
    )
    data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK,_data_cfg_str = load_data(cfg)
    data_augmenter,_data_augmenter_cfg_str = build_data_augmenter(
        cfg,
        CHANNEL_TIMESTEP_MASK,
        aux.get("channel_schema"),
        cfg.data.get("batch_multiplier", 1),
    )
    loss_time_channel_mask = expand_channel_timestep_mask_for_loss(
        cfg, CHANNEL_TIMESTEP_MASK, aux.get("channel_schema")
    )
    target_timepoints = [f"t{time}h" for time in list(cfg.data.timesteps)[1:]]
    if cfg.data.get("duplicate_final_timestep", False):
        target_timepoints.append(f"{target_timepoints[-1]}_steady")
    run_name = build_filename(
        cfg,
        _model_cfg_str,
        _data_cfg_str,
        _data_augmenter_cfg_str,
    )
    run_name += f"_{opt_name}"


    trainer_class = (
        NCA_sycl_Trainer if cfg.model.family == "NCA_sycl" else NCA_Trainer
    )
    trainer_kwargs = {}
    if cfg.model.family == "NCA_sycl":
        trainer_kwargs["SYCL_FUSED_STEPS"] = cfg.trainer.get(
            "sycl_fused_steps", 2
        )
        trainer_kwargs["SYCL_SYNCHRONIZE_CUSTOM_CALLS"] = cfg.trainer.get(
            "sycl_synchronize_custom_calls", False
        )
        trainer_kwargs["SYCL_STRICT_STAGE_SYNCHRONIZATION"] = cfg.trainer.get(
            "sycl_strict_stage_synchronization", False
        )
        trainer_kwargs["SYCL_REGULARISER_REDUCTION"] = cfg.trainer.get(
            "sycl_regulariser_reduction", "atomic"
        )
        trainer_kwargs["SYCL_PMEAN_LOSS"] = cfg.trainer.get(
            "sycl_pmean_loss", True
        )
        trainer_kwargs["SYCL_PMEAN_REGULARISERS"] = cfg.trainer.get(
            "sycl_pmean_regularisers", True
        )
        trainer_kwargs["SYCL_SERIALIZE_CUSTOM_CALLS"] = cfg.trainer.get(
            "sycl_serialize_custom_calls", False
        )
    trainer = trainer_class(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        BOUNDARY_MASK=boundary_mask,
        CHANNEL_NAMES=CHANNEL_NAMES,
        CHANNEL_SCHEMA=aux.get("channel_schema"),
        TIMEPOINT_NAMES=target_timepoints,
        LOSS_TIME_CHANNEL_MASK=loss_time_channel_mask,
        DATA_AUGMENTER=data_augmenter,  # pyright: ignore[reportArgumentType]
        GRAD_LOSS=cfg.trainer.grad_loss,
        SHARDING=cfg.trainer.get("sharding", None),
        BATCH_MODE=cfg.trainer.get("batch_mode", "tree"),
        MODEL_DIRECTORY=MODEL_SAVE_PATH + cfg.logging.wandb.group + "/", # type: ignore
        **trainer_kwargs,
    )

    trainer.train(
        t=cfg.run.t,
        iters=cfg.run.iterations,
        REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
        WARMUP=cfg.run.warmup,
        optimiser=optimiser,
        LEARNING_RATE_SCHEDULE=learning_rate_schedule,
        WRITE_IMAGES=cfg.run.write_images,
        LOSS_FUNC_STR=cfg.loss.primary,
        KNOCKOUT_ARGS={
            "channel": cfg.knockout.channel,
            "time": cfg.knockout.time
        },
        wandb_args={
            "project":cfg.logging.wandb.project,
            "group":cfg.logging.wandb.group,
            # "group":"baseline-9ch-train-1",
            "tags":build_tags(cfg),
            "name":run_name
        },
        LOSS_ARGS=build_loss_args(cfg, overrides={"D": 3}),
        
        LOG_EVERY=cfg.trainer.log_every,
        CLEAR_CACHE_EVERY=cfg.trainer.clear_cache_every,
        LOOP_AUTODIFF=cfg.trainer.loop_autodiff,
        POOL_ADMISSION_CONFIG=build_pool_admission_config(cfg),
        SINGULAR_VALUE_LOGGING_CONFIG=cfg.logging.get("singular_values", None),
        JAX_TRACE=cfg.trainer.get("jax_trace", False),
        key=train_key,
    )
