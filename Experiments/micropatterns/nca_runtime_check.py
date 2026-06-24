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
        f"_{cfg.system.precision}"
        f"_loop{cfg.trainer.loop_autodiff}"
        f"_gpu{cfg.system.gpu}"
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
    from NCA.trainer.optimizer import build_optimizer
    from Experiments.config_helpers import build_loss_args, build_pool_admission_config, build_tags
    from Experiments.micropatterns.config_helpers import build_data_augmenter, load_data, build_model
    
    
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model,_model_cfg_str = build_model(cfg, key=model_key)
    # optimiser,_ = build_optimizer(cfg)
    # model = build_model(cfg)
    optimiser,opt_name = build_optimizer(cfg)
    data,_,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK,_data_cfg_str = load_data(cfg)
    data_augmenter,_data_augmenter_cfg_str = build_data_augmenter(cfg)
    run_name = build_filename(
        cfg,
        _model_cfg_str,
        _data_cfg_str,
        _data_augmenter_cfg_str,
    )
    run_name += f"_{opt_name}"


    trainer = NCA_Trainer(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        BOUNDARY_MASK=boundary_mask,
        LOSS_TIME_CHANNEL_MASK=None,
        DATA_AUGMENTER=data_augmenter,  # pyright: ignore[reportArgumentType]
        GRAD_LOSS=cfg.trainer.grad_loss,
        MODEL_DIRECTORY=MODEL_SAVE_PATH + cfg.logging.wandb.group + "/", # type: ignore
    )

    trainer.train(
        t=cfg.run.t,
        iters=cfg.run.iterations,
        REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
        WARMUP=cfg.run.warmup,
        optimiser=optimiser,
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
        key=train_key,
    )
