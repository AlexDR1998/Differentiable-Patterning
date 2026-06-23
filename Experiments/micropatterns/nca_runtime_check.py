import sys
import os
from dotenv import load_dotenv

load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
sys.path.append(CODE_PATH)  # type: ignore
os.chdir(CODE_PATH) # type: ignore

def build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str):
    loss_str = "_".join(cfg.loss.primary).lower()
    loss_str += f"_vgg{cfg.loss.vgg_internal.lower()}"
    loss_str += f"_lcm{cfg.loss.regulariser_coeffs.latent_channel_match}"
    loss_str += f"_cg{cfg.loss.regulariser_coeffs.contiguous_growth}"
    if cfg.loss.random_crop:
        loss_str += "_rc"
    if cfg.loss.get("random_channel_shuffle", False):
        loss_str += "_chshuffle"
    if cfg.system.xla_flags is not None:
        _xla_str = "xla_flags_"+"".join(list(cfg.system.xla_flags))
    else:
        _xla_str = ""
    pool_enabled = cfg.trainer.get("pool_admission_enabled", True)
    runtime_str = (
        f"runtime_t{cfg.run.t}"
        f"_{cfg.system.precision}"
        f"_loop{cfg.trainer.loop_autodiff}"
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
    from Experiments.config_helpers import build_tags
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
        LOSS_ARGS={
            "metric":cfg.loss.vgg_internal,
            "channels":None,
            "experiment_groups":None,
            "S":1024,
            "K":5,
            "D":3,
            "sharpen":True,
            "epsilon":0.1,
            "internal_loss_func":"l2",
            "samples":128,
            "random_crop":cfg.loss.random_crop,
            "random_channel_shuffle":cfg.loss.get("random_channel_shuffle", False),
            "layers":cfg.loss.layers
        },
        
        LOG_EVERY=cfg.trainer.log_every,
        CLEAR_CACHE_EVERY=cfg.trainer.clear_cache_every,
        LOOP_AUTODIFF=cfg.trainer.loop_autodiff,
        POOL_ADMISSION_CONFIG={
            "enabled": cfg.trainer.get("pool_admission_enabled", True),
            "relative_threshold": cfg.trainer.get("pool_admission_relative_threshold", 1.25),
            "previous_relative_threshold": cfg.trainer.get("pool_admission_previous_relative_threshold", 1.10),
            "absolute_threshold": cfg.trainer.get("pool_admission_absolute_threshold", None),
            "ema_decay": cfg.trainer.get("pool_admission_ema_decay", 0.95),
            "warmup": cfg.trainer.get("pool_admission_warmup", None),
        },
        key=train_key,
    )
