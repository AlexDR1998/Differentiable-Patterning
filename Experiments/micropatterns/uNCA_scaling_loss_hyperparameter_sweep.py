import sys
import os
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32" ) # Decent speedup on H100
import equinox as eqx
import optax
from dotenv import load_dotenv
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
sys.path.append(CODE_PATH)  # type: ignore
os.chdir(CODE_PATH) # type: ignore


from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.optimizer import build_optimizer
from Experiments.config_helpers import build_loss_args, build_pool_admission_config, build_tags
from Experiments.micropatterns.config_helpers import build_data_augmenter, build_loss_filename, load_data, build_model


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str):
    loss_str = build_loss_filename(
        cfg,
        include_layers=cfg.model.family in {"uNCA", "isouNCA"},
        include_loss_args=any("ott" in loss_name for loss_name in _as_list(cfg.loss.primary)),
    )
    train_str = (
        f"train_{cfg.run.scaling}"
        f"_t{cfg.run.t}"
        f"_lr{cfg.optimiser.learn_rate}"
        f"_dr{cfg.optimiser.decay_rate}"
    )
    return "_".join([
        model_cfg_str,
        # data_cfg_str,
        # data_augmenter_cfg_str,
        loss_str,
        train_str,
    ])


def run(cfg):
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model,_model_cfg_str = build_model(cfg, key=model_key)
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
        LOSS_ARGS=build_loss_args(cfg),
        
        LOG_EVERY=cfg.trainer.log_every,
        CLEAR_CACHE_EVERY=cfg.trainer.clear_cache_every,
        LOOP_AUTODIFF=cfg.trainer.loop_autodiff,
        POOL_ADMISSION_CONFIG=build_pool_admission_config(cfg),
        key=train_key,
    )
