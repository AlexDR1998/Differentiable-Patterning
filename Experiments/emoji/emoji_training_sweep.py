import os
import sys

import jax
from dotenv import load_dotenv

load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
if CODE_PATH is not None:
    sys.path.append(CODE_PATH)
    os.chdir(CODE_PATH)

from Experiments.config_helpers import (
    build_loss_args,
    build_model,
    build_pool_admission_config,
    build_tags,
    set_matmul_precision,
)
from Experiments.emoji.config_helpers import (
    build_data_augmenter,
    build_filename,
    load_data,
    resolve_run_t,
)
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.optimizer import build_optimizer


def run(cfg):
    if MODEL_SAVE_PATH is None:
        raise ValueError("MODEL_SAVE_PATH must be set for emoji training.")

    set_matmul_precision(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)

    model, model_cfg_str = build_model(cfg, key=model_key)
    optimiser, opt_name = build_optimizer(cfg)
    data, data_cfg_str = load_data(cfg)
    data_augmenter, data_augmenter_cfg_str = build_data_augmenter(cfg)

    run_name = build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str)
    run_name += f"_{opt_name}"

    trainer = NCA_Trainer(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        DATA_AUGMENTER=data_augmenter,
        BOUNDARY_MODE=cfg.trainer.boundary_mode,
        SHARDING=cfg.trainer.sharding,
        GRAD_LOSS=cfg.trainer.grad_loss,
        OBS_CHANNELS=cfg.data.observed_channels,
        DATA_CHANNELS=cfg.data.data_channels,
        LOSS_TIME_CHANNEL_MASK=cfg.trainer.loss_time_channel_mask,
        MODEL_DIRECTORY=os.path.join(MODEL_SAVE_PATH, cfg.logging.wandb.group, ""),
    )

    trainer.train(
        t=resolve_run_t(cfg),
        iters=cfg.run.iterations,
        REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
        WARMUP=cfg.run.warmup,
        optimiser=optimiser,
        WRITE_IMAGES=cfg.run.write_images,
        LOSS_FUNC_STR=cfg.loss.primary,
        LOSS_ARGS=build_loss_args(cfg),
        KNOCKOUT_ARGS={
            "channel": cfg.knockout.channel,
            "time": cfg.knockout.time,
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
        SPARSE_PRUNING=cfg.run.sparse_pruning,
        TARGET_SPARSITY=cfg.run.target_sparsity,
        key=train_key,
    )
