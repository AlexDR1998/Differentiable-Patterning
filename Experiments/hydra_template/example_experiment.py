from __future__ import annotations
import jax.numpy as jnp
import jax.random as jr
import jax 
from omegaconf import OmegaConf
# import argparse
print(jax.default_backend())
print(jax.devices())


def run(cfg) -> None:
    trainer_init_kwargs = {
        "BOUNDARY_MODE": cfg.trainer.boundary_mode,
        "SHARDING": cfg.trainer.sharding,
        "GRAD_LOSS": cfg.trainer.grad_loss,
        "OBS_CHANNELS": cfg.data.observed_channels,
        "DATA_CHANNELS": cfg.data.data_channels,
        "LOSS_TIME_CHANNEL_MASK": cfg.trainer.loss_time_channel_mask,
        "MODEL_DIRECTORY": cfg.trainer.model_directory,
        "LOG_DIRECTORY": cfg.trainer.log_directory,
    }

    train_kwargs = {
        "t": cfg.run.t,
        "iters": cfg.run.iterations,
        "WARMUP": cfg.run.warmup,
        "LOG_EVERY": cfg.run.log_every,
        "CLEAR_CACHE_EVERY": cfg.run.clear_cache_every,
        "WRITE_IMAGES": cfg.run.write_images,
        "LOOP_AUTODIFF": cfg.run.loop_autodiff,
        "SPARSE_PRUNING": cfg.run.sparse_pruning,
        "TARGET_SPARSITY": cfg.run.target_sparsity,
        "LOSS_FUNC_STR": [cfg.loss.primary],
        "LOSS_ARGS": OmegaConf.to_container(cfg.loss.args, resolve=True),
        "REGULARISER_COEFFS": OmegaConf.to_container(cfg.loss.regulariser_coeffs, resolve=True),
        "KNOCKOUT_ARGS": OmegaConf.to_container(cfg.knockout, resolve=True),
        "wandb_args": OmegaConf.to_container(cfg.logging.wandb, resolve=True),
    }

    summary = {
        "seed": cfg.seed,
        "experiment": cfg.experiment.name,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "optimiser": OmegaConf.to_container(cfg.optimiser, resolve=True),
        "trainer_init_kwargs": trainer_init_kwargs,
        "train_kwargs": train_kwargs,
    }

    print("Resolved high-level NCA experiment config:")
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))
    A = jr.normal(jr.PRNGKey(0), (100, 100))
    B = jr.normal(jr.PRNGKey(1), (100, 100))
    C = jnp.dot(A, B)
    print(C.sum())