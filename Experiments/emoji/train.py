"""Config-driven emoji NCA training entrypoint."""

import os

from Experiments.config_helpers import _cfg_get


def run(cfg):
    import jax
    from dotenv import load_dotenv

    from Experiments.config_helpers import build_model
    from NCA.registry import create_model_id, evaluation_input_provenance
    from Experiments.emoji.config_helpers import (
        build_data_augmenter,
        build_filename,
        load_data,
        resolve_run_t,
    )
    from Experiments.nca_training import run_training
    from NCA.trainer.context import TrainerContext
    from NCA.trainer.optimizer import build_optimizer

    load_dotenv()
    model_root = _cfg_get(_cfg_get(cfg, "model_store", None), "root", None)
    if not model_root:
        raise ValueError("model_store.root must be set for emoji training.")

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    data, data_name = load_data(cfg.data)
    model, model_name = build_model(cfg.model, key=model_key)
    _, optimiser_name, _ = build_optimizer(
        cfg.training.optimizer,
        cfg.training.loop.iterations,
        return_schedule=True,
    )
    augmenter, augmenter_name = build_data_augmenter(cfg.data)
    run_name = f"{build_filename(cfg, model_name, data_name, augmenter_name)}_{optimiser_name}"

    context = TrainerContext(
        run_name=run_name,
        storage_id=create_model_id(cfg),
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        data_augmenter=augmenter,
        observed_channels=cfg.data.emoji.observed_channels,
        data_channels=cfg.data.emoji.data_channels,
        loss_time_channel_mask=cfg.training.trainer.loss_time_channel_mask,
        evaluation_input=evaluation_input_provenance(data),
    )
    return run_training(
        cfg,
        model=model,
        data=data,
        context=context,
        key=train_key,
        timesteps=resolve_run_t(cfg),
    )
