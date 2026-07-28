"""Config-driven emoji NCA training entrypoint."""

import os


def run(cfg):
    import jax
    from dotenv import load_dotenv

    from Experiments.config_helpers import build_loss_args, build_model
    from Experiments.emoji.config_helpers import (
        build_data_augmenter,
        build_filename,
        load_data,
        resolve_run_t,
    )
    from Experiments.nca_training import build_trainer, train_model
    from NCA.trainer.optimizer import build_optimizer

    load_dotenv()
    model_root = os.getenv("MODEL_SAVE_PATH")
    if model_root is None:
        raise ValueError("MODEL_SAVE_PATH must be set for emoji training.")

    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model, model_name = build_model(cfg, key=model_key)
    optimiser, optimiser_name, schedule = build_optimizer(cfg, return_schedule=True)
    data, data_name = load_data(cfg)
    augmenter, augmenter_name = build_data_augmenter(cfg)
    run_name = f"{build_filename(cfg, model_name, data_name, augmenter_name)}_{optimiser_name}"

    trainer = build_trainer(
        cfg,
        model=model,
        data=data,
        run_name=run_name,
        data_augmenter=augmenter,
        model_directory=os.path.join(model_root, cfg.logging.wandb.group, ""),
        OBS_CHANNELS=cfg.data.observed_channels,
        DATA_CHANNELS=cfg.data.data_channels,
        LOSS_TIME_CHANNEL_MASK=cfg.trainer.loss_time_channel_mask,
    )
    return train_model(
        cfg,
        trainer=trainer,
        optimiser=optimiser,
        learning_rate_schedule=schedule,
        loss_args=build_loss_args(cfg),
        run_name=run_name,
        key=train_key,
        timesteps=resolve_run_t(cfg),
    )
