import jax

from Experiments.config import config_to_dict
from Experiments.config_helpers import (
    build_wandb_tags,
    loss_names,
    loss_weights,
    open_registry_bundle,
    resolve_store_root,
)
from Experiments.impulse.config_helpers import (
    build_impulse_optimiser,
    build_intervention,
    build_objective,
    build_pair_source,
    load_impulse_data,
    loss_args_from_config,
    pixel_steps,
)
from Experiments.impulse.records import create_impulse_id, publish_impulse_record
from NCA.trainer.impulse import NCAImpulseOptimiser
from NCA.trainer.logging.impulse_wandb_log import ImpulseLogger


def run(cfg):
    """Load a trained NCA, optimise an intervention, and save it as an impulse record.

    The record goes into the model store the model came from
    (``Experiments/impulse/records.py``).
    """

    key = jax.random.PRNGKey(cfg.seed)
    model_key, intervention_key, train_key = jax.random.split(key, 3)
    bundle = open_registry_bundle(cfg.checkpoint.model_id, cfg.checkpoint.store_root)
    model = bundle.load_model(key=model_key)
    # By default, start from the same data the model was trained on.
    data_config = bundle.config.data if cfg.data is None else cfg.data
    observed_channels = data_config.emoji.observed_channels
    trajectories = load_impulse_data(data_config, model)
    pair_source = build_pair_source(cfg.impulse, model, trajectories)
    intervention = build_intervention(
        cfg.impulse.intervention,
        observed_channels,
        model,
        trajectories,
        intervention_key,
    )

    impulse_cfg = cfg.impulse
    config = config_to_dict(cfg)
    # The W&B run is named after the record it will produce
    impulse_id = create_impulse_id(cfg)
    logger = None
    if cfg.logging.backend == "wandb":
        logger = ImpulseLogger(
            trajectories,
            observed_channels,
            wandb_config={
                "project": cfg.logging.wandb.project,
                "group": cfg.logging.wandb.group,
                "tags": build_wandb_tags(cfg),
                "name": impulse_id,
                "config": config,
            },
        )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=pair_source,
        intervention=intervention,
        objective=build_objective(cfg.impulse.objective),
        optimiser=build_impulse_optimiser(cfg.impulse.optimiser),
        observed_channels=observed_channels,
        rollout_steps=impulse_cfg.rollout.steps,
        loss_window=impulse_cfg.rollout.loss_window,
        loss_names=loss_names(impulse_cfg.loss),
        loss_args=loss_args_from_config(cfg.impulse.loss),
        component_weights=loss_weights(impulse_cfg.loss),
        loss_channels=impulse_cfg.loss.terms[0].channels or "observed",
        regulariser_coefficients=dict(impulse_cfg.loss.regularisers),
        scan_kind=impulse_cfg.rollout.scan_kind,
        resample_every=impulse_cfg.resample_every,
        evaluation_frame_every=impulse_cfg.rollout.evaluation_frame_every,
        checkpoints=impulse_cfg.rollout.checkpoints,
        logger=logger,
    )
    result = optimiser.train(
        iterations=impulse_cfg.iterations,
        batch_size=impulse_cfg.batch_size,
        key=train_key,
        evaluation_steps=impulse_cfg.rollout.evaluation_steps,
        log_every=impulse_cfg.log_every,
    )

    wandb_run_id = None
    if logger is not None:
        logger.log_result(result, pixel_steps=pixel_steps(impulse_cfg.intervention))
        wandb_run_id = getattr(logger.run, "id", None)
        logger.finish()

    record = publish_impulse_record(
        store_root=resolve_store_root(cfg.checkpoint.store_root),
        impulse_id=impulse_id,
        cfg=cfg,
        bundle=bundle,
        data_config=data_config,
        result=result,
        observed_channels=observed_channels,
        wandb_run_id=wandb_run_id,
    )
    print(f"Saved impulse record to {record.path}")
    return result
