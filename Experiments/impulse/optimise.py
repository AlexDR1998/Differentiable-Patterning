import jax

from Experiments.config import config_to_dict
from Experiments.config_helpers import (
    build_wandb_tags,
    loss_names,
    loss_weights,
    open_registry_bundle,
    resolve_store_root,
)
from Experiments.emoji.multi_attractor import emoji_name
from Experiments.impulse.config_helpers import (
    build_impulse_optimiser,
    build_intervention,
    build_objective,
    build_pair_source,
    build_restarts,
    load_impulse_data,
    loss_args_from_config,
    pixel_steps,
)
from Experiments.impulse.records import create_impulse_id, publish_impulse_record
from NCA.trainer.impulse import NCAImpulseOptimiser
from NCA.trainer.logging.impulse_wandb_log import ImpulseLogger


def multi_attractor_targets(data_config, impulse_cfg, trajectories, observed_channels):
    """Target patterns [K, observed channels, H, W] and names of a multi-attractor
    model, with the source and target of this run marked, and the index of this
    run's target among them; (None, None, None) otherwise."""

    if data_config.emoji.task != "multi_attractor":
        return None, None, None
    pair_names = [emoji_name(pair.target) for pair in data_config.emoji.pairs]
    names = list(dict.fromkeys(pair_names))
    targets = trajectories[[pair_names.index(name) for name in names], -1, :observed_channels]
    roles = {
        pair_names[impulse_cfg.pair_source.source_index]: "source",
        pair_names[impulse_cfg.pair_source.target_index]: "target",
    }
    labels = [f"{name} ({roles[name]})" if name in roles else name for name in names]
    target = names.index(pair_names[impulse_cfg.pair_source.target_index])
    return targets, labels, target


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
    attractors, attractor_names, target_attractor = multi_attractor_targets(
        data_config, impulse_cfg, trajectories, observed_channels
    )
    restart_interventions, scout_iterations = build_restarts(
        impulse_cfg.intervention,
        impulse_cfg.iterations,
        observed_channels,
        model,
        trajectories,
        intervention_key,
    )
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
            attractor_names=attractor_names,
        )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=pair_source,
        intervention=intervention,
        objective=build_objective(cfg.impulse.objective, attractors, target_attractor),
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
        attractors=attractors,
        curriculum=impulse_cfg.rollout.curriculum,
        restart_interventions=restart_interventions,
        scout_iterations=scout_iterations,
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
        logger.log_result(
            result,
            pixel_steps=pixel_steps(impulse_cfg.intervention),
            frame_every=impulse_cfg.rollout.evaluation_frame_every,
            rollout_steps=impulse_cfg.rollout.steps,
        )
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
