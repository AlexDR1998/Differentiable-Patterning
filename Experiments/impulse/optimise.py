import json

import equinox as eqx
import jax
import numpy as np

from Experiments.config import config_to_dict
from Experiments.config_helpers import (
    build_wandb_tags,
    loss_names,
    loss_weights,
    open_registry_bundle,
)
from Experiments.impulse.config_helpers import (
    build_impulse_optimiser,
    build_intervention,
    build_objective,
    build_pair_source,
    load_impulse_data,
    loss_args_from_config,
    resolve_output_directory,
)
from Experiments.model_registry import _config_digest
from NCA.trainer.impulse import NCAImpulseOptimiser
from NCA.trainer.logging.impulse_wandb_log import ImpulseLogger


def run(cfg):
    """Load a trained NCA, optimise an intervention, and save float outputs."""

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
    # The config hash keeps runs of one sweep from overwriting each other.
    run_name = f"{bundle.id}_{impulse_cfg.objective.type}_cfg{_config_digest(config)[:8]}"
    logger = None
    if cfg.logging.backend == "wandb":
        logger = ImpulseLogger(
            trajectories,
            observed_channels,
            wandb_config={
                "project": cfg.logging.wandb.project,
                "group": cfg.logging.wandb.group,
                "tags": build_wandb_tags(cfg),
                "name": run_name,
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
        logger=logger,
    )
    result = optimiser.train(
        iterations=impulse_cfg.iterations,
        batch_size=impulse_cfg.batch_size,
        key=train_key,
        evaluation_steps=impulse_cfg.rollout.evaluation_steps,
        log_every=impulse_cfg.log_every,
    )

    if logger is not None:
        logger.log_result(result)
        logger.finish()

    output_directory = resolve_output_directory(cfg.impulse.output)
    eqx.tree_serialise_leaves(output_directory / f"{run_name}.eqx", result.best_intervention)
    np.savez_compressed(
        output_directory / f"{run_name}.npz",
        initial_states=np.asarray(result.initial_states),
        target_states=np.asarray(result.target_states),
        perturbed_initial_states=np.asarray(result.perturbed_initial_states),
        final_states=np.asarray(result.final_states),
        baseline_trajectory=np.asarray(result.baseline_trajectory),
        perturbed_trajectory=np.asarray(result.perturbed_trajectory),
    )
    summary = {
        "model_id": bundle.id,
        "bundle_path": str(bundle.path),
        "best_step": result.best_step,
        "best_loss": result.best_loss,
        "evaluation_steps": impulse_cfg.rollout.evaluation_steps,
        "evaluation_loss": result.evaluation_loss,
        "baseline_evaluation_loss": result.baseline_evaluation_loss,
        "config": config,
    }
    with (output_directory / f"{run_name}.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved impulse result to {output_directory / run_name}")
    return result
