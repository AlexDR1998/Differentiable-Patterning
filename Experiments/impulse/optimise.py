import json

import equinox as eqx
import jax
import numpy as np

from Experiments.config_helpers import load_model_checkpoint, set_matmul_precision
from Experiments.config_helpers import loss_names, loss_weights
from Experiments.impulse.config_helpers import (
    build_impulse_optimizer,
    build_intervention,
    build_objective,
    build_pair_source,
    load_impulse_data,
    loss_args_from_config,
    resolve_output_directory,
)
from NCA.trainer.impulse import NCAImpulseOptimiser


def run(cfg):
    """Load a trained NCA, optimise an intervention, and save float outputs."""

    set_matmul_precision(cfg.runtime)
    key = jax.random.PRNGKey(cfg.seed)
    model_key, intervention_key, train_key = jax.random.split(key, 3)
    model, _, checkpoint_path = load_model_checkpoint(
        cfg.model,
        cfg.checkpoint,
        key=model_key,
    )
    trajectories = load_impulse_data(cfg.data, model)
    pair_source = build_pair_source(cfg.impulse, model, trajectories)
    intervention = build_intervention(
        cfg.impulse.intervention,
        cfg.data.emoji.observed_channels,
        model,
        trajectories,
        intervention_key,
    )

    impulse_cfg = cfg.impulse
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=pair_source,
        intervention=intervention,
        objective=build_objective(cfg.impulse.objective),
        optimiser=build_impulse_optimizer(cfg.impulse.optimiser),
        observed_channels=cfg.data.emoji.observed_channels,
        rollout_steps=impulse_cfg.rollout.steps,
        loss_names=loss_names(impulse_cfg.loss),
        loss_args=loss_args_from_config(cfg.impulse.loss),
        component_weights=loss_weights(impulse_cfg.loss),
        loss_channels=impulse_cfg.loss.terms[0].channels or "observed",
        regulariser_coefficients=dict(impulse_cfg.loss.regularisers),
        scan_kind=impulse_cfg.rollout.scan_kind,
        resample_every=impulse_cfg.resample_every,
    )
    result = optimiser.train(
        iterations=impulse_cfg.iterations,
        batch_size=impulse_cfg.batch_size,
        key=train_key,
        evaluation_steps=impulse_cfg.rollout.evaluation_steps,
        log_every=impulse_cfg.log_every,
    )

    output_directory = resolve_output_directory(cfg.impulse.output)
    run_name = f"{checkpoint_path.stem}_{impulse_cfg.objective.type}"
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
        "checkpoint": str(checkpoint_path),
        "best_step": result.best_step,
        "best_loss": result.best_loss,
    }
    with (output_directory / f"{run_name}.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved impulse result to {output_directory / run_name}")
    return result
