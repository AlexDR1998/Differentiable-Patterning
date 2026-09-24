"""Optax driver for Fourier-radial inverse design."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from NCA.inverse_design.geometry import coordinate_grid
from NCA.inverse_design.initial_state import build_initial_state
from NCA.inverse_design.objectives import (
    CellTypeObjective,
    GeometryPenalties,
    geometry_loss,
)
from NCA.inverse_design.result import OptimisationResult
from NCA.inverse_design.rollout import rollout_final


@dataclass(frozen=True)
class OptimisationConfig:
    iterations: int
    learning_rate: float
    total_steps: int
    mask_softness: float = 0.04
    boundary_mode: str = "soft"
    gradient_clip: float = 1.0
    resample_initial_condition: bool = False

    def __post_init__(self):
        if self.iterations < 1 or self.total_steps < 0:
            raise ValueError("iterations must be positive and total_steps nonnegative")
        if self.learning_rate <= 0 or self.mask_softness <= 0:
            raise ValueError("learning rate and mask softness must be positive")


def optimise_geometry(
    model,
    geometry,
    sample_biology,
    output_shape: tuple[int, int],
    biological_channels: int,
    key,
    objective: CellTypeObjective,
    config: OptimisationConfig,
    penalties: GeometryPenalties = GeometryPenalties(),
) -> OptimisationResult:
    """Optimise geometry while keeping the NCA and sampled patches frozen.

    ``sample_biology(key)`` must return ``[biological_channels, H, W]``. By
    default it is called once. Optional per-step resampling folds the iteration
    into the initial-condition key while remaining reproducible.
    """

    sample_key, rollout_key = jax.random.split(key)
    fixed_biology = sample_biology(sample_key)
    if fixed_biology.shape != (biological_channels, *output_shape):
        raise ValueError("sample_biology returned an unexpected shape")
    grid = coordinate_grid(output_shape, dtype=fixed_biology.dtype)
    total_channels = int(model.N_CHANNELS)

    def evaluate(candidate, biology, current_rollout_key):
        occupancy = candidate.occupancy(grid, config.mask_softness)
        initial_state = build_initial_state(
            biology,
            occupancy,
            total_channels,
            boundary_mode=config.boundary_mode,
        )
        final_state = rollout_final(
            model,
            initial_state,
            occupancy[None],
            config.boundary_mode,
            current_rollout_key,
            config.total_steps,
        )
        loss, components = geometry_loss(
            candidate, final_state, occupancy, objective, penalties
        )
        return loss, (components, occupancy, final_state)

    value_and_grad = eqx.filter_value_and_grad(evaluate, has_aux=True)
    optimiser = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(config.learning_rate),
    )
    optimiser_state = optimiser.init(eqx.filter(geometry, eqx.is_inexact_array))
    histories = []
    biology = fixed_biology
    final_occupancy = None
    final_state = None

    for iteration in range(config.iterations):
        if config.resample_initial_condition:
            biology = sample_biology(jax.random.fold_in(sample_key, iteration))
        iteration_key = jax.random.fold_in(rollout_key, iteration)
        (loss, (components, final_occupancy, final_state)), gradients = value_and_grad(
            geometry, biology, iteration_key
        )
        updates, optimiser_state = optimiser.update(
            gradients, optimiser_state, geometry
        )
        geometry = eqx.apply_updates(geometry, updates)
        histories.append({**components, "loss": loss})

    # Return outputs corresponding to the updated rather than pre-update geometry.
    _, (_, final_occupancy, final_state) = evaluate(
        geometry, biology, jax.random.fold_in(rollout_key, config.iterations)
    )
    component_history = {
        name: jnp.stack([entry[name] for entry in histories])
        for name in histories[0]
    }
    return OptimisationResult(
        geometry=geometry,
        loss_history=component_history["loss"],
        component_history=component_history,
        final_occupancy=final_occupancy,
        final_state=final_state,
        sampled_biology=biology,
    )
