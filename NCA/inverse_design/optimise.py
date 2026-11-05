"""Optax driver for Fourier-radial inverse design."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from NCA.inverse_design.density import CellDensity, CellGaps
from NCA.inverse_design.geometry import coordinate_grid
from NCA.inverse_design.initial_state import build_initial_state
from NCA.inverse_design.objectives import (
    CellTypeShareObjective,
    GeometryPenalties,
    border_fraction,
    geometry_loss,
)
from NCA.inverse_design.result import BinaryEvaluation, OptimisationResult
from NCA.inverse_design.rollout import rollout_final, rollout_trajectory
from NCA.trainer.data_augmenter.transforms import add_noise


@dataclass(frozen=True)
class OptimisationConfig:
    iterations: int
    learning_rate: float
    total_steps: int
    batch_size: int = 1
    # The mask edge width. It shrinks geometrically from mask_softness to
    # final_mask_softness (None: constant), towards the binary masks the
    # model was trained on.
    mask_softness: float = 0.04
    final_mask_softness: float | None = None
    # Gaussian noise blended into the initial state, as in training
    noise_strength: float = 0.0
    boundary_mode: str = "soft"
    gradient_clip: float = 1.0
    # Keep the mean radius, and so the colony area, fixed
    fixed_radius: bool = False
    # false keeps the geometry as given (a density-only control)
    optimise_geometry: bool = True
    # With a cell density passed to optimise_geometry: learn it (else keep it fixed)
    optimise_density: bool = True
    # Binary-mask evaluation every log_every iterations (0: only at the start
    # and end), over evaluation_seeds rollouts
    log_every: int = 0
    evaluation_seeds: int = 8
    scan_kind: str = "lax"
    checkpoints: int | None = None

    def __post_init__(self):
        if self.iterations < 1 or self.total_steps < 0 or self.batch_size < 1:
            raise ValueError("iterations and batch_size must be positive and total_steps nonnegative")
        if self.learning_rate <= 0 or self.mask_softness <= 0:
            raise ValueError("learning rate and mask softness must be positive")
        if self.final_mask_softness is not None and self.final_mask_softness <= 0:
            raise ValueError("final_mask_softness must be positive")
        if self.log_every < 0 or self.evaluation_seeds < 1:
            raise ValueError("log_every must be nonnegative and evaluation_seeds positive")

    def softness_at(self, iteration: int) -> float:
        if self.final_mask_softness is None:
            return self.mask_softness
        fraction = iteration / max(self.iterations - 1, 1)
        return self.mask_softness * (self.final_mask_softness / self.mask_softness) ** fraction


def _initial_state(model, occupancy, biology, key, config, gaps=None, density=None):
    """Masked texture, hidden channels and noise. With ``density``, the
    texture first gets gaps between cells (``CellGaps.apply``)."""

    if density is not None:
        key, coverage_key = jax.random.split(key)
        biology = gaps.apply(biology, density, coverage_key)
    state = build_initial_state(
        biology, occupancy, int(model.N_CHANNELS), boundary_mode=config.boundary_mode
    )
    if config.noise_strength > 0:
        state = add_noise(state, config.noise_strength, key)
    return state


def _rollout(model, occupancy, biology, key, config, scan_kind, checkpoints, gaps=None, density=None):
    noise_key, rollout_key = jax.random.split(key)
    state = _initial_state(model, occupancy, biology, noise_key, config, gaps, density)
    return rollout_final(
        model,
        state,
        occupancy[None],
        config.boundary_mode,
        rollout_key,
        config.total_steps,
        scan_kind=scan_kind,
        checkpoints=checkpoints,
    )


def _split_trainable(design, config):
    """``(trainable parameters, everything else)`` of a ``(geometry, density)`` pair."""

    geometry, density = jax.tree_util.tree_map(eqx.is_inexact_array, design)
    if not config.optimise_geometry:
        geometry = jax.tree_util.tree_map(lambda _: False, geometry)
    elif config.fixed_radius:
        geometry = eqx.tree_at(lambda g: g.raw_radius, geometry, replace=False)
    if density is not None and not config.optimise_density:
        density = jax.tree_util.tree_map(lambda _: False, density)
    return eqx.partition(design, (geometry, density))


@eqx.filter_jit
def _train_step(design, optimiser_state, model, biology, gaps, keys, softness, *, objective, penalties, config, optimiser):
    """One update of ``design`` = ``(geometry, density or None)`` from a batch of textures ``[B, C, H, W]``."""

    grid = coordinate_grid(biology.shape[-2:], dtype=biology.dtype)
    parameters, frozen = _split_trainable(design, config)

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_and_components(parameters):
        candidate, density = eqx.combine(parameters, frozen)
        occupancy = candidate.occupancy(grid, softness)

        def single(biology, key):
            final_state = _rollout(
                model, occupancy, biology, key, config, config.scan_kind, config.checkpoints,
                gaps, density,
            )
            return geometry_loss(candidate, final_state, occupancy, objective, penalties)

        losses, components = jax.vmap(single)(biology, keys)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, components)

    (loss, components), gradients = loss_and_components(parameters)
    updates, optimiser_state = optimiser.update(gradients, optimiser_state, parameters)
    design = eqx.apply_updates(design, updates)
    return design, optimiser_state, components


def _binary_mask(geometry, biology):
    grid = coordinate_grid(biology.shape[-2:], dtype=biology.dtype)
    return (geometry.level_set(grid) < 0).astype(biology.dtype)


@eqx.filter_jit
def _binary_rollouts(model, geometry, biology, keys, gaps, density, *, objective, config):
    mask = _binary_mask(geometry, biology)

    def single(biology, key):
        final_state = _rollout(model, mask, biology, key, config, "lax", None, gaps, density)
        score, _ = objective.score(final_state, mask, hard=True)
        return final_state, score

    final_states, scores = jax.vmap(single)(biology, keys)
    return mask, final_states, scores


def _check_density(gaps, density):
    if (gaps is None) != (density is None):
        raise ValueError("Pass both gaps and density, or neither")


def evaluate_binary(
    model, geometry, biology_bank, key, objective, config,
    gaps: CellGaps | None = None, density: CellDensity | None = None,
) -> BinaryEvaluation:
    """Roll the model out ``config.evaluation_seeds`` times on the binary mask.

    Rollout ``i`` starts from texture ``i mod K`` of ``biology_bank``
    ``[K, C, H, W]``, with gaps between cells for ``density`` if given.
    Pass the same ``key`` to compare geometries (and densities) on the same
    textures, gap patterns, noise and NCA updates.
    """

    _check_density(gaps, density)
    seeds = config.evaluation_seeds
    biology = biology_bank[np.arange(seeds) % len(biology_bank)]
    mask, final_states, scores = _binary_rollouts(
        model, geometry, biology, jax.random.split(key, seeds), gaps, density,
        objective=objective, config=config,
    )
    return BinaryEvaluation(
        mask=mask,
        final_states=final_states,
        scores=np.asarray(scores),
        border_fraction=float(border_fraction(mask)),
        density=None if density is None else float(density.value),
    )


@eqx.filter_jit
def _binary_trajectory(model, mask, biology, key, gaps, density, *, config):
    noise_key, rollout_key = jax.random.split(key)
    state = _initial_state(model, mask, biology, noise_key, config, gaps, density)
    return rollout_trajectory(
        model, state, mask[None], config.boundary_mode, rollout_key, config.total_steps
    )


def binary_trajectories(model, geometry, biology_bank, key, config, seeds, gaps=None, density=None):
    """Every state of the first ``seeds`` rollouts of ``evaluate_binary``.

    Uses the same textures and random numbers as ``evaluate_binary`` with
    this ``key`` (and ``gaps``, ``density``), so the last frames are that
    evaluation's final states. Returns the mask ``[H, W]`` and the states
    ``[seeds, total_steps + 1, C, H, W]`` as numpy arrays. Rollouts run one
    at a time to save memory.
    """

    _check_density(gaps, density)
    if not 1 <= seeds <= config.evaluation_seeds:
        raise ValueError("seeds must be between 1 and config.evaluation_seeds")
    biology_bank = jnp.asarray(biology_bank)
    mask = _binary_mask(geometry, biology_bank[0])
    keys = jax.random.split(key, config.evaluation_seeds)
    trajectories = [
        np.asarray(_binary_trajectory(
            model, mask, biology_bank[index % len(biology_bank)], keys[index], gaps, density,
            config=config,
        ))
        for index in range(seeds)
    ]
    return np.asarray(mask), np.stack(trajectories)


def optimise_geometry(
    model,
    geometry,
    biology_bank,
    key,
    objective: CellTypeShareObjective,
    config: OptimisationConfig,
    penalties: GeometryPenalties = GeometryPenalties(),
    log_fn=None,
    evaluation_key=None,
    gaps: CellGaps | None = None,
    density: CellDensity | None = None,
) -> OptimisationResult:
    """Optimise the geometry, and with ``density`` the cell density, while keeping the NCA frozen.

    ``biology_bank`` ``[K, C, H, W]`` holds initial textures over the whole
    grid; each iteration draws ``config.batch_size`` of them with new noise
    and NCA keys. With ``gaps`` and ``density`` the textures are the filled
    ones of ``fill_reference_gaps`` and get new gap patterns each iteration;
    ``config.optimise_density`` says whether the density is learned. The
    geometry (and density) with the best binary-mask score among the
    evaluations is returned as ``best_geometry`` (``best_density``).

    ``log_fn(iteration, metrics, evaluation)`` is called after every update
    with scalar metrics, and with a ``BinaryEvaluation`` (else None) when
    one was made. Evaluation ``i`` scores the design after ``i`` updates.
    Every evaluation uses ``evaluation_key`` (None: one split from ``key``),
    so pass the key used for ``evaluate_binary`` of other geometries to
    compare them on the same random numbers.
    """

    _check_density(gaps, density)
    if not config.optimise_geometry and (density is None or not config.optimise_density):
        raise ValueError("Nothing to optimise: the geometry is fixed and there is no density to learn")
    biology_bank = jnp.asarray(biology_bank)
    if biology_bank.ndim != 4:
        raise ValueError("biology_bank must have shape [K, C, H, W]")
    batch_key, default_evaluation_key = jax.random.split(key)
    if evaluation_key is None:
        evaluation_key = default_evaluation_key
    optimiser = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(config.learning_rate),
    )
    design = (geometry, density)
    optimiser_state = optimiser.init(_split_trainable(design, config)[0])
    sign = 1.0 if objective.maximise else -1.0

    evaluations = []
    evaluation_iterations = []
    best = (None, -np.inf, None)

    def evaluate(iteration, design):
        nonlocal best
        evaluation = evaluate_binary(
            model, design[0], biology_bank, evaluation_key, objective, config, gaps, design[1]
        )
        evaluations.append(evaluation)
        evaluation_iterations.append(iteration)
        if sign * evaluation.mean > best[1]:
            best = (iteration, sign * evaluation.mean, design)
        return evaluation

    histories = []
    for iteration in range(config.iterations):
        evaluation = None
        if iteration == 0 or (config.log_every and iteration % config.log_every == 0):
            evaluation = evaluate(iteration, design)
        iteration_key = jax.random.fold_in(batch_key, iteration)
        choice_key, rollout_key = jax.random.split(iteration_key)
        indices = jax.random.randint(choice_key, (config.batch_size,), 0, len(biology_bank))
        softness = config.softness_at(iteration)
        density_before = design[1]
        design, optimiser_state, components = _train_step(
            design,
            optimiser_state,
            model,
            biology_bank[indices],
            gaps,
            jax.random.split(rollout_key, config.batch_size),
            jnp.asarray(softness, dtype=biology_bank.dtype),
            objective=objective,
            penalties=penalties,
            config=config,
            optimiser=optimiser,
        )
        histories.append(components)
        if log_fn is not None:
            metrics = {name: float(value) for name, value in components.items()}
            metrics["mask_softness"] = softness
            if density_before is not None:
                # The density these components were computed with
                metrics["density"] = float(density_before.value)
            log_fn(iteration, metrics, evaluation)

    evaluation = evaluate(config.iterations, design)
    if log_fn is not None:
        log_fn(config.iterations, {}, evaluation)

    component_history = {
        name: jnp.stack([entry[name] for entry in histories]) for name in histories[0]
    }
    best_iteration, _, (best_geometry, best_density) = best
    return OptimisationResult(
        geometry=design[0],
        best_geometry=best_geometry,
        best_iteration=best_iteration,
        loss_history=component_history["loss"],
        component_history=component_history,
        evaluation_iterations=tuple(evaluation_iterations),
        evaluations=tuple(evaluations),
        density=design[1],
        best_density=best_density,
    )
