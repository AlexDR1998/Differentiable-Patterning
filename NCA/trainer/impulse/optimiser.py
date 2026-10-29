from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from Common.trainer.loss_table import build_loss_functions
from NCA.trainer.impulse.outcomes import outcome_fractions, switch_steps
from NCA.trainer.impulse.regularisers import intervention_metrics, weighted_regulariser
from NCA.trainer.impulse.rollout import identity_boundary, run_nca_batch, run_with_intervention
from NCA.trainer.impulse.types import ImpulseBatch, ImpulseResult


# Rollout-length curriculum: training runs at 1/4, 2/4, 3/4 and then all of
# the rollout (and loss window), for an equal share of the iterations each.
CURRICULUM_STAGES = 4


@dataclass
class _Start:
    """One starting intervention: its optimisation state and the best intervention so far."""

    intervention: object
    opt_state: object
    batch: object
    locations: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    # Logger calls held back while scouting, replayed if this start is chosen
    log: list | None = None


def _select_channels(states, channel_mode, observed_channels):
    """Select the state channels used by the target loss."""

    if channel_mode == "observed":
        return states[:, :observed_channels]
    if channel_mode == "hidden":
        return states[:, observed_channels:]
    if channel_mode == "all":
        return states
    if isinstance(channel_mode, (list, tuple)):
        return states[:, jnp.asarray(channel_mode)]
    raise ValueError("loss_channels must be 'observed', 'hidden', 'all', or a channel list")


@dataclass
class NCAImpulseOptimiser:
    """Optimise one shared intervention while keeping the NCA model frozen.

    Pair sources define where initial and target states come from. Objectives
    define whether the intervention targets, disrupts, or preserves an outcome.
    The returned intervention is the one with the lowest objective on a fixed
    validation batch after the evaluation rollout, checked every log_every steps.
    """

    model: object
    pair_source: object
    intervention: object
    objective: object
    optimiser: optax.GradientTransformation
    observed_channels: int
    rollout_steps: int
    # Average the target loss over the last loss_window states of the rollout,
    # so that the intervention must reach the target and stay there.
    loss_window: int = 1
    loss_functions: object = None
    loss_names: object = ("l2",)
    loss_args: dict | None = None
    component_weights: object = None
    loss_channels: object = "observed"
    regulariser_coefficients: dict | None = None
    boundary_callback: object = identity_boundary
    scan_kind: str = "lax"
    resample_every: int = 0
    # The saved evaluation trajectories keep the observed channels of every
    # n-th state only, so long rollouts of large lattices fit in memory.
    evaluation_frame_every: int = 1
    # Saved states of a checkpointed rollout (see run_nca_batch)
    checkpoints: int | None = None
    # [K, observed channels, H, W] patterns of a multi-attractor model. When
    # given, validation also reports the fraction of rollouts ending at each
    # (see outcomes.py; tolerance as in the multi-attractor explorer).
    attractors: object = None
    attractor_tolerance: float = 0.25
    # Train on shorter rollouts first (see CURRICULUM_STAGES)
    curriculum: bool = False
    # Other starting interventions (e.g. other random centres). Each start is
    # optimised for scout_iterations, then only the one with the lowest
    # validation objective continues.
    restart_interventions: tuple = ()
    scout_iterations: int = 0
    logger: object = None

    def __post_init__(self):
        """Validate configuration and construct named loss functions."""

        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if self.loss_window < 1:
            raise ValueError("loss_window must be at least 1")
        if self.loss_window > max(self.rollout_steps, 1):
            raise ValueError("loss_window cannot be longer than rollout_steps")
        if self.resample_every < 0:
            raise ValueError("resample_every must be non-negative")
        if self.curriculum and self.rollout_steps < CURRICULUM_STAGES:
            raise ValueError(f"A curriculum needs at least {CURRICULUM_STAGES} rollout steps")
        if self.restart_interventions and self.scout_iterations < 1:
            raise ValueError("scout_iterations must be positive when there are restarts")
        # The intervention must end before the loss window starts, also in
        # the shortest curriculum stage
        steps, window = self._stage_rollout(0, 1)
        last_step = max(steps - window + 1, 1)
        if self.intervention.duration > last_step:
            raise ValueError(
                f"The intervention lasts {self.intervention.duration} steps but must end "
                f"before the loss window: at most rollout_steps - loss_window + 1 = {last_step}"
                + (" in the first curriculum stage" if self.curriculum else "")
            )
        if self.loss_functions is None:
            self.loss_functions = build_loss_functions(self.loss_names, self.loss_args or {})
        else:
            self.loss_functions = list(self.loss_functions)
        if not self.loss_functions:
            raise ValueError("At least one loss function is required")
        if self.component_weights is None:
            self.component_weights = jnp.ones((len(self.loss_functions),))
        else:
            self.component_weights = jnp.asarray(self.component_weights)
        if len(self.component_weights) != len(self.loss_functions):
            raise ValueError("component_weights must match the number of loss functions")
        if jnp.any(self.component_weights < 0):
            raise ValueError("component_weights cannot be negative")
        if float(jnp.sum(self.component_weights)) == 0.0:
            raise ValueError("component_weights must contain a positive value")

    def _stage_rollout(self, index, iterations):
        """Rollout steps and loss window used at iteration ``index``."""

        if not self.curriculum:
            return self.rollout_steps, self.loss_window
        stage = min(CURRICULUM_STAGES * index // iterations, CURRICULUM_STAGES - 1) + 1
        return (
            self.rollout_steps * stage // CURRICULUM_STAGES,
            max(self.loss_window * stage // CURRICULUM_STAGES, 1),
        )

    def _loss_components(self, predictions, targets, key):
        """Return one per-sample loss array for each configured loss function."""

        predictions = _select_channels(predictions, self.loss_channels, self.observed_channels)
        targets = _select_channels(targets, self.loss_channels, self.observed_channels)
        components = []
        for index, loss_function in enumerate(self.loss_functions):
            loss_key = jax.random.fold_in(key, index)
            components.append(
                loss_function(predictions, targets, key=loss_key, where=None, cache=None)
            )
        return jnp.stack(components)

    def _weighted_loss(self, components):
        """Combine loss components into one per-sample loss."""

        weights = self.component_weights[:, None]
        return jnp.sum(weights * components, axis=0) / jnp.sum(weights)

    def _switch_rate(self, final_states, batch):
        """Fraction of final states nearer the target than the source state.

        Distances are the mean squared error of the observed channels. For
        stable_attractor pairs the source is the grown source attractor; for
        the other pair sources it is the start state.
        """

        def distance(states, references):
            difference = states[:, : self.observed_channels] - references[:, : self.observed_channels]
            return jnp.mean(difference**2, axis=(-3, -2, -1))

        nearer_target = distance(final_states, batch.target_states) < distance(
            final_states, batch.initial_states
        )
        return float(jnp.mean(nearer_target))

    def _loss_curve(self, trajectory, targets, key):
        """Mean target loss of every saved frame [batch, frames, observed channels, H, W].

        None when the loss uses hidden channels, which the saved frames do not keep.
        """

        if self.loss_channels != "observed":
            return None
        frames = trajectory.shape[1]
        if frames == 0:
            return np.zeros((0,))
        # [batch, frames, ...] -> [batch * frames, ...], each target repeated frames times
        predictions = trajectory.reshape((-1, *trajectory.shape[2:]))
        targets = jnp.repeat(targets[:, : self.observed_channels], frames, axis=0)
        losses = self._weighted_loss(self._loss_components(predictions, targets, key))
        return np.asarray(jnp.mean(losses.reshape((-1, frames)), axis=0))

    def evaluate(self, intervention, batch, key, rollout_steps=None, loss_window=None):
        """Evaluate an intervention without updating its parameters.

        ``rollout_steps`` and ``loss_window`` default to the configured ones
        (the curriculum passes shorter ones).
        """

        rollout_steps = self.rollout_steps if rollout_steps is None else rollout_steps
        loss_window = self.loss_window if loss_window is None else loss_window
        # The first delta perturbs the initial state; later ones are added
        # during the first rollout segment.
        deltas = intervention.deltas(batch.initial_states)
        perturbed = batch.initial_states + deltas[0]
        forcing = deltas[1:] if len(deltas) > 1 else None
        rollout = dict(
            boundary_callback=self.boundary_callback,
            scan_kind=self.scan_kind,
            checkpoints=self.checkpoints,
        )
        if loss_window == 1:
            final_states = run_nca_batch(
                self.model, perturbed, rollout_steps, key, forcing=forcing, **rollout
            )
            predictions = final_states
            components = self._loss_components(final_states, batch.target_states, key)
        else:
            # Run up to the loss window without keeping states, then keep only
            # the window (and only the observed channels, when the loss uses no others).
            states = run_nca_batch(
                self.model,
                perturbed,
                rollout_steps - loss_window,
                key,
                forcing=forcing,
                **rollout,
            )
            final_states, window = run_nca_batch(
                self.model,
                states,
                loss_window,
                jax.random.fold_in(key, 1),
                return_trajectory=True,
                trajectory_channels=(
                    self.observed_channels if self.loss_channels == "observed" else None
                ),
                **rollout,
            )
            # [batch, window, ...] -> [batch * window, ...], each target repeated window times
            predictions = window.reshape((-1, *window.shape[2:]))
            targets = jnp.repeat(batch.target_states, loss_window, axis=0)
            components = self._loss_components(predictions, targets, key)
            components = components.reshape((len(components), len(perturbed), loss_window))
            components = jnp.mean(components, axis=-1)
        per_sample_loss = self._weighted_loss(components)
        target_loss = jnp.mean(per_sample_loss)
        metrics = intervention_metrics(batch.initial_states, deltas)
        penalty = weighted_regulariser(metrics, self.regulariser_coefficients)
        total_loss = self.objective(target_loss, penalty, metrics, predictions)
        return {
            "total_loss": total_loss,
            "target_loss": target_loss,
            "regulariser": penalty,
            "loss_components": components,
            "loss_per_sample": per_sample_loss,
            "intervention_metrics": metrics,
            "perturbed_initial_states": perturbed,
            "final_states": final_states,
        }

    def _make_step(self, rollout_steps, loss_window):
        """Build the compiled intervention update function for one rollout length."""

        @eqx.filter_jit
        def step(intervention, opt_state, initial_states, target_states, key):
            batch = ImpulseBatch(initial_states, target_states)

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_and_aux(candidate):
                values = self.evaluate(candidate, batch, key, rollout_steps, loss_window)
                return values["total_loss"], values

            (_, values), gradients = loss_and_aux(intervention)
            # Before any clipping by the optimiser
            values["gradient_norms"] = {
                "total": optax.global_norm(gradients),
                "location": jnp.linalg.norm(gradients.location),
                "values": jnp.linalg.norm(gradients.values),
            }
            parameters, static = eqx.partition(intervention, eqx.is_inexact_array)
            updates, opt_state = self.optimiser.update(gradients, opt_state, parameters)
            parameters = eqx.apply_updates(parameters, updates)
            intervention = eqx.combine(parameters, static)
            return intervention, opt_state, values

        return step

    def _make_validation(self, batch, steps, key):
        """Build a function that scores an intervention on a fixed validation batch.

        It returns the switch rate, target loss and objective after a
        ``steps`` rollout, with the same batch and key every time, so changes
        come from the intervention only. With ``attractors``, ``outcomes`` is
        the fraction of rollouts ending at each attractor, then other, died
        and diverged.
        """

        @eqx.filter_jit
        def rollout(intervention):
            final_states = run_with_intervention(
                self.model,
                intervention,
                batch.initial_states,
                steps,
                key,
                boundary_callback=self.boundary_callback,
            )
            loss = jnp.mean(
                self._weighted_loss(self._loss_components(final_states, batch.target_states, key))
            )
            metrics = intervention_metrics(batch.initial_states, intervention.deltas(batch.initial_states))
            penalty = weighted_regulariser(metrics, self.regulariser_coefficients)
            return final_states, loss, self.objective(loss, penalty, metrics, final_states)

        def validate(intervention):
            final_states, loss, objective = rollout(intervention)
            values = {
                "switch_rate": self._switch_rate(final_states, batch),
                "evaluation_loss": float(loss),
                "evaluation_objective": float(objective),
            }
            if self.attractors is not None:
                values["outcomes"] = outcome_fractions(
                    final_states[:, : self.observed_channels], self.attractors, self.attractor_tolerance
                )
            return values

        return validate

    def train(self, iterations, batch_size, key, evaluation_steps=None, log_every=100):
        """Optimise an intervention and return its best evaluated state.

        Every ``log_every`` steps the current intervention is scored on a fixed
        validation batch after an ``evaluation_steps`` rollout; best means the
        lowest validation objective. Pair batches are regenerated at
        ``resample_every`` when that value is positive.
        """

        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.restart_interventions and self.scout_iterations >= iterations:
            raise ValueError("scout_iterations must be smaller than iterations")

        sample_key, train_key = jax.random.split(key)
        batch = self.pair_source.sample(batch_size, self.model, sample_key)
        evaluation_steps = self.rollout_steps if evaluation_steps is None else evaluation_steps
        if evaluation_steps % self.evaluation_frame_every:
            raise ValueError(
                f"evaluation_steps ({evaluation_steps}) must be a multiple of "
                f"evaluation_frame_every ({self.evaluation_frame_every})"
            )
        validation_batch = self.pair_source.sample(batch_size, self.model, jax.random.fold_in(key, 1))
        validate = self._make_validation(validation_batch, evaluation_steps, jax.random.fold_in(key, 2))
        # One compiled step per rollout length
        steps = {}
        # Only local and patch interventions have a centre that moves
        track_location = self.intervention.mode["spatial"] in ("local", "patch")

        def log(start, method, *arguments):
            if self.logger is None:
                return
            if start.log is None:
                getattr(self.logger, method)(*arguments)
            else:
                start.log.append((method, arguments))

        def advance(start, first, stop):
            """Optimise ``start`` from iteration ``first`` up to ``stop``."""

            for index in range(first, stop):
                step_key = jax.random.fold_in(train_key, index)
                if self.resample_every and index > 0 and index % self.resample_every == 0:
                    start.batch = self.pair_source.sample(batch_size, self.model, step_key)
                evaluated_intervention = start.intervention
                if track_location:
                    start.locations.append(np.asarray(evaluated_intervention.get_location()))
                rollout = self._stage_rollout(index, iterations)
                if rollout not in steps:
                    steps[rollout] = self._make_step(*rollout)
                start.intervention, start.opt_state, metrics = steps[rollout](
                    evaluated_intervention,
                    start.opt_state,
                    start.batch.initial_states,
                    start.batch.target_states,
                    step_key,
                )
                if index % log_every:
                    continue
                validation = validate(evaluated_intervention)
                # The first check always counts, even if its score is NaN
                if "intervention" not in start.best or validation["evaluation_objective"] < start.best["score"]:
                    start.best = {
                        "score": validation["evaluation_objective"],
                        "intervention": evaluated_intervention,
                        "batch": start.batch,
                        "step": index,
                        "loss": float(metrics["total_loss"]),
                        "metrics": metrics,
                    }
                switch_rate = self._switch_rate(metrics["final_states"], start.batch)
                log(start, "log_training", {**metrics, "switch_rate": switch_rate}, index, log_every)
                log(start, "log_validation", validation, index)

        def new_start(intervention, scouting):
            parameters, _ = eqx.partition(intervention, eqx.is_inexact_array)
            return _Start(intervention, self.optimiser.init(parameters), batch, log=[] if scouting else None)

        candidates = [self.intervention, *self.restart_interventions]
        restart_objectives = None
        if len(candidates) == 1:
            start = new_start(self.intervention, scouting=False)
            first = 0
        else:
            # Scout every start for the same iterations (and batches), then
            # continue the one with the lowest validation objective.
            scouts = [new_start(candidate, scouting=True) for candidate in candidates]
            for scout in scouts:
                advance(scout, 0, self.scout_iterations)
            restart_objectives = np.array(
                [validate(scout.intervention)["evaluation_objective"] for scout in scouts]
            )
            start = scouts[int(np.argmin(restart_objectives))]
            for method, arguments in start.log:
                getattr(self.logger, method)(*arguments)
            start.log = None
            first = self.scout_iterations
        advance(start, first, iterations)

        intervention = start.intervention
        if track_location:
            start.locations.append(np.asarray(intervention.get_location()))
        final_validation = validate(intervention)
        if "intervention" not in start.best or final_validation["evaluation_objective"] < start.best["score"]:
            final_metrics = self.evaluate(intervention, start.batch, jax.random.fold_in(train_key, iterations))
            start.best = {
                "score": final_validation["evaluation_objective"],
                "intervention": intervention,
                "batch": start.batch,
                "step": iterations,
                "loss": float(final_metrics["total_loss"]),
                "metrics": final_metrics,
            }
        best_intervention = start.best["intervention"]
        best_batch = start.best["batch"]
        locations = start.locations

        evaluation_key = jax.random.fold_in(train_key, iterations + 1)
        perturbed = best_intervention(best_batch.initial_states)
        final_states, perturbed_trajectory = run_with_intervention(
            self.model,
            best_intervention,
            best_batch.initial_states,
            evaluation_steps,
            evaluation_key,
            boundary_callback=self.boundary_callback,
            return_trajectory=True,
            scan_kind=self.scan_kind,
            trajectory_every=self.evaluation_frame_every,
            trajectory_channels=self.observed_channels,
        )
        baseline_final_states, baseline_trajectory = run_nca_batch(
            self.model,
            best_batch.initial_states,
            evaluation_steps,
            evaluation_key,
            boundary_callback=self.boundary_callback,
            return_trajectory=True,
            scan_kind=self.scan_kind,
            trajectory_every=self.evaluation_frame_every,
            trajectory_channels=self.observed_channels,
        )
        # Target loss after the longer evaluation rollout, with and without the
        # intervention, to check whether a switch persists.
        evaluation_loss = self._weighted_loss(
            self._loss_components(final_states, best_batch.target_states, evaluation_key)
        )
        baseline_evaluation_loss = self._weighted_loss(
            self._loss_components(baseline_final_states, best_batch.target_states, evaluation_key)
        )
        evaluation_loss_curve = self._loss_curve(perturbed_trajectory, best_batch.target_states, evaluation_key)
        baseline_loss_curve = self._loss_curve(baseline_trajectory, best_batch.target_states, evaluation_key)
        return ImpulseResult(
            evaluation_switch_rate=self._switch_rate(final_states, best_batch),
            baseline_switch_rate=self._switch_rate(baseline_final_states, best_batch),
            best_intervention=best_intervention,
            best_step=start.best["step"],
            best_loss=start.best["loss"],
            best_validation_objective=start.best["score"],
            restart_validation_objectives=restart_objectives,
            metrics=start.best["metrics"],
            initial_states=best_batch.initial_states,
            target_states=best_batch.target_states,
            perturbed_initial_states=perturbed,
            final_states=final_states,
            baseline_trajectory=baseline_trajectory,
            perturbed_trajectory=perturbed_trajectory,
            evaluation_loss=float(jnp.mean(evaluation_loss)),
            baseline_evaluation_loss=float(jnp.mean(baseline_evaluation_loss)),
            location_history=np.stack(locations) if track_location else None,
            switch_steps=switch_steps(
                perturbed_trajectory,
                best_batch.initial_states,
                best_batch.target_states,
                self.evaluation_frame_every,
            ),
            evaluation_loss_curve=evaluation_loss_curve,
            baseline_loss_curve=baseline_loss_curve,
        )
