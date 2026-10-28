from dataclasses import dataclass

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
        # The intervention must end before the loss window starts
        last_step = max(self.rollout_steps - self.loss_window + 1, 1)
        if self.intervention.duration > last_step:
            raise ValueError(
                f"The intervention lasts {self.intervention.duration} steps but must end "
                f"before the loss window: at most rollout_steps - loss_window + 1 = {last_step}"
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

    def evaluate(self, intervention, batch, key):
        """Evaluate an intervention without updating its parameters."""

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
        if self.loss_window == 1:
            final_states = run_nca_batch(
                self.model, perturbed, self.rollout_steps, key, forcing=forcing, **rollout
            )
            components = self._loss_components(final_states, batch.target_states, key)
        else:
            # Run up to the loss window without keeping states, then keep only
            # the window (and only the observed channels, when the loss uses no others).
            states = run_nca_batch(
                self.model,
                perturbed,
                self.rollout_steps - self.loss_window,
                key,
                forcing=forcing,
                **rollout,
            )
            final_states, window = run_nca_batch(
                self.model,
                states,
                self.loss_window,
                jax.random.fold_in(key, 1),
                return_trajectory=True,
                trajectory_channels=(
                    self.observed_channels if self.loss_channels == "observed" else None
                ),
                **rollout,
            )
            # [batch, window, ...] -> [batch * window, ...], each target repeated window times
            predictions = window.reshape((-1, *window.shape[2:]))
            targets = jnp.repeat(batch.target_states, self.loss_window, axis=0)
            components = self._loss_components(predictions, targets, key)
            components = components.reshape((len(components), len(perturbed), self.loss_window))
            components = jnp.mean(components, axis=-1)
        per_sample_loss = self._weighted_loss(components)
        target_loss = jnp.mean(per_sample_loss)
        metrics = intervention_metrics(batch.initial_states, deltas)
        penalty = weighted_regulariser(metrics, self.regulariser_coefficients)
        total_loss = self.objective(target_loss, penalty, metrics)
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

    def _make_step(self):
        """Build the compiled intervention update function."""

        @eqx.filter_jit
        def step(intervention, opt_state, initial_states, target_states, key):
            batch = ImpulseBatch(initial_states, target_states)

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_and_aux(candidate):
                values = self.evaluate(candidate, batch, key)
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

        It returns the switch rate and target loss after a ``steps`` rollout,
        with the same batch and key every time, so changes come from the
        intervention only. With ``attractors``, ``outcomes`` is the fraction
        of rollouts ending at each attractor, then other, died and diverged.
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
            loss = self._weighted_loss(self._loss_components(final_states, batch.target_states, key))
            return final_states, jnp.mean(loss)

        def validate(intervention):
            final_states, loss = rollout(intervention)
            values = {"switch_rate": self._switch_rate(final_states, batch), "evaluation_loss": float(loss)}
            if self.attractors is not None:
                values["outcomes"] = outcome_fractions(
                    final_states[:, : self.observed_channels], self.attractors, self.attractor_tolerance
                )
            return values

        return validate

    def train(self, iterations, batch_size, key, evaluation_steps=None, log_every=100):
        """Optimise an intervention and return its best evaluated state.

        Best means the lowest configured scalar objective. Pair batches are
        regenerated at ``resample_every`` when that value is positive.
        """

        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        sample_key, train_key = jax.random.split(key)
        batch = self.pair_source.sample(batch_size, self.model, sample_key)
        parameters, _ = eqx.partition(self.intervention, eqx.is_inexact_array)
        opt_state = self.optimiser.init(parameters)
        step = self._make_step()
        evaluation_steps = self.rollout_steps if evaluation_steps is None else evaluation_steps
        if evaluation_steps % self.evaluation_frame_every:
            raise ValueError(
                f"evaluation_steps ({evaluation_steps}) must be a multiple of "
                f"evaluation_frame_every ({self.evaluation_frame_every})"
            )
        validate = None
        if self.logger is not None:
            # A separate fixed batch, scored with the evaluation rollout at each log step
            validation_batch = self.pair_source.sample(batch_size, self.model, jax.random.fold_in(key, 1))
            validate = self._make_validation(validation_batch, evaluation_steps, jax.random.fold_in(key, 2))

        intervention = self.intervention
        best_intervention = intervention
        best_batch = batch
        best_step = -1
        best_loss = float("inf")
        best_metrics = None
        # Only local and patch interventions have a centre that moves
        track_location = intervention.mode["spatial"] in ("local", "patch")
        locations = []

        for index in range(iterations):
            step_key = jax.random.fold_in(train_key, index)
            if self.resample_every and index > 0 and index % self.resample_every == 0:
                batch = self.pair_source.sample(batch_size, self.model, step_key)
            evaluated_intervention = intervention
            if track_location:
                locations.append(np.asarray(intervention.get_location()))
            intervention, opt_state, metrics = step(
                intervention,
                opt_state,
                batch.initial_states,
                batch.target_states,
                step_key,
            )
            scalar_loss = float(metrics["total_loss"])
            if scalar_loss < best_loss:
                best_loss = scalar_loss
                best_intervention = evaluated_intervention
                best_batch = batch
                best_step = index
                best_metrics = metrics
            if self.logger is not None and index % log_every == 0:
                switch_rate = self._switch_rate(metrics["final_states"], batch)
                self.logger.log_training({**metrics, "switch_rate": switch_rate}, index, log_every)
                self.logger.log_validation(validate(evaluated_intervention), index)

        if track_location:
            locations.append(np.asarray(intervention.get_location()))
        final_key = jax.random.fold_in(train_key, iterations)
        final_metrics = self.evaluate(intervention, batch, final_key)
        final_loss = float(final_metrics["total_loss"])
        if final_loss < best_loss:
            best_loss = final_loss
            best_intervention = intervention
            best_batch = batch
            best_step = iterations
            best_metrics = final_metrics

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
            best_step=best_step,
            best_loss=best_loss,
            metrics=best_metrics,
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
