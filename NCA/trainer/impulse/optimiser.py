from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from Common.trainer.loss import build_loss_functions
from NCA.trainer.impulse.regularisers import intervention_metrics, weighted_regulariser
from NCA.trainer.impulse.rollout import identity_boundary, run_nca_batch
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
    loss_functions: object = None
    loss_names: object = ("l2",)
    loss_args: dict | None = None
    component_weights: object = None
    loss_channels: object = "observed"
    regulariser_coefficients: dict | None = None
    boundary_callback: object = identity_boundary
    scan_kind: str = "lax"
    resample_every: int = 0
    logger: object = None

    def __post_init__(self):
        """Validate configuration and construct named loss functions."""

        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if self.resample_every < 0:
            raise ValueError("resample_every must be non-negative")
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

    def evaluate(self, intervention, batch, key):
        """Evaluate an intervention without updating its parameters."""

        perturbed = intervention(batch.initial_states)
        final_states = run_nca_batch(
            self.model,
            perturbed,
            self.rollout_steps,
            key,
            boundary_callback=self.boundary_callback,
            scan_kind=self.scan_kind,
        )
        components = self._loss_components(final_states, batch.target_states, key)
        weights = self.component_weights[:, None]
        per_sample_loss = jnp.sum(weights * components, axis=0) / jnp.sum(weights)
        target_loss = jnp.mean(per_sample_loss)
        metrics = intervention_metrics(batch.initial_states, perturbed)
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
            parameters, static = eqx.partition(intervention, eqx.is_inexact_array)
            updates, opt_state = self.optimiser.update(gradients, opt_state, parameters)
            parameters = eqx.apply_updates(parameters, updates)
            intervention = eqx.combine(parameters, static)
            return intervention, opt_state, values

        return step

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

        intervention = self.intervention
        best_intervention = intervention
        best_batch = batch
        best_step = -1
        best_loss = float("inf")
        best_metrics = None

        for index in range(iterations):
            step_key = jax.random.fold_in(train_key, index)
            if self.resample_every and index > 0 and index % self.resample_every == 0:
                batch = self.pair_source.sample(batch_size, self.model, step_key)
            evaluated_intervention = intervention
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
                self.logger.log_training(metrics, index, log_every)

        final_key = jax.random.fold_in(train_key, iterations)
        final_metrics = self.evaluate(intervention, batch, final_key)
        final_loss = float(final_metrics["total_loss"])
        if final_loss < best_loss:
            best_loss = final_loss
            best_intervention = intervention
            best_batch = batch
            best_step = iterations
            best_metrics = final_metrics

        evaluation_steps = self.rollout_steps if evaluation_steps is None else evaluation_steps
        evaluation_key = jax.random.fold_in(train_key, iterations + 1)
        perturbed = best_intervention(best_batch.initial_states)
        final_states, perturbed_trajectory = run_nca_batch(
            self.model,
            perturbed,
            evaluation_steps,
            evaluation_key,
            boundary_callback=self.boundary_callback,
            return_trajectory=True,
            scan_kind=self.scan_kind,
        )
        _, baseline_trajectory = run_nca_batch(
            self.model,
            best_batch.initial_states,
            evaluation_steps,
            evaluation_key,
            boundary_callback=self.boundary_callback,
            return_trajectory=True,
            scan_kind=self.scan_kind,
        )
        return ImpulseResult(
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
        )
