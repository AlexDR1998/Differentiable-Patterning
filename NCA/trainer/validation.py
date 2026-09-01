"""Side-effect-free held-out replicate evaluation during NCA training."""

from copy import copy

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from Common.model.boundary import hard_boundary, model_boundary
from Common.trainer.variation_metrics import grouped_variation_metrics
from NCA.trainer.step import _batch_loss


class ValidationEvaluator:
    """Compile a conditional rollout loss that never updates training state."""

    def __init__(self, trainer, setup, data, boundary_mask, loss_mask, key):
        if trainer.sharding not in (None, 1):
            raise ValueError(
                "Held-out validation currently requires trainer.sharding=null or 1"
            )
        data = jnp.asarray(data)
        self.trainer = copy(trainer)
        self.trainer.batch_count = data.shape[0]
        self.trainer.intervention_times = context_times = (
            trainer.context.validation_intervention_times
        )
        self.trainer.nodal_channel = (
            None
            if context_times is None
            or trainer.channel_schema is None
            or "NODAL" not in trainer.channel_schema.state_channels
            else trainer.channel_schema.state_channels.index("NODAL")
        )
        self.trainer.diagnostic_boundary_mask = boundary_mask
        callback_type = (
            model_boundary
            if trainer.config.training.trainer.boundary_mode == "soft"
            else hard_boundary
        )
        self.trainer.boundary_callbacks = [
            callback_type(boundary_mask[index]) for index in range(data.shape[0])
        ]
        self.trainer.loss_time_channel_mask = list(
            rearrange(jnp.asarray(loss_mask), "b n c -> b n c () ()")
        )
        self.trainer.loss_cache = [None] * data.shape[0]
        self.setup = setup
        self.key = key

        schema = trainer.channel_schema
        if schema is None:
            observed = data[:, :-1, : trainer.observed_channels]
        else:
            observed = data[:, :-1, schema.primary_measurements]
        padding = trainer.channels - observed.shape[2]
        states = jnp.pad(observed, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
        self.states = [trainer.model.prepare_pool_state(value) for value in states]
        self.targets = [value for value in data[:, 1:]]
        self.execution = self.trainer._training_execution()
        self._compiled = eqx.filter_jit(self._evaluate)

    def _loss_metrics(self, model, states, loss_weights, prefix):
        prediction = jnp.stack(states)[:, :, : self.trainer.observed_channels]
        target = jnp.stack(self.targets)[:, :, : self.trainer.data_channels]
        if self.setup.is_multi_target:
            arguments = {
                **self.setup.loss_arguments,
                "multi_target_weights": loss_weights.multi_target,
            }
            losses, components = self.execution.multi_target_loss(
                prediction,
                target,
                jnp.asarray(self.trainer.diagnostic_boundary_mask)[0, 0],
                self.trainer.channel_schema,
                self.setup.multi_target_params,
                self.key,
                arguments,
            )
            metrics = {f"{prefix}/loss": jnp.mean(losses)}
            for name, value in components.items():
                if name.startswith("raw/"):
                    metrics[f"{prefix}/loss_component_raw/{name[4:]}"] = jnp.mean(value)
                elif name.startswith("group/"):
                    metrics[f"{prefix}/loss_detail/{name[6:]}"] = jnp.mean(value)
                else:
                    metrics[f"{prefix}/loss_component/{name}"] = jnp.mean(value)
            metrics.update(self._variation_metrics(prediction, target, prefix))
            return metrics

        keys = jr.split(self.key, self.trainer.batch_count)
        losses = jnp.asarray([
            _batch_loss(
                self.trainer,
                self.setup,
                model,
                state,
                target,
                mask,
                cache,
                loss_key,
                loss_weights.terms,
            )
            for state, target, mask, cache, loss_key in zip(
                states,
                self.targets,
                self.trainer.loss_time_channel_mask,
                self.trainer.loss_cache,
                keys,
            )
        ])
        metrics = {f"{prefix}/loss": jnp.mean(losses)}
        metrics.update(self._variation_metrics(prediction, target, prefix))
        return metrics

    def _variation_metrics(self, prediction, target, prefix):
        if self.trainer.batch_count < 2 or self.trainer.channel_schema is None:
            return {}
        values = grouped_variation_metrics(
            prediction,
            target,
            jnp.asarray(self.trainer.diagnostic_boundary_mask)[0, 0],
            self.trainer.channel_schema,
            radial_bins=self.setup.loss_arguments.get("radial_bins", 16),
        )
        metrics = {}
        by_name = {}
        by_group_and_name = {}
        for (group_name, _time_index, name), value in values.items():
            by_group_and_name.setdefault((group_name, name), []).append(value)
            by_name.setdefault(name, []).append(value)
        for (group_name, name), metric_values in by_group_and_name.items():
            metrics[f"{prefix}/variation/{group_name}/{name}"] = jnp.mean(
                jnp.stack(metric_values)
            )
        for name, metric_values in by_name.items():
            metrics[f"{prefix}/variation/{name}"] = jnp.mean(
                jnp.stack(metric_values)
            )
        return metrics

    def _evaluate(self, model, loss_weights):
        batched_model = self.trainer._make_batched_nca(model)

        def no_regularisers(totals, before, after, context, key, skip=()):
            return totals

        _, states, _ = self.trainer._run_nca_steps(
            model,
            batched_model,
            self.states,
            {},
            self.setup.timesteps,
            self.key,
            self.trainer.config.training.trainer.loop_autodiff,
            no_regularisers,
            self.execution,
        )
        metrics = self._loss_metrics(model, states, loss_weights, "validation")
        if not self.trainer.config.training.trainer.validation_rollout:
            return metrics

        rollout_states = [state[:1] for state in self.states]
        snapshots = []
        for transition in range(self.targets[0].shape[0]):
            rollout_model = self.trainer._make_batched_nca(
                model, time_offset=transition
            )
            _, rollout_states, _ = self.trainer._run_nca_steps(
                model,
                rollout_model,
                rollout_states,
                {},
                self.setup.timesteps,
                jr.fold_in(self.key, transition + 1),
                self.trainer.config.training.trainer.loop_autodiff,
                no_regularisers,
                self.execution,
            )
            snapshots.append(rollout_states)
        rollout_predictions = [
            jnp.concatenate([snapshot[batch] for snapshot in snapshots], axis=0)
            for batch in range(self.trainer.batch_count)
        ]
        metrics.update(
            self._loss_metrics(
                model, rollout_predictions, loss_weights, "validation_rollout"
            )
        )
        return metrics

    def __call__(self, model, loss_weights):
        return self._compiled(model, loss_weights)


def build_validation_evaluator(trainer, setup):
    context = trainer.context
    if context.validation_data is None:
        return None
    return ValidationEvaluator(
        trainer,
        setup,
        context.validation_data,
        context.validation_boundary_mask,
        context.validation_loss_time_channel_mask,
        jr.fold_in(setup.key, 0x56414C),
    )


__all__ = ["ValidationEvaluator", "build_validation_evaluator"]
