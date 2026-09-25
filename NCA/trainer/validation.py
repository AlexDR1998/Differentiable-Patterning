"""Held-out replicate evaluation during NCA training; never changes training state."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from Common.model.boundary import hard_boundary, model_boundary
from Common.trainer.variation_metrics import grouped_variation_metrics
from NCA.trainer.step import batch_loss, batch_model, multi_target_losses, run_nca_steps


class ValidationEvaluator:
    """Compiled rollout and loss on the validation replicates.

    It uses the trainer only for settings shared with training (channel
    counts, schema, loss options); the validation batches have their own
    boundary callbacks, loss masks and knockout times.
    """

    def __init__(self, trainer, setup, data, boundary_mask, loss_mask, key):
        if trainer.sharding not in (None, 1):
            raise ValueError(
                "Held-out validation currently requires trainer.sharding=null or 1"
            )
        data = jnp.asarray(data)
        self.trainer = trainer
        self.setup = setup
        self.key = key
        self.batch_count = data.shape[0]
        self.boundary_mask = boundary_mask
        callback_type = (
            model_boundary
            if trainer.trainer_config.boundary_mode == "soft"
            else hard_boundary
        )
        self.boundary_callbacks = [
            callback_type(boundary_mask[index]) for index in range(self.batch_count)
        ]
        self.loss_masks = list(rearrange(jnp.asarray(loss_mask), "b n c -> b n c () ()"))
        schema = trainer.channel_schema
        self.intervention_times = trainer.context.validation_intervention_times
        self.nodal_channel = (
            None
            if self.intervention_times is None
            or schema is None
            or "NODAL" not in schema.state_channels
            else schema.state_channels.index("NODAL")
        )

        if schema is None:
            observed = data[:, :-1, : trainer.observed_channels]
        else:
            observed = data[:, :-1, schema.primary_measurements]
        padding = trainer.channels - observed.shape[2]
        states = jnp.pad(observed, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
        self.states = [trainer.model.prepare_pool_state(value) for value in states]
        self.targets = [value for value in data[:, 1:]]
        self._compiled = eqx.filter_jit(self._evaluate)

    def _loss_metrics(self, model, states, loss_weights, prefix):
        trainer = self.trainer
        prediction = jnp.stack(states)[:, :, : trainer.observed_channels]
        target = jnp.stack(self.targets)[:, :, : trainer.data_channels]
        if self.setup.is_multi_target:
            losses, components = multi_target_losses(
                trainer,
                self.setup,
                states,
                self.targets,
                jnp.asarray(self.boundary_mask)[0, 0],
                self.loss_masks,
                self.intervention_times,
                self.key,
                loss_weights,
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

        keys = jr.split(self.key, self.batch_count)
        losses = jnp.asarray([
            batch_loss(
                trainer,
                self.setup,
                model,
                state,
                target_batch,
                mask,
                None,  # target features are only cached for the training data
                loss_key,
                loss_weights.terms,
            )
            for state, target_batch, mask, loss_key in zip(
                states, self.targets, self.loss_masks, keys
            )
        ])
        metrics = {f"{prefix}/loss": jnp.mean(losses)}
        metrics.update(self._variation_metrics(prediction, target, prefix))
        return metrics

    def _variation_metrics(self, prediction, target, prefix):
        if self.batch_count < 2 or self.trainer.channel_schema is None:
            return {}
        values = grouped_variation_metrics(
            prediction,
            target,
            jnp.asarray(self.boundary_mask)[0, 0],
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

    def _rollout(self, model, states, schedule, key, time_offset=0):
        batched = batch_model(
            model,
            self.intervention_times,
            self.nodal_channel,
            self.setup.intervention_observation_times,
            time_offset=time_offset,
        )
        _, states, _ = run_nca_steps(
            model,
            batched,
            states,
            {},
            schedule,
            key,
            self.trainer.trainer_config.loop_autodiff,
            lambda totals, before, after, context, step_key: totals,
            self.boundary_callbacks,
        )
        return states

    def _evaluate(self, model, loss_weights):
        schedule = self.setup.interval_schedule
        if not schedule.is_uniform and schedule.n_slots != self.targets[0].shape[0]:
            raise ValueError(
                f"Validation data has {self.targets[0].shape[0]} transitions but the "
                f"interval schedule has {schedule.n_slots}"
            )
        states = self._rollout(model, self.states, schedule, self.key)
        metrics = self._loss_metrics(model, states, loss_weights, "validation")
        if not self.trainer.trainer_config.validation_rollout:
            return metrics

        # Sequential rollout from the first image: each transition starts where
        # the previous one ended and runs its own step count.
        rollout_states = [state[:1] for state in self.states]
        snapshots = []
        for transition in range(self.targets[0].shape[0]):
            rollout_states = self._rollout(
                model,
                rollout_states,
                schedule.for_slot(transition) if not schedule.is_uniform else schedule,
                jr.fold_in(self.key, transition + 1),
                time_offset=transition,
            )
            snapshots.append(rollout_states)
        rollout_predictions = [
            jnp.concatenate([snapshot[batch] for snapshot in snapshots], axis=0)
            for batch in range(self.batch_count)
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
