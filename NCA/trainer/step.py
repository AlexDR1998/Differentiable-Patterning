"""The compiled training step: roll the NCA out, compute the loss, update the model.

Everything here is traced by JAX. Python-side decisions (pool admission,
logging, checkpoints) live in ``runner.py``.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from einops import einsum, rearrange, repeat

from Common.trainer.loss_multi_target import multi_target_loss
from NCA.trainer.interval_schedule import IntervalSchedule, uniform_schedule
from NCA.trainer.intervention import (
    apply_model_with_blocked_channel,
    nodal_read_block_mask,
)
from NCA.trainer.objective import combine_loss_components


LOSS_DTYPE = jnp.float32


class TrainState(NamedTuple):
    """Everything that changes from one training iteration to the next."""

    model: Any
    states: Any
    targets: Any
    optimizer_state: Any
    key: Any
    loss_weights: Any


class StepOutput(NamedTuple):
    state: TrainState
    loss: Any
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def batch_model(
    model,
    intervention_times=None,
    nodal_channel=None,
    observation_times=None,
    time_offset=0,
):
    """Apply ``model`` to a list of batches, each of shape [N, C, H, W].

    With ``intervention_times`` (one per batch), the NODAL channel is blocked
    from the knockout time onward. ``observation_times`` maps knockout hours to
    time slots for non-uniform schedules (``None`` keeps 12-hour slots).
    """
    if intervention_times is None:
        apply_with_boundary = jax.vmap(
            model, in_axes=(0, None, 0), out_axes=0, axis_name="N"
        )
        return lambda x, callbacks, key_array: jtu.tree_map(
            apply_with_boundary, x, callbacks, key_array
        )

    if nodal_channel is None:
        raise ValueError("NODAL intervention requires a NODAL state channel")
    intervention_times = tuple(intervention_times)

    def apply_interventions(x, callbacks, key_array):
        if len(x) != len(intervention_times):
            raise ValueError("Intervention times must match the outer training batch")
        outputs = []
        for states, callback, keys, knockout_time in zip(
            x, callbacks, key_array, intervention_times
        ):
            blocked = nodal_read_block_mask(
                knockout_time,
                states.shape[0],
                time_offset=time_offset,
                observation_times=observation_times,
            )
            outputs.append(jax.vmap(
                lambda state, item_key, is_blocked: apply_model_with_blocked_channel(
                    model, state, callback, item_key, nodal_channel, is_blocked,
                ),
                in_axes=(0, 0, 0),
                out_axes=0,
                axis_name="N",
            )(states, keys, blocked))
        return type(x)(outputs)

    return apply_interventions


def run_nca_steps(
    model,
    batched_model,
    states,
    regulariser_totals,
    schedule,
    key,
    loop_autodiff,
    apply_regularisers,
    boundary_callbacks,
):
    """Run the NCA forward with ``eqx.internal.scan``; return (key, states, totals).

    ``schedule`` is either an int (every time slot runs that many steps) or an
    :class:`IntervalSchedule`. For a non-uniform schedule the scan runs to the
    longest slot and each slot keeps its state once it has taken its own step
    count, so slot ``i`` ends exactly where an unmasked ``steps[i]``-step scan
    would. Regularisers are still evaluated on every scan iteration; finished
    slots contribute their held state, and (non-uniform only)
    ``context["active_slots"]`` marks which slots actually updated.
    """
    schedule = schedule if isinstance(schedule, IntervalSchedule) else uniform_schedule(schedule, 1)
    batch_count = len(states)
    slot_count = states[0].shape[0]
    if schedule.is_uniform:
        active_steps = None
    elif schedule.n_slots != slot_count:
        raise ValueError(
            f"Interval schedule has {schedule.n_slots} slots but the rollout has {slot_count}"
        )
    else:
        active_steps = jnp.asarray(schedule.steps, dtype=jnp.int32)

    def hold_finished(new, old, active):
        active = active.reshape(active.shape + (1,) * (new.ndim - 1))
        return jnp.where(active, new, old)

    def nca_step(carry, j):
        step_key, state, totals = carry
        step_key = jr.fold_in(step_key, j)
        key_array = list(jr.randint(
            step_key,
            shape=(batch_count, slot_count, 2),
            minval=0,
            maxval=2_147_483_647,
            dtype=jnp.uint32,
        ))
        new_state = batched_model(state, boundary_callbacks, key_array)
        context = {
            "model": batched_model,
            "boundary_state_selector": model.boundary_regulariser_state,
        }
        if active_steps is not None:
            active = j < active_steps
            new_state = jtu.tree_map(
                lambda new, old: hold_finished(new, old, active), new_state, state
            )
            context["active_slots"] = active
        totals = apply_regularisers(totals, state, new_state, context, step_key)
        return (step_key, new_state, totals), None

    carry, _ = eqx.internal.scan(
        nca_step,
        (key, states, regulariser_totals),
        xs=jnp.arange(schedule.scan_length),
        kind=loop_autodiff,
    )
    return carry


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _gradient_features(model, values):
    perception = jax.vmap(model.perception, in_axes=0, out_axes=0)
    channel_count = values.shape[1]
    features = perception(values)
    return features.at[:, channel_count:].set(
        0.1 * features[:, channel_count:]
    )


def _channel_mask(trainer, setup, component, time_mask):
    selected = (setup.loss_channels == component) | (setup.loss_channels == -1)
    mask_channels = time_mask.shape[1]
    if mask_channels == selected.shape[0]:
        pass
    elif (
        trainer.channel_schema is not None
        and mask_channels == trainer.channel_schema.n_measurement_channels
        and selected.shape[0] == trainer.channel_schema.n_state_channels
    ):
        selected = selected[jnp.asarray(trainer.channel_schema.target_to_state)]
    elif mask_channels % selected.shape[0] == 0:
        selected = repeat(
            selected, "c -> (groups c)", groups=mask_channels // selected.shape[0]
        )
    else:
        raise ValueError(
            "Loss-time mask has an incompatible channel layout: "
            f"{mask_channels} mask channels for {selected.shape[0]} state channels"
        )
    selected = rearrange(selected.astype(jnp.float32), "c -> c () ()")
    return einsum(
        time_mask, selected, "n c w h, c w h -> n c w h"
    ).astype(jnp.bool_)


def batch_loss(
    trainer, setup, model, states, targets, time_mask, cache, key, component_weights
):
    """Weighted sum of the configured loss terms for one batch."""
    predicted = states[:, : trainer.observed_channels]
    expected = targets[:, : trainer.data_channels]
    if trainer.grad_loss:
        predicted = _gradient_features(model, predicted)
        expected = _gradient_features(model, expected)
    losses = []
    for index, loss_function in enumerate(setup.loss_functions):
        component_key = jr.fold_in(key, index)
        losses.append(
            loss_function(
                predicted,
                expected,
                component_key,
                _channel_mask(trainer, setup, index, time_mask),
                cache,
            )
        )
    return combine_loss_components(losses, component_weights)


def multi_target_losses(trainer, setup, states, targets, boundary, measurement_masks,
                        intervention_times, key, loss_weights):
    """Multi-target loss per batch and its components, for all batches at once."""
    loss_arguments = {
        **setup.loss_arguments,
        "multi_target_weights": loss_weights.multi_target,
    }
    return multi_target_loss(
        jnp.stack(states)[:, :, : trainer.observed_channels],
        jnp.stack(targets)[:, :, : trainer.data_channels],
        boundary,
        trainer.channel_schema,
        setup.multi_target_params,
        key,
        loss_arguments,
        measurement_mask=jnp.stack(measurement_masks)[..., 0, 0],
        assignment_groups=intervention_times,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def build_train_step(trainer, setup):
    """Build the jitted function that maps one TrainState to a StepOutput."""

    def apply_regularisers(totals, before, after, context, key):
        aux = {
            "boundary_callbacks": trainer.boundary_callbacks,
            "observed_channels": trainer.observed_channels,
            **context,
        }
        for name, function in setup.regulariser_functions.items():
            totals[name] += function(before, after, aux, key)
        return totals

    def objective(differentiable, static, states, targets, key, loss_weights):
        model = eqx.combine(differentiable, static)
        batched = batch_model(
            model,
            trainer.intervention_times,
            trainer.nodal_channel,
            setup.intervention_observation_times,
        )
        regulariser_totals = {
            name: jnp.zeros(len(states), dtype=LOSS_DTYPE)
            for name in setup.regulariser_coefficients
        }
        key, states, regulariser_totals = run_nca_steps(
            model,
            batched,
            states,
            regulariser_totals,
            setup.interval_schedule,
            key,
            trainer.trainer_config.loop_autodiff,
            apply_regularisers,
            trainer.boundary_callbacks,
        )
        diagnostics = {}
        if setup.is_multi_target:
            losses, components = multi_target_losses(
                trainer,
                setup,
                states,
                targets,
                jnp.asarray(trainer.diagnostic_boundary_mask)[0, 0],
                trainer.loss_time_channel_mask,
                trainer.intervention_times,
                key,
                loss_weights,
            )
            for name, value in components.items():
                if name.startswith("raw/"):
                    diagnostics[
                        f"loss_component_raw/{name.removeprefix('raw/')}"
                    ] = jnp.mean(value)
                elif name.startswith("group/"):
                    diagnostics[f"loss_detail/{name.removeprefix('group/')}"] = value
                else:
                    diagnostics[f"loss_component/{name}"] = jnp.mean(value)
        else:
            losses = jnp.asarray(
                jtu.tree_map(
                    lambda state, target, mask, cache, loss_key: batch_loss(
                        trainer,
                        setup,
                        model,
                        state,
                        target,
                        mask,
                        cache,
                        loss_key,
                        loss_weights.terms,
                    ),
                    states,
                    targets,
                    trainer.loss_time_channel_mask,
                    setup.loss_cache,
                    list(jr.split(key, trainer.batch_count)),
                )
            )
        # Regularisers accumulate once per scan iteration, so normalise by the
        # scan length (``t`` for uniform schedules, the longest slot otherwise).
        regulariser_losses = {
            name: coefficient * jnp.mean(regulariser_totals[name]) / setup.timesteps
            for name, coefficient in setup.regulariser_coefficients.items()
        }
        regulariser_total = (
            jnp.sum(jnp.stack(tuple(regulariser_losses.values())))
            if regulariser_losses
            else jnp.array(0.0, dtype=LOSS_DTYPE)
        )
        mean_loss = jnp.mean(losses) + regulariser_total
        return mean_loss, (states, losses, regulariser_losses, diagnostics)

    def train_step(state: TrainState):
        differentiable, static = state.model.partition()
        (loss, auxiliary), gradients = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(
            differentiable,
            static,
            state.states,
            state.targets,
            state.key,
            state.loss_weights,
        )
        states, losses, regulariser_losses, diagnostics = auxiliary
        updates, optimizer_state = setup.optimiser.update(
            gradients, state.optimizer_state, differentiable
        )
        model = eqx.apply_updates(state.model, updates)
        metrics = {
            "loss": loss,
            "states": states,
            "losses": losses,
            **regulariser_losses,
            **diagnostics,
        }
        metrics.update(
            {
                f"loss_weight/term_{index}_{name}": state.loss_weights.terms[index]
                for index, name in enumerate(setup.loss_names)
            }
        )
        metrics.update(
            {
                f"loss_weight/{name}": value
                for name, value in state.loss_weights.multi_target.items()
            }
        )
        return StepOutput(
            TrainState(
                model,
                states,
                state.targets,
                optimizer_state,
                state.key,
                state.loss_weights,
            ),
            loss,
            metrics,
        )

    return eqx.filter_jit(train_step)
