"""Pure differentiated NCA training step."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from einops import einsum, rearrange, repeat

from NCA.trainer.objective import combine_loss_components
from NCA.trainer.state import StepOutput, TrainState


LOSS_DTYPE = jnp.float32


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


def _batch_loss(
    trainer, setup, model, states, targets, time_mask, cache, key, component_weights
):
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


def build_train_step(trainer, setup):
    """Build one explicit state-to-output function for JIT compilation."""

    execution = setup.execution

    def apply_regularisers(totals, before, after, context, key, skip=()):
        aux = {
            "boundary_callbacks": execution.boundary_callbacks(),
            "observed_channels": trainer.observed_channels,
            **context,
        }
        for name, function in setup.regulariser_functions.items():
            if name not in skip:
                totals[name] += function(before, after, aux, key)
        return totals

    def objective(differentiable, static, states, targets, key, loss_weights):
        model = eqx.combine(differentiable, static)
        batched_model = trainer._make_batched_nca(model)
        regulariser_totals = {
            name: jnp.zeros(len(states), dtype=LOSS_DTYPE)
            for name in setup.regulariser_coefficients
        }
        key, states, regulariser_totals = trainer._run_nca_steps(
            model,
            batched_model,
            states,
            regulariser_totals,
            setup.timesteps,
            key,
            trainer.config.training.trainer.loop_autodiff,
            apply_regularisers,
            execution,
        )
        diagnostics = {}
        if setup.is_multi_target:
            boundary = jnp.asarray(trainer.diagnostic_boundary_mask)[0, 0]
            loss_arguments = {
                **setup.loss_arguments,
                "multi_target_weights": loss_weights.multi_target,
            }
            losses, components = execution.multi_target_loss(
                jnp.stack(states)[:, :, : trainer.observed_channels],
                jnp.stack(targets)[:, :, : trainer.data_channels],
                boundary,
                trainer.channel_schema,
                setup.multi_target_params,
                key,
                loss_arguments,
            )
            diagnostics = {}
            for name, value in components.items():
                if name.startswith("raw/"):
                    diagnostics[
                        f"loss_component_raw/{name.removeprefix('raw/')}"
                    ] = jnp.mean(value)
                elif not name.startswith("group/"):
                    diagnostics[f"loss_component/{name}"] = jnp.mean(value)
            diagnostics.update(
                {
                    f"loss_detail/{name.removeprefix('group/')}": value
                    for name, value in components.items()
                    if name.startswith("group/")
                }
            )
        else:
            losses = jnp.asarray(
                jtu.tree_map(
                    lambda state, target, mask, cache, loss_key: _batch_loss(
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
                    execution.loss_time_channel_mask(),
                    execution.loss_cache(),
                    list(jr.split(key, trainer.batch_count)),
                )
            )
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
        mean_loss, regulariser_losses = execution.synchronise_loss(
            mean_loss, regulariser_losses
        )
        return mean_loss, (states, losses, regulariser_losses, diagnostics)

    def train_step(state: TrainState):
        differentiable, static = state.model.partition()
        transformed_objective = execution.transform_loss(objective)
        (loss, auxiliary), gradients = eqx.filter_value_and_grad(
            transformed_objective, has_aux=True
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

    return execution.transform_step(train_step)
