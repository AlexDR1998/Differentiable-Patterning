"""Permutation-invariant loss for small batches of snapshot measurements."""

from itertools import combinations, permutations

import jax
import jax.numpy as jnp
from einops import rearrange

from Common.trainer.loss import masked_channel_correlations, radial_profiles


def init_texture_params(key, spatial_shape, metric="l2", samples=128):
    """Initialise the existing LPIPS texture model once before training."""
    from Common.trainer import loss_vgg

    model = loss_vgg.lpips_variants[metric]
    sample = jnp.zeros((1, *spatial_shape, 3), dtype=loss_vgg.VGG_DTYPE)
    params = model.init(key, sample, sample, key, aux={"samples": samples})
    return loss_vgg.cast_params_bf16(params)


def _pairwise_l2(x, y, weights):
    weights = jnp.asarray(weights, x.dtype)
    shape = (1, 1, 1) + weights.shape
    error = (x[:, None] - y[None]) ** 2 * weights.reshape(shape)
    normalizer = jnp.broadcast_to(weights, x.shape[2:]).sum()
    return error.sum(axis=tuple(range(3, error.ndim))) / normalizer


def _channel_means(x, mask):
    pixels = x.reshape(*x.shape[:3], -1)
    spatial_mask = mask.reshape(-1).astype(x.dtype)
    count = jnp.maximum(spatial_mask.sum(), 1.0)
    return (pixels * spatial_mask).sum(-1) / count


def _texture_cost(x, y, params, metric, samples, size, key):
    from Common.trainer import loss_vgg

    size = min(size, x.shape[-2], x.shape[-1])
    if x.shape[-2:] != (size, size):
        shape = (*x.shape[:-2], size, size)
        x, y = jax.image.resize(x, shape, "linear"), jax.image.resize(y, shape, "linear")
    padding = (-x.shape[2]) % 3
    x = jnp.pad(x, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
    y = jnp.pad(y, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
    x = rearrange(x, "b n (p c) h w -> b n p h w c", c=3)
    y = rearrange(y, "b n (p c) h w -> b n p h w c", c=3)
    shape = (x.shape[0], y.shape[0], *x.shape[1:])
    x = jnp.broadcast_to(x[:, None], shape).reshape(-1, *x.shape[-3:])
    y = jnp.broadcast_to(y[None], shape).reshape(-1, *y.shape[-3:])
    loss = loss_vgg.lpips_variants[metric].apply(
        params, x, y, key, aux={"samples": samples}
    )
    return loss.reshape(shape[:4] + (-1,)).mean((3, 4))


def _assignment(cost, components, mode, tau):
    batch_count = cost.shape[0]
    orders = jnp.asarray(tuple(permutations(range(batch_count))))
    assigned = lambda values: jax.vmap(
        lambda order: values[jnp.arange(batch_count), order].mean(0)
    )(orders)
    values = assigned(cost)
    component_values = {name: assigned(value) for name, value in components.items()}
    if mode == "hard":
        selected = jnp.argmin(values, axis=0)
        take = lambda value: value[selected, jnp.arange(value.shape[1])]
        zeros = jnp.zeros(values.shape[1])
        return values.min(0), {name: take(value) for name, value in component_values.items()}, zeros, zeros
    if mode != "softmin" or tau <= 0:
        raise ValueError("assignment must be 'hard' or a positive-temperature 'softmin'")
    probabilities = jax.nn.softmax(-values / tau, axis=0)
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1e-12), axis=0)
    expected = {
        name: jnp.sum(probabilities * value, axis=0)
        for name, value in component_values.items()
    }
    loss = -tau * (jax.nn.logsumexp(-values / tau, axis=0) - jnp.log(values.shape[0]))
    return loss, expected, tau * (jnp.log(values.shape[0]) - entropy), entropy


def _weights(args):
    """Return multi-target component weights with defaults."""
    return {
        "texture": 1.0,
        "channel_mean": 1.0,
        "radial": 1.0,
        "correlation": 1.0,
        **args.get("multi_target_weights", {}),
    }


def multi_target_pairwise_costs(
    prediction, target, boundary, schema, params, key, args
):
    """Build prediction-row by target-column costs for each experiment group."""
    weights = _weights(args)
    metric = args.get("metric", "l2")
    samples = args.get("samples", 128)
    texture_size = args.get("texture_size", 128)
    radial_bins = args.get("radial_bins", 16)
    pair_weights = dict(zip(schema.co_measurement_pairs, schema.correlation_pair_weights))
    group_costs = []
    group_components = []

    for group_index, targets in enumerate(schema.group_measurement_indices):
        states = tuple(schema.target_to_state[channel] for channel in targets)
        x, y = prediction[:, :, states], target[:, :, targets]
        x, y = x * boundary, y * boundary
        local_pairs = tuple(combinations(range(len(targets)), 2))
        pair_array = jnp.asarray(local_pairs)
        x_mean, y_mean = _channel_means(x, boundary), _channel_means(y, boundary)
        x_radial, _ = radial_profiles(x, boundary, radial_bins)
        y_radial, _ = radial_profiles(y, boundary, radial_bins)
        x_corr, _ = masked_channel_correlations(x, boundary)
        y_corr, _ = masked_channel_correlations(y, boundary)
        x_corr = x_corr[..., pair_array[:, 0], pair_array[:, 1]]
        y_corr = y_corr[..., pair_array[:, 0], pair_array[:, 1]]
        channel_weights = jnp.asarray([schema.measurement_weights[c] for c in targets])
        correlation_weights = jnp.asarray([
            pair_weights[tuple(sorted((targets[left], targets[right])))]
            for left, right in local_pairs
        ])
        components = {
            "channel_mean": _pairwise_l2(x_mean, y_mean, channel_weights),
            "radial": _pairwise_l2(x_radial, y_radial, channel_weights[:, None]),
            "correlation": _pairwise_l2(x_corr, y_corr, correlation_weights),
        }
        if weights["texture"]:
            components["texture"] = _texture_cost(
                x, y, params, metric, samples, texture_size,
                jax.random.fold_in(key, group_index)
            )
        else:
            components["texture"] = jnp.zeros_like(components["channel_mean"])
        group_costs.append(
            sum(weights[name] * value for name, value in components.items())
        )
        group_components.append(components)

    return jnp.stack(group_costs), {
        name: jnp.stack([group[name] for group in group_components])
        for name in weights
    }


def multi_target_assignment(costs, components, schema, args):
    """Assign complete square pairwise matrices and return loss diagnostics."""
    weights = _weights(args)
    group_losses = []
    group_components = []
    assignment_regularisation = []
    assignment_entropy = []
    groups = zip(*(components[name] for name in weights))
    for cost, group in zip(costs, groups):
        group = dict(zip(weights, group))
        loss, assigned, regularisation, entropy = _assignment(
            cost,
            group,
            args.get("assignment", "hard"),
            args.get("assignment_tau", 0.05),
        )
        group_losses.append(loss)
        group_components.append(assigned)
        assignment_regularisation.append(regularisation)
        assignment_entropy.append(entropy)

    components = {
        name: jnp.stack([group[name] for group in group_components]).mean(0)
        for name in weights
    }
    diagnostics = {name: weights[name] * value for name, value in components.items()}
    diagnostics["assignment_regularisation"] = jnp.stack(assignment_regularisation).mean(0)
    diagnostics["assignment_entropy"] = jnp.stack(assignment_entropy).mean(0)
    for group_index, group_name in enumerate(schema.group_names):
        diagnostics[f"group/{group_name}/total"] = group_losses[group_index]
        for name in weights:
            diagnostics[f"group/{group_name}/{name}"] = (
                weights[name] * group_components[group_index][name]
            )
    return jnp.stack(group_losses).mean(0), jax.lax.stop_gradient(diagnostics)


def multi_target_loss(prediction, target, boundary, schema, params, key, args):
    """Match unordered batches independently for each time and experiment group."""
    costs, components = multi_target_pairwise_costs(
        prediction, target, boundary, schema, params, key, args
    )
    return multi_target_assignment(costs, components, schema, args)
