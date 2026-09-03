"""Permutation-invariant loss for small batches of snapshot measurements."""

from itertools import combinations, permutations, product

import jax
import jax.numpy as jnp
from einops import rearrange

from Common.trainer.loss import masked_channel_correlations, radial_profiles
from Common.trainer.loss_components import MULTI_TARGET_WEIGHT_DEFAULTS


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


def _pairwise_group_l2(x, y, weights, mask):
    """Weighted pixelwise L2 costs within one co-measured channel group."""
    weights = jnp.asarray(weights, x.dtype)
    channel_weights = weights.reshape((1, 1, 1, -1, 1, 1))
    spatial_mask = mask.astype(x.dtype).reshape((1, 1, 1, 1, *mask.shape))
    error = (x[:, None] - y[None]) ** 2 * channel_weights * spatial_mask
    normalizer = jnp.maximum(weights.sum() * spatial_mask.sum(), 1.0)
    return error.sum(axis=(-3, -2, -1)) / normalizer


def _channel_means(x, mask):
    pixels = x.reshape(*x.shape[:3], -1)
    spatial_mask = mask.reshape(-1).astype(x.dtype)
    count = jnp.maximum(spatial_mask.sum(), 1.0)
    return (pixels * spatial_mask).sum(-1) / count


def _texture_cost(
    x,
    y,
    params,
    metric,
    samples,
    key,
    random_channel_shuffle=False,
    random_crop=False,
):
    from Common.trainer import loss_vgg

    padding = (-x.shape[2]) % 3
    x = jnp.pad(x, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
    y = jnp.pad(y, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)))
    if random_channel_shuffle:
        # Keep prediction and target channels paired while changing their RGB
        # grouping for this perceptual comparison.  Permuting after padding
        # matches the VGG loss behaviour, including treatment of pad channels.
        permutation = jax.random.permutation(
            jax.random.fold_in(key, 0), x.shape[2]
        )
        x = jnp.take(x, permutation, axis=2)
        y = jnp.take(y, permutation, axis=2)
    x = rearrange(x, "b n (p c) h w -> b n p h w c", c=3)
    y = rearrange(y, "b n (p c) h w -> b n p h w c", c=3)
    shape = (x.shape[0], y.shape[0], *x.shape[1:])
    x = jnp.broadcast_to(x[:, None], shape).reshape(-1, *x.shape[-3:])
    y = jnp.broadcast_to(y[None], shape).reshape(-1, *y.shape[-3:])
    if random_crop:
        # Crop every prediction-target/RGB-triplet comparison with the same
        # offset for x and y, as required for a meaningful perceptual cost.
        crop_key = jax.random.fold_in(key, 1)
        x = loss_vgg._random_crop_to_vgg_input(x[:, None], crop_key)[:, 0]
        y = loss_vgg._random_crop_to_vgg_input(y[:, None], crop_key)[:, 0]
    loss = loss_vgg.lpips_variants[metric].apply(
        params, x, y, key, aux={"samples": samples}
    )
    return loss.reshape(shape[:4] + (-1,)).mean((3, 4))


def _condition_preserving_orders(batch_count, assignment_groups):
    if assignment_groups is None:
        return tuple(permutations(range(batch_count)))
    assignment_groups = tuple(assignment_groups)
    if len(assignment_groups) != batch_count:
        raise ValueError("Assignment groups must match the multi-target batch size")
    unique_groups = tuple(dict.fromkeys(assignment_groups))
    group_indices = tuple(
        tuple(index for index, value in enumerate(assignment_groups) if value == group)
        for group in unique_groups
    )
    orders = []
    for group_permutations in product(
        *(tuple(permutations(indices)) for indices in group_indices)
    ):
        order = list(range(batch_count))
        for indices, permuted in zip(group_indices, group_permutations):
            for index, target_index in zip(indices, permuted):
                order[index] = target_index
        orders.append(tuple(order))
    return tuple(orders)


def _assignment(
    cost, components, mode, tau, target_mask=None, assignment_groups=None
):
    batch_count = cost.shape[0]
    orders = jnp.asarray(
        _condition_preserving_orders(batch_count, assignment_groups)
    )
    if target_mask is None:
        target_mask = jnp.ones((batch_count, cost.shape[-1]), dtype=cost.dtype)
    target_mask = jnp.asarray(target_mask, dtype=cost.dtype)

    def assigned(values):
        def assign_order(order):
            weights = target_mask[order]
            selected = values[jnp.arange(batch_count), order]
            return (selected * weights).sum(0) / jnp.maximum(weights.sum(0), 1.0)

        return jax.vmap(assign_order)(orders)
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
        **MULTI_TARGET_WEIGHT_DEFAULTS,
        **args.get("multi_target_weights", {}),
    }


def multi_target_pairwise_costs(
    prediction, target, boundary, schema, params, key, args
):
    """Build prediction-row by target-column costs for each experiment group."""
    if prediction.shape[2] != schema.n_state_channels:
        raise ValueError(
            f"Schema {schema.name!r} expects {schema.n_state_channels} state "
            f"channels, got {prediction.shape[2]}"
        )
    schema.validate_measurement_channel_count(target.shape[2])
    weights = _weights(args)
    metric = args.get("metric", "l2")
    samples = args.get("samples", 128)
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
            "l2": _pairwise_group_l2(x, y, channel_weights, boundary),
            "channel_mean": _pairwise_l2(x_mean, y_mean, channel_weights),
            "radial": _pairwise_l2(x_radial, y_radial, channel_weights[:, None]),
            "correlation": _pairwise_l2(x_corr, y_corr, correlation_weights),
        }
        texture_enabled = (
            args["texture_enabled"]
            if "texture_enabled" in args
            else bool(weights["texture"])
        )
        if texture_enabled:
            components["texture"] = _texture_cost(
                x, y, params, metric, samples,
                jax.random.fold_in(key, group_index),
                random_channel_shuffle=args.get("random_channel_shuffle", False),
                random_crop=args.get("random_crop", False),
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


def multi_target_assignment(
    costs,
    components,
    schema,
    args,
    measurement_mask=None,
    assignment_groups=None,
):
    """Assign complete square pairwise matrices and return loss diagnostics."""
    weights = _weights(args)
    group_losses = []
    group_components = []
    assignment_regularisation = []
    assignment_entropy = []
    if measurement_mask is None:
        group_masks = [None] * len(schema.experiment_groups)
    else:
        measurement_mask = jnp.asarray(measurement_mask, dtype=bool)
        expected_shape = (costs.shape[2], costs.shape[3], schema.n_measurement_channels)
        if measurement_mask.shape != expected_shape:
            raise ValueError(
                "Multi-target measurement mask must have shape "
                f"[target_batch, time, measurement]={expected_shape}, got "
                f"{measurement_mask.shape}"
            )
        group_masks = [
            jnp.all(measurement_mask[:, :, channels], axis=-1)
            for channels in schema.group_measurement_indices
        ]
    groups = zip(*(components[name] for name in weights))
    for cost, group, group_mask in zip(costs, groups, group_masks):
        group = dict(zip(weights, group))
        loss, assigned, regularisation, entropy = _assignment(
            cost,
            group,
            args.get("assignment", "hard"),
            args.get("assignment_tau", 0.05),
            group_mask,
            assignment_groups,
        )
        group_losses.append(loss)
        group_components.append(assigned)
        assignment_regularisation.append(regularisation)
        assignment_entropy.append(entropy)

    group_valid = jnp.stack(
        [
            jnp.ones_like(group_losses[0], dtype=bool)
            if mask is None
            else jnp.any(mask, axis=0)
            for mask in group_masks
        ]
    )
    valid_count = jnp.maximum(group_valid.sum(0), 1)
    components = {
        name: (
            jnp.stack([group[name] for group in group_components]) * group_valid
        ).sum(0)
        / valid_count
        for name in weights
    }
    diagnostics = {name: weights[name] * value for name, value in components.items()}
    diagnostics.update({f"raw/{name}": value for name, value in components.items()})
    diagnostics["assignment_regularisation"] = (
        jnp.stack(assignment_regularisation) * group_valid
    ).sum(0) / valid_count
    diagnostics["assignment_entropy"] = (
        jnp.stack(assignment_entropy) * group_valid
    ).sum(0) / valid_count
    for group_index, group_name in enumerate(schema.group_names):
        diagnostics[f"group/{group_name}/total"] = group_losses[group_index]
        for name in weights:
            diagnostics[f"group/{group_name}/{name}"] = (
                weights[name] * group_components[group_index][name]
            )
    loss = (jnp.stack(group_losses) * group_valid).sum(0) / valid_count
    return loss, jax.lax.stop_gradient(diagnostics)


def multi_target_loss(
    prediction,
    target,
    boundary,
    schema,
    params,
    key,
    args,
    measurement_mask=None,
    assignment_groups=None,
):
    """Match unordered batches independently for each time and experiment group."""
    costs, components = multi_target_pairwise_costs(
        prediction, target, boundary, schema, params, key, args
    )
    return multi_target_assignment(
        costs,
        components,
        schema,
        args,
        measurement_mask=measurement_mask,
        assignment_groups=assignment_groups,
    )
