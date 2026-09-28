"""Permutation-invariant metrics for replicate-to-replicate variation."""

import jax.numpy as jnp


def _radial_profiles(values, spatial_mask, radial_bins, epsilon=1e-8):
    spatial_mask = jnp.asarray(spatial_mask).reshape(values.shape[-2:]).astype(bool)
    width, height = values.shape[-2:]
    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(width), jnp.arange(height), indexing="ij"
    )
    mask = spatial_mask.astype(values.dtype)
    centre_x = jnp.sum(grid_x * mask) / jnp.maximum(mask.sum(), 1.0)
    centre_y = jnp.sum(grid_y * mask) / jnp.maximum(mask.sum(), 1.0)
    radius = jnp.sqrt((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2)
    radius /= jnp.maximum(
        jnp.max(jnp.where(spatial_mask, radius, 0.0)), epsilon
    )
    indices = jnp.minimum(
        (radius * radial_bins).astype(jnp.int32), radial_bins - 1
    )
    annuli = (
        indices[None] == jnp.arange(radial_bins)[:, None, None]
    ) & spatial_mask
    counts = jnp.sum(annuli, axis=(-1, -2))
    profiles = jnp.einsum(
        "...chw,rhw->...cr", values, annuli.astype(values.dtype)
    )
    return profiles / jnp.maximum(counts, 1.0)


def _masked_channel_correlations(values, spatial_mask, epsilon=1e-8):
    mask = jnp.asarray(spatial_mask).reshape(-1).astype(values.dtype)
    pixels = values.reshape(*values.shape[:-2], -1)
    means = jnp.sum(pixels * mask, axis=-1) / jnp.maximum(mask.sum(), 1.0)
    centred = (pixels - means[..., None]) * mask
    covariance = jnp.einsum("...cp,...dp->...cd", centred, centred)
    squared_norms = jnp.sum(centred**2, axis=-1)
    denominator = jnp.sqrt(
        squared_norms[..., :, None] * squared_norms[..., None, :] + epsilon
    )
    return covariance / denominator


def _summary_features(values, boundary, radial_bins):
    """Return equally scaled mean, radial, and correlation feature blocks."""

    mask = jnp.asarray(boundary, dtype=values.dtype)
    pixel_count = jnp.maximum(mask.sum(), 1.0)
    means = (values * mask).sum(axis=(-1, -2)) / pixel_count
    radial = _radial_profiles(values, boundary, radial_bins)
    correlations = _masked_channel_correlations(values, boundary)
    pair_i, pair_j = jnp.triu_indices(values.shape[-3], 1)
    correlations = correlations[..., pair_i, pair_j]

    blocks = [
        means.reshape(values.shape[0], -1),
        radial.reshape(values.shape[0], -1),
    ]
    if correlations.shape[-1]:
        blocks.append(correlations.reshape(values.shape[0], -1))
    return jnp.concatenate(
        [block / jnp.sqrt(block.shape[-1]) for block in blocks], axis=-1
    )


def _pairwise_distances(left, right):
    differences = left[:, None] - right[None]
    return jnp.sqrt(jnp.maximum(jnp.sum(differences**2, axis=-1), 0.0))


def replicate_variation_metrics(prediction, target, boundary, radial_bins=16):
    """Compare two equally sized, unordered replicate feature populations.

    Inputs have shape ``[replicate, channel, x, y]``. Pairwise-distance
    correlation compares the sorted distance spectra, keeping the statistic
    invariant to replicate ordering when biological replicate identities are
    not paired across staining experiments.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Variation metrics require matching prediction and target shapes; "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.shape[0] < 2:
        raise ValueError("Variation metrics require at least two replicates")

    predicted_features = _summary_features(prediction, boundary, radial_bins)
    target_features = _summary_features(target, boundary, radial_bins)
    predicted_distances = _pairwise_distances(
        predicted_features, predicted_features
    )
    target_distances = _pairwise_distances(target_features, target_features)
    cross_distances = _pairwise_distances(predicted_features, target_features)

    pair_i, pair_j = jnp.triu_indices(prediction.shape[0], 1)
    predicted_pairs = jnp.sort(predicted_distances[pair_i, pair_j])
    target_pairs = jnp.sort(target_distances[pair_i, pair_j])
    epsilon = jnp.asarray(1e-8, dtype=prediction.dtype)
    dispersion_ratio = predicted_pairs.mean() / jnp.maximum(
        target_pairs.mean(), epsilon
    )

    predicted_centred = predicted_pairs - predicted_pairs.mean()
    target_centred = target_pairs - target_pairs.mean()
    denominator = jnp.sqrt(
        jnp.sum(predicted_centred**2) * jnp.sum(target_centred**2)
    )
    distance_correlation = jnp.where(
        denominator > epsilon,
        jnp.sum(predicted_centred * target_centred) / denominator,
        0.0,
    )
    energy_distance = jnp.maximum(
        2.0 * cross_distances.mean()
        - predicted_distances.mean()
        - target_distances.mean(),
        0.0,
    )
    return {
        "dispersion_ratio": dispersion_ratio,
        "pairwise_distance_correlation": distance_correlation,
        "energy_distance": energy_distance,
    }


def grouped_variation_metrics(
    prediction, target, boundary, schema, radial_bins=16
):
    """Compute replicate variation metrics independently by group and time."""

    results = {}
    for group_name, target_indices in zip(
        schema.group_names, schema.group_measurement_indices
    ):
        state_indices = tuple(schema.target_to_state[index] for index in target_indices)
        for time_index in range(prediction.shape[1]):
            metrics = replicate_variation_metrics(
                prediction[:, time_index, state_indices],
                target[:, time_index, target_indices],
                boundary,
                radial_bins,
            )
            for name, value in metrics.items():
                results[(group_name, time_index, name)] = value
    return results


__all__ = ["grouped_variation_metrics", "replicate_variation_metrics"]
