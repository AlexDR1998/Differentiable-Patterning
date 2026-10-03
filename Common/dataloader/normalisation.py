"""Linear per-channel intensity rescaling with clipping.

Each channel is mapped to [0, 1] by ``(x - low) / (high - low)`` and clipped,
where ``low`` and ``high`` are percentiles of that channel's pixels. The
percentiles are taken over all timesteps of a trajectory together, so every
timestep shares one linear scale and changes over time are kept.
"""

import numpy as np

# How the clipping bounds of one channel are shared between trajectories
# (one trajectory = one condition and replicate, over all its timesteps).
NORMALISATION_MODES = {
    "replicate_mean": "per channel, bounds averaged over replicates",
    "per_replicate": "per channel and replicate",
    "pooled": "per channel, percentiles of all pixels pooled (as the loader)",
}


def percentile_bins(
    samples, low_percentile, high_percentile, mode="replicate_mean", reference=None
):
    """Return clipping bounds ``{trajectory: (low, high)}`` for one channel.

    ``samples`` maps each trajectory (any hashable key) to a 1D array of the
    channel's pixel values over all of that trajectory's timesteps.

    - ``"per_replicate"``: each trajectory uses the percentiles of its own pixels.
    - ``"replicate_mean"``: percentiles are found per trajectory, then averaged,
      and every trajectory uses the average.
    - ``"pooled"``: every trajectory uses the percentiles of all pixels
      together. Pixels shared by several trajectories count once per trajectory.

    ``reference`` optionally maps a trajectory to another trajectory whose
    pixels set its bounds, for example each knockout replicate to the matching
    control replicate. A knockout then keeps any change in overall intensity
    relative to control, instead of having it normalised away. With
    ``"per_replicate"`` a trajectory takes the bounds of its reference. In the
    shared modes only reference trajectories enter the average or the pool.
    Trajectories not listed in ``reference`` are their own reference.
    """
    if mode not in NORMALISATION_MODES:
        raise ValueError(
            f"Unknown normalisation mode {mode!r}; choose from {tuple(NORMALISATION_MODES)}"
        )
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must increase within [0, 100]")
    samples = {key: np.asarray(values).reshape(-1) for key, values in samples.items()}
    samples = {key: values for key, values in samples.items() if values.size}
    if not samples:
        raise ValueError("No pixel values were given")
    reference = {key: (reference or {}).get(key, key) for key in samples}
    missing = sorted({str(source) for source in reference.values() if source not in samples})
    if missing:
        raise ValueError("No pixel values for reference trajectories: " + ", ".join(missing))
    # Reference trajectories in their order in ``samples``.
    sources = [key for key in samples if key in set(reference.values())]
    percentiles = (low_percentile, high_percentile)

    if mode == "pooled":
        bounds = np.percentile(np.concatenate([samples[key] for key in sources]), percentiles)
        return {key: tuple(float(value) for value in bounds) for key in samples}
    per_source = {
        key: tuple(float(value) for value in np.percentile(samples[key], percentiles))
        for key in sources
    }
    if mode == "per_replicate":
        return {key: per_source[reference[key]] for key in samples}
    mean_bounds = tuple(
        float(value) for value in np.mean(list(per_source.values()), axis=0)
    )
    return {key: mean_bounds for key in samples}


def rescale(values, low, high):
    """Map ``low`` to 0 and ``high`` to 1, clipping everything outside."""
    span = max(high - low, 1e-6)
    return np.clip((np.asarray(values, dtype=np.float32) - low) / span, 0.0, 1.0)
