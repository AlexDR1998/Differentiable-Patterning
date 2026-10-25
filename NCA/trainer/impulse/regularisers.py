import jax.numpy as jnp


def intervention_metrics(initial_states, deltas):
    """Measure an intervention from its per-step deltas ``[steps, 1 or batch, C, H, W]``.

    Each step is averaged over the lattice and the steps are summed, so l1, l2
    and smoothness measure the total intervention and do not favour spreading
    it over more steps. With one step this is the delta applied to the state.
    """

    lattice = tuple(range(1, deltas.ndim))
    horizontal = deltas[..., 1:, :] - deltas[..., :-1, :]
    vertical = deltas[..., :, 1:] - deltas[..., :, :-1]
    perturbed_states = initial_states + deltas[0]
    return {
        "l1": jnp.sum(jnp.mean(jnp.abs(deltas), axis=lattice)),
        # The epsilon keeps the RMS gradient finite at an all-zero intervention.
        "l2": jnp.sqrt(jnp.sum(jnp.mean(deltas**2, axis=lattice)) + 1e-12),
        "linf": jnp.max(jnp.abs(deltas)),
        "smoothness": jnp.sum(
            jnp.mean(horizontal**2, axis=lattice) + jnp.mean(vertical**2, axis=lattice)
        ),
        # Only the first step: later deltas land on states the NCA has changed.
        "state_range": jnp.mean(jnp.maximum(-perturbed_states, 0.0))
        + jnp.mean(jnp.maximum(perturbed_states - 1.0, 0.0)),
    }


def weighted_regulariser(metrics, coefficients=None):
    """Combine intervention metrics using a dictionary of scalar coefficients."""

    coefficients = coefficients or {}
    unknown = set(coefficients) - set(metrics)
    if unknown:
        raise ValueError(f"Unknown intervention regularisers: {sorted(unknown)}")
    total = jnp.asarray(0.0)
    for name, coefficient in coefficients.items():
        total = total + float(coefficient) * metrics[name]
    return total
