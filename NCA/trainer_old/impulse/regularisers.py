import jax.numpy as jnp


def intervention_metrics(initial_states, perturbed_states):
    """Measure the perturbation actually applied to the NCA state."""

    delta = perturbed_states - initial_states
    horizontal = delta[..., 1:, :] - delta[..., :-1, :]
    vertical = delta[..., :, 1:] - delta[..., :, :-1]
    return {
        "l1": jnp.mean(jnp.abs(delta)),
        # The epsilon keeps the RMS gradient finite at an all-zero intervention.
        "l2": jnp.sqrt(jnp.mean(delta**2) + 1e-12),
        "linf": jnp.max(jnp.abs(delta)),
        "smoothness": jnp.mean(horizontal**2) + jnp.mean(vertical**2),
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
