"""Resolve typed loss configuration into the existing JAX loss primitives."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


LOSS_DTYPE = jnp.float32


def resolve_loss_component_weights(weights, loss_count):
    if weights is None:
        return jnp.ones((loss_count,), dtype=LOSS_DTYPE)
    weights = list(weights)
    if len(weights) != loss_count:
        raise ValueError(
            "loss term weights must have one value per configured loss "
            f"({loss_count} expected, got {len(weights)})"
        )
    if any(float(weight) < 0 for weight in weights):
        raise ValueError("loss term weights cannot contain negative weights")
    if not any(float(weight) > 0 for weight in weights):
        raise ValueError("loss term weights must contain at least one positive weight")
    return jnp.asarray(weights, dtype=LOSS_DTYPE)


def combine_loss_components(losses, weights):
    losses = jnp.stack(losses)
    weights = jnp.asarray(weights, dtype=losses.dtype)
    return jnp.sum(losses * weights[:, None], axis=0) / jnp.sum(weights)


@dataclass(frozen=True)
class ResolvedObjective:
    names: tuple[str, ...]
    arguments: dict[str, Any]
    regulariser_coefficients: dict[str, float]


def resolve_objective(loss_config, overrides=None) -> ResolvedObjective:
    """Validate shared loss options once, before entering compiled code."""

    terms = tuple(loss_config.terms)
    arguments = {
        "component_weights": [float(term.weight) for term in terms],
    }
    ignored = {"type", "weight"}
    for term in terms:
        for name, value in term.items():
            if name in ignored or value is None:
                continue
            runtime_name = "internal_loss_func" if name == "metric" else name
            if name == "metric" and "vgg" in term.type:
                runtime_name = "metric"
            if runtime_name in arguments and arguments[runtime_name] != value:
                raise ValueError(
                    "Loss terms require conflicting values for "
                    f"{runtime_name!r}"
                )
            arguments[runtime_name] = value
    arguments.setdefault("channels", None)
    arguments.setdefault("experiment_groups", None)
    if overrides:
        arguments.update(overrides)
    return ResolvedObjective(
        names=tuple(term.type for term in terms),
        arguments=arguments,
        regulariser_coefficients={
            name: float(value)
            for name, value in loss_config.regularisers.items()
        },
    )
