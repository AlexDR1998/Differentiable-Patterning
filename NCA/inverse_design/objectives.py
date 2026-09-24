"""Smooth cell-fate objectives and geometry regularisation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True)
class GeometryPenalties:
    target_area_fraction: float | None = None
    area_weight: float = 0.0
    perimeter_weight: float = 0.0
    coefficient_weight: float = 0.0


@dataclass(frozen=True)
class CellTypeObjective:
    """A differentiable relaxation of high/low marker fate rules.

    Rules have value 1 for high, -1 for low, and 0 for ignored markers.
    ``target_rule`` therefore has one entry per ``marker_channel``.
    """

    marker_channels: tuple[int, ...]
    marker_thresholds: tuple[float, ...]
    target_rule: tuple[int, ...]
    temperature: float = 0.05
    amount_weight: float = 0.0

    def __post_init__(self):
        length = len(self.marker_channels)
        if len(self.marker_thresholds) != length or len(self.target_rule) != length:
            raise ValueError("marker channels, thresholds, and rules must align")
        if any(rule not in (-1, 0, 1) for rule in self.target_rule):
            raise ValueError("target rules must contain only -1, 0, or 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def probability(self, state: Array) -> Array:
        values = state[jnp.asarray(self.marker_channels)]
        thresholds = jnp.asarray(self.marker_thresholds, dtype=values.dtype)[:, None, None]
        rules = jnp.asarray(self.target_rule, dtype=values.dtype)[:, None, None]
        signed_margin = rules * (values - thresholds) / self.temperature
        condition = jnp.where(rules == 0, 1.0, jax.nn.sigmoid(signed_margin))
        return jnp.prod(condition, axis=0)

    def score(self, state: Array, occupancy: Array) -> tuple[Array, dict[str, Array]]:
        probability = self.probability(state)
        target_amount = jnp.mean(occupancy * probability)
        prevalence = jnp.sum(occupancy * probability) / jnp.maximum(
            jnp.sum(occupancy), 1.0e-6
        )
        score = prevalence + self.amount_weight * target_amount
        return score, {
            "score": score,
            "target_prevalence": prevalence,
            "target_amount": target_amount,
        }


def geometry_loss(
    geometry,
    state: Array,
    occupancy: Array,
    objective: CellTypeObjective,
    penalties: GeometryPenalties,
) -> tuple[Array, dict[str, Array]]:
    """Return minimisation loss and named scalar components."""

    score, components = objective.score(state, occupancy)
    area = jnp.mean(occupancy)
    area_penalty = jnp.asarray(0.0)
    if penalties.target_area_fraction is not None:
        area_penalty = jnp.square(area - penalties.target_area_fraction)
    vertical = jnp.mean(jnp.abs(occupancy[1:] - occupancy[:-1]))
    horizontal = jnp.mean(jnp.abs(occupancy[:, 1:] - occupancy[:, :-1]))
    perimeter = vertical + horizontal
    coefficient_norm = jnp.mean(
        jnp.square(geometry.coefficients_cos)
        + jnp.square(geometry.coefficients_sin)
    )
    regularisation = (
        penalties.area_weight * area_penalty
        + penalties.perimeter_weight * perimeter
        + penalties.coefficient_weight * coefficient_norm
    )
    loss = -score + regularisation
    return loss, {
        **components,
        "loss": loss,
        "area_fraction": area,
        "area_penalty": area_penalty,
        "perimeter": perimeter,
        "coefficient_norm": coefficient_norm,
    }
