"""Smooth cell-type objectives and geometry regularisation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from Common.dataloader.cell_types import label_names, pattern_label_shares


@dataclass(frozen=True)
class GeometryPenalties:
    target_area_fraction: float | None = None
    area_weight: float = 0.0
    perimeter_weight: float = 0.0
    coefficient_weight: float = 0.0
    # Penalises mask in the outermost two pixels, where it would wrap round
    border_weight: float = 0.0


def border_fraction(occupancy: Array, width: int = 2) -> Array:
    """Mean of ``occupancy`` over the outermost ``width`` pixels of the grid."""

    inside = jnp.zeros(occupancy.shape, dtype=bool).at[width:-width, width:-width].set(True)
    return jnp.sum(jnp.where(inside, 0.0, occupancy)) / jnp.sum(~inside)


@dataclass(frozen=True)
class CellTypeShareObjective:
    """Share of one cell type in a colony, with the rules of a cell type file.

    Each pixel's markers are high with probability
    ``sigmoid((value - threshold) / temperature)``, independently. Every
    high/low pattern of the markers then counts towards the target label
    with the weight ``pattern_label_shares`` gives it, so clauses joined by
    "or", clause weights, "Several" and "Other" all follow the rule file.
    With ``hard=True`` (or as the temperature goes to 0) this is
    ``Common.dataloader.cell_types.pixel_label_shares``.

    ``pattern_shares`` holds the target label's share of each pattern, where
    bit ``i`` of the pattern is marker ``i`` (in ``marker_channels`` order).
    ``maximise=False`` turns the objective round, as a control.
    """

    marker_channels: tuple[int, ...]
    marker_thresholds: tuple[float, ...]
    pattern_shares: tuple[float, ...]
    temperature: float = 0.05
    amount_weight: float = 0.0
    maximise: bool = True

    def __post_init__(self):
        length = len(self.marker_channels)
        if len(self.marker_thresholds) != length:
            raise ValueError("marker channels and thresholds must align")
        if len(self.pattern_shares) != 2**length:
            raise ValueError("pattern_shares needs one entry per high/low marker pattern")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @classmethod
    def from_rules(cls, cell_type_rules, state_channel_names, target, **options):
        """Build the objective for label ``target`` of a ``CellTypeRules``.

        Markers are found among the model's state channels by name, as
        ``marker_values`` does for model states.
        """

        names = list(state_channel_names)
        missing = [marker for marker in cell_type_rules.markers if marker not in names]
        if missing:
            raise KeyError(f"Markers {missing} are not among the state channels {names}")
        labels = label_names(cell_type_rules)
        if target not in labels:
            raise ValueError(f"Unknown cell type {target!r}; choose one of {labels}")
        shares = pattern_label_shares(cell_type_rules)[:, labels.index(target)]
        return cls(
            marker_channels=tuple(names.index(marker) for marker in cell_type_rules.markers),
            marker_thresholds=tuple(
                float(cell_type_rules.thresholds[marker]) for marker in cell_type_rules.markers
            ),
            pattern_shares=tuple(float(share) for share in shares),
            **options,
        )

    def probability(self, state: Array, hard: bool = False) -> Array:
        """Per-pixel share ``[H, W]`` of the target label."""

        values = state[jnp.asarray(self.marker_channels)]
        thresholds = jnp.asarray(self.marker_thresholds, dtype=values.dtype)[:, None, None]
        if hard:
            high = (values > thresholds).astype(values.dtype)
        else:
            high = jax.nn.sigmoid((values - thresholds) / self.temperature)
        markers = len(self.marker_channels)
        bits = (np.arange(2**markers)[:, None] >> np.arange(markers)) & 1
        pattern_probability = jnp.prod(
            jnp.where(bits[:, :, None, None] == 1, high[None], 1.0 - high[None]),
            axis=1,
        )
        shares = jnp.asarray(self.pattern_shares, dtype=values.dtype)
        return jnp.tensordot(shares, pattern_probability, axes=1)

    def score(
        self, state: Array, occupancy: Array, hard: bool = False
    ) -> tuple[Array, dict[str, Array]]:
        probability = self.probability(state, hard)
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
    objective: CellTypeShareObjective,
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
    border = border_fraction(occupancy)
    regularisation = (
        penalties.area_weight * area_penalty
        + penalties.perimeter_weight * perimeter
        + penalties.coefficient_weight * coefficient_norm
        + penalties.border_weight * border
    )
    loss = (-score if objective.maximise else score) + regularisation
    return loss, {
        **components,
        "loss": loss,
        "area_fraction": area,
        "area_penalty": area_penalty,
        "perimeter": perimeter,
        "coefficient_norm": coefficient_norm,
        "border_fraction": border,
    }
