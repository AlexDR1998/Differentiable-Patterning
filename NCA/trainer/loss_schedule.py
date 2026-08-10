"""Pure, fixed-structure schedules for loss component weights."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from Common.trainer.loss_components import MULTI_TARGET_WEIGHT_DEFAULTS


class ScheduledLossWeights(NamedTuple):
    terms: object
    multi_target: dict[str, object]


def schedule_factor(schedule, iteration, total_iterations):
    """Evaluate a configured multiplier without changing JAX tree structure."""
    dtype = jnp.float32
    if schedule is None:
        return jnp.asarray(1.0, dtype=dtype)
    initial = jnp.asarray(schedule.initial_factor, dtype=dtype)
    final = jnp.asarray(schedule.final_factor, dtype=dtype)
    if schedule.type == "constant":
        return initial

    last_iteration = max(int(total_iterations) - 1, 1)
    progress = jnp.asarray(iteration, dtype=dtype) / last_iteration
    transition = (progress - schedule.start_fraction) / (
        schedule.end_fraction - schedule.start_fraction
    )
    transition = jnp.clip(transition, 0.0, 1.0)
    if schedule.type == "cosine":
        transition = 0.5 - 0.5 * jnp.cos(jnp.pi * transition)
    return initial + (final - initial) * transition


def _multi_target_base_weights(term):
    configured = term.multi_target_weights or {}
    return {
        name: float(configured.get(name, default))
        for name, default in MULTI_TARGET_WEIGHT_DEFAULTS.items()
    }


def validate_loss_schedules(loss_config):
    allowed = set(MULTI_TARGET_WEIGHT_DEFAULTS)
    for index, term in enumerate(loss_config.terms):
        schedules = getattr(term, "multi_target_schedules", None) or {}
        configured = getattr(term, "multi_target_weights", None) or {}
        if term.type != "multi_target" and schedules:
            raise ValueError(
                f"loss term {index} is not multi_target but defines component schedules"
            )
        unknown = (set(schedules) | set(configured)) - allowed
        if unknown:
            raise ValueError(
                "Unknown multi-target loss components: "
                f"{sorted(unknown)}"
            )
        negative = {name: value for name, value in configured.items() if value < 0}
        if negative:
            raise ValueError(
                f"Multi-target loss weights cannot be negative: {negative}"
            )


def build_loss_weight_schedule(loss_config, total_iterations):
    """Return a callable producing all effective weights for one iteration."""
    validate_loss_schedules(loss_config)
    terms = tuple(loss_config.terms)
    multi_target_terms = [term for term in terms if term.type == "multi_target"]
    if len(multi_target_terms) > 1:
        raise ValueError("Only one multi_target loss term can be scheduled")
    multi_target_term = multi_target_terms[0] if multi_target_terms else None

    def weight_schedule(iteration):
        term_weights = jnp.stack(
            [
                jnp.asarray(float(term.weight), dtype=jnp.float32)
                * schedule_factor(term.schedule, iteration, total_iterations)
                for term in terms
            ]
        )
        multi_target = {}
        if multi_target_term is not None:
            schedules = multi_target_term.multi_target_schedules or {}
            for name, base_weight in _multi_target_base_weights(
                multi_target_term
            ).items():
                multi_target[name] = jnp.asarray(
                    base_weight
                    * schedule_factor(
                        schedules.get(name), iteration, total_iterations
                    ),
                    dtype=jnp.float32,
                )
        return ScheduledLossWeights(term_weights, multi_target)

    return weight_schedule


def final_transition_iteration(loss_config, total_iterations):
    """Last iteration at which any configured loss weight is still changing."""
    schedules = [
        term.schedule for term in loss_config.terms if term.schedule is not None
    ]
    for term in loss_config.terms:
        schedules.extend(
            (getattr(term, "multi_target_schedules", None) or {}).values()
        )
    changing = [schedule for schedule in schedules if schedule.type != "constant"]
    if not changing:
        return 0
    return min(
        int(total_iterations) - 1,
        max(
            round(schedule.end_fraction * (int(total_iterations) - 1))
            for schedule in changing
        ),
    )
