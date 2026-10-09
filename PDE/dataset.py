"""Simulate PDE trajectories as NCA training data, shaped [batches, time, channels, H, W]."""

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange

from PDE.catalogue import NORMALISATIONS, PDE_MODELS, build_pde
from PDE.initial_conditions import make_initial_condition
from PDE.model.solver.semidiscrete_solver import PDE_solver


@dataclass(frozen=True)
class PdeTrajectory:
    data: np.ndarray  # [B, T, C, H, W], observed channels only
    observation_times: tuple[float, ...]
    channel_names: tuple[str, ...]


def normalise(Y, mode):
    """Rescale to [0, 1] over the whole array (minmax) or per channel (channel_minmax)."""
    if mode == "none":
        return Y
    axes = None if mode == "minmax" else (0, 1, 3, 4)
    lo = jnp.min(Y, axis=axes, keepdims=True)
    hi = jnp.max(Y, axis=axes, keepdims=True)
    return (Y - lo) / jnp.maximum(hi - lo, 1e-12)


def simulate_trajectories(
    model,
    key,
    *,
    parameters=None,
    batches=2,
    size=64,
    padding="CIRCULAR",
    dx=1.0,
    dt=0.1,
    solver="heun",
    t_end=1000.0,
    frames=12,
    observed_channels=None,
    initial_condition="uniform",
    initial_condition_scale=1.0,
    initial_condition_offset=0.0,
    initial_condition_blur=0,
    normalisation="minmax",
    max_steps=100_000,
):
    """Solve ``model`` from ``batches`` initial conditions and keep ``frames`` evenly spaced snapshots.

    The first frame is the initial condition. ``observed_channels`` (names) picks the
    channels the NCA must reproduce; the rest of the PDE state is hidden from it.
    """
    channel_names = PDE_MODELS[model].channel_names
    observed_channels = tuple(observed_channels or channel_names)
    unknown = set(observed_channels) - set(channel_names)
    if unknown:
        raise ValueError(f"PDE {model!r} has channels {channel_names}, not {sorted(unknown)}")
    if normalisation not in NORMALISATIONS:
        raise ValueError(f"Unknown normalisation {normalisation!r}; choose from {NORMALISATIONS}")

    x0 = make_initial_condition(
        initial_condition, key, batches, len(channel_names), size,
        scale=initial_condition_scale, offset=initial_condition_offset, blur_passes=initial_condition_blur,
    )
    func = build_pde(model, padding, dx, parameters)
    v_func = eqx.filter_vmap(func, in_axes=(None, 0, None), out_axes=0)  # Parallelise over batches
    ts = jnp.linspace(0.0, t_end, frames)
    _, Y = PDE_solver(v_func, dt=dt, SOLVER=solver, max_steps=max_steps)(ts, x0)
    Y = rearrange(Y, "T B C X Y -> B T C X Y")
    if not bool(jnp.all(jnp.isfinite(Y))):
        raise FloatingPointError(f"PDE {model!r} diverged; try a smaller dt or larger dx")

    Y = Y[:, :, jnp.array([channel_names.index(name) for name in observed_channels])]
    Y = normalise(Y, normalisation)
    return PdeTrajectory(
        data=np.asarray(Y, dtype=np.float32),
        observation_times=tuple(float(t) for t in np.asarray(ts)),
        channel_names=observed_channels,
    )
