"""Initial conditions for PDE simulations, all shaped [batches, channels, size, size]."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from Common.model.spatial_operators import Ops
from PDE.catalogue import INITIAL_CONDITIONS


def blur(x, passes, kernel_scale=2):
    """Smooth each channel with a periodic Gaussian average, ``passes`` times."""
    if passes == 0:
        return x
    average = eqx.filter_vmap(Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=kernel_scale).Average)
    for _ in range(passes):
        x = average(x)
    return x


def uniform(key, batches, channels, size, scale=1.0, offset=0.0):
    return jr.uniform(key, (batches, channels, size, size)) * scale + offset


def square(batches, channels, size, scale=1.0, offset=0.0):
    """A centred square of side size/3 at ``scale + offset`` on a background of ``offset``."""
    lo, hi = size // 3, size - size // 3
    x = jnp.full((batches, channels, size, size), offset)
    return x.at[:, :, lo:hi, lo:hi].add(scale)


def gray_scott_shapes(batches, size):
    """Deterministic shapes in B (a square ring, or a square plus two lines), with A = 1 - B.

    These are the initial conditions of the thesis Gray-Scott runs; batches alternate between the two.
    """
    ring = jnp.zeros((size, size))
    ring = ring.at[size//6:5*size//6, size//6:5*size//6].set(1.0)
    ring = ring.at[size//4:3*size//4, size//4:3*size//4].set(0.0)
    lines = jnp.zeros((size, size))
    lines = lines.at[size//4:size//4 + size//6, size//4:size//4 + size//6].set(1.0)
    lines = lines.at[:, 3*size//4:3*size//4 + max(1, size//12)].set(1.0)
    lines = lines.at[3*size//4:3*size//4 + max(1, size//20), :].set(1.0)
    B = jnp.stack([(ring, lines)[b % 2] for b in range(batches)])[:, None]
    B = blur(B, 1)
    return jnp.concatenate((1 - B, B), axis=1)


def gray_scott_spots(key, batches, size, threshold=0.51):
    """Random blobs: thresholded smoothed noise in B, with A = 1 - B."""
    noise = blur(jr.uniform(key, (batches, 1, size, size)), 5, kernel_scale=3)
    B = blur(jnp.where(noise > threshold, 1.0, 0.0), 1)
    return jnp.concatenate((1 - B, B), axis=1)


def make_initial_condition(name, key, batches, channels, size, scale=1.0, offset=0.0, blur_passes=0):
    """Build the named initial condition, then blur it ``blur_passes`` times."""
    if name == "uniform":
        x = uniform(key, batches, channels, size, scale, offset)
    elif name == "square":
        x = square(batches, channels, size, scale, offset)
    elif name in ("gray_scott_shapes", "gray_scott_spots"):
        if channels != 2:
            raise ValueError(f"{name} needs a 2-channel PDE, got {channels} channels")
        x = gray_scott_shapes(batches, size) if name == "gray_scott_shapes" else gray_scott_spots(key, batches, size)
    else:
        raise ValueError(f"Unknown initial condition {name!r}; choose from {INITIAL_CONDITIONS}")
    return blur(x, blur_passes)
