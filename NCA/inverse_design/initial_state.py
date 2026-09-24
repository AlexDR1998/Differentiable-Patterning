"""Key-driven patch sampling and geometry-conditioned NCA states."""

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True)
class PatchSamplingConfig:
    patch_size: int = 9
    measurement_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")


def _transform_patch(patch: Array, transform: Array) -> Array:
    transforms = (
        lambda value: value,
        lambda value: jnp.rot90(value, 1, axes=(-2, -1)),
        lambda value: jnp.rot90(value, 2, axes=(-2, -1)),
        lambda value: jnp.rot90(value, 3, axes=(-2, -1)),
        lambda value: value[..., ::-1],
        lambda value: jnp.rot90(value[..., ::-1], 1, axes=(-2, -1)),
        lambda value: jnp.rot90(value[..., ::-1], 2, axes=(-2, -1)),
        lambda value: jnp.rot90(value[..., ::-1], 3, axes=(-2, -1)),
    )
    return jax.lax.switch(transform, transforms, patch)


def _valid_patch_centres(mask: Array, patch_size: int) -> Array:
    kernel = jnp.ones((1, 1, patch_size, patch_size), dtype=jnp.float32)
    counts = jax.lax.conv_general_dilated(
        mask[None, None].astype(jnp.float32),
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )[0, 0]
    return counts >= patch_size * patch_size - 0.5


def sample_circular_patches(
    reference: Array,
    reference_mask: Array,
    output_shape: tuple[int, int],
    key: Array,
    config: PatchSamplingConfig = PatchSamplingConfig(),
) -> Array:
    """Tile random real patches into a rectangular initial texture.

    Patch centres and dihedral transforms are selected exclusively by ``key``.
    Co-measured channels in a group share selections. The discrete sample is
    explicitly stopped from participating in automatic differentiation.
    """

    reference = jnp.asarray(reference)
    reference_mask = jnp.asarray(reference_mask, dtype=bool)
    if reference.ndim != 3 or reference_mask.shape != reference.shape[-2:]:
        raise ValueError("expected reference [C,H,W] and matching mask [H,W]")
    patch_size = config.patch_size
    if patch_size > min(reference_mask.shape):
        raise ValueError("patch_size exceeds the reference shape")
    groups: Sequence[Sequence[int]] = config.measurement_groups or (
        tuple(range(reference.shape[0])),
    )
    flattened = tuple(index for group in groups for index in group)
    if sorted(flattened) != list(range(reference.shape[0])):
        raise ValueError("measurement_groups must partition all reference channels")

    valid = _valid_patch_centres(reference_mask, patch_size)
    if not bool(jnp.any(valid)):
        raise ValueError("reference mask contains no complete source patch")
    centre_logits = jnp.where(valid.reshape(-1), 0.0, -jnp.inf)
    half = patch_size // 2
    output = jnp.zeros((reference.shape[0], *output_shape), dtype=reference.dtype)
    tiles_y = (output_shape[0] + patch_size - 1) // patch_size
    tiles_x = (output_shape[1] + patch_size - 1) // patch_size
    group_keys = jax.random.split(key, len(groups))

    for group, group_key in zip(groups, group_keys):
        indices = jnp.asarray(group)
        tile_keys = jax.random.split(group_key, tiles_y * tiles_x)
        for tile_index, tile_key in enumerate(tile_keys):
            centre_key, transform_key = jax.random.split(tile_key)
            flat_centre = jax.random.categorical(centre_key, centre_logits)
            centre_y, centre_x = jnp.divmod(flat_centre, reference_mask.shape[1])
            patch = jax.lax.dynamic_slice(
                reference,
                (0, centre_y - half, centre_x - half),
                (reference.shape[0], patch_size, patch_size),
            )[indices]
            patch = _transform_patch(
                patch, jax.random.randint(transform_key, (), 0, 8)
            )
            tile_y, tile_x = divmod(tile_index, tiles_x)
            top, left = tile_y * patch_size, tile_x * patch_size
            height = min(patch_size, output_shape[0] - top)
            width = min(patch_size, output_shape[1] - left)
            output = output.at[indices, top : top + height, left : left + width].set(
                patch[:, :height, :width]
            )
    return jax.lax.stop_gradient(output)


def build_initial_state(
    sampled_biology: Array,
    occupancy: Array,
    total_channels: int,
    *,
    boundary_mode: str = "soft",
) -> Array:
    """Insert masked biology, zero hidden state, and an optional mask channel."""

    sampled_biology = jnp.asarray(sampled_biology)
    if sampled_biology.shape[-2:] != occupancy.shape:
        raise ValueError("biology and occupancy spatial shapes differ")
    required = sampled_biology.shape[0] + (boundary_mode == "soft")
    if total_channels < required:
        raise ValueError("total_channels cannot hold biology and boundary channels")
    if boundary_mode not in ("none", "hard", "soft"):
        raise ValueError(f"unknown boundary mode {boundary_mode!r}")
    state = jnp.zeros((total_channels, *occupancy.shape), dtype=sampled_biology.dtype)
    state = state.at[: sampled_biology.shape[0]].set(
        sampled_biology * occupancy[None]
    )
    if boundary_mode == "soft":
        state = state.at[-1].set(occupancy)
    return state
