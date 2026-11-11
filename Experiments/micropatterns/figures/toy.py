"""Toy patterning used by the method schematics (illustration only, not data)."""

import numpy as np

from Experiments.micropatterns.figures.drawing import composite
from Experiments.micropatterns.figures.style import marker_colour

# Toy fates, drawn like the SOX2 / TBXT / SOX17 stainings
TOY_COLOURS = tuple(marker_colour(marker) for marker in ("SOX2", "TBXT", "SOX17"))


def toy_patterning(size, frame_count, steps_per_frame, seed):
    """Toy patterning process made only of local updates, for illustration.

    A signal enters at the colony edge and spreads by diffusion (a 4-neighbour
    Laplacian); each pixel then relaxes its three fates towards levels set by the
    local signal: high → SOX17-like, middle → TBXT-like, low → SOX2-like. Returns
    fates [F, 3, size, size] and the colony mask.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((size, size))
    centre = (size - 1) / 2
    mask = np.hypot(rows - centre, cols - centre) <= size / 2 - 0.5
    shifts = ((1, 0), (-1, 0), (1, 1), (-1, 1))
    neighbour_masks = [np.roll(mask, s, axis=a) for s, a in shifts]
    neighbour_count = sum(neighbour_masks)
    rim = mask & (neighbour_count < 4)
    # Fixed per-pixel jitter, so ring borders are not perfectly round
    jitter = 0.06 * rng.normal(size=(size, size))
    rate = 0.06 * (1 + 0.5 * rng.uniform(-1, 1, size=(size, size)))

    signal = np.zeros((size, size))
    fates = np.zeros((3, size, size))
    fates[0] = np.clip(0.85 + 0.1 * rng.normal(size=(size, size)), 0, 1)
    frames = [fates * mask]
    for _ in range(frame_count - 1):
        for _ in range(steps_per_frame):
            # Zero-flux Laplacian: only neighbours inside the colony
            laplacian = sum(np.roll(signal, s, axis=a) * m for (s, a), m in zip(shifts, neighbour_masks))
            laplacian = laplacian - neighbour_count * signal
            signal = (signal + 0.2 * laplacian - 0.02 * signal) * mask
            signal[rim] += 0.3 * (1.0 - signal[rim])
            level = signal + jitter
            high = 1 / (1 + np.exp(-(level - 0.5) / 0.04))
            middle = (1 / (1 + np.exp(-(level - 0.22) / 0.04))) * (1 - high)
            target = np.stack([np.clip(1 - high - middle, 0, 1), middle, high])
            fates = fates + rate * (target - fates)
        frames.append(fates * mask)
    return np.stack(frames), mask


def toy_rgba(fates, mask):
    """RGBA [H, W, 4] of toy fates [3, H, W] on black, transparent outside the colony."""
    rgb = composite(np.clip(fates, 0, 1), TOY_COLOURS)
    return np.concatenate([rgb, mask[..., None].astype(float)], axis=-1)
