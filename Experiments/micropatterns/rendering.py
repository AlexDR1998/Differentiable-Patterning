"""CMY composites of micropattern NCA states.

Each panel shows three state channels as cyan, magenta and yellow, so a
pixel high in one marker takes that colour and overlaps mix.
"""

import numpy as np


def render_cmy_composite(states, state_channel_names):
    """Build the thesis-style row of three CMY composite panels."""

    triplets = (
        ("SOX2", "TBXT", "SOX17"),
        ("CER1", "LEFTY", "NODAL"),
        ("FOXA2", "LEF1", "SMAD23"),
    )
    missing = sorted(
        {name for triplet in triplets for name in triplet}
        .difference(state_channel_names)
    )
    if missing:
        raise ValueError(
            "Three-panel CMY rendering requires state channels: "
            + ", ".join(missing)
        )

    panels = []
    for cyan_name, magenta_name, yellow_name in triplets:
        cyan, magenta, yellow = (
            np.clip(states[:, state_channel_names.index(name)], 0.0, 1.0)
            for name in (cyan_name, magenta_name, yellow_name)
        )
        panels.append(np.stack(
            (
                0.5 * (magenta + yellow),
                0.5 * (cyan + yellow),
                0.5 * (cyan + magenta),
            ),
            axis=-1,
        ))
    return np.concatenate(panels, axis=2)


def render_fate_cmy(states, state_channel_names, renormalise=False):
    """Render SOX2 cyan, TBXT magenta, and SOX17 yellow."""

    channels = [
        np.clip(states[:, state_channel_names.index(name)], 0.0, 1.0)
        for name in ("SOX2", "TBXT", "SOX17")
    ]
    if renormalise:
        normalised_channels = []
        for channel in channels:
            channel_min = np.min(channel, axis=(-2, -1), keepdims=True)
            channel_max = np.max(channel, axis=(-2, -1), keepdims=True)
            channel_range = channel_max - channel_min
            normalised_channels.append(np.where(
                channel_range > 0,
                (channel - channel_min) / np.maximum(channel_range, 1e-12),
                0.0,
            ))
        channels = normalised_channels
    cyan, magenta, yellow = channels
    return np.stack(
        (
            0.5 * (magenta + yellow),
            0.5 * (cyan + yellow),
            0.5 * (cyan + magenta),
        ),
        axis=-1,
    )
