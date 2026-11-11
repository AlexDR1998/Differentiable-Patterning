"""CMY composites of micropattern NCA states.

Each panel shows three state channels as cyan, magenta and yellow, so a
pixel high in one marker takes that colour and overlaps mix. Cell type
maps colour each pixel by its label shares from a cell type file.
"""

import numpy as np

from Common.dataloader.cell_types import label_names, marker_values, pixel_label_shares
from Experiments.micropatterns.figures.style import CMY_TRIPLETS, cell_type_rgb


def render_cmy_composite(states, state_channel_names):
    """Build the thesis-style row of three CMY composite panels."""

    triplets = CMY_TRIPLETS
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


def render_cell_types(states, state_channel_names, cell_type_rules, mask):
    """RGB ``[..., H, W, 3]`` of model states ``[..., C, H, W]`` coloured by cell type.

    Each pixel blends the cell-type colours (``figures/style.py``) of its label
    shares (``pixel_label_shares``); outside ``mask`` ``[H, W]`` it is black.
    """

    values = marker_values(states, state_channel_names, cell_type_rules, axis=-3)
    shares = pixel_label_shares(values, cell_type_rules, mask)
    rgb = sum(
        shares[label][..., None] * np.asarray(cell_type_rgb(label))
        for label in label_names(cell_type_rules)
    )
    return np.clip(rgb, 0.0, 1.0)


def cell_type_prevalences(states, state_channel_names, cell_type_rules, mask):
    """``{label: share of the colony}`` of each state ``[..., C, H, W]``, as arrays ``[...]``."""

    values = marker_values(states, state_channel_names, cell_type_rules, axis=-3)
    mask = np.asarray(mask, dtype=bool)
    shares = pixel_label_shares(values, cell_type_rules, mask)
    return {
        label: share.sum(axis=(-2, -1)) / max(mask.sum(), 1)
        for label, share in shares.items()
    }
