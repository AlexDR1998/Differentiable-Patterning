"""Synthetic 0 h initial textures made from measured colonies.

Two kinds, set per projection in the texture settings file (``texture_settings.py``):

- ``synthesise_texture``: a texture over the whole grid made from the
  measured colonies (patch resampling or IAAFT). It keeps the colonies' own
  gaps between cells, so its cell density is the measured one.
- ``cell_states``: whole measured cells placed at a chosen density
  (``CellSettings``). ``tiled`` grows cells from seed points without
  overlap (``tile_cells``); ``layers`` places layers of cells and keeps
  them in a random order up to the density (``NCA/inverse_design/density.py``).
  Each staining panel is placed on its own: the panels were imaged on
  different colonies, so in the data their cells do not line up (only
  ``cell_fate_s1`` holds the 0 h cell markers SOX2 and LMBR). Every panel
  reaches each density by itself. Panels with no cell structure at 0 h
  (``CellSettings.noise_panels``, the RNA panel) get no cells: their
  channels are measured pixels drawn at random (``fill_noise_panels``).

Tuned in ``synthetic_textured_ic_explorer.py``; used by inverse design and
the density × KO sweep.
"""

import numpy as np

from Experiments.micropatterns.cell_counts import (
    nuclear_cell_library,
    rank_softness,
)
from Experiments.micropatterns.reference import cell_library, combine_reference
from Experiments.micropatterns.texture_settings import scaled_patch_geometry
from Experiments.micropatterns.texture_synthesis import (
    layer_cell_counts,
    normalise_histograms_and_covariance,
    place_cells,
    scatter_seeds,
    synthesise_patch_texture,
    synthesise_reference_texture,
    tile_cells,
)


# ---------------------------------------------------------------- textures with the measured gaps


def synthesise_texture(reference, texture, shape, seed, downsample):
    """One texture ``[measurement channels, H, W]`` over a whole grid ``shape``.

    Each staining panel comes from a random replicate of ``reference``
    (``combine_reference``) that measured it. ``texture`` is a ``texture``
    section of the settings file; its patch sizes are rescaled to the data's
    ``downsample``.
    """

    groups = tuple(tuple(group) for group in reference["schema"].group_measurement_indices)
    full = np.ones(shape, dtype=bool)
    measurements, support = combine_reference(reference, "sample_groups", 0, seed)
    if texture["mode"] == "patch":
        size, overlap, jitter = scaled_patch_geometry(
            texture["patch_size"], texture["patch_overlap"], texture["patch_jitter"],
            texture["patch_downsample"], downsample,
        )
        tile, *_ = synthesise_patch_texture(
            measurements, support, shape, groups, size, overlap,
            texture["interior_fraction"], jitter, texture["candidates"], seed,
        )
    elif texture["mode"] == "iaaft":
        tile = synthesise_reference_texture(
            measurements, support, shape, full, texture["iaaft_iterations"], seed
        )
    else:
        raise ValueError(f"Unknown texture mode {texture['mode']!r}; use patch or iaaft")
    if texture["normalise"]:
        tile = normalise_histograms_and_covariance(
            tile, measurements, support, full, groups, texture["normalise_iterations"]
        )
    return np.asarray(tile, dtype=np.float32)


# ---------------------------------------------------------------- whole cells


def replicate_cell_library(colonies, index, settings, raw=None):
    """``(library, measured covered share)``: the pieces placed by ``cell_states``.

    ``colonies`` are 0 h colonies as ``cell_library`` reads them. With
    ``index``, pieces come from that colony, and staining panels it did not
    measure take theirs from the others; with None, from every colony.
    Pieces are segments (``cell_library``) or, for ``settings.pieces ==
    "nuclei"``, real cells from the nuclei (``nuclear_cell_library``);
    ``raw`` then holds its other keyword arguments (``source_files``,
    ``dataset_root``, ``alignment``, ``downsample``, ...). The library also
    holds ``pixels``: per panel, every measured pixel inside the colonies
    used ``[n, panel channels]`` (for ``settings.noise_panels``).
    """

    available = np.array(colonies["available"], dtype=bool)
    if index is not None:
        others = np.arange(len(available)) != index
        for indices in colonies["schema"].group_measurement_indices:
            indices = np.asarray(indices, dtype=int)
            if np.all(available[index, indices]):
                available[np.ix_(others, indices)] = False
    colonies = {**colonies, "available": available}
    if settings.pieces == "segments":
        library = cell_library(
            colonies, settings.markers, settings.split_distance,
            settings.interior_fraction, settings.min_cell_area,
        )
    else:
        if raw is None:
            raise ValueError("Real cells need the raw images (raw=...)")
        library = nuclear_cell_library(
            colonies,
            markers=settings.markers,
            split_distance=settings.split_distance,
            interior_fraction=settings.interior_fraction,
            min_area=settings.min_cell_area,
            expansion_um=settings.expansion_um,
            **settings.nucleus_options(),
            **raw,
        )
    library["pixels"] = [
        np.concatenate([
            np.asarray(colonies["measurements"][_b])[np.asarray(_indices, dtype=int)][:, colonies["support"][_b]].T
            for _b in np.flatnonzero(np.all(available[:, np.asarray(_indices, dtype=int)], axis=1))
        ]).astype(np.float32)
        for _indices in colonies["schema"].group_measurement_indices
    ]
    own = [
        _row["coverage"] for _row in library["coverage"]
        if _row["panel"] in settings.density_panels
        and (index is None or _row["replicate"] == colonies["replicates"][index])
    ]
    # The colony's own covered share in the density panels (else that of the colonies used)
    if not own:
        own = [_row["coverage"] for _row in library["coverage"] if _row["panel"] in settings.density_panels]
    measured = float(np.mean(own)) if own else library["density"]
    return library, measured


def cell_states(library, schema, mask, targets, by_share, seed, settings, per_piece=1.0):
    """Initial states of whole cells in ``mask`` ``[H, W]``, one per target.

    ``targets`` are cells in the colony, or covered shares with
    ``by_share``. ``per_piece`` is the real cells per placed piece (1 for
    real cells). Returns a dict with ``initial_states`` ``[D, C, H, W]``
    (state channels, 0 outside ``mask``), the ``cells`` and covered
    ``shares`` reached, ``capacity`` (the most cells, or the largest share,
    the placement holds), ``cell_weights`` ``[D, H, W]`` (each pixel's share
    of the cells, summing to the cells), the measured ``cell_area``,
    ``squeeze`` (the mean area of a cell at the highest density over the
    measured one), ``panel_cells`` ``[D, P]`` (cells per staining panel,
    NaN for panels without cells) and the number of ``layers`` (0 for
    tiled). Cells are
    those that show in the state: for layers, cells hidden under others are
    not counted (``visible_cells``).
    """

    if settings.placement == "tiled":
        states = tiled_cell_states(library, schema, mask, targets, by_share, seed, settings)
    else:
        states = layered_cell_states(library, schema, mask, targets, by_share, seed, settings, per_piece)
    states["initial_states"] = fill_noise_panels(states["initial_states"], library, schema, mask, seed, settings)
    return states


def noise_panel_ids(library, schema, settings):
    """Indices of ``settings.noise_panels`` among the library's staining panels."""

    names = list(schema.group_names)
    unknown = set(settings.noise_panels) - set(names)
    if unknown:
        raise ValueError(f"Unknown noise panels {sorted(unknown)}; the panels are {names}")
    ids = {names.index(_name) for _name in settings.noise_panels}
    if len(ids) >= len(library["panels"]):
        raise ValueError("At least one staining panel must hold cells")
    return ids


def fill_noise_panels(states, library, schema, mask, seed, settings):
    """``states`` ``[D, C, H, W]`` with the channels of the noise panels drawn from measured pixels.

    Each pixel in ``mask`` takes a measured pixel of the panel (all its
    channels together, so their joint distribution is kept), drawn at
    random. The same draw is used at every density.
    """

    states = np.array(states, copy=True)
    primary = np.asarray(schema.primary_measurements, dtype=int)
    panel_of_channel = channel_panels(library, schema)
    for panel in sorted(noise_panel_ids(library, schema, settings)):
        channels = np.flatnonzero(panel_of_channel == panel)
        if not len(channels):
            continue
        indices = list(np.asarray(library["panels"][panel][0], dtype=int))
        pixels = library["pixels"][panel][:, [indices.index(int(primary[_c])) for _c in channels]]
        drawn = pixels[np.random.default_rng([int(seed), 2, panel]).integers(len(pixels), size=int(mask.sum()))]
        for channel, values in zip(channels, drawn.T):
            states[:, channel][:, mask] = values
    return states


def _panel_table(columns, panels, rows):
    """``[rows, panels]`` from ``(panel, values)`` columns; NaN for panels without cells."""

    table = np.full((rows, panels), np.nan)
    for panel, values in columns:
        table[:, panel] = values
    return table


def channel_panels(library, schema):
    """The staining panel of each state channel, ``[C]``."""

    panel_of = {
        int(_index): _panel for _panel, (_indices, _) in enumerate(library["panels"]) for _index in _indices
    }
    return np.asarray([panel_of[int(_index)] for _index in schema.primary_measurements], dtype=int)


def tiled_cell_states(library, schema, mask, targets, by_share, seed, settings):
    """``cell_states`` with cells packed without overlap (``tile_cells``).

    Each staining panel is tiled on its own (its own seeds), as the panels
    were imaged on different colonies and their cells do not line up. A
    panel's cells are grown from seeds spread evenly over ``mask``, out to
    ``settings.reach_scale`` times the radius of its measured cells (from
    their mean area), and each takes the mean values of one of its measured
    cells, at random. The seeds come in one random order, so the densities
    share their first cells. Every panel that supplies state channels
    reaches each target; ``cells``, ``shares``, ``cell_weights`` and
    ``squeeze`` are means over those panels and ``capacity`` the least of
    them.
    """

    mask = np.asarray(mask, dtype=bool)
    targets = np.asarray(targets, dtype=float)
    factor = int(settings.tile_factor)
    primary = np.asarray(schema.primary_measurements, dtype=int)
    panel_of_channel = channel_panels(library, schema)
    gap_values = np.asarray(library["gap_values"], dtype=np.float32)
    colony_area = float(np.count_nonzero(mask))
    fine_mask = np.kron(mask, np.ones((factor, factor), dtype=bool))
    states = np.broadcast_to(
        gap_values[None, :, None, None], (len(targets), len(gap_values), *mask.shape)
    ).copy()
    found = []
    noise = noise_panel_ids(library, schema, settings)
    for panel, (indices, cells) in enumerate(library["panels"]):
        channels = np.flatnonzero(panel_of_channel == panel)
        if not len(channels) or panel in noise:
            continue
        area = float(np.mean([np.count_nonzero(_m) for _m, _ in cells]))
        reach = float(settings.reach_scale) * np.sqrt(area / np.pi)
        if by_share:
            # Enough cells to cover the colony, squeezed to a third of their area
            wanted = int(np.ceil(3.0 * colony_area / (np.pi * reach**2)))
        else:
            wanted = int(np.ceil(targets.max()))
        seeds = scatter_seeds(
            fine_mask, wanted, float(settings.seed_spacing) * np.sqrt(fine_mask.sum() / max(wanted, 1)),
            [int(seed), panel],
        ) / factor
        # Each cell's values: the mean of one of the panel's measured cells
        rng = np.random.default_rng([int(seed), 1, panel])
        position = {int(_index): _k for _k, _index in enumerate(indices)}
        means = np.stack([_values[:, _mask].mean(axis=1) for _mask, _values in cells])
        values = means[rng.integers(len(cells), size=len(seeds))][:, [position[int(primary[_c])] for _c in channels]]

        def _tile(count):
            return tile_cells(seeds[:count], reach, values[:count], gap_values[channels], mask, factor)

        def _share(tiled):
            return float(np.mean(tiled[1][mask]))

        if by_share:
            capacity = _share(_tile(len(seeds)))
            counts = []
            for target in targets:
                # Fewest cells covering the target (coverage grows with the cells)
                low, high = 0, len(seeds)
                while low < high:
                    middle = (low + high) // 2
                    if _share(_tile(middle)) >= target:
                        high = middle
                    else:
                        low = middle + 1
                counts.append(low)
        else:
            capacity = float(len(seeds))
            counts = [min(int(round(_target)), len(seeds)) for _target in targets]
        tiled = [_tile(_count) for _count in counts]
        states[:, channels] = np.stack([_t[0] for _t in tiled])
        cells_found = np.asarray([_t[3] for _t in tiled], dtype=float)
        shares = np.asarray([_share(_t) for _t in tiled])
        found.append({
            "panel": panel,
            "cells": cells_found,
            "shares": shares,
            "weights": np.stack([_t[2] for _t in tiled]),
            "capacity": capacity,
            "area": area,
            "squeeze": shares[-1] * colony_area / max(cells_found[-1], 1.0) / area,
        })
    return {
        "initial_states": states * mask[None, None],
        "cell_weights": np.mean([_f["weights"] for _f in found], axis=0) * mask,
        "cells": np.mean([_f["cells"] for _f in found], axis=0),
        "panel_cells": _panel_table([(_f["panel"], _f["cells"]) for _f in found], len(library["panels"]), len(targets)),
        "shares": np.mean([_f["shares"] for _f in found], axis=0),
        "capacity": min(_f["capacity"] for _f in found),
        "cell_area": float(np.mean([_f["area"] for _f in found])),
        "squeeze": float(np.mean([_f["squeeze"] for _f in found])),
        "layers": 0,
    }


def placed_cell_texture(
    library, schema, shape, settings, seed, highest_density=0.0, max_layers=10,
    mask=None, highest_cells=0.0, layer_cache=None,
):
    """``(packed texture, CellGaps, cells per layer, uncropped ranks, margin)``:
    whole cells placed at random over ``shape``, in state channel order
    (``NCA/inverse_design/density.py``). Each staining panel is placed on
    its own (``place_cells`` with ``independent_panels``), and each layer
    from its own seed ``[seed, layer]``, so a layer is the same however many
    are placed. ``layer_cache`` (a dict) keeps placed layers between calls.

    Cells are placed over a margin around the grid and cropped, so the
    grid edge has as many cells as the centre. One layer of
    non-overlapping cells jams well below full coverage, so layers are
    added (from ``settings.layers`` up to ``max_layers``) until together
    they can cover ``highest_density`` and hold ``highest_cells`` cells in
    ``mask`` ``[H, W]``. Placement is layer by layer from ``seed``, so the
    extra layers go on top of the same cells. The cells per layer are
    the placed cells counted in ``mask`` (None without a mask), hidden or
    not. The rank maps ``[L, P, H + 2 margin, W + 2 margin]`` are returned
    before cropping, so cells at the grid edge keep their whole area.
    """
    import jax.numpy as jnp

    from NCA.inverse_design.density import CellGaps, pack_cells

    panels = library["panels"]
    # Noise panels get no cells (fill_noise_panels draws their channels)
    noise = noise_panel_ids(library, schema, settings)
    placed_ids = [_panel for _panel in range(len(panels)) if _panel not in noise]
    margin = max(max(_mask.shape) for _panel in placed_ids for _mask, _ in panels[_panel][1])

    layer_cache = {} if layer_cache is None else layer_cache

    def _layer(layer):
        if layer not in layer_cache:
            textures, placed_ranks, coverage = place_cells(
                [panels[_panel] for _panel in placed_ids],
                len(library["measurement_gap_values"]),
                (shape[0] + 2 * margin, shape[1] + 2 * margin),
                1,
                settings.attempts,
                [int(seed), layer],
                size_bias=settings.size_bias,
                # The panels were imaged on different colonies: their cells do not line up
                independent_panels=True,
                first_layer=layer,
            )
            ranks = np.full((1, len(panels), *placed_ranks.shape[-2:]), -1.0, dtype=np.float32)
            ranks[:, placed_ids] = placed_ranks
            layer_cache[layer] = textures, ranks, coverage
        return layer_cache[layer]

    def _place(layers):
        placed = [_layer(_layer_index) for _layer_index in range(layers)]
        return tuple(np.concatenate([_p[_part] for _p in placed]) for _part in range(3))

    padded_mask = None if mask is None else np.pad(np.asarray(mask, dtype=bool), margin)

    def _cells(ranks):
        return None if padded_mask is None else layer_cell_counts(ranks[:, placed_ids], padded_mask)

    layers = int(settings.layers)
    textures, ranks, coverage = _place(layers)
    cells = _cells(ranks)
    uncovered = float(np.prod(1.0 - coverage))
    per_layer = 1.0 - uncovered ** (1.0 / layers)
    needed = layers
    if 1.0 - uncovered < highest_density and per_layer > 0.0:
        # Layers are about equally full and independent: 1 - (1 - c) ** L.
        # The small margin covers layers that come out a little emptier.
        target = min(float(highest_density) + 0.02, 0.999)
        needed = int(np.ceil(np.log(1.0 - target) / np.log(1.0 - per_layer)))
    if cells is not None and float(np.sum(cells)) < highest_cells and np.mean(cells) > 0.0:
        needed = max(needed, int(np.ceil(1.02 * highest_cells / float(np.mean(cells)))))
    if needed > layers and layers < max_layers:
        textures, ranks, coverage = _place(min(max(needed, layers + 1), max_layers))
        cells = _cells(ranks)
    crop = (
        slice(None), slice(None),
        slice(margin, margin + shape[0]), slice(margin, margin + shape[1]),
    )
    primary = np.asarray(schema.primary_measurements, dtype=int)
    gaps = CellGaps(
        gap_values=jnp.asarray(library["gap_values"]),
        channel_panels=jnp.asarray(channel_panels(library, schema)),
        layer_coverage=jnp.asarray(coverage),
        panels=len(panels),
        softness=rank_softness(settings.softness_cells, ranks[:, placed_ids]),
    )
    packed = pack_cells(jnp.asarray(textures[crop][:, primary]), jnp.asarray(ranks[crop]))
    return packed, gaps, cells, ranks, margin


def cell_ids(ranks):
    """``(ids [L, P, H, W], panel of each id)`` for rank maps ``[L, P, H, W]``.

    Within a layer every placed cell has its own rank (shared by its panels),
    so cells are told apart by layer, panel and rank. ``ids`` is -1 where no
    cell is.
    """

    ranks = np.asarray(ranks)
    ids = np.full(ranks.shape, -1, dtype=np.int64)
    panel_of = []
    for layer in range(ranks.shape[0]):
        for panel in range(ranks.shape[1]):
            placed = ranks[layer, panel] >= 0
            _, inverse = np.unique(ranks[layer, panel][placed], return_inverse=True)
            ids[layer, panel][placed] = inverse + len(panel_of)
            panel_of.extend([panel] * (int(inverse.max()) + 1 if inverse.size else 0))
    return ids, np.asarray(panel_of, dtype=int)


def visible_cells(ids, panel_of, kept, mask, visible_fraction):
    """The cells that show in a layered initial state, per staining panel.

    ``ids`` and ``panel_of`` come from ``cell_ids``, ``kept`` ``[L, P, H, W]``
    is how much each cell pixel is kept (``CellGaps.kept_ranks``) and
    ``mask`` ``[H, W]`` the colony, all on the same (uncropped) grid. As in
    ``CellGaps.apply``, layer 0 is drawn on top, so a pixel of a layer ``l``
    cell shows with weight ``kept_l * prod_{j<l} (1 - kept_j)``. A cell
    counts when these weights inside ``mask`` add up to at least
    ``visible_fraction`` of its area: hidden cells, cells mostly outside the
    colony and cells faded at the density cut are left out.

    Returns a dict with, per panel, ``cells`` ``[P]``; ``kept_cells``
    ``[P]`` (counted the same way but hidden ones included); the ``covered``
    share of ``mask`` ``[P]``; the ``visible_area`` of the counted cells
    ``[P]``; ``weights`` ``[P, H, W]`` (every counted cell spreads a weight
    of 1 over the pixels it shows on, so each panel's sums to its cells);
    and ``owners`` ``[P, H, W]``, the cell showing most on each pixel (its
    id, -2 if that cell does not count, -1 where cells show less than half).
    """

    kept = np.asarray(kept, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    panels = ids.shape[1]
    above = np.cumprod(np.concatenate([np.ones_like(kept[:1]), 1.0 - kept[:-1]]), axis=0)
    shown = kept * above * mask
    placed = ids >= 0
    count = len(panel_of)
    area = np.bincount(ids[placed], minlength=count)
    seen = np.bincount(ids[placed], weights=shown[placed], minlength=count)
    counted = (area > 0) & (seen >= float(visible_fraction) * area)
    held = np.bincount(ids[placed], weights=(kept * mask)[placed], minlength=count)
    kept_counted = (area > 0) & (held >= float(visible_fraction) * area)
    share = np.zeros_like(shown)
    share[placed] = np.where(counted[ids[placed]], shown[placed] / np.maximum(seen[ids[placed]], 1e-12), 0.0)
    top = np.argmax(shown, axis=0)[None]
    top_id = np.take_along_axis(ids, top, axis=0)[0]
    top_shown = np.take_along_axis(shown, top, axis=0)[0]
    owners = np.where(counted[np.maximum(top_id, 0)], top_id, -2)
    owners = np.where((top_id >= 0) & (top_shown >= 0.5), owners, -1)
    return {
        "cells": np.bincount(panel_of[counted], minlength=panels).astype(float),
        "kept_cells": np.bincount(panel_of[kept_counted], minlength=panels).astype(float),
        "covered": (1.0 - np.prod(1.0 - kept, axis=0))[:, mask].mean(axis=1),
        "visible_area": np.bincount(panel_of[counted], weights=seen[counted], minlength=panels),
        "weights": share.sum(axis=0),
        "owners": owners,
    }


def layered_cell_states(library, schema, mask, targets, by_share, seed, settings, per_piece=1.0, max_layers=10):
    """``cell_states`` with layers of cells that may overlap (``placed_cell_texture``).

    Each staining panel is placed on its own and gets its own density
    (``CellDensity``), found by search so that the cells that show
    (``visible_cells``) or the covered share reach each target. The cells
    kept are those of the lowest random ranks, so the densities share their
    first cells. Layers are added until the highest target fits.
    ``cells``, ``placed_cells`` (hidden cells included), ``shares``,
    ``cell_weights`` and ``squeeze`` are means over the panels that supply
    state channels, and ``capacity`` the least of them. Also returns
    ``panel_cells`` and ``densities`` ``[D, P]`` and ``owners`` ``[D, P, H,
    W]`` (``visible_cells``).
    """
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    from NCA.inverse_design.density import CellDensity

    mask = np.asarray(mask, dtype=bool)
    targets = np.asarray(targets, dtype=float)
    key = jr.PRNGKey(seed)
    panels = len(library["panels"])

    @eqx.filter_jit
    def _kept_panel(gaps, ranks, density):
        return gaps.kept_ranks(ranks, density, key)

    # Panels whose cells show in the state (not the noise panels)
    noise = noise_panel_ids(library, schema, settings)
    shown = np.asarray([_p for _p in np.unique(channel_panels(library, schema)) if _p not in noise])
    wanted = 0.0 if by_share else float(targets.max()) / per_piece
    layer_cache = {}
    for _attempt in range(10):
        packed, gaps, layer_pieces, ranks, margin = placed_cell_texture(
            library, schema, mask.shape, settings, seed,
            highest_density=float(targets.max()) if by_share else 0.0,
            max_layers=max_layers, mask=mask, highest_cells=wanted, layer_cache=layer_cache,
        )
        ids, panel_of = cell_ids(ranks)
        padded = np.pad(mask, margin)
        top = min(float(gaps.max_density), 0.999)
        layers = int(gaps.layer_coverage.shape[0])
        rank_maps = jnp.asarray(ranks)

        def _found(densities):
            """``(kept, visible_cells)`` with one density per panel."""
            kept = np.concatenate([
                np.asarray(_kept_panel(gaps, rank_maps[:, _panel : _panel + 1], CellDensity(float(_d))))
                for _panel, _d in enumerate(densities)
            ], axis=1)
            return kept, visible_cells(ids, panel_of, kept, padded, settings.visible_fraction)

        def _values(found):
            return found["covered"] if by_share else found["cells"]

        _, full = _found(np.full(panels, top))
        least = float(np.min(_values(full)[shown]))
        # Hidden cells do not count, so more layers can be needed than placement expects
        if by_share or least >= wanted or layers >= max_layers or least <= 0.0:
            break
        wanted = 1.05 * wanted * float(np.sum(layer_pieces)) / least

    def _solve(target):
        """Lowest density per panel reaching ``target`` (pieces or covered share), by bisection."""
        low, high = np.full(panels, 1e-3), np.full(panels, top)
        for _ in range(24):
            middle = 0.5 * (low + high)
            value = _values(_found(middle)[1])
            reached = value >= target
            high, low = np.where(reached, middle, high), np.where(reached, low, middle)
            # Cell counts are whole numbers: the first densities reaching the target will do
            if not by_share and np.all((reached & (value < target + 1))[shown]):
                break
        return high

    densities = np.stack([_solve(_t if by_share else _t / per_piece) for _t in targets])
    results = [_found(_d) for _d in densities]
    crop = (Ellipsis, slice(margin, margin + mask.shape[0]), slice(margin, margin + mask.shape[1]))
    textures, _ = gaps.unpack(packed)
    areas = np.asarray([np.mean([np.count_nonzero(_m) for _m, _ in _cells]) for _, _cells in library["panels"]])
    found = [_f for _, _f in results]
    last = found[-1]
    return {
        "initial_states": np.stack([
            np.asarray(gaps.composite(textures, jnp.asarray(_kept[crop]))) * mask[None] for _kept, _ in results
        ]),
        "cell_weights": np.stack([_f["weights"][shown].mean(axis=0)[crop] * mask for _f in found]),
        "cells": np.asarray([_f["cells"][shown].mean() for _f in found]) * per_piece,
        "panel_cells": _panel_table(
            [(_p, np.asarray([_f["cells"][_p] for _f in found]) * per_piece) for _p in shown],
            panels, len(targets),
        ),
        "placed_cells": np.asarray([_f["kept_cells"][shown].mean() for _f in found]) * per_piece,
        "shares": np.asarray([_f["covered"][shown].mean() for _f in found]),
        "densities": densities,
        "capacity": float(np.min(_values(full)[shown])) * (1.0 if by_share else per_piece),
        "cell_area": float(np.mean(areas)),
        # Mean shown area of a counted cell at the highest density, over the panel's measured cell area
        "squeeze": float(np.mean((last["visible_area"] / np.maximum(last["cells"], 1.0) / areas)[shown])),
        "layers": layers,
        "owners": np.stack([_f["owners"][crop] for _f in found]),
    }
