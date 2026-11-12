# /// script
# dependencies = [
#   "marimo",
#   "jax",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "scikit-image",
# ]
# ///

"""Synthetic textured initial conditions: tune them on measured 0 h colonies and save them.

Run from the repository root with:

    marimo edit Experiments/micropatterns/synthetic_textured_ic_explorer.py

Data come from ``DATA_PATH_BASE/260726_nca_dataset`` (control colonies at
0 h). The settings are saved per projection to the texture settings file
(``texture_settings.py``), which inverse design, the density × KO sweep and
the mask explorers read. The generation code is ``synthetic_ic.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv

    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_projections import PROJECTIONS
    from Experiments.micropatterns.cell_counts import (
        default_alignment,
        reference_cell_counts,
        summarise_counts,
        um_per_pixel,
    )
    from Experiments.micropatterns.synthetic_ic import (
        cell_states,
        replicate_cell_library,
        synthesise_texture,
    )
    from Experiments.micropatterns.texture_settings import (
        DEFAULT_TEXTURE_SETTINGS_FILE,
        CellSettings,
        load_cell_settings,
        load_texture_settings,
        read_texture_settings_file,
        save_texture_settings,
        scaled_patch_geometry,
    )

    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env", override=False)
    DENSITY_UNITS = {
        "Cells per mm²": "per_mm2",
        "Cells in the colony": "per_colony",
        "Covered share": "share",
    }

    def circle_mask(shape, area):
        """Centred disc of about ``area`` pixels."""
        rows, cols = np.indices(shape)
        distance = np.hypot(rows - (shape[0] - 1) / 2, cols - (shape[1] - 1) / 2)
        return distance <= np.sqrt(area / np.pi)

    def radial_power(image, mask):
        """Radially averaged power spectrum of ``image`` inside ``mask`` (mean removed), summing to 1."""
        values = np.where(mask, image - image[mask].mean(), 0.0)
        power = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
        rows, cols = np.indices(power.shape)
        radius = np.hypot(rows - power.shape[0] // 2, cols - power.shape[1] // 2).astype(int)
        spectrum = np.bincount(radius.ravel(), power.ravel()) / np.maximum(np.bincount(radius.ravel()), 1)
        spectrum = spectrum[1 : min(power.shape) // 2]
        return spectrum / max(spectrum.sum(), 1e-12)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Synthetic textured initial conditions

    - Synthetic 0 h initial textures made from measured control colonies, compared with the data
    - **Texture**: patch resampling or IAAFT over the whole grid; keeps the measured gaps between cells
    - **Cells**: whole measured cells placed at chosen cell densities (tiled: no overlap; layers: may overlap)
    - **Save** writes both for the projection; used by inverse design (`texture`), the density × KO sweep (`cells`) and the mask explorers
    """)
    return


@app.cell(hide_code=True)
def _():
    _data_base = os.environ.get("DATA_PATH_BASE")
    data_root = mo.ui.text(
        str(Path(_data_base) / "260726_nca_dataset") if _data_base else "../Data/260726_nca_dataset",
        label="260726 dataset folder",
        full_width=True,
    )
    _default_file = DEFAULT_TEXTURE_SETTINGS_FILE
    if _default_file.is_relative_to(Path.cwd()):
        _default_file = _default_file.relative_to(Path.cwd())
    settings_path = mo.ui.text(str(_default_file), label="Texture settings file", full_width=True)
    projection = mo.ui.dropdown(list(PROJECTIONS), value="slice", label="Projection")
    downsample = mo.ui.dropdown([2, 4, 8, 16], value=8, label="Downsample")
    replicate_count = mo.ui.slider(1, 4, value=3, step=1, label="Replicates loaded")
    mo.vstack([
        data_root,
        settings_path,
        mo.hstack([projection, downsample, replicate_count], justify="start"),
    ])
    return data_root, downsample, projection, replicate_count, settings_path


@app.function(hide_code=True)
# Cached, so returning to settings already seen does not reload
@mo.lru_cache(maxsize=4)
def load_colonies(root, projection, downsample, replicates):
    """Control colonies at 0 h, as ``cell_library`` reads them, preprocessed as projection
    training runs are (the micropattern base config, cleaning included, which sum projections need)."""
    from Experiments.micropatterns.figures.data import load_dataset, projection_data_config

    dataset = load_dataset(
        root, projection_data_config(projection, int(downsample)), ("ctrl",), tuple(range(int(replicates))),
    )
    return {
        "measurements": np.asarray(dataset.data[:, 0]),
        "available": np.asarray(dataset.measurement_mask[:, 0], dtype=bool),
        "support": np.asarray(dataset.boundary_mask[:, 0], dtype=bool),
        "schema": dataset.aux["channel_schema"],
        "replicates": tuple(dataset.aux["batch_replicates"]),
        "source_files": np.asarray(dataset.aux["source_files"])[:, 0],
    }


@app.cell(hide_code=True)
def _(data_root, downsample, projection, replicate_count):
    with mo.status.spinner("Loading the 0 h control colonies"):
        colonies = load_colonies(
            str(Path(data_root.value).expanduser()), projection.value,
            int(downsample.value), int(replicate_count.value),
        )
    schema = colonies["schema"]
    state_names = tuple(schema.state_channels)
    primary = np.asarray(schema.primary_measurements, dtype=int)
    grid_shape = colonies["measurements"].shape[-2:]
    alignment = default_alignment()
    pixel_um = um_per_pixel(alignment.pattern_radius, int(downsample.value))
    # One circle for every synthetic colony, of the mean measured area
    colony = circle_mask(grid_shape, float(np.mean(colonies["support"].sum(axis=(1, 2)))))
    colony_mm2 = float(colony.sum()) * pixel_um**2 * 1e-6
    mo.md(
        f"- Replicates {', '.join(map(str, colonies['replicates']))}; grid {grid_shape[0]} × {grid_shape[1]}; "
        f"{pixel_um:.2f} µm per pixel; colony {colony_mm2:.3f} mm²"
    )
    return (
        colonies,
        colony,
        colony_mm2,
        grid_shape,
        pixel_um,
        primary,
        schema,
        state_names,
    )


@app.cell(hide_code=True)
def _(downsample, projection, settings_path):
    # Start values: the saved settings for the projection, else the defaults
    _path = Path(settings_path.value).expanduser()
    try:
        saved_texture = load_texture_settings(_path, projection.value)
        saved_cells = load_cell_settings(_path, projection.value)
        _note = read_texture_settings_file(_path)[projection.value].get("tuned_on")
        _view = mo.md(f"- Starting from the saved {projection.value} settings (tuned on: {_note})")
    except (KeyError, ValueError, OSError) as _error:
        saved_texture = {
            "mode": "patch", "patch_size": 11, "patch_overlap": 7, "patch_jitter": 3,
            "patch_downsample": None, "interior_fraction": 0.8, "candidates": 24,
            "iaaft_iterations": 25, "normalise": False, "normalise_iterations": 4,
        }
        saved_cells = CellSettings()
        _view = mo.callout(f"Starting from the defaults: {_error}", kind="warn")
    # Saved patch sizes at the loaded downsample
    saved_patch = scaled_patch_geometry(
        saved_texture["patch_size"], saved_texture["patch_overlap"], saved_texture["patch_jitter"],
        saved_texture["patch_downsample"], int(downsample.value),
    )
    _view
    return saved_cells, saved_patch, saved_texture


@app.cell(hide_code=True)
def _(saved_patch, saved_texture):
    texture_mode = mo.ui.dropdown(
        {"Patch resampling": "patch", "IAAFT": "iaaft"},
        value="IAAFT" if saved_texture["mode"] == "iaaft" else "Patch resampling",
        label="Texture",
    )
    patch_size = mo.ui.slider(3, 31, value=min(saved_patch[0], 31), step=2, label="Patch width (px)")
    patch_overlap = mo.ui.slider(0, 30, value=min(saved_patch[1], 30), step=1, label="Patch overlap (px)")
    patch_jitter = mo.ui.slider(0, 12, value=min(saved_patch[2], 12), step=1, label="Patch jitter (px)")
    patch_interior = mo.ui.slider(
        0.4, 1.0, value=saved_texture["interior_fraction"], step=0.01, label="Patch source radius"
    )
    patch_candidates = mo.ui.slider(1, 64, value=saved_texture["candidates"], step=1, label="Candidates")
    iaaft_iterations = mo.ui.slider(
        0, 100, value=5 * round(saved_texture["iaaft_iterations"] / 5), step=5, label="IAAFT iterations"
    )
    normalise = mo.ui.switch(value=saved_texture["normalise"], label="Match histograms + covariance")
    normalise_iterations = mo.ui.slider(
        1, 12, value=saved_texture["normalise_iterations"], step=1, label="Normalisation iterations"
    )
    mo.vstack([
        mo.md("### Texture (measured gaps)"),
        mo.hstack([texture_mode, patch_size, patch_overlap, patch_jitter], justify="start"),
        mo.hstack([patch_interior, patch_candidates, iaaft_iterations], justify="start"),
        mo.hstack([normalise, normalise_iterations], justify="start"),
    ])
    return (
        iaaft_iterations,
        normalise,
        normalise_iterations,
        patch_candidates,
        patch_interior,
        patch_jitter,
        patch_overlap,
        patch_size,
        texture_mode,
    )


@app.cell(hide_code=True)
def _(colonies, data_root, saved_cells, schema):
    from Experiments.micropatterns.nucleus_segmentation import stored_methods

    _cells = saved_cells
    _markers = sorted({_channel.marker for _channel in schema.measurement_channels})
    placement = mo.ui.dropdown(
        {"Tiled (no overlap)": "tiled", "Layers (may overlap)": "layers"},
        value="Tiled (no overlap)" if _cells.placement == "tiled" else "Layers (may overlap)",
        label="Placement",
    )
    pieces = mo.ui.dropdown(
        {"Real cells (nuclei)": "nuclei", "Segments (model resolution)": "segments"},
        value="Real cells (nuclei)" if _cells.pieces == "nuclei" else "Segments (model resolution)",
        label="Pieces",
    )
    markers = mo.ui.multiselect(
        _markers, value=[_m for _m in _cells.markers if _m in _markers], label="Cell markers"
    )
    split_distance = mo.ui.number(0, 20, value=_cells.split_distance, step=1, label="Split distance (px)")
    min_cell_area = mo.ui.number(1, 100, value=_cells.min_cell_area, step=1, label="Min piece area (px)")
    cell_interior = mo.ui.slider(
        0.3, 1.0, value=round(_cells.interior_fraction, 2), step=0.05, label="Piece source radius"
    )
    expansion_um = mo.ui.number(0.0, 10.0, value=_cells.expansion_um, step=0.5, label="Reach beyond nucleus (µm)")
    nucleus_diameter = mo.ui.number(
        2.0, 20.0, value=_cells.nucleus_diameter_um, step=0.5, label="Nucleus diameter (µm)"
    )
    nucleus_seed_spacing = mo.ui.slider(
        0.1, 1.0, value=min(max(_cells.nucleus_seed_spacing, 0.1), 1.0), step=0.05, show_value=True,
        label="Split nuclei closer than (× diameter)",
    )
    nucleus_threshold = mo.ui.slider(
        0.5, 1.5, value=min(max(_cells.nucleus_threshold, 0.5), 1.5), step=0.05, show_value=True,
        label="Stain threshold (× Otsu)",
    )
    _split_methods = {
        "Where blobs narrow (distance)": "distance",
        "Inside each LMBR rim (rim)": "rim",
        "Stain maxima (intensity)": "intensity",
    }
    nucleus_split_method = mo.ui.dropdown(
        _split_methods,
        value=next(_k for _k, _v in _split_methods.items() if _v == _cells.nucleus_split_method),
        label="Split touching nuclei",
    )
    nucleus_roundness = mo.ui.slider(
        0.0, 0.98, value=min(max(_cells.nucleus_roundness, 0.0), 0.98), step=0.01, show_value=True,
        label="Split again if less round than (0: off)",
    )
    # Stored segmentations (nucleus_segmentation.py) of the 0 h SOX2/LMBR images
    _stored = {"None (the nucleus settings)": "", **{
        _m: _m for _m in stored_methods(Path(data_root.value).expanduser())
    }}
    segmentation = mo.ui.dropdown(
        _stored,
        value=next((_k for _k, _v in _stored.items() if _v == _cells.segmentation), next(iter(_stored))),
        label="Stored segmentation (0 h SOX2/LMBR images)",
    )
    # Size limits, in areas of the set nucleus diameter (as the Nuclei histogram)
    nucleus_smallest = mo.ui.slider(
        0.0, 1.5, value=min(max(_cells.nucleus_smallest, 0.0), 1.5), step=0.05, show_value=True,
        label="Drop nuclei smaller than (× nucleus area)",
    )
    nucleus_largest = mo.ui.slider(
        0.0, 8.0, value=min(max(_cells.nucleus_largest, 0.0), 8.0), step=0.25, show_value=True,
        label="Split nuclei larger than (× nucleus area; 0: off)",
    )
    reach_scale = mo.ui.slider(
        0.3, 2.0, value=min(max(_cells.reach_scale, 0.3), 2.0), step=0.05, label="Cell reach (× measured radius)"
    )
    seed_spacing = mo.ui.slider(
        0.0, 1.0, value=min(max(_cells.seed_spacing, 0.0), 1.0), step=0.05, label="Seed spacing"
    )
    tile_factor = mo.ui.dropdown([1, 2, 4, 8], value=_cells.tile_factor, label="Fine grid factor")
    layers = mo.ui.number(1, 10, value=_cells.layers, step=1, label="Layers")
    attempts = mo.ui.number(1, 50, value=_cells.attempts, step=1, label="Attempts per cell")
    softness_cells = mo.ui.number(0.05, 5.0, value=_cells.softness_cells, step=0.05, label="Cut softness (cells)")
    size_bias = mo.ui.slider(
        0.0, 3.0, value=min(max(_cells.size_bias, 0.0), 3.0), step=0.1, label="Smaller cells in deeper layers"
    )
    visible_fraction = mo.ui.slider(
        0.1, 1.0, value=min(max(_cells.visible_fraction, 0.1), 1.0), step=0.05, label="Counted when this share shows"
    )
    noise_panels = mo.ui.multiselect(
        list(schema.group_names),
        value=[_p for _p in _cells.noise_panels if _p in schema.group_names],
        label="Noise panels (measured pixels, no cells)",
    )
    density_panels = mo.ui.multiselect(
        list(schema.group_names),
        value=[_p for _p in _cells.density_panels if _p in schema.group_names],
        label="Panels giving the measured density",
    )
    piece_replicate = mo.ui.dropdown(
        {"All replicates": -1, **{f"Replicate {_r}": _i for _i, _r in enumerate(colonies["replicates"])}},
        value="All replicates",
        label="Pieces from",
    )
    mo.vstack([
        mo.md(
            "### Cells (chosen density)\n\n"
            "- Each staining panel is placed on its own (imaged on different colonies, so cells do not line up) "
            "and reaches each density by itself"
        ),
        mo.hstack([placement, pieces, markers, piece_replicate], justify="start"),
        mo.hstack([noise_panels, density_panels], justify="start"),
        mo.hstack([split_distance, min_cell_area, cell_interior, expansion_um], justify="start"),
        mo.md("- Nuclei (cell numbers, and real-cell pieces; check them under Nuclei below):"),
        mo.hstack([nucleus_diameter, nucleus_seed_spacing, nucleus_threshold], justify="start"),
        mo.hstack([nucleus_split_method, nucleus_roundness], justify="start"),
        mo.hstack([nucleus_smallest, nucleus_largest], justify="start"),
        segmentation,
        mo.md("- Tiled only:"),
        mo.hstack([reach_scale, seed_spacing, tile_factor], justify="start"),
        mo.md("- Layers only:"),
        mo.hstack([layers, attempts, softness_cells], justify="start"),
        mo.hstack([size_bias, visible_fraction], justify="start"),
    ])
    return (
        attempts,
        cell_interior,
        density_panels,
        expansion_um,
        layers,
        markers,
        min_cell_area,
        noise_panels,
        nucleus_diameter,
        nucleus_largest,
        nucleus_roundness,
        nucleus_seed_spacing,
        nucleus_smallest,
        nucleus_split_method,
        nucleus_threshold,
        piece_replicate,
        pieces,
        placement,
        reach_scale,
        seed_spacing,
        segmentation,
        size_bias,
        softness_cells,
        split_distance,
        tile_factor,
        visible_fraction,
    )


@app.cell(hide_code=True)
def _():
    density_unit = mo.ui.dropdown(DENSITY_UNITS, value="Cells per mm²", label="Density as")
    return (density_unit,)


@app.cell(hide_code=True)
def _(density_unit):
    _start, _stop, _step = {
        "per_mm2": (1000.0, 8000.0, 100.0),
        "per_colony": (200.0, 1500.0, 10.0),
        "share": (0.1, 0.9, 0.01),
    }[density_unit.value]
    density_start = mo.ui.number(0.0, 50000.0, value=_start, step=_step, label="Lowest")
    density_stop = mo.ui.number(0.0, 50000.0, value=_stop, step=_step, label="Highest")
    density_count = mo.ui.number(1, 12, value=5, step=1, label="Densities")
    seed = mo.ui.number(0, 100000, value=0, step=1, label="Seed")
    mo.vstack([
        mo.md("### Densities"),
        mo.hstack([density_unit, density_start, density_stop, density_count, seed], justify="start"),
    ])
    return density_count, density_start, density_stop, seed


@app.cell(hide_code=True)
def _(
    attempts,
    cell_interior,
    density_panels,
    downsample,
    expansion_um,
    iaaft_iterations,
    layers,
    markers,
    min_cell_area,
    noise_panels,
    normalise,
    normalise_iterations,
    nucleus_diameter,
    nucleus_largest,
    nucleus_roundness,
    nucleus_seed_spacing,
    nucleus_smallest,
    nucleus_split_method,
    nucleus_threshold,
    patch_candidates,
    patch_interior,
    patch_jitter,
    patch_overlap,
    patch_size,
    pieces,
    placement,
    reach_scale,
    seed_spacing,
    segmentation,
    size_bias,
    softness_cells,
    split_distance,
    texture_mode,
    tile_factor,
    visible_fraction,
):
    # The current controls, as saved to the settings file
    texture_settings = {
        "mode": texture_mode.value,
        "patch_size": int(patch_size.value),
        "patch_overlap": int(patch_overlap.value),
        "patch_jitter": int(patch_jitter.value),
        # Pixel sizes are at the loaded downsample
        "patch_downsample": int(downsample.value),
        "interior_fraction": float(patch_interior.value),
        "candidates": int(patch_candidates.value),
        "iaaft_iterations": int(iaaft_iterations.value),
        "normalise": bool(normalise.value),
        "normalise_iterations": int(normalise_iterations.value),
    }
    cell_settings = None
    settings_error = None
    try:
        cell_settings = CellSettings(
            placement=placement.value,
            pieces=pieces.value,
            markers=tuple(markers.value),
            split_distance=int(split_distance.value),
            min_cell_area=int(min_cell_area.value),
            interior_fraction=float(cell_interior.value),
            expansion_um=float(expansion_um.value),
            nucleus_diameter_um=float(nucleus_diameter.value),
            nucleus_seed_spacing=float(nucleus_seed_spacing.value),
            nucleus_threshold=float(nucleus_threshold.value),
            nucleus_split_method=nucleus_split_method.value,
            nucleus_roundness=float(nucleus_roundness.value),
            nucleus_smallest=float(nucleus_smallest.value),
            nucleus_largest=float(nucleus_largest.value),
            segmentation=segmentation.value,
            reach_scale=float(reach_scale.value),
            seed_spacing=float(seed_spacing.value),
            tile_factor=int(tile_factor.value),
            layers=int(layers.value),
            attempts=int(attempts.value),
            softness_cells=float(softness_cells.value),
            size_bias=float(size_bias.value),
            visible_fraction=float(visible_fraction.value),
            noise_panels=tuple(noise_panels.value),
            density_panels=tuple(density_panels.value),
        )
    except ValueError as _error:
        settings_error = str(_error)
    mo.callout(settings_error, kind="danger") if settings_error else None
    return cell_settings, texture_settings


@app.cell(hide_code=True)
def _():
    @mo.lru_cache(maxsize=8)
    def piece_library(colony_key, index, piece_key, root, downsample):
        """``replicate_cell_library`` for the colonies ``load_colonies(*colony_key)``.

        Cached on the settings that change the pieces only.
        """
        _colonies = load_colonies(*colony_key)
        _settings = CellSettings(**dict(piece_key))
        return replicate_cell_library(
            _colonies, None if index < 0 else index, _settings,
            raw=None if _settings.pieces == "segments" else {
                "source_files": _colonies["source_files"],
                "dataset_root": Path(root),
                "alignment": default_alignment(),
                "downsample": downsample,
                "cache_dir": cache_dir_from_environment(),
            },
        )

    @mo.lru_cache(maxsize=8)
    def measured_counts(colony_key, root, markers, split_distance, min_area, nucleus_key):
        """Nuclei counted in the raw 0 h images (``reference_cell_counts``); ``nucleus_key``
        is ``CellSettings.nucleus_options`` as a tuple of pairs."""
        _colonies = load_colonies(*colony_key)
        _rows = reference_cell_counts(
            _colonies, _colonies["source_files"], Path(root), default_alignment(), markers,
            split_distance, min_area, cache_dir=cache_dir_from_environment(), **dict(nucleus_key),
        )
        return _rows

    return measured_counts, piece_library


@app.cell(hide_code=True)
def _(
    cell_settings,
    data_root,
    downsample,
    measured_counts,
    piece_library,
    piece_replicate,
    projection,
    replicate_count,
):
    mo.stop(cell_settings is None)
    _root = str(Path(data_root.value).expanduser())
    _colony_key = (_root, projection.value, int(downsample.value), int(replicate_count.value))
    _piece_key = tuple(
        (_name, getattr(cell_settings, _name))
        for _name in (
            "pieces", "markers", "split_distance", "min_cell_area", "interior_fraction", "expansion_um", "noise_panels",
            "nucleus_diameter_um", "nucleus_seed_spacing", "nucleus_threshold", "nucleus_split_method",
            "nucleus_roundness", "nucleus_smallest", "nucleus_largest", "density_panels", "segmentation",
        )
    )
    with mo.status.spinner("Cutting out cells"):
        library, measured_share = piece_library(
            _colony_key, int(piece_replicate.value), _piece_key, _root, int(downsample.value)
        )
    count_rows = count_summary = None
    count_error = None
    try:
        with mo.status.spinner("Counting nuclei in the raw 0 h images"):
            count_rows = measured_counts(
                _colony_key, _root, cell_settings.markers,
                cell_settings.split_distance, cell_settings.min_cell_area,
                tuple(cell_settings.nucleus_options().items()),
            )
        # The measured density comes from the density panels only (the table below shows every panel)
        count_summary = summarise_counts([
            _row for _row in count_rows if _row["panel"] in cell_settings.density_panels
        ])
    except (OSError, ValueError, KeyError) as _error:
        count_error = str(_error)
    _pieces = sum(len(_cells) for _, _cells in library["panels"])
    _area = np.mean([np.count_nonzero(_m) for _, _cells in library["panels"] for _m, _ in _cells])
    _lines = [
        f"- {_pieces} pieces over {len(library['panels'])} staining panels, mean area {_area:.1f} px",
        f"- Measured covered share: {measured_share:.3f}",
    ]
    if count_summary is not None:
        _lines.append(
            f"- Measured nuclei ({', '.join(cell_settings.density_panels)}): "
            f"{count_summary['cells_per_mm2']:.0f} per mm², "
            f"{count_summary['cells_per_segment']:.2f} per model-resolution segment"
        )
    mo.vstack([
        mo.md("\n".join(_lines)),
        mo.callout(f"No nuclear counts: {count_error}", kind="warn") if count_error else mo.md(""),
        mo.accordion({
            "Covered share per colony": mo.ui.table(library["coverage"], selection=None),
            **({"Nuclei per colony": mo.ui.table(count_rows, selection=None)} if count_rows else {}),
        }),
    ])
    return count_summary, library, measured_share


@app.cell(hide_code=True)
def _(colonies, schema):
    # Each measured panel of each loaded replicate has its own raw image
    _options = {
        f"Replicate {_r} · {_group}": (_i, _k)
        for _i, _r in enumerate(colonies["replicates"])
        for _k, _group in enumerate(schema.group_names)
        if str(colonies["source_files"][_i][_k])
    }
    nuclei_image = mo.ui.dropdown(_options, value=next(iter(_options)), label="Image")
    nuclei_flag = mo.ui.slider(
        0.5, 1.0, value=0.9, step=0.01, show_value=True, label="Outline in red if less round than"
    )
    mo.vstack([
        mo.md(
            "### Nuclei\n\n"
            "- The nuclei found in a raw nuclear stain (LMBR, or DAPI for the RNA panel) with the nucleus "
            "controls above; these give the measured cell numbers\n"
            "- Click or drag on the colony to move the zoom box; one colour per nucleus, so a merge shows as "
            "one colour over two nuclei\n"
            "- Roundness is solidity (area over convex hull area); nuclei are round, so low values are "
            "usually merged ones (only shown here; re-splitting is a nucleus control above)"
        ),
        mo.hstack([nuclei_image, nuclei_flag], justify="start"),
    ])
    return nuclei_flag, nuclei_image


@app.cell(hide_code=True)
def _(colonies, data_root, nuclei_image, schema):
    from Common.dataloader.micropattern_260726 import read_nuclear_channel
    from Experiments.micropatterns.box_picker import BoxPicker, png_data_url
    from Experiments.micropatterns.cell_counts import colony_mask

    # The stain alone (independent of the nucleus settings), so changing them keeps the box where it is
    _index, _panel = nuclei_image.value
    nuclei_path = str(colonies["source_files"][_index][_panel])
    nuclei_group = schema.group_names[_panel]
    with mo.status.spinner("Reading the nuclear stain"):
        _stain = read_nuclear_channel(nuclei_path, nuclei_group)
    _colony = colony_mask(nuclei_path, Path(data_root.value).expanduser(), default_alignment(), _stain.shape)
    _rows, _cols = np.flatnonzero(_colony.any(axis=1)), np.flatnonzero(_colony.any(axis=0))
    # Cropped to the colony: stain scaled to [0, 1] inside it, and the colony mask
    nuclei_crop = (slice(_rows[0], _rows[-1] + 1), slice(_cols[0], _cols[-1] + 1))
    _low, _high = np.percentile(_stain[_colony], [1.0, 99.5])
    nuclei_grey = np.clip((_stain[nuclei_crop] - _low) / max(_high - _low, 1e-12), 0.0, 1.0)
    nuclei_colony = _colony[nuclei_crop]
    nuclei_pixel_um = um_per_pixel(default_alignment().pattern_radius)
    nuclei_picker = mo.ui.anywidget(BoxPicker(
        background=png_data_url(np.where(nuclei_colony, nuclei_grey, 0.35 * nuclei_grey)),
        image_um=nuclei_grey.shape[1] * nuclei_pixel_um,
    ))
    return (
        nuclei_colony,
        nuclei_crop,
        nuclei_grey,
        nuclei_group,
        nuclei_path,
        nuclei_picker,
        nuclei_pixel_um,
    )


@app.cell(hide_code=True)
def _(cell_settings, data_root, nuclei_crop, nuclei_group, nuclei_path):
    from Experiments.micropatterns.cell_counts import colony_nuclei

    mo.stop(cell_settings is None)
    with mo.status.spinner("Finding nuclei"):
        _labels, _, _ = colony_nuclei(
            nuclei_path, nuclei_group, Path(data_root.value).expanduser(), default_alignment(),
            cache_dir=cache_dir_from_environment(), **cell_settings.nucleus_options(),
        )
    from skimage.measure import regionprops

    nuclei_labels = _labels[nuclei_crop]
    # Solidity of nucleus k at index k - 1
    nuclei_solidity = np.asarray([_region.solidity for _region in regionprops(nuclei_labels)])
    return nuclei_labels, nuclei_solidity


@app.cell(hide_code=True)
def _(
    cell_settings,
    nuclei_colony,
    nuclei_flag,
    nuclei_grey,
    nuclei_image,
    nuclei_labels,
    nuclei_picker,
    nuclei_pixel_um,
    nuclei_solidity,
):
    from skimage.segmentation import find_boundaries

    _areas = np.bincount(nuclei_labels.ravel())[1:]
    _count = len(_areas)
    _expected = np.pi * (cell_settings.nucleus_diameter_um / 2.0 / nuclei_pixel_um) ** 2
    _not_round = np.flatnonzero(nuclei_solidity < nuclei_flag.value) + 1
    # The zoomed region at full resolution: each nucleus filled in its own colour, outlined
    # in that colour, or in red if it is not round
    _box = nuclei_picker.value
    _half = 0.5 * float(_box["size_um"]) / nuclei_pixel_um
    _height, _width = nuclei_grey.shape
    _cx, _cy = float(_box["x"]) * _width, float(_box["y"]) * _height
    _x0, _x1 = int(max(_cx - _half, 0)), int(min(_cx + _half, _width))
    _y0, _y1 = int(max(_cy - _half, 0)), int(min(_cy + _half, _height))
    _labels = nuclei_labels[_y0:_y1, _x0:_x1]
    _palette = np.random.default_rng(0).uniform(0.25, 1.0, size=(_count + 1, 3))
    _rgb = np.repeat(nuclei_grey[_y0:_y1, _x0:_x1, None], 3, axis=-1)
    _inside = _labels > 0
    _rgb[_inside] = 0.7 * _rgb[_inside] + 0.3 * _palette[_labels[_inside]]
    _edges = find_boundaries(_labels, mode="inner")
    _rgb[_edges] = _palette[_labels[_edges]]
    _rgb[_edges & np.isin(_labels, _not_round)] = (1.0, 0.1, 0.1)
    _rgb[~nuclei_colony[_y0:_y1, _x0:_x1]] *= 0.35

    _figure, _axes = plt.subplots(
        1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.2, 0.8, 0.8]}
    )
    _axes[0].imshow(_rgb, interpolation="nearest")
    _axes[0].set_title(f"{nuclei_image.selected_key}, {float(_box['size_um']):.0f} µm across", fontsize=9)
    _axes[0].set_xticks([])
    _axes[0].set_yticks([])
    _top = max(5.0, 1.5 * cell_settings.nucleus_largest)
    _axes[1].hist(_areas / _expected, bins=np.linspace(0, _top, 51), color="#555555")
    _axes[1].axvline(1.0, color="#0072B2", lw=1, label="set nucleus diameter")
    _axes[1].axvline(cell_settings.nucleus_smallest, color="#888888", lw=1, ls=":", label="dropped below")
    if cell_settings.nucleus_largest > 0:
        _axes[1].axvline(cell_settings.nucleus_largest, color="#D55E00", lw=1, ls=":", label="split above")
    _axes[1].set_xlabel("Nucleus area / area of the set diameter")
    _axes[1].set_ylabel("Nuclei")
    _axes[1].legend(fontsize=7)
    _axes[2].hist(nuclei_solidity, bins=np.linspace(0.5, 1.0, 51), color="#555555")
    _axes[2].axvline(nuclei_flag.value, color="#D55E00", lw=1, ls="--", label="red outline below")
    if cell_settings.nucleus_roundness > 0:
        _axes[2].axvline(cell_settings.nucleus_roundness, color="#0072B2", lw=1, label="split again below")
    _axes[2].set_xlabel("Roundness (solidity)")
    _axes[2].legend(fontsize=7)
    _figure.tight_layout()
    _colony_mm2 = float(nuclei_colony.sum()) * nuclei_pixel_um**2 * 1e-6
    mo.vstack([
        mo.hstack([nuclei_picker, _figure], justify="start", align="center"),
        mo.md(
            f"- {_count} nuclei in {_colony_mm2:.3f} mm²: {_count / _colony_mm2:.0f} per mm² "
            f"({cell_settings.nucleus_split_method} split"
            + (f", split again below roundness {cell_settings.nucleus_roundness:.2f}"
               if cell_settings.nucleus_roundness > 0 else "")
            + ")\n"
            f"- Mean nucleus area {np.mean(_areas) * nuclei_pixel_um**2:.0f} µm² "
            f"(set diameter gives {_expected * nuclei_pixel_um**2:.0f} µm²); "
            f"{len(_not_round)} less round than {nuclei_flag.value:.2f} ({len(_not_round) / max(_count, 1):.0%})"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    cell_settings,
    colonies,
    colony,
    colony_mm2,
    count_summary,
    density_count,
    density_start,
    density_stop,
    density_unit,
    downsample,
    grid_shape,
    library,
    measured_share,
    primary,
    schema,
    seed,
    texture_settings,
):
    unit = density_unit.value
    densities = np.linspace(float(density_start.value), float(density_stop.value), int(density_count.value))
    mo.stop(
        unit != "share" and count_summary is None,
        mo.callout("Densities in cells need the nuclear counts; use the covered share.", kind="warn"),
    )
    # Real cells per placed piece: 1 for real cells, else measured per segment
    per_piece = library.get(
        "cells_per_segment", None if count_summary is None else count_summary["cells_per_segment"]
    )
    mo.stop(per_piece is None, mo.callout("Segments need the nuclear counts.", kind="warn"))
    measured_density = {
        "share": measured_share,
        "per_mm2": None if count_summary is None else count_summary["cells_per_mm2"],
        "per_colony": None if count_summary is None else count_summary["cells_per_mm2"] * colony_mm2,
    }[unit]
    with mo.status.spinner("Placing cells"):
        placed = cell_states(
            library, schema, colony,
            densities if unit == "share" else densities * (colony_mm2 if unit == "per_mm2" else 1.0),
            unit == "share", int(seed.value), cell_settings, per_piece,
        )
    with mo.status.spinner("Synthesising the texture"):
        texture = synthesise_texture(
            colonies, texture_settings, tuple(grid_shape), int(seed.value), int(downsample.value)
        )[primary] * colony[None]
    return densities, measured_density, placed, texture, unit


@app.cell(hide_code=True)
def _(state_names):
    channel = mo.ui.dropdown(
        list(state_names), value="SOX2" if "SOX2" in state_names else state_names[0], label="Channel"
    )
    channel
    return (channel,)


@app.cell(hide_code=True)
def _(
    channel,
    colonies,
    colony,
    densities,
    placed,
    primary,
    state_names,
    texture,
    texture_settings,
    unit,
):
    # One channel: measured colonies, the texture and the cells at each density, on one colour scale
    _c = state_names.index(channel.value)
    _m = int(primary[_c])
    _measured = [
        (np.where(_support, _image[_m], np.nan), f"Measured · replicate {_r}")
        for _image, _support, _available, _r in zip(
            colonies["measurements"], colonies["support"], colonies["available"], colonies["replicates"]
        )
        if _available[_m]
    ]
    _values = np.concatenate([_image[np.isfinite(_image)] for _image, _ in _measured]) if _measured else np.zeros(1)
    _low, _high = np.percentile(_values, [1, 99])
    _texture_label = (
        f"IAAFT {texture_settings['iaaft_iterations']}" if texture_settings["mode"] == "iaaft"
        else f"patch {texture_settings['patch_size']} px"
    )
    _synthetic = [(np.where(colony, texture[_c], np.nan), f"Texture ({_texture_label})")]
    _format = "{:.2f}" if unit == "share" else "{:.0f}"
    _synthetic += [
        (
            np.where(colony, _state[_c], np.nan),
            f"{_format.format(_d)} · {_cells:.0f} cells · share {_share:.2f}",
        )
        for _d, _state, _cells, _share in zip(densities, placed["initial_states"], placed["cells"], placed["shares"])
    ]
    _columns = max(len(_measured), len(_synthetic))
    _figure, _axes = plt.subplots(2, _columns, figsize=(4.4 * _columns, 9.2), squeeze=False,dpi=300)
    _cmap = plt.get_cmap("magma").copy()
    _cmap.set_bad("#d9d9d9")
    for _row, _panels in enumerate((_measured, _synthetic)):
        for _column in range(_columns):
            _axis = _axes[_row, _column]
            _axis.set_xticks([])
            _axis.set_yticks([])
            if _column >= len(_panels):
                _axis.axis("off")
                continue
            _image, _title = _panels[_column]
            _axis.imshow(_image, cmap=_cmap, vmin=_low, vmax=_high, origin="lower")
            _axis.set_title(_title, fontsize=8)
    _figure.suptitle(f"{channel.value} · colour scale: 1st–99th percentile of the measured colonies", fontsize=10)
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(densities, measured_density):
    # Which synthetic image to show with every channel
    _options = {"Texture": -1}
    _options.update({f"Density {_i + 1} ({_d:g})": _i for _i, _d in enumerate(densities)})
    _closest = (
        int(np.argmin(np.abs(densities - measured_density))) if measured_density is not None else 0
    )
    all_channels_choice = mo.ui.dropdown(
        _options, value=list(_options)[_closest + 1], label="Synthetic image (starts nearest the measured density)"
    )
    replicate_choice = mo.ui.number(1, 4, value=1, step=1, label="Measured replicate (index)")
    mo.hstack([all_channels_choice, replicate_choice], justify="start")
    return all_channels_choice, replicate_choice


@app.cell(hide_code=True)
def _(
    all_channels_choice,
    colonies,
    colony,
    placed,
    primary,
    replicate_choice,
    state_names,
    texture,
):
    # Every channel: one measured colony (grey: not measured) and one synthetic image
    _index = min(int(replicate_choice.value), len(colonies["replicates"])) - 1
    _choice = int(all_channels_choice.value)
    _synthetic = texture if _choice < 0 else placed["initial_states"][_choice]
    _figure, _axes = plt.subplots(2, len(state_names), figsize=(2.0 * len(state_names), 4.4), squeeze=False)
    _cmap = plt.get_cmap("magma").copy()
    _cmap.set_bad("#d9d9d9")
    for _c, _name in enumerate(state_names):
        _m = int(primary[_c])
        _support = colonies["support"][_index]
        _measured = np.where(_support, colonies["measurements"][_index, _m], np.nan)
        if not colonies["available"][_index, _m]:
            _measured = np.full_like(_measured, np.nan)
        _finite = colonies["measurements"][:, _m][colonies["support"] & colonies["available"][:, _m, None, None]]
        _low, _high = np.percentile(_finite, [1, 99]) if _finite.size else (0.0, 1.0)
        for _row, _image in enumerate((_measured, np.where(colony, _synthetic[_c], np.nan))):
            _axes[_row, _c].imshow(_image, cmap=_cmap, vmin=_low, vmax=_high, origin="lower")
            _axes[_row, _c].set_xticks([])
            _axes[_row, _c].set_yticks([])
        _axes[0, _c].set_title(_name, fontsize=9)
    _axes[0, 0].set_ylabel(f"Replicate {colonies['replicates'][_index]}")
    _axes[1, 0].set_ylabel(all_channels_choice.selected_key)
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(all_channels_choice, cell_settings, colony, placed, schema):
    # Layers: the cells counted in the chosen image, one colour each, per staining panel (not the noise panels)
    _choice = int(all_channels_choice.value)
    mo.stop("owners" not in placed or _choice < 0)
    _panels = [_i for _i, _name in enumerate(schema.group_names) if _name not in cell_settings.noise_panels]
    _owners = placed["owners"][_choice][_panels]
    _rng = np.random.default_rng(0)
    _palette = _rng.uniform(0.15, 0.95, size=(int(_owners.max()) + 1, 3))
    _figure, _axes = plt.subplots(1, len(_owners), figsize=(2.6 * len(_owners), 2.9), squeeze=False)
    for _axis, _map, _name, _count in zip(
        _axes[0], _owners, [schema.group_names[_i] for _i in _panels], placed["panel_cells"][_choice][_panels]
    ):
        _rgb = np.ones((*_map.shape, 3))
        _rgb[_map >= 0] = _palette[_map[_map >= 0]]
        _rgb[_map == -2] = 0.55
        _rgb[~colony] = 0.85
        _axis.imshow(_rgb, origin="lower")
        _axis.set_title(f"{_name}: {_count:.0f} cells", fontsize=8)
        _axis.set_xticks([])
        _axis.set_yticks([])
    _figure.tight_layout()
    mo.vstack([
        _figure,
        mo.md(
            "- Colours: counted cells; dark grey: shows too little to count (mostly hidden or faded); white: gaps"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    all_channels_choice,
    channel,
    colonies,
    colony,
    pixel_um,
    placed,
    primary,
    state_names,
    texture,
):
    # Value histograms and radial power spectra of the chosen channel inside the colony
    _c = state_names.index(channel.value)
    _m = int(primary[_c])
    _choice = int(all_channels_choice.value)
    _curves = [
        (f"Measured · replicate {_r}", _image[_m], _support)
        for _image, _support, _available, _r in zip(
            colonies["measurements"], colonies["support"], colonies["available"], colonies["replicates"]
        )
        if _available[_m]
    ]
    _curves.append(("Texture", texture[_c], colony))
    if _choice >= 0:
        _curves.append((all_channels_choice.selected_key, placed["initial_states"][_choice][_c], colony))
    _figure, (_hist_axis, _power_axis) = plt.subplots(1, 2, figsize=(11, 3.6))
    _bins = np.linspace(
        min(float(_v[_s].min()) for _, _v, _s in _curves), max(float(_v[_s].max()) for _, _v, _s in _curves), 50
    )
    for _label, _image, _mask in _curves:
        _style = "--" if _label.startswith("Measured") else "-"
        _hist_axis.hist(_image[_mask], bins=_bins, density=True, histtype="step", linestyle=_style, label=_label)
        _spectrum = radial_power(_image, _mask)
        _frequency = np.arange(1, len(_spectrum) + 1) / (_image.shape[0] * pixel_um)
        _power_axis.loglog(_frequency, _spectrum, _style, label=_label)
    _hist_axis.set_xlabel(f"{channel.value} value")
    _hist_axis.set_ylabel("Density")
    _power_axis.set_xlabel("Spatial frequency (cycles / µm)")
    _power_axis.set_ylabel("Share of power")
    _power_axis.legend(fontsize=7)
    _figure.tight_layout()
    _figure
    return


@app.cell(hide_code=True)
def _(colony_mm2, densities, measured_density, placed, unit):
    # The densities asked for and reached
    # Layers: cells that show, and the cells kept with hidden ones included
    _placed = placed.get("placed_cells", [None] * len(densities))
    _rows = [
        {
            "asked": float(_d),
            "cells in colony": float(_cells),
            "cells per mm²": float(_cells) / colony_mm2,
            "covered share": float(_share),
            **({} if _kept is None else {"placed (incl. hidden)": float(_kept)}),
        }
        for _d, _cells, _share, _kept in zip(densities, placed["cells"], placed["shares"], _placed)
    ]
    _capacity = placed["capacity"] / (colony_mm2 if unit == "per_mm2" else 1.0)
    _lines = [f"- The placement holds at most {_capacity:.3g} ({unit})"]
    if measured_density is not None:
        _lines.append(f"- Measured: {measured_density:.3g} ({unit})")
    if np.isfinite(placed["squeeze"]):
        _lines.append(f"- At the highest density cells have {placed['squeeze']:.0%} of the measured cell area")
    if placed["layers"]:
        _lines.append(f"- {placed['layers']} layers")
    mo.vstack([mo.md("\n".join(_lines)), mo.ui.table(_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(projection):
    save_button = mo.ui.run_button(label=f"Save settings for {projection.value}")
    return (save_button,)


@app.cell(hide_code=True)
def _(
    cell_settings,
    downsample,
    piece_replicate,
    projection,
    replicate_count,
    save_button,
    seed,
    settings_path,
    texture_settings,
):
    _path = Path(settings_path.value).expanduser()
    _notes = []
    if save_button.value and cell_settings is not None:
        save_texture_settings(
            _path, projection.value, texture_settings, cell_settings,
            note={
                "condition": "ctrl",
                "downsample": int(downsample.value),
                "replicates": int(replicate_count.value),
                "pieces_from": piece_replicate.selected_key,
                "seed": int(seed.value),
            },
        )
        _notes.append(mo.callout(f"Saved the {projection.value} settings to `{_path}`", kind="success"))
    try:
        _saved = read_texture_settings_file(_path)
    except ValueError as _error:
        _saved = {}
        _notes.append(mo.callout(str(_error), kind="danger"))
    _rows = [
        {
            "projection": _name,
            "texture": _section.get("texture", {}).get("mode"),
            "placement": _section.get("cells", {}).get("placement"),
            "pieces": _section.get("cells", {}).get("pieces"),
            "tuned at downsample": _section.get("tuned_on", {}).get("downsample"),
            "saved": _section.get("tuned_on", {}).get("saved"),
        }
        for _name, _section in _saved.items()
    ]
    mo.vstack([
        mo.md(
            "### Save\n\n"
            "- Writes the texture and cells controls above for this projection\n"
            "- Pixel sizes are saved at the loaded downsample; patch sizes are rescaled where used"
        ),
        save_button,
        *_notes,
        mo.ui.table(_rows, selection=None),
    ])
    return



@app.cell(hide_code=True)
def _():
    check_store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(repository_root / "models")), label="Model store", full_width=True
    )
    check_selection = mo.ui.text(
        "", label="Model selection YAML (from model_registry_explorer.py)", full_width=True
    )
    mo.vstack([
        mo.md(
            "## Sanity check with a trained NCA\n\n"
            "- Rolls a model out from the true 0 h state of one colony, and from a synthetic one made with the "
            "cells controls above at that colony's measured cell density (its nuclei)\n"
            "- Data are loaded and scaled as the model was trained; both rollouts use the same seed"
        ),
        check_store_root,
        check_selection,
    ])
    return check_selection, check_store_root


@app.cell(hide_code=True)
def _(check_selection, check_store_root, projection):
    from Experiments.micropatterns.figures.data import load_selection_records

    mo.stop(not check_selection.value.strip(), mo.md("- Give a model selection to run the check"))
    _path = Path(check_selection.value.strip()).expanduser()
    _path = _path if _path.is_absolute() else repository_root / _path
    _records = [
        _r for _r in load_selection_records(_path, check_store_root.value.strip())
        if _r["projection"] == projection.value
    ]
    mo.stop(not _records, mo.callout(f"The selection has no {projection.value} models.", kind="warn"))
    check_records = {_r["model_id"]: _r for _r in _records}
    check_model = mo.ui.dropdown(
        {_r["option_label"]: _r["model_id"] for _r in _records}, value=_records[0]["option_label"], label="Model"
    )
    check_condition = mo.ui.dropdown(
        {"Control": "baseline", "NODAL KO from 24 h": "ko_24h", "NODAL KO from 0 h": "ko_0h"},
        value="Control",
        label="Condition",
    )
    check_replicate = mo.ui.number(1, 8, value=1, step=1, label="Replicate")
    check_seed = mo.ui.number(0, 100000, value=0, step=1, label="Rollout seed")
    run_check = mo.ui.run_button(label="Run both rollouts")
    mo.hstack([check_model, check_condition, check_replicate, check_seed, run_check], justify="start")
    return check_condition, check_model, check_records, check_replicate, check_seed, run_check


@app.cell(hide_code=True)
def _(
    cell_settings,
    check_condition,
    check_model,
    check_records,
    check_replicate,
    check_seed,
    check_store_root,
    data_root,
    downsample,
    replicate_count,
    run_check,
    seed,
):
    import jax.random as jr

    from Experiments.micropatterns.density_ko_sweep import (
        bundle_alignment,
        load_condition_data,
        load_training_histogram_bins,
        replicate_cell_counts,
        stacked_colonies,
    )
    from Experiments.micropatterns.ko_evaluation import evaluate_condition
    from Experiments.model_registry import ModelRegistry

    mo.stop(not run_check.value, mo.md("- Pick a model, condition and replicate, then run"))
    mo.stop(cell_settings is None, mo.callout("Fix the cells controls first.", kind="warn"))
    _root = Path(data_root.value).expanduser().parent
    _dataset = _root / "260726_nca_dataset"
    _bundle = ModelRegistry(Path(check_store_root.value.strip()).expanduser()).get(check_model.value)
    _model = _bundle.load_model(key=jr.PRNGKey(int(check_seed.value)), implementation="portable")
    _replicate = int(check_replicate.value)
    _condition = check_condition.value
    with mo.status.spinner("Loading the colonies as the model was trained"):
        _bins = load_training_histogram_bins(_bundle, _root)
        _loaded = {}
        # The other replicates give pieces for staining panels the chosen one did not measure
        for _r in sorted({*range(1, int(replicate_count.value) + 1), _replicate}):
            try:
                _loaded[_r] = load_condition_data(_bundle, _root, _condition, _r, _bins)
            except (IndexError, KeyError, ValueError, OSError):
                if _r == _replicate:
                    raise
    _data = _loaded[_replicate]
    _replicates = tuple(_loaded)
    _colonies = stacked_colonies([_loaded[_r] for _r in _replicates], _replicates)
    _alignment, _radius_scale, _pixel_um = bundle_alignment(_bundle)
    with mo.status.spinner("Cutting out cells and counting nuclei"):
        _library, _ = replicate_cell_library(
            _colonies, _replicates.index(_replicate), cell_settings,
            raw=None if cell_settings.pieces == "segments" else {
                "source_files": _colonies["source_files"],
                "dataset_root": _dataset,
                "alignment": _alignment,
                "downsample": int(_bundle.config.data.downsample),
                "mask_radius_scale": _radius_scale,
                "cache_dir": cache_dir_from_environment(),
            },
        )
        _counts = replicate_cell_counts(
            reference_cell_counts(
                _colonies, _colonies["source_files"], _dataset, _alignment, cell_settings.markers,
                cell_settings.split_distance, cell_settings.min_cell_area,
                mask_radius_scale=_radius_scale, cache_dir=cache_dir_from_environment(),
                **cell_settings.nucleus_options(),
            ),
            _replicate,
            cell_settings.density_panels,
        )
    # The synthetic colony has the true colony's outline and its measured cell density
    _mask = np.asarray(_data["boundary"]).squeeze() > 0.5
    _target = _counts["cells_per_mm2"] * float(_mask.sum()) * _pixel_um**2 * 1e-6
    with mo.status.spinner("Placing cells"):
        _placed = cell_states(
            _library, _colonies["schema"], _mask, [_target], False, int(seed.value), cell_settings,
            _library.get("cells_per_segment", _counts["cells_per_segment"]),
        )
    with mo.status.spinner("Rolling out the NCA from both initial states"):
        _true, _labels = evaluate_condition(_model, _bundle, _data, _condition, int(check_seed.value))
        _synthetic, _ = evaluate_condition(
            _model, _bundle, {**_data, "initial_state": _placed["initial_states"][0]}, _condition,
            int(check_seed.value),
        )
    _common = {
        "target": np.asarray(_data["target"]),
        "measured": np.asarray(_data["measured"]),
        "channel_names": tuple(_data["channel_names"]),
        "time_labels": tuple(_labels),
        "boundary": _mask,
    }
    check_rollouts = {
        "true": {**_common, "prediction": np.asarray(_true)},
        "synthetic": {**_common, "prediction": np.asarray(_synthetic)},
    }
    check_label = f"Replicate {_replicate}\n{check_condition.selected_key}"
    _record = check_records[check_model.value]
    _notes = []
    if int(_record["downsample"]) != int(downsample.value):
        _notes.append(mo.callout(
            f"The model uses downsample {_record['downsample']}, the cells controls were set at "
            f"{downsample.value}: pixel sizes (split distance, min area) may not carry over.",
            kind="warn",
        ))
    mo.vstack([
        mo.md(
            f"- Measured density of replicate {_replicate} ({', '.join(cell_settings.density_panels)}): "
            f"{_counts['cells_per_mm2']:.0f} cells per mm² "
            f"({_target:.0f} cells in its colony)\n"
            f"- Synthetic initial state: {_placed['cells'][0]:.0f} cells ({cell_settings.placement}), "
            f"covered share {_placed['shares'][0]:.2f}"
        ),
        *_notes,
    ])
    return check_label, check_rollouts


@app.cell(hide_code=True)
def _(check_rollouts):
    _labels = list(check_rollouts["true"]["time_labels"])
    check_times = mo.ui.multiselect(_labels, value=_labels, label="Timesteps")
    check_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    check_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    mo.hstack([check_times, check_gamma, check_size], justify="start")
    return check_gamma, check_size, check_times


@app.cell(hide_code=True)
def _(check_gamma, check_label, check_rollouts, check_size, check_times):
    from Experiments.micropatterns.figures.drawing import trajectory_figure
    from Experiments.micropatterns.figures.style import ACCENT, DATA, INK, MODEL, paper_style

    mo.stop(not check_times.value, mo.callout("Pick at least one timestep.", kind="warn"))
    _labels = list(check_rollouts["true"]["time_labels"])
    _keep = [_i for _i, _t in enumerate(_labels) if _t in check_times.value]

    def _at_times(rollout):
        return {
            **rollout,
            **{_key: rollout[_key][_keep] for _key in ("prediction", "target", "measured")},
            "time_labels": tuple(_labels[_i] for _i in _keep),
        }

    _rollouts = {_key: _at_times(_rollout) for _key, _rollout in check_rollouts.items()}
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _rollouts["true"]["channel_names"] if "/" in _n))
    # One block of rows: (rollout, source, row label, row label colour)
    _blocks = [(check_label, INK, [
        ("true", "target", "Data", DATA),
        ("true", "prediction", "NCA (true IC)", MODEL),
        ("synthetic", "prediction", "NCA (synthetic IC)", ACCENT),
    ])]
    with paper_style():
        _figure = trajectory_figure(_rollouts, _blocks, _groups, check_gamma.value, check_size.value, True)
    _figure
    return


if __name__ == "__main__":
    app.run()
