"""Schematic of the NODAL knockout problem, drawn with real micropattern data.

Run with:
    marimo edit Experiments/micropatterns/nodal_ko_schematic.py

The dataset root defaults to ``$DATA_PATH_BASE/260726_nca_dataset``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="nodal knockout schematic")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv
    from matplotlib.colors import is_color_like
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Circle, FancyBboxPatch, Wedge

    load_dotenv()

    # Channel colours, added like light on a black background
    COLOURS = {
        "off": None,
        "cyan": (0.0, 0.78, 0.9),
        "magenta": (0.92, 0.12, 0.72),
        "yellow": (1.0, 0.78, 0.0),
        "red": (0.9, 0.15, 0.1),
        "green": (0.15, 0.8, 0.2),
        "blue": (0.15, 0.35, 1.0),
        "grey": (0.6, 0.6, 0.6),
        "white": (1.0, 1.0, 1.0),
    }
    MARKER_COLOURS = {
        "SOX2": "cyan", "SOX17": "yellow", "TBXT": "magenta", "FOXA2": "cyan", "LMBR": "off",
        "CER1": "yellow", "LEFTY": "cyan", "NODAL": "magenta", "LEF1": "green", "SMAD23": "magenta",
    }

    THEMES = {
        "light": {"bg": "white", "fg": "#222222", "muted": "#9a9a9a", "accent": "#d62728", "model": "#1f77b4"},
        "dark": {"bg": "#111111", "fg": "#eeeeee", "muted": "#777777", "accent": "#ff5a5a", "model": "#4fa3e0"},
    }

    # Knockout hour of each condition (None: control). Before the knockout the
    # loader substitutes the control images (see source_condition in
    # Common/dataloader/micropattern_260726.py).
    KO_HOURS = {"ctrl": None, "sl24": 24, "sl0": 0}
    TIMEPOINTS = (0, 12, 24, 36, 48)
    EXPERIMENT_GROUPS = ("cell_fate_s1", "cell_fate_s2", "rna_expression", "protein_response")
    GROUP_LABELS = {"cell_fate_s1": "Fate S1", "cell_fate_s2": "Fate S2", "rna_expression": "RNA",
                    "protein_response": "Protein"}

    # Per-projection cleaning, copied from
    # conf/experiments/micropattern_projections_training_sweep.yaml
    PROJECTION_CLEANING = {
        "max": {"horizontal_stripe_widths": {}, "stripe_block": 0},
        "sum": {"horizontal_stripe_widths": {"cell_fate_s1/TBXT": 10, "cell_fate_s2/TBXT": 10}, "stripe_block": 270},
        "slice": {"horizontal_stripe_widths": {}, "stripe_block": 0},
    }


@app.function(hide_code=True)
def knockout_data_config(groups, projection, downsample, normalisation=None):
    """Typed data config: the micropattern base config (the standard preprocessing) with the
    given experiment groups, projection and downsampling. ``normalisation`` None keeps the
    base config's."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.micropatterns.config_helpers import REPOSITORY_ROOT, measurement_count

    base = OmegaConf.load(REPOSITORY_ROOT / "Experiments" / "micropatterns" / "conf" / "base_config.yaml")
    channels = measurement_count(groups, base.data.micropattern.excluded_measurements)
    overrides = {"data": {
        "dataset": "micropatterns_260726",
        "downsample": downsample,
        "micropattern": {
            "data_channels": channels,
            "experiment_groups": list(groups),
            "projection": projection,
            "cleaning": {**PROJECTION_CLEANING[projection],
                         **({} if normalisation is None else {"normalisation": normalisation})},
        },
    }}
    return load_experiment_config(OmegaConf.merge(base, overrides)).data


@app.function(hide_code=True)
def training_data_config(path, index=0):
    """Data config of a training run: a model bundle folder, a generated
    ``manifest.yaml`` (entry ``index``, as ``run_config.py --index``) or a single config file."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.config_workflow import load_config_from_entry, resolve_manifest_index
    from Experiments.model_registry import open_model_bundle

    path = Path(path).expanduser()
    if path.is_dir():
        return open_model_bundle(path).config.data
    cfg = OmegaConf.load(path)
    if "configs" in cfg:
        manifest = OmegaConf.to_container(cfg, resolve=True)
        _, entry = resolve_manifest_index(manifest, index=index, worker_index=0, worker_count=1)
        cfg = load_config_from_entry(entry, manifest_dir=path.parent)
    return load_experiment_config(cfg).data


@app.function(hide_code=True)
def training_bounds(data_config, root):
    """Clipping bounds ``{channel: (low, high)}`` that training fits, or None when
    every trajectory sets its own (per_replicate normalisation).

    Runs the training data loading (``load_train_validation_data``) on the
    training replicates, so the bounds match training exactly.
    """
    from dataclasses import replace

    from Experiments.micropatterns.config_helpers import load_train_validation_data

    micropattern = data_config.micropattern
    if micropattern.cleaning.enabled and not micropattern.cleaning.shares_bounds:
        return None
    config = replace(data_config, micropattern=replace(micropattern, validation_replicates=None, pool_copies=1))
    (_, aux, names, *_), _ = load_train_validation_data(config, impath=str(root))
    return dict(zip(names, np.asarray(aux["histogram_bins"]).tolist()))


@app.function(hide_code=True)
def with_groups(data_config, groups):
    """``data_config`` loading only the experiment groups ``groups``.

    Bounds are per channel, so this changes no channel's preprocessing.
    """
    from dataclasses import replace

    from Experiments.micropatterns.config_helpers import measurement_count

    channels = measurement_count(groups, data_config.micropattern.excluded_measurements)
    return replace(data_config, micropattern=replace(
        data_config.micropattern, experiment_groups=tuple(groups), data_channels=channels))


@app.function(hide_code=True)
def load_knockout_trajectories(data_config, root, replicates, histogram_bins=None):
    """One trajectory per condition, ``replicates`` mapping condition → zero-based replicate.

    Returns ``{condition: {"images": [T, C, H, W], "measured": [T, G], "substituted": [T, G],
    "mask": [H, W]}}``, the channel names and the experiment group of each channel. The
    groups share one colony mask. ``histogram_bins`` (``{channel: (low, high)}``) fixes the
    clipping bounds; without it they are fitted to the loaded trajectories.
    """
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.micropatterns.config_helpers import _cleaning_for_loader, _excluded_images

    micropattern = data_config.micropattern
    conditions = tuple(KO_HOURS)
    indices = tuple(sorted(set(replicates.values())))
    dataset = load_micropattern_260726(
        root,
        conditions=conditions,
        projection=micropattern.projection,
        timesteps=TIMEPOINTS,
        downsample=data_config.downsample,
        replicate_indices=indices,
        histogram_bins=histogram_bins,
        experiment_groups=tuple(micropattern.experiment_groups),
        excluded_measurements=micropattern.excluded_measurements,
        hist_eqs=micropattern.histogram_percentiles,
        intensity_factors=micropattern.intensity_factors,
        cleaning=_cleaning_for_loader(micropattern.cleaning),
        excluded_images=_excluded_images(micropattern),
        cache_dir=cache_dir_from_environment(),
    )
    channel_groups = [name.split("/")[0] for name in dataset.channel_names]
    groups = list(dict.fromkeys(channel_groups))
    trajectories = {}
    for condition_index, condition in enumerate(conditions):
        batch = condition_index * len(indices) + indices.index(replicates[condition])
        measured = np.asarray(dataset.measurement_mask[batch])
        trajectories[condition] = {
            "images": np.asarray(dataset.data[batch]),
            "measured": np.stack([measured[:, [g == group for g in channel_groups]].any(axis=-1)
                                  for group in groups], axis=-1),
            "substituted": np.asarray(dataset.aux["is_substituted"][batch]),
            "mask": np.asarray(dataset.boundary_mask[batch, 0]),
        }
    return trajectories, dataset.channel_names, channel_groups


@app.function(hide_code=True)
def composite(channels, colours):
    """RGB image [..., 3] on black from normalised channels [C, ...], one colour (or None) per channel."""
    rgb = np.zeros(channels.shape[1:] + (3,))
    for channel, colour in zip(channels, colours):
        if colour is not None:
            rgb += channel[..., None] * np.asarray(colour)
    return np.clip(rgb, 0.0, 1.0)


@app.function(hide_code=True)
def colony_rgba(image, mask, colours, gamma=1.0):
    """RGBA of a [C, H, W] image in [0, 1], cropped to its colony mask and transparent outside it."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    rgb = composite(np.clip(image[:, crop[0], crop[1]], 0.0, 1.0) ** gamma, colours)
    return np.concatenate([rgb, mask[crop][..., None].astype(float)], axis=-1)


@app.function(hide_code=True)
def sector_angles(index, count):
    """Start and end angle (degrees, anticlockwise from +x) of sector ``index`` of ``count``.

    The first sector starts at the top, so two sectors are the left and right halves.
    """
    start = 90.0 + index * 360.0 / count
    return start, start + 360.0 / count


@app.class_definition(hide_code=True)
class SectorHandle:
    """Legend handle for sector ``index`` of ``count`` (see sector_angles)."""

    def __init__(self, index, count):
        self.index, self.count = index, count


@app.class_definition(hide_code=True)
class SectorHandler(HandlerBase):
    """Draws a SectorHandle as a small disc with its sector filled."""

    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        centre = (width / 2 - xdescent, height / 2 - ydescent)
        radius = 0.75 * height
        start, stop = sector_angles(handle.index, handle.count)
        return [
            Circle(centre, radius, fill=False, edgecolor="#999999", lw=0.6, transform=trans),
            Wedge(centre, radius, start, stop, facecolor="#dddddd", edgecolor="none", transform=trans),
        ]


@app.function(hide_code=True)
def sector_map(shape, count):
    """Sector index [H, W] of each pixel of an image drawn with row 0 at the top."""
    rows, cols = np.indices(shape)
    angle = np.degrees(np.arctan2((shape[0] - 1) / 2 - rows, cols - (shape[1] - 1) / 2))
    return (np.mod(angle - 90.0, 360.0) // (360.0 / count)).astype(int).clip(0, count - 1)


@app.function(hide_code=True)
def sliced_rgba(group_rgbas):
    """One RGBA [H, W, 4] taking sector k from the k-th RGBA of ``group_rgbas``."""
    stacked = np.stack(group_rgbas)
    index = sector_map(stacked.shape[1:3], len(group_rgbas))
    return np.take_along_axis(stacked, index[None, ..., None], axis=0)[0]


@app.function(hide_code=True)
def draw_colony(ax, rgba, centre, size, theme, alpha=1.0, edge=None):
    """Colony disc of diameter ``size`` at ``centre`` (data units), on black."""
    x, y = centre
    faded = rgba.copy()
    faded[..., 3] *= alpha
    ax.add_patch(Circle(centre, size / 2, facecolor="black", edgecolor="none", alpha=alpha, zorder=3))
    ax.imshow(faded, extent=(x - size / 2, x + size / 2, y - size / 2, y + size / 2), origin="upper",
              interpolation="antialiased", zorder=4)
    ax.add_patch(Circle(centre, size / 2, fill=False, edgecolor=edge or theme["muted"], lw=0.6, alpha=alpha,
                        zorder=5))


@app.function(hide_code=True)
def draw_placeholder(ax, centre, size, theme, text, colour=None):
    """Dashed empty disc for a colony that was not measured (or is unknown)."""
    colour = colour or theme["muted"]
    ax.add_patch(Circle(centre, size / 2, facecolor=theme["bg"], edgecolor=colour, lw=0.7, ls=(0, (2, 2)),
                        zorder=3))
    ax.text(*centre, text, ha="center", va="center", color=colour, fontsize=9, zorder=5)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NODAL knockout schematic

    - Rows: control, NODAL inhibited from 24h, from 0h; real colonies at 0–48h
    - Lower strip: 48h outcome against knockout time; only 0h and 24h are measured, the model can probe any time
    - Several stainings: stacked discs, or one disc split into sectors
    - Preprocessing defaults to a shared scale: training cleaning (`conf/base_config.yaml`), but one intensity scale per channel for every condition, so intensity changes between conditions are shown as measured
    - "training" modes show the data as the model saw it; with `per_replicate` each trajectory fills its own range, so intensities are not comparable between conditions
    - Figures are saved to `figures/schematics/`
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_root = mo.ui.text(
        value=str(Path(os.getenv("DATA_PATH_BASE", "")) / "260726_nca_dataset"),
        label="Dataset root",
        full_width=True,
    )
    groups = mo.ui.multiselect(options=list(EXPERIMENT_GROUPS), value=["cell_fate_s1"], label="Stainings")
    projection = mo.ui.dropdown(options=list(PROJECTION_CLEANING), value="max", label="Projection")
    downsample = mo.ui.dropdown(options={"4": 4, "8": 8, "16": 16}, value="8", label="Downsample")
    # training (standard): conf/base_config.yaml, shared by all micropattern runs
    preprocessing = mo.ui.dropdown(
        options=["shared scale (compare conditions)", "training (standard)", "training config file",
                 "custom normalisation"],
        value="shared scale (compare conditions)", label="Preprocessing")
    training_path = mo.ui.text(placeholder="model bundle folder, generated manifest.yaml or config .yaml",
                               label="Training config", full_width=True)
    manifest_index = mo.ui.number(start=0, stop=10000, step=1, value=0, label="Manifest index")
    # pooled keeps intensities comparable between conditions
    normalisation = mo.ui.dropdown(options=["pooled", "per_replicate", "replicate_mean"], value="pooled",
                                   label="Normalisation")
    replicates = mo.ui.dictionary({
        condition: mo.ui.number(start=1, stop=20, step=1, value=1, label=f"{condition} replicate")
        for condition in KO_HOURS
    })
    mo.vstack([
        mo.md("**Data**"),
        dataset_root,
        mo.hstack([groups, preprocessing, projection, downsample], justify="start"),
        mo.md("- Training config file (projection and downsampling then come from the file):"),
        mo.hstack([training_path, manifest_index], widths=[4, 1]),
        mo.md("- Custom normalisation:"),
        normalisation,
        mo.hstack(list(replicates.values()), justify="start"),
    ])
    return (
        dataset_root,
        downsample,
        groups,
        manifest_index,
        normalisation,
        preprocessing,
        projection,
        replicates,
        training_path,
    )


@app.cell(hide_code=True)
def _(
    dataset_root,
    downsample,
    groups,
    manifest_index,
    normalisation,
    preprocessing,
    projection,
    training_path,
):
    mo.stop(not groups.value, mo.callout("Select at least one staining.", kind="warn"))
    _groups = [g for g in EXPERIMENT_GROUPS if g in groups.value]
    if preprocessing.value == "shared scale (compare conditions)":
        # pooled: one set of bounds per channel from all loaded trajectories,
        # so every condition and timepoint goes through the same linear map
        data_config = knockout_data_config(_groups, projection.value, downsample.value, "pooled")
        fixed_bounds = None
        _bounds_source = "one per channel, pooled over all loaded trajectories and shared by every condition"
    elif preprocessing.value == "custom normalisation":
        data_config = knockout_data_config(_groups, projection.value, downsample.value, normalisation.value)
        fixed_bounds = None
        _bounds_source = "fitted to the trajectories loaded here"
    else:
        if preprocessing.value == "training (standard)":
            _training = knockout_data_config(EXPERIMENT_GROUPS, projection.value, downsample.value)
        else:
            mo.stop(not training_path.value, mo.callout("Type the path of a training config.", kind="warn"))
            _training = training_data_config(training_path.value, int(manifest_index.value))
        mo.stop(_training.dataset != "micropatterns_260726",
                mo.callout(f"The config uses `{_training.dataset}`, not `micropatterns_260726`.", kind="warn"))
        _trained = _training.micropattern.experiment_groups or EXPERIMENT_GROUPS
        _missing = [g for g in _groups if g not in _trained]
        mo.stop(_missing, mo.callout(f"Not in the training data: {', '.join(_missing)}.", kind="warn"))
        with mo.status.spinner("Fitting the training normalisation (training replicates)"):
            fixed_bounds = training_bounds(_training, dataset_root.value)
        data_config = with_groups(_training, _groups)
        _bounds_source = ("each trajectory's own (per_replicate, as in training)" if fixed_bounds is None
                          else "fitted on the training replicates, as in training")
    _micropattern = data_config.micropattern
    _cleaning = _micropattern.cleaning
    _notes = [
        f"projection `{_micropattern.projection}`, downsample {data_config.downsample}",
        f"cleaning {'on' if _cleaning.enabled else 'off (legacy loader)'}"
        + (f": hot pixels, stripes, rolling-ball background, alignment `{Path(_cleaning.alignment_file).name}`"
           if _cleaning.enabled else ""),
        f"normalisation `{_cleaning.normalisation}`, percentiles {tuple(_micropattern.histogram_percentiles)}"
        + (f", per channel {dict(_cleaning.channel_percentiles)}" if _cleaning.channel_percentiles else ""),
        f"clipping bounds {_bounds_source}",
    ]
    if fixed_bounds is None and _cleaning.enabled and not _cleaning.shares_bounds:
        _notes.append("**intensities are not comparable between conditions**: each trajectory is scaled to its own "
                      "range (use the shared scale to compare them)")
    if tuple(_micropattern.timesteps) != TIMEPOINTS:
        _notes.append(f"training timesteps {tuple(_micropattern.timesteps)}; shown here: {TIMEPOINTS}")
    mo.md("**Preprocessing**\n\n" + "\n".join(f"- {n}" for n in _notes))
    return data_config, fixed_bounds


@app.cell(hide_code=True)
def _(data_config, dataset_root, fixed_bounds, replicates):
    with mo.status.spinner("Loading control and knockout trajectories"):
        trajectories, channel_names, channel_groups = load_knockout_trajectories(
            data_config, dataset_root.value, {c: int(r) - 1 for c, r in replicates.value.items()}, fixed_bounds
        )
    _rows = [
        f"`{c}`: measured at {', '.join(f'{t}h' for t, m, s in zip(TIMEPOINTS, d['measured'].any(axis=1), d['substituted'].any(axis=1)) if m and not s) or 'none'}"
        for c, d in trajectories.items()
    ]
    mo.md("Own images (not substituted control) per condition:\n\n" + "\n".join(f"- {r}" for r in _rows))
    return channel_groups, channel_names, trajectories


@app.cell(hide_code=True)
def _(channel_groups, channel_names):
    _several = len(set(channel_groups)) > 1
    channel_colours = mo.ui.array([
        mo.ui.dropdown(
            options=list(COLOURS),
            value=MARKER_COLOURS.get(name.split("/")[-1], "grey"),
            label=f"{GROUP_LABELS[group]} {name.split('/')[-1]}" if _several else name.split("/")[-1],
        )
        for name, group in zip(channel_names, channel_groups)
    ])
    mo.vstack([mo.md("**Channels**"), mo.hstack(list(channel_colours), justify="start")])
    return (channel_colours,)


@app.cell(hide_code=True)
def _():
    background = mo.ui.dropdown(options=list(THEMES), value="light", label="Page background")
    # 1 keeps displayed brightness proportional to normalised intensity
    gamma = mo.ui.slider(start=0.3, stop=2.0, step=0.05, value=1.0, label="Gamma (1 = linear)", show_value=True)
    colony_size = mo.ui.slider(start=6.0, stop=11.0, step=0.5, value=9.0, label="Colony size (h)",
                               show_value=True)
    label_space = mo.ui.slider(start=10.0, stop=40.0, step=1.0, value=16.0, label="Label space (h)",
                               show_value=True)
    figure_width = mo.ui.slider(start=4.0, stop=12.0, step=0.25, value=7.0, label="Width (in)", show_value=True)
    pre_knockout = mo.ui.dropdown(options=["branch from control", "faded copies"], value="branch from control",
                                  label="Before knockout")
    show_model = mo.ui.switch(value=False, label="Knockout-time strip")
    group_layout = mo.ui.dropdown(options=["stacked", "slices"], value="slices",
                                  label="Several stainings shown as")
    model_times = mo.ui.text(value="12, 36", label="Unmeasured knockout times (h)")
    accent_colour = mo.ui.text(value="", placeholder="theme", label="Inhibitor colour")
    model_colour = mo.ui.text(value="", placeholder="theme", label="Model colour")
    inhibitor = mo.ui.text(value="NODAL inhibitor", label="Inhibitor label")
    condition_labels = mo.ui.dictionary({
        "ctrl": mo.ui.text(value="Control", label="ctrl"),
        "sl24": mo.ui.text(value="NODAL KO\nat 24h", label="sl24"),
        "sl0": mo.ui.text(value="NODAL KO\nat 0h", label="sl0"),
    })
    title_a = mo.ui.text(value="Inhibiting NODAL at different times changes the 48h pattern",
                         label="Title (top)", full_width=True)
    title_b = mo.ui.text(value="Experiments only measure 2 knockout times; the model can probe any",
                         label="Title (strip)", full_width=True)
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([background, gamma, colony_size, label_space, figure_width], justify="start"),
        mo.hstack([pre_knockout, group_layout, show_model, model_times], justify="start"),
        mo.hstack([accent_colour, model_colour, inhibitor], justify="start"),
        mo.md("**Labels** (`\\n` for a new line)"),
        mo.hstack(list(condition_labels.values()), justify="start"),
        title_a,
        title_b,
    ])
    return (
        accent_colour,
        background,
        colony_size,
        condition_labels,
        figure_width,
        gamma,
        group_layout,
        inhibitor,
        label_space,
        model_colour,
        model_times,
        pre_knockout,
        show_model,
        title_a,
        title_b,
    )


@app.cell(hide_code=True)
def _(
    accent_colour,
    background,
    channel_colours,
    channel_groups,
    channel_names,
    colony_size,
    condition_labels,
    figure_width,
    gamma,
    group_layout,
    inhibitor,
    label_space,
    model_colour,
    model_times,
    pre_knockout,
    show_model,
    title_a,
    title_b,
    trajectories,
):
    _theme = dict(THEMES[background.value])
    # Any matplotlib colour; blank or invalid keeps the theme colour
    for _key, _text in (("accent", accent_colour.value), ("model", model_colour.value)):
        if is_color_like(_text.strip()):
            _theme[_key] = _text.strip()
    _colours = [COLOURS[c] for c in channel_colours.value]
    _groups = list(dict.fromkeys(channel_groups))
    _n = len(_groups)
    _slices = group_layout.value == "slices" and _n > 1
    _stacked = group_layout.value == "stacked" and _n > 1
    _s = colony_size.value
    _tile_gap = 0.8
    # Height of one timepoint: a column of discs when stacked, else one disc
    _cell = _n * _s + (_n - 1) * _tile_gap if _stacked else _s
    _half = _cell / 2
    _row_step = _cell + 3.5
    _end = TIMEPOINTS[-1]
    _left = -_s / 2 - label_space.value
    _label_x = -_s / 2 - (7 if _stacked else 2)
    _branch = pre_knockout.value == "branch from control"
    _label = lambda text: text.replace("\\n", "\n")

    # One RGBA per condition, timepoint and staining (only its own channels coloured)
    _group_colours = [[c if g == group else None for c, g in zip(_colours, channel_groups)] for group in _groups]
    _rgba = {
        c: [[colony_rgba(image, d["mask"], gc, gamma.value) for gc in _group_colours] for image in d["images"]]
        for c, d in trajectories.items()
    }

    def _tile_y(y, k):
        return y + _half - _s / 2 - k * (_s + _tile_gap)

    def _draw_cell(condition, i, x, y, alpha=1.0, edge=None):
        """All stainings of one condition at timepoint i, centred at (x, y)."""
        _measured = trajectories[condition]["measured"][i]
        if _slices:
            if not _measured.any():
                draw_placeholder(_ax, (x, y), _s, _theme, "–")
                return
            # Unmeasured stainings (possibly filled with a replicate mean by the loader) stay black
            _tiles = [t if m else t * np.array([0.0, 0.0, 0.0, 1.0]) for t, m in zip(_rgba[condition][i], _measured)]
            draw_colony(_ax, sliced_rgba(_tiles), (x, y), _s, _theme, alpha=alpha, edge=edge)
            for _k in range(_n):
                _start, _stop = sector_angles(_k, _n)
                _ax.plot([x, x + _s / 2 * np.cos(np.radians(_start))], [y, y + _s / 2 * np.sin(np.radians(_start))],
                         color="white", lw=0.6, alpha=alpha, zorder=5)
                if not _measured[_k]:
                    _middle = np.radians((_start + _stop) / 2)
                    _ax.text(x + 0.25 * _s * np.cos(_middle), y + 0.25 * _s * np.sin(_middle), "–",
                             ha="center", va="center", color="#bbbbbb", fontsize=9, zorder=6)
            return
        for _k in range(_n):
            _y = _tile_y(y, _k) if _stacked else y
            if _measured[_k]:
                draw_colony(_ax, _rgba[condition][i][_k], (x, _y), _s, _theme, alpha=alpha, edge=edge)
            else:
                draw_placeholder(_ax, (x, _y), _s, _theme, "–")
            if not _stacked:
                break

    def _draw_unknown(x, y):
        for _k in range(_n if _stacked else 1):
            draw_placeholder(_ax, (x, _tile_y(y, _k) if _stacked else y), _s, _theme, "?", colour=_theme["model"])

    def _draw_tile_labels(y):
        if _stacked:
            for _k, _group in enumerate(_groups):
                _ax.text(-_s / 2 - 0.8, _tile_y(y, _k), GROUP_LABELS[_group], ha="right", va="center",
                         fontsize=5.5, color=_theme["muted"])

    # Everything is drawn on one axis whose x unit is hours, so the time axis
    # of the experiment and the knockout-time axis of the strip line up.
    _row_y = {c: -i * _row_step for i, c in enumerate(KO_HOURS)}
    _bottom_a = _row_y["sl0"] - _half
    # Room for one legend row per staining
    _legend_rows = _n if _n > 1 else 1
    _top = _half + 9 + (_legend_rows - 1) * 11 * (_s / 2 + label_space.value + _end + _s / 2 + 4) / (
        figure_width.value * 72)

    _strip_image_y = _bottom_a - 7 - _half
    _strip_axis_y = _strip_image_y - _half - 2
    _measured_y = _strip_axis_y - 6
    _model_y = _measured_y - 4
    _bottom = (_model_y - 3) if show_model.value else (_bottom_a - 2)
    _right = _end + _s / 2 + 4

    with plt.rc_context({"font.size": 7, "font.family": "sans-serif", "text.color": _theme["fg"]}):
        _height = figure_width.value * (_top - _bottom) / (_right - _left)
        ko_schematic = plt.figure(figsize=(figure_width.value, _height), facecolor=_theme["bg"], dpi=300)
        _ax = ko_schematic.add_axes((0, 0, 1, 1))
        _ax.set_xlim(_left, _right)
        _ax.set_ylim(_bottom, _top)
        _ax.set_aspect("equal")
        _ax.axis("off")

        # Title and time axis
        _ax.text(_left + 1, _top - 1, title_a.value, ha="left", va="top", fontsize=8, fontweight="bold")
        _axis_y = _half + 2.5
        _ax.plot([TIMEPOINTS[0] - _s / 2, _end + _s / 2], [_axis_y] * 2, color=_theme["fg"], lw=0.7)
        for _t in TIMEPOINTS:
            _ax.plot([_t, _t], [_axis_y, _axis_y + 0.8], color=_theme["fg"], lw=0.7)
            _ax.text(_t, _axis_y + 1.2, f"{_t}h", ha="center", va="bottom")
        _ax.text(TIMEPOINTS[0] - _s / 2 - 1, _axis_y, "time", ha="right", va="center", color=_theme["muted"])

        for _condition, _ko in KO_HOURS.items():
            _y = _row_y[_condition]
            _data = trajectories[_condition]
            _ax.text(_label_x, _y, _label(condition_labels.value[_condition]), ha="right",
                     va="center", fontsize=7, color=_theme["fg"] if _ko is None else _theme["accent"])
            _draw_tile_labels(_y)
            # Trajectory line, behind the colonies; a knockout row branches off
            # the control row at its knockout time.
            _start = TIMEPOINTS[0] if _ko is None or not _branch else _ko
            _xs, _ys = [_start, _end], [_y, _y]
            if _ko is not None and _branch:
                _xs, _ys = [_ko, _ko, _end], [_row_y["ctrl"], _y, _y]
            _ax.plot(_xs, _ys, color=_theme["muted"] if _ko is None else _theme["accent"], lw=0.9, zorder=1,
                     solid_joinstyle="round")
            if _ko is not None:
                # Inhibitor band from the knockout time to the end
                _ax.add_patch(FancyBboxPatch((_ko, _y - _half - 0.8), _end + _s / 2 + 0.8 - _ko, _cell + 1.6,
                                             boxstyle="round,pad=0,rounding_size=1.5", facecolor=_theme["accent"],
                                             alpha=0.1, edgecolor="none", zorder=0))
                _ax.plot([_ko, _ko], [_y - _half - 0.8, _y + _half + 0.8], color=_theme["accent"], lw=1.2,
                         zorder=2)
                _ax.text(_ko + 0.6, _y + _half + 1.0, f"+ {inhibitor.value} from {_ko}h", ha="left",
                         va="bottom", fontsize=6, color=_theme["accent"])
            for _i, _t in enumerate(TIMEPOINTS):
                if _data["substituted"][_i].any():
                    if not _branch:
                        _draw_cell(_condition, _i, _t, _y, alpha=0.25)
                else:
                    _draw_cell(_condition, _i, _t, _y)

        # The 48h outcomes, boxed together
        _ax.add_patch(FancyBboxPatch((_end - _s / 2 - 1.2, _bottom_a - 1.2), _s + 2.4,
                                     _row_y["ctrl"] + _half - _bottom_a + 2.4,
                                     boxstyle="round,pad=0,rounding_size=2", fill=False, edgecolor=_theme["fg"],
                                     lw=0.8, ls=(0, (3, 2)), zorder=6))
        _ax.text(_end + _s / 2 + 2.8, (_row_y["ctrl"] + _row_y["sl0"]) / 2, "48h fate pattern", rotation=90,
                 ha="center", va="center", fontsize=6.5)

        if show_model.value:
            # 48h outcome against knockout time: measured at 0h and 24h, and the
            # control stands in for a knockout too late to act (48h).
            _ax.text(_left + 1, _strip_image_y + _half + 3, title_b.value, ha="left", va="center", fontsize=8,
                     fontweight="bold")
            _anchors = {_ko if _ko is not None else _end: c for c, _ko in KO_HOURS.items()}
            for _x, _condition in _anchors.items():
                _draw_cell(_condition, len(TIMEPOINTS) - 1, _x, _strip_image_y, edge=_theme["fg"])
            try:
                _unknown = [float(v) for v in model_times.value.split(",") if v.strip()]
            except ValueError:
                _unknown = []
            for _x in _unknown:
                if 0 <= _x <= _end and _x not in _anchors:
                    _draw_unknown(_x, _strip_image_y)
            _ax.text(_label_x, _strip_image_y, "48h pattern", ha="right", va="center")
            _draw_tile_labels(_strip_image_y)

            # Knockout-time axis
            _ax.plot([0, _end], [_strip_axis_y] * 2, color=_theme["fg"], lw=0.7)
            for _t in TIMEPOINTS:
                _ax.plot([_t, _t], [_strip_axis_y, _strip_axis_y - 0.8], color=_theme["fg"], lw=0.7)
                _ax.text(_t, _strip_axis_y - 1.2, f"{_t}h" if _t < _end else f"{_t}h (≈ ctrl)", ha="center",
                         va="top", fontsize=6)
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _strip_axis_y, "knockout time", ha="right", va="center",
                     color=_theme["muted"])

            # What is measured, and what the model can probe
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _measured_y, "experiments", ha="right", va="center",
                     color=_theme["accent"])
            for _ko in (0, 24):
                _ax.plot(_ko, _measured_y, "o", ms=5, color=_theme["accent"], zorder=4)
            _ax.plot(_end, _measured_y, "o", ms=5, mfc=_theme["bg"], mec=_theme["muted"], zorder=4)
            _ax.text(_end + 1.5, _measured_y, "no KO", ha="left", va="center", fontsize=6, color=_theme["muted"])
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _model_y, "model", ha="right", va="center",
                     color=_theme["model"])
            _ax.plot([0, _end], [_model_y] * 2, color=_theme["model"], lw=4, alpha=0.35, solid_capstyle="round")
            _ax.plot(np.arange(0, _end + 1, 2), [_model_y] * (_end // 2 + 1), "o", ms=1.6, color=_theme["model"])
            _ax.text(_end + 1.5, _model_y, "any time", ha="left", va="center", fontsize=6, color=_theme["model"])

        # Channel colours on black, like the images: one row per staining, led
        # by its name (and, for slices, a disc showing its sector).
        _rows = []
        for _k, _group in enumerate(_groups):
            _entries = [(plt.Line2D([], [], color=c, lw=2), n.split("/")[-1])
                        for n, c, g in zip(channel_names, _colours, channel_groups) if g == _group and c is not None]
            if _n > 1:
                _header = SectorHandle(_k, _n) if _slices else plt.Line2D([], [], lw=0)
                _entries.insert(0, (_header, GROUP_LABELS[_group]))
            _rows.append(_entries)
        _width = max(len(r) for r in _rows)
        _blank = (plt.Line2D([], [], lw=0), "")
        _rows = [r + [_blank] * (_width - len(r)) for r in _rows]
        # Legend columns fill top to bottom, so entries go in column by column
        _entries = [_rows[r][c] for c in range(_width) for r in range(len(_rows))]
        ko_schematic.legend([h for h, _ in _entries], [l for _, l in _entries], loc="upper right",
                            ncol=max(1, _width), fontsize=6.5, bbox_to_anchor=(0.995, 0.995),
                            labelcolor="white", frameon=True, fancybox=False, facecolor="black",
                            edgecolor="black", framealpha=1.0, handlelength=1.2, columnspacing=0.8,
                            handler_map={SectorHandle: SectorHandler()})
    ko_schematic
    return (ko_schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="nodal_knockout_schematic", label="File name")
    save_figure = mo.ui.run_button(label="Save PDF + PNG")
    mo.hstack([figure_name, save_figure], justify="start")
    return figure_name, save_figure


@app.cell
def _(figure_name, ko_schematic, save_figure):
    mo.stop(not save_figure.value)
    _directory = Path(mo.notebook_dir()).parents[1] / "figures" / "schematics"
    _directory.mkdir(parents=True, exist_ok=True)
    for _suffix in ("pdf", "png"):
        ko_schematic.savefig(_directory / f"{figure_name.value}.{_suffix}", dpi=400,
                             facecolor=ko_schematic.get_facecolor())
    mo.md(f"Saved `{figure_name.value}.pdf` and `.png` to `{_directory}`")
    return


if __name__ == "__main__":
    app.run()
