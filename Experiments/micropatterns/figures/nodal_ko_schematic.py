"""Schematic of the NODAL knockout problem, drawn with real micropattern data.

Run with:
    marimo edit Experiments/micropatterns/figures/nodal_ko_schematic.py

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
    from matplotlib.patches import FancyBboxPatch

    from Experiments.micropatterns.figures.data import (
        KO_HOURS,
        PROJECTION_CLEANING,
        TIMEPOINTS,
        knockout_data_config,
        load_knockout_trajectories,
        run_data_config,
        training_bounds,
        with_groups,
    )
    from Experiments.micropatterns.figures.drawing import (
        colony_rgba,
        draw_colony,
        draw_placeholder,
        draw_sectors,
        draw_time_axis,
        marker_legend_entries,
        rows_to_columns,
        sliced_rgba,
    )
    from Experiments.micropatterns.figures.style import (
        ACCENT,
        CONDITION_LABELS,
        EXPERIMENT_GROUPS,
        FONT,
        GROUP_LABELS,
        INK,
        LINE,
        MODEL,
        MUTED,
        PAGE,
        UNMEASURED,
        WIDTH,
        group_colours,
        legend_options,
        paper_style,
        save_figure,
        sector_angles,
    )

    load_dotenv()


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
            _training = run_data_config(training_path.value, int(manifest_index.value))
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
def _():
    # 1 keeps displayed brightness proportional to normalised intensity
    gamma = mo.ui.slider(start=0.3, stop=2.0, step=0.05, value=1.0, label="Gamma (1 = linear)", show_value=True)
    colony_size = mo.ui.slider(start=6.0, stop=11.0, step=0.5, value=9.0, label="Colony size (h)",
                               show_value=True)
    label_space = mo.ui.slider(start=10.0, stop=40.0, step=1.0, value=16.0, label="Label space (h)",
                               show_value=True)
    figure_width = mo.ui.dropdown(options={"double column": WIDTH["double"], "single column": WIDTH["single"]},
                                  value="double column", label="Width")
    pre_knockout = mo.ui.dropdown(options=["branch from control", "faded copies"], value="branch from control",
                                  label="Before knockout")
    show_model = mo.ui.switch(value=False, label="Knockout-time strip")
    group_layout = mo.ui.dropdown(options=["stacked", "slices"], value="slices",
                                  label="Several stainings shown as")
    model_times = mo.ui.text(value="12, 36", label="Unmeasured knockout times (h)")
    inhibitor = mo.ui.text(value="NODAL inhibitor", label="Inhibitor label")
    title_a = mo.ui.text(value="Inhibiting NODAL at different times changes the 48h pattern",
                         label="Title (top)", full_width=True)
    title_b = mo.ui.text(value="Experiments only measure 2 knockout times; the model can probe any",
                         label="Title (strip)", full_width=True)
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([gamma, colony_size, label_space, figure_width], justify="start"),
        mo.hstack([pre_knockout, group_layout, show_model, model_times, inhibitor], justify="start"),
        title_a,
        title_b,
    ])
    return (
        colony_size,
        figure_width,
        gamma,
        group_layout,
        inhibitor,
        label_space,
        model_times,
        pre_knockout,
        show_model,
        title_a,
        title_b,
    )


@app.cell(hide_code=True)
def _(
    channel_groups,
    channel_names,
    colony_size,
    figure_width,
    gamma,
    group_layout,
    inhibitor,
    label_space,
    model_times,
    pre_knockout,
    show_model,
    title_a,
    title_b,
    trajectories,
):
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
    _width = figure_width.value

    # One RGBA per condition, timepoint and staining (only its own channels coloured)
    _group_colours = [group_colours(channel_names, group) for group in _groups]
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
                draw_placeholder(_ax, (x, y), _s, "–")
                return
            # Unmeasured stainings (possibly filled with a replicate mean by the loader) stay black
            _tiles = [t if m else t * np.array([0.0, 0.0, 0.0, 1.0]) for t, m in zip(_rgba[condition][i], _measured)]
            draw_colony(_ax, sliced_rgba(_tiles), (x, y), _s, alpha=alpha, edge=edge)
            draw_sectors(_ax, (x, y), _s / 2, [False] * _n, alpha=alpha)
            for _k in range(_n):
                if not _measured[_k]:
                    _middle = np.radians(sum(sector_angles(_k, _n)) / 2)
                    _ax.text(x + 0.25 * _s * np.cos(_middle), y + 0.25 * _s * np.sin(_middle), "–",
                             ha="center", va="center", color=UNMEASURED["face"], fontsize=FONT["label"], zorder=6)
            return
        for _k in range(_n):
            _y = _tile_y(y, _k) if _stacked else y
            if _measured[_k]:
                draw_colony(_ax, _rgba[condition][i][_k], (x, _y), _s, alpha=alpha, edge=edge)
            else:
                draw_placeholder(_ax, (x, _y), _s, "–")
            if not _stacked:
                break

    def _draw_unknown(x, y):
        for _k in range(_n if _stacked else 1):
            draw_placeholder(_ax, (x, _tile_y(y, _k) if _stacked else y), _s, "?", colour=MODEL)

    def _draw_tile_labels(y):
        if _stacked:
            for _k, _group in enumerate(_groups):
                _ax.text(-_s / 2 - 0.8, _tile_y(y, _k), GROUP_LABELS[_group], ha="right", va="center",
                         fontsize=FONT["min"], color=MUTED)

    # Everything is drawn on one axis whose x unit is hours, so the time axis
    # of the experiment and the knockout-time axis of the strip line up.
    _row_y = {c: -i * _row_step for i, c in enumerate(KO_HOURS)}
    _bottom_a = _row_y["sl0"] - _half
    # Room for one legend row per staining
    _legend_rows = _n if _n > 1 else 1
    _top = _half + 9 + (_legend_rows - 1) * 11 * (_s / 2 + label_space.value + _end + _s / 2 + 4) / (_width * 72)

    _strip_image_y = _bottom_a - 7 - _half
    _strip_axis_y = _strip_image_y - _half - 2
    _measured_y = _strip_axis_y - 6
    _model_y = _measured_y - 4
    _bottom = (_model_y - 3) if show_model.value else (_bottom_a - 2)
    _right = _end + _s / 2 + 4

    with paper_style():
        _height = _width * (_top - _bottom) / (_right - _left)
        ko_schematic = plt.figure(figsize=(_width, _height))
        _ax = ko_schematic.add_axes((0, 0, 1, 1))
        _ax.set_xlim(_left, _right)
        _ax.set_ylim(_bottom, _top)
        _ax.set_aspect("equal")
        _ax.axis("off")

        # Title and time axis
        _ax.text(_left + 1, _top - 1, title_a.value, ha="left", va="top", fontsize=FONT["title"], fontweight="bold")
        _axis_y = _half + 2.5
        draw_time_axis(_ax, TIMEPOINTS, _axis_y, _s)
        _ax.text(TIMEPOINTS[0] - _s / 2 - 1, _axis_y, "time", ha="right", va="center", color=MUTED)

        for _condition, _ko in KO_HOURS.items():
            _y = _row_y[_condition]
            _data = trajectories[_condition]
            _ax.text(_label_x, _y, CONDITION_LABELS[_condition].replace(" from", "\nfrom"), ha="right",
                     va="center", fontsize=FONT["label"], color=INK if _ko is None else ACCENT)
            _draw_tile_labels(_y)
            # Trajectory line, behind the colonies; a knockout row branches off
            # the control row at its knockout time.
            _start = TIMEPOINTS[0] if _ko is None or not _branch else _ko
            _xs, _ys = [_start, _end], [_y, _y]
            if _ko is not None and _branch:
                _xs, _ys = [_ko, _ko, _end], [_row_y["ctrl"], _y, _y]
            _ax.plot(_xs, _ys, color=MUTED if _ko is None else ACCENT, lw=LINE["data"], zorder=1,
                     solid_joinstyle="round")
            if _ko is not None:
                # Inhibitor band from the knockout time to the end
                _ax.add_patch(FancyBboxPatch((_ko, _y - _half - 0.8), _end + _s / 2 + 0.8 - _ko, _cell + 1.6,
                                             boxstyle="round,pad=0,rounding_size=1.5", facecolor=ACCENT,
                                             alpha=0.1, edgecolor="none", zorder=0))
                _ax.plot([_ko, _ko], [_y - _half - 0.8, _y + _half + 0.8], color=ACCENT, lw=LINE["emphasis"],
                         zorder=2)
                _ax.text(_ko + 0.6, _y + _half + 1.0, f"+ {inhibitor.value} from {_ko}h", ha="left",
                         va="bottom", fontsize=FONT["annotation"], color=ACCENT)
            for _i, _t in enumerate(TIMEPOINTS):
                if _data["substituted"][_i].any():
                    if not _branch:
                        _draw_cell(_condition, _i, _t, _y, alpha=0.25)
                else:
                    _draw_cell(_condition, _i, _t, _y)

        # The 48h outcomes, boxed together
        _ax.add_patch(FancyBboxPatch((_end - _s / 2 - 1.2, _bottom_a - 1.2), _s + 2.4,
                                     _row_y["ctrl"] + _half - _bottom_a + 2.4,
                                     boxstyle="round,pad=0,rounding_size=2", fill=False, edgecolor=INK,
                                     lw=LINE["axis"], ls=(0, (3, 2)), zorder=6))
        _ax.text(_end + _s / 2 + 2.8, (_row_y["ctrl"] + _row_y["sl0"]) / 2, "48h fate pattern", rotation=90,
                 ha="center", va="center", fontsize=FONT["annotation"])

        if show_model.value:
            # 48h outcome against knockout time: measured at 0h and 24h, and the
            # control stands in for a knockout too late to act (48h).
            _ax.text(_left + 1, _strip_image_y + _half + 3, title_b.value, ha="left", va="center",
                     fontsize=FONT["title"], fontweight="bold")
            _anchors = {_ko if _ko is not None else _end: c for c, _ko in KO_HOURS.items()}
            for _x, _condition in _anchors.items():
                _draw_cell(_condition, len(TIMEPOINTS) - 1, _x, _strip_image_y, edge=INK)
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
            _ax.plot([0, _end], [_strip_axis_y] * 2, color=INK, lw=LINE["axis"])
            for _t in TIMEPOINTS:
                _ax.plot([_t, _t], [_strip_axis_y, _strip_axis_y - 0.8], color=INK, lw=LINE["axis"])
                _ax.text(_t, _strip_axis_y - 1.2, f"{_t}h" if _t < _end else f"{_t}h (≈ ctrl)", ha="center",
                         va="top", fontsize=FONT["tick"])
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _strip_axis_y, "knockout time", ha="right", va="center",
                     color=MUTED)

            # What is measured, and what the model can probe
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _measured_y, "experiments", ha="right", va="center", color=ACCENT)
            for _ko in (0, 24):
                _ax.plot(_ko, _measured_y, "o", ms=5, color=ACCENT, zorder=4)
            _ax.plot(_end, _measured_y, "o", ms=5, mfc=PAGE, mec=MUTED, zorder=4)
            _ax.text(_end + 1.5, _measured_y, "no KO", ha="left", va="center", fontsize=FONT["annotation"],
                     color=MUTED)
            _ax.text(TIMEPOINTS[0] - _s / 2 - 2, _model_y, "model", ha="right", va="center", color=MODEL)
            _ax.plot([0, _end], [_model_y] * 2, color=MODEL, lw=4, alpha=0.35, solid_capstyle="round")
            _ax.plot(np.arange(0, _end + 1, 2), [_model_y] * (_end // 2 + 1), "o", ms=1.6, color=MODEL)
            _ax.text(_end + 1.5, _model_y, "any time", ha="left", va="center", fontsize=FONT["annotation"],
                     color=MODEL)

        # Channel colours on black, like the images: one row per staining, led by its
        # name (and, for slices, a disc showing its sector).
        _legend = rows_to_columns(marker_legend_entries(channel_names, _groups, _groups if _slices else (), "black"))
        ko_schematic.legend(*_legend[:2], ncol=_legend[2], loc="upper right", bbox_to_anchor=(0.995, 0.995),
                            **legend_options("black"))
    ko_schematic
    return (ko_schematic,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="nodal_knockout_schematic", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_button = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_button], justify="start")
    return figure_name, save_button, save_svg


@app.cell
def _(figure_name, ko_schematic, save_button, save_svg):
    mo.stop(not save_button.value)
    _paths = save_figure(ko_schematic, figure_name.value, "schematics", svg=save_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
