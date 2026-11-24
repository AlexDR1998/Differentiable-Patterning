"""Density × NODAL-KO figure: predicted cell-type prevalence over plating density
and knockout time, with the 150K training colonies and the held-out 250K colonies.

Run with:
    marimo edit Experiments/micropatterns/figures/density_ko_figure.py

- Data: ``nodal_ko_eval.py``, section "Held-out test: 250K plating", button
  "Save for density × KO figure"; it writes ``figures/density_ko/``.
- Model and data use the cell type file's rules restricted to the S2 panel
  (SOX17, TBXT, FOXA2), as the 250K colonies have only those markers.
- Optional panel d: colony images at chosen points. NCA images need a rollout
  (button; JAX, a GPU helps), cached in ``figures/density_ko/cache/``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="density × KO figure")

with app.setup:
    import hashlib
    import json
    import os
    from dataclasses import replace
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from dotenv import load_dotenv
    from matplotlib.patches import Patch

    from Common.dataloader.cell_types import load_cell_type_rules, marker_values, pixel_label_shares
    from Experiments.micropatterns import density_test_data as heldout
    from Experiments.micropatterns.density_ko_sweep import settings_from_dict
    from Experiments.micropatterns.figures.data import cache_name, cached, cell_type_rgba
    from Experiments.micropatterns.figures.drawing import colony_rgba, draw_placeholder
    from Experiments.micropatterns.ko_evaluation import cell_edges, ko_time_label, ko_time_ticks, uncertainty_band
    from Experiments.micropatterns.figures.style import (
        ACCENT,
        BAND_ALPHA,
        CELL_TYPE_COLOURS,
        CMAP,
        FIGURE_ROOT,
        FONT,
        INK,
        LINE,
        MUTED,
        PAGE,
        REPOSITORY_ROOT,
        cell_type_colour,
        figure_width,
        label_panel,
        legend_options,
        marker_colour,
        new_figure,
        paper_style,
        save_figure,
    )

    FIGURE_DIRECTORY = FIGURE_ROOT / "density_ko"
    MODEL_CSV = FIGURE_DIRECTORY / "model_prevalence.csv"
    MEASURED_CSV = FIGURE_DIRECTORY / "measured_prevalence.csv"
    SETTINGS_JSON = FIGURE_DIRECTORY / "settings.json"
    # Marker images of the measured colonies (index table and arrays)
    IMAGES_CSV = FIGURE_DIRECTORY / "measured_images.csv"
    IMAGES_NPZ = FIGURE_DIRECTORY / "measured_images.npz"
    CACHE_DIRECTORY = FIGURE_DIRECTORY / "cache"
    # Channels of the marker images, in the order of the figure channels of the S2 panel
    IMAGE_MARKERS = ("FOXA2", "TBXT", "SOX17")
    IMAGE_ROWS = ("True data", "NCA")

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)

    # Marker and legend label of each measured data set (column "dataset")
    DATASET_MARKERS = {"150K": "s", "250K": "o"}
    DATASET_LABELS = {"150K": "150K plating (training data)", "250K": "250K plating (held out)"}
    PREVALENCE_LABEL = "Share of colony pixels"


@app.function(hide_code=True)
def summarise_model(model, mode):
    """Mean and band (``uncertainty_band`` ``mode``) over the runs at each density,
    KO time and cell type; ``density`` is the mean density the runs used."""
    rows = []
    for (density_index, hour, cell_type), values in model.groupby(
        ["density_index", "knockout_hour", "cell_type"], sort=True
    ):
        mean, lower, upper = uncertainty_band(values["prevalence"], mode)
        rows.append({
            "density_index": density_index, "knockout_hour": hour, "cell_type": cell_type,
            "density": values["density"].mean(), "mean": mean, "lower": lower, "upper": upper,
            "runs": values["run"].nunique(),
        })
    return pd.DataFrame(rows)


@app.function(hide_code=True)
def summarise_measured(measured):
    """Mean and SD over the colonies of each data set, KO time and cell type."""
    return (
        measured.groupby(["dataset", "knockout_hour", "cell_type"], sort=False)
        .agg(density=("density", "mean"), mean=("share", "mean"), sd=("share", "std"), colonies=("share", "size"))
        .fillna({"sd": 0.0})
        .reset_index()
    )


@app.function(hide_code=True)
def format_density(value, unit):
    return f"{value:.2f}" if unit == "share" else f"{value:,.0f}"


@app.function(hide_code=True)
def parse_image_points(text, snapshot):
    """Image points from ``"<plating>@<KO hour>"`` separated by commas, e.g.
    ``"150K@24, 250K@36, 0.5@48"``. The plating is in thousands of cells
    ("150K") or a multiple of the 150K density ("0.5"); a KO hour at or after
    the end time (or none) is the baseline. Returns ``[(label, scale, hour)]``."""
    points = []
    for part in text.split(","):
        if not part.strip():
            continue
        density, _, hour = part.partition("@")
        density = density.strip().upper()
        scale = float(density[:-1]) / 150.0 if density.endswith("K") else float(density)
        hour = min(float(hour) if hour.strip() else snapshot, snapshot)
        points.append((f"{scale * 150:g}K", scale, hour))
    return points


@app.function(hide_code=True)
def colony_tile(values, mask, rules, style, resolve_several):
    """RGBA of one colony from ``{marker: image}``: cell types with ``rules``
    (as in panel a), or the FOXA2 / TBXT / SOX17 composite."""
    if style == "cell types":
        shares = pixel_label_shares(values, rules, mask, resolve_several=resolve_several)
        return cell_type_rgba(shares, CELL_TYPE_COLOURS, mask)
    channels = np.stack([np.asarray(values[marker]) for marker in IMAGE_MARKERS])
    return colony_rgba(channels, np.asarray(mask, dtype=bool), [marker_colour(m) for m in IMAGE_MARKERS])


@app.function(hide_code=True)
def draw_tag(axis, number, xy, xycoords="data", offset=(6, 6), leader=None):
    """Numbered circle that links a colony image to its point; ``leader`` is the
    colour of a thin line from the circle to ``xy`` (None: no line)."""
    axis.annotate(
        str(number), xy=xy, xycoords=xycoords, xytext=offset, textcoords="offset points",
        ha="center", va="center", fontsize=FONT["min"], fontweight="bold", color=INK,
        bbox={"boxstyle": "circle,pad=0.2", "facecolor": PAGE, "edgecolor": INK, "linewidth": LINE["thin"]},
        arrowprops=None if leader is None else {
            "arrowstyle": "-", "color": leader, "linewidth": LINE["thin"], "shrinkA": 0, "shrinkB": 0,
        },
        zorder=6, annotation_clip=False,
    )


@app.function(hide_code=True)
def draw_figure(summary, measured, settings, cell_types, slice_ko, slice_density_index, colour_max,
                show_slices, show_band, image_points=(), image_rows=(), image_style="cell types"):
    """Heatmaps (one per cell type) above a density slice and a KO-time slice,
    and optionally colony images (one column per point of ``image_points``,
    one row per entry of ``image_rows``), tagged by number in every panel.

    Each image point is ``{"number", "label", "density", "hour", "tiles":
    {row: RGBA, or a text when there is no image}}``.
    """
    snapshot = float(settings["snapshot_hour"])
    unit, density_label = settings["density_unit"], settings["density_label"]
    densities = summary.groupby("density_index")["density"].mean().sort_values()
    hours = np.sort(summary["knockout_hour"].unique())
    slice_density = float(densities[slice_density_index])
    ticks, tick_labels = ko_time_ticks(snapshot)
    shown = measured[measured["cell_type"].isin(cell_types)]
    # Swept density nearest to each data point, so the KO-time slice shows the
    # data of its own density
    nearest = densities.index[np.abs(shown["density"].to_numpy()[:, None] - densities.to_numpy()).argmin(axis=1)]

    # Image tiles at most ~0.9 inch, at least six columns wide
    image_rows = list(image_rows) if len(image_points) else []
    base = 0.6 * figure_width("double")
    columns = max(len(image_points), 6)
    tile = min(0.9, 0.92 * figure_width("double") / columns)
    extra = len(image_rows) * tile + 0.45 if image_rows else 0.0
    figure = new_figure("double", height=base + extra)
    if image_rows:
        outer = figure.add_gridspec(3, 1, height_ratios=(base / 2, base / 2, extra), hspace=0.62)
    else:
        outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.62)
    top = outer[0].subgridspec(1, len(cell_types) + 1, width_ratios=[1.0] * len(cell_types) + [0.05], wspace=0.2)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.0, 1.0, 0.62), wspace=0.32)

    # a: heatmaps, with the measured shares as markers on the same colour scale
    heatmap_axes = []
    for column, cell_type in enumerate(cell_types):
        axis = figure.add_subplot(top[0, column], sharey=heatmap_axes[0] if heatmap_axes else None)
        heatmap_axes.append(axis)
        grid = (
            summary[summary["cell_type"] == cell_type]
            .pivot(index="density_index", columns="knockout_hour", values="mean")
            .loc[densities.index, hours]
        )
        mesh = axis.pcolormesh(
            cell_edges(hours), cell_edges(densities.to_numpy()), grid.to_numpy(),
            shading="flat", cmap=CMAP["heatmap"], vmin=0.0, vmax=colour_max, rasterized=True,
        )
        for dataset, marker in DATASET_MARKERS.items():
            points = shown[(shown["dataset"] == dataset) & (shown["cell_type"] == cell_type)]
            points = points.groupby("knockout_hour", as_index=False).agg(density=("density", "mean"), share=("share", "mean"))
            axis.scatter(
                points["knockout_hour"], points["density"], c=points["share"], cmap=CMAP["heatmap"],
                vmin=0.0, vmax=colour_max, marker=marker, s=22, edgecolors="white",
                linewidths=LINE["axis"], zorder=3, clip_on=False,
            )
        if show_slices:
            axis.axvline(slice_ko, color=ACCENT, linewidth=LINE["thin"], linestyle="--")
            axis.axhline(slice_density, color=ACCENT, linewidth=LINE["thin"], linestyle="--")
        for point in image_points:
            draw_tag(axis, point["number"], (point["hour"], point["density"]), leader=PAGE)
        axis.set_title(cell_type, fontsize=FONT["label"], color=INK, pad=3)
        axis.set_xticks(ticks, tick_labels)
        axis.set_xlabel("NODAL KO time")
        if column == 0:
            axis.set_ylabel(density_label)
        else:
            axis.tick_params(labelleft=False)
    colour_bar = figure.colorbar(mesh, cax=figure.add_subplot(top[0, -1]))
    colour_bar.set_label(PREVALENCE_LABEL)
    colour_bar.outline.set_linewidth(LINE["axis"])
    runs = int(summary["runs"].max())
    label_panel(
        heatmap_axes[0], "a", f"Predicted prevalence at {snapshot:g} h",
        f"NCA mean of {runs} runs; markers: measured colonies", offset=(0, 13),
    )

    def _model_line(axis, curve, x_column, colour):
        curve = curve.sort_values(x_column)
        axis.plot(curve[x_column], curve["mean"], color=colour, linewidth=LINE["data"])
        if show_band:
            axis.fill_between(curve[x_column], curve["lower"], curve["upper"], color=colour,
                              alpha=BAND_ALPHA, linewidth=0)

    def _data_points(axis, points, x_column, colour, shift):
        # Shifted a little sideways per cell type, so markers at one x stay apart
        for dataset, rows in points.groupby("dataset"):
            axis.errorbar(
                rows[x_column] + shift, rows["mean"], yerr=rows["sd"], linestyle="none",
                marker=DATASET_MARKERS[dataset], markersize=4, markerfacecolor=colour, markeredgecolor=INK,
                markeredgewidth=LINE["thin"], ecolor=colour, elinewidth=LINE["thin"], capsize=1.5, zorder=3,
            )

    def _shift(cell_type, step):
        return (cell_types.index(cell_type) - (len(cell_types) - 1) / 2) * step

    density_step = 0.012 * float(densities.max() - densities.min())
    measured_summary = summarise_measured(shown)
    near_summary = summarise_measured(shown[nearest == slice_density_index])

    # b: against density at one KO time
    density_axis = figure.add_subplot(bottom[0, 0])
    for cell_type in cell_types:
        colour = cell_type_colour(cell_type)
        _model_line(
            density_axis,
            summary[(summary["cell_type"] == cell_type) & np.isclose(summary["knockout_hour"], slice_ko)],
            "density", colour,
        )
        _data_points(
            density_axis,
            measured_summary[(measured_summary["cell_type"] == cell_type)
                             & np.isclose(measured_summary["knockout_hour"], slice_ko)],
            "density", colour, _shift(cell_type, density_step),
        )
    for name, value in (("150K", settings["training_density"]), ("250K", settings["test_density"])):
        density_axis.axvline(value, color=MUTED, linewidth=LINE["hairline"], linestyle=":", zorder=0)
        density_axis.text(value, 1.0, name, transform=density_axis.get_xaxis_transform(), ha="center",
                          va="bottom", fontsize=FONT["annotation"], color=MUTED)
    density_axis.set(xlabel=density_label, ylabel=PREVALENCE_LABEL, ylim=(0.0, None))
    # Image points on this slice, tagged just above the density axis
    for point in image_points:
        if np.isclose(point["hour"], slice_ko):
            draw_tag(density_axis, point["number"], (point["density"], 0.0), ("data", "axes fraction"),
                     offset=(0, 7), leader=INK)
    ko_text = ko_time_label(slice_ko, snapshot)
    label_panel(density_axis, "b", "Against density",
                "baseline (no KO)" if ko_text == "Baseline" else f"NODAL KO at {ko_text}", offset=(0, 10))

    # c: against KO time at one density, with the data whose nearest swept density it is
    ko_axis = figure.add_subplot(bottom[0, 1], sharey=density_axis)
    for cell_type in cell_types:
        colour = cell_type_colour(cell_type)
        _model_line(
            ko_axis,
            summary[(summary["cell_type"] == cell_type) & (summary["density_index"] == slice_density_index)],
            "knockout_hour", colour,
        )
        _data_points(ko_axis, near_summary[near_summary["cell_type"] == cell_type], "knockout_hour", colour,
                     _shift(cell_type, 0.6))
    for point in image_points:
        point_index = densities.index[np.abs(densities.to_numpy() - point["density"]).argmin()]
        if point_index == slice_density_index:
            draw_tag(ko_axis, point["number"], (point["hour"], 0.0), ("data", "axes fraction"),
                     offset=(0, 7), leader=INK)
    ko_axis.set_xticks(ticks, tick_labels)
    ko_axis.set(xlabel="NODAL KO time")
    ko_axis.tick_params(labelleft=False)
    label_panel(ko_axis, "c", "Against KO time",
                f"{format_density(slice_density, unit)} {density_label.lower()}", offset=(0, 10))

    # Legend: cell types, model band, data sets, slice lines
    legend_axis = figure.add_subplot(bottom[0, 2])
    legend_axis.axis("off")
    handles = [Line2D([], [], color=cell_type_colour(c), linewidth=LINE["data"]) for c in cell_types]
    labels = list(cell_types)
    if show_band:
        handles.append(Patch(facecolor=MUTED, alpha=BAND_ALPHA * 2, linewidth=0))
        labels.append("NCA mean ± spread over runs")
    present = [d for d in DATASET_MARKERS if d in set(shown["dataset"])]
    handles += [Line2D([], [], linestyle="none", marker=DATASET_MARKERS[d], markersize=4,
                       markerfacecolor="white", markeredgecolor=INK, markeredgewidth=LINE["thin"]) for d in present]
    labels += [f"{DATASET_LABELS[d]}, mean ± SD" if d == "150K" else DATASET_LABELS[d] for d in present]
    if show_slices:
        handles.append(Line2D([], [], color=ACCENT, linewidth=LINE["thin"], linestyle="--"))
        labels.append("Slices shown in b and c")
    if image_rows and image_style == "cell types":
        # Image colours of the labels without a line above (e.g. Several, Other)
        extra_labels = [c for c in settings["cell_types"] if c not in cell_types]
        handles += [Patch(facecolor=cell_type_colour(c), edgecolor=INK, linewidth=LINE["thin"]) for c in extra_labels]
        labels += [f"{c} (images)" for c in extra_labels]
    elif image_rows:
        handles += [Patch(facecolor=marker_colour(m), edgecolor=INK, linewidth=LINE["thin"]) for m in IMAGE_MARKERS]
        labels += [f"{m} (images)" for m in IMAGE_MARKERS]
    if image_rows:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=PAGE,
                              markeredgecolor=INK, markeredgewidth=LINE["thin"]))
        labels.append("Point of colony image (d)")
    legend_axis.legend(handles, labels, loc="center left", **legend_options())

    # d: colony images, one column per point
    if image_rows:
        grid = outer[2].subgridspec(len(image_rows), columns, wspace=0.08, hspace=0.12)
        for column, point in enumerate(image_points):
            for row, name in enumerate(image_rows):
                axis = figure.add_subplot(grid[row, column])
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                content = point["tiles"].get(name)
                if isinstance(content, np.ndarray):
                    axis.imshow(content, interpolation="none")
                else:
                    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
                    draw_placeholder(axis, (0.5, 0.5), 0.9, "")
                    axis.text(0.5, 0.5, content or "", ha="center", va="center",
                              fontsize=FONT["min"], color=MUTED, zorder=6)
                if column == 0:
                    axis.set_ylabel(name)
                if row == 0:
                    ko_text = ko_time_label(point["hour"], snapshot)
                    axis.set_title(f"{point['label']} · {'baseline' if ko_text == 'Baseline' else 'KO ' + ko_text}",
                                   fontsize=FONT["annotation"], fontweight="normal", color=INK, pad=2)
                    draw_tag(axis, point["number"], (0.0, 1.0), "axes fraction", offset=(4, -4))
                if row == 0 and column == 0:
                    label_panel(axis, "d", f"Colonies at {snapshot:g} h",
                                "cell types (S2 rules)" if image_style == "cell types" else "FOXA2 / TBXT / SOX17",
                                offset=(0, 12))
    return figure


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Density × NODAL-KO figure
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **a:** predicted prevalence over plating density and NODAL KO time (heatmaps), with the measured colonies on the same colour scale: squares for the 150K training colonies (KO 0h, KO 24h, baseline at the end time), circles for the held-out 250K colonies
    - **b, c:** 1D slices through the heatmaps, at one KO time and at one density, with the measured colonies that lie on them
    - **d** (optional): colony images at 48 h, true data and NCA, at chosen points; numbered circles show each point in a–c
    - Model and data use the S2-panel rules (SOX17, TBXT, FOXA2); see `docs/finetune_ablation.md` for the held-out test
    - Data: `Experiments/micropatterns/nodal_ko_eval.py`, *Held-out test: 250K plating* → *Save for density × KO figure*
    - Style: `figures/style.py` (`docs/figures.md`)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data
    """)
    return


@app.cell(hide_code=True)
def _():
    _missing = [p.name for p in (MODEL_CSV, MEASURED_CSV, SETTINGS_JSON) if not p.is_file()]
    mo.stop(
        bool(_missing),
        mo.callout(
            f"Missing in `figures/density_ko/`: {', '.join(_missing)}. Save them from `nodal_ko_eval.py` "
            "(*Held-out test: 250K plating* → *Save for density × KO figure*).",
            kind="warn",
        ),
    )
    model_table = pd.read_csv(MODEL_CSV)
    measured_table = pd.read_csv(MEASURED_CSV)
    figure_settings = json.loads(SETTINGS_JSON.read_text())
    mo.md(
        f"- Projection **{figure_settings['projection']}**, groups by *{figure_settings['group_by']}*, "
        f"cell type file `{figure_settings['cell_type_file']}`\n"
        f"- {model_table['run'].nunique()} sweep runs; measured colonies: "
        + ", ".join(f"{_d} {_n}" for _d, _n in measured_table.groupby("dataset")["colony"].nunique().items())
        + "\n"
        f"- 250K normalisation: {figure_settings['normalisation_250k']}; rings {figure_settings['rings']}; "
        f"resolve multiple matches {figure_settings['resolve_several']:g}"
    )
    return figure_settings, measured_table, model_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Content
    """)
    return


@app.cell(hide_code=True)
def _(figure_settings, model_table):
    _groups = list(dict.fromkeys(model_table["group"]))
    figure_group = mo.ui.dropdown(_groups, value=_groups[0], label="Model group")
    figure_cell_types = mo.ui.multiselect(
        figure_settings["cell_types"],
        value=[_c for _c in figure_settings["cell_types"] if _c not in ("Several", "Other")],
        label="Cell types",
    )
    figure_uncertainty = mo.ui.dropdown(
        {"SD over runs": "std", "SEM over runs": "sem", "95% interval": "percentile", "None": "none"},
        value="SD over runs", label="Band",
    )
    figure_colour_max = mo.ui.dropdown({"0 to 1": "one", "0 to largest shown": "max"}, value="0 to 1",
                                       label="Colour scale")
    figure_show_slices = mo.ui.switch(value=True, label="Mark the slices on the heatmaps")
    mo.hstack([figure_group, figure_cell_types, figure_uncertainty, figure_colour_max, figure_show_slices],
              justify="start", wrap=True)
    return figure_cell_types, figure_colour_max, figure_group, figure_show_slices, figure_uncertainty


@app.cell(hide_code=True)
def _(figure_group, figure_uncertainty, model_table):
    figure_summary = summarise_model(model_table[model_table["group"] == figure_group.value], figure_uncertainty.value)
    return (figure_summary,)


@app.cell(hide_code=True)
def _(figure_settings, figure_summary):
    _snapshot = float(figure_settings["snapshot_hour"])
    _hours = sorted(figure_summary["knockout_hour"].unique())
    _ko_options = {ko_time_label(_h, _snapshot): float(_h) for _h in _hours}
    figure_slice_ko = mo.ui.dropdown(
        _ko_options,
        # 24 h: both the 150K KO 24h and the 250K KO 24h colonies lie on this slice
        value=next((_k for _k, _h in _ko_options.items() if np.isclose(_h, 24.0)), next(iter(_ko_options))),
        label="KO time of slice b",
    )
    _densities = figure_summary.groupby("density_index")["density"].mean().sort_values()
    _unit = figure_settings["density_unit"]
    _density_options = {format_density(_v, _unit): int(_i) for _i, _v in _densities.items()}
    # Default: the swept density nearest the 150K training density
    _nearest = int(_densities.index[np.abs(_densities.to_numpy() - figure_settings["training_density"]).argmin()])
    figure_slice_density = mo.ui.dropdown(
        _density_options,
        value=next(_k for _k, _i in _density_options.items() if _i == _nearest),
        label=f"Density of slice c ({figure_settings['density_label'].lower()})",
    )
    mo.hstack([figure_slice_ko, figure_slice_density], justify="start")
    return figure_slice_density, figure_slice_ko


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Colony images
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Points are `plating@KO hour`, separated by commas: `150K@24`, `250K@36`, or a multiple of the 150K density (`0.5@48`); 48 (the end time) is the baseline
    - Defaults: the three 150K training conditions and the first and last 250K KO times
    - True data exists only at the measured points (150K: KO 0h, KO 24h, baseline; 250K: its KO times); elsewhere the tile says *not measured*
    - NCA images: one model of the chosen group, with the cell placement and seeds of the density × KO sweep (`density_test_data.test_point_states`); every replicate and run is rolled out, and *NCA sample* picks one
    """)
    return


@app.cell(hide_code=True)
def _(figure_group, model_table):
    _models = list(dict.fromkeys(model_table.loc[model_table["group"] == figure_group.value, "model_id"]))
    figure_show_images = mo.ui.switch(value=True, label="Show colony images (panel d)")
    figure_image_text = mo.ui.text(
        "150K@0, 150K@24, 150K@48, 250K@24, 250K@36", label="Points", full_width=True,
    )
    figure_image_rows = mo.ui.multiselect(list(IMAGE_ROWS), value=list(IMAGE_ROWS), label="Rows")
    figure_image_style = mo.ui.dropdown(
        {"Cell types (S2 rules)": "cell types", "Markers (FOXA2, TBXT, SOX17)": "markers"},
        value="Cell types (S2 rules)", label="Images show",
    )
    figure_image_colony = mo.ui.number(1, 4, value=1, step=1, label="True-data colony (150K)")
    figure_image_model = mo.ui.dropdown(_models, value=_models[0], label="NCA model")
    figure_image_sample = mo.ui.number(1, 1000, value=1, step=1, label="NCA sample")
    figure_rollout = mo.ui.run_button(label="Roll out NCA at these points")
    mo.vstack([
        mo.hstack([figure_show_images, figure_image_rows, figure_image_style], justify="start"),
        figure_image_text,
        mo.hstack([figure_image_colony, figure_image_model, figure_image_sample, figure_rollout], justify="start"),
    ])
    return (
        figure_image_colony,
        figure_image_model,
        figure_image_rows,
        figure_image_sample,
        figure_image_style,
        figure_image_text,
        figure_rollout,
        figure_show_images,
    )


@app.cell(hide_code=True)
def _(figure_image_text, figure_settings):
    try:
        figure_points = parse_image_points(figure_image_text.value, float(figure_settings["snapshot_hour"]))
        _view = mo.md("")
    except ValueError:
        figure_points = []
        _view = mo.callout("Could not read the points; write them as `150K@24, 250K@36`.", kind="warn")
    _view
    return (figure_points,)


@app.cell(hide_code=True)
def _(figure_settings):
    # The S2 rules of the saved data: the cell type file with the thresholds used when saving
    _rules = load_cell_type_rules(REPOSITORY_ROOT / figure_settings["cell_type_file"])
    _rules = replace(_rules, thresholds={**_rules.thresholds, **figure_settings["thresholds"]})
    figure_rules, _, _ = heldout.stain_rules(_rules)
    # Marker images of the measured colonies (saved since the image panel was added)
    if IMAGES_CSV.is_file() and IMAGES_NPZ.is_file():
        with np.load(IMAGES_NPZ) as _file:
            figure_measured_images = (pd.read_csv(IMAGES_CSV), {_key: _file[_key] for _key in _file.files})
    else:
        figure_measured_images = None
    return figure_measured_images, figure_rules


@app.cell(hide_code=True)
def _(figure_image_model, figure_image_rows, figure_points, figure_rollout, figure_settings, figure_show_images):
    figure_nca_states = None
    _view = mo.md("")
    _wanted = figure_show_images.value and "NCA" in figure_image_rows.value and figure_points
    if _wanted and "sweep_settings" not in figure_settings:
        _view = mo.callout("Save the data again from `nodal_ko_eval.py` to get NCA images.", kind="warn")
    elif _wanted:
        _snapshot = float(figure_settings["snapshot_hour"])
        # Baseline points are rolled out without a knockout
        _points = [(_l, _s, None if _h >= _snapshot else _h) for _l, _s, _h in figure_points]
        _key = hashlib.sha1(json.dumps(
            [figure_image_model.value, [[_s, _h] for _, _s, _h in _points], figure_settings["sweep_settings"]],
            sort_keys=True,
        ).encode()).hexdigest()[:12]
        _name = cache_name("nca_points", _key)

        def _roll_out():
            _values = dict(figure_settings["sweep_settings"])
            _values["data_root"] = _values["data_root"] or os.environ.get("DATA_PATH_BASE", "")
            _values["store_root"] = _values["store_root"] or os.environ.get("MODEL_STORE_ROOT", "")
            _settings = settings_from_dict(_values)
            from Experiments.model_registry import ModelRegistry

            _bundle = ModelRegistry(Path(_settings.store_root).expanduser()).get(figure_image_model.value)
            return heldout.test_point_states(
                _bundle, _settings, _points, Path(_settings.data_root).expanduser(), max_batch=_settings.max_batch,
            )

        if (CACHE_DIRECTORY / _name).is_file() or figure_rollout.value:
            with mo.status.spinner("Rolling out the NCA at the image points"):
                figure_nca_states = cached(_name, _roll_out, directory=CACHE_DIRECTORY)
        else:
            _view = mo.callout(
                "Click *Roll out NCA at these points* for the NCA row (cached in `figures/density_ko/cache/`).",
                kind="info",
            )
    _view
    return (figure_nca_states,)


@app.cell(hide_code=True)
def _(
    figure_image_colony,
    figure_image_rows,
    figure_image_sample,
    figure_image_style,
    figure_measured_images,
    figure_nca_states,
    figure_points,
    figure_rules,
    figure_settings,
    figure_show_images,
):
    _resolve = float(figure_settings["resolve_several"])
    _training = float(figure_settings["training_density"])
    _rows = [_r for _r in IMAGE_ROWS if _r in figure_image_rows.value] if figure_show_images.value else []
    figure_image_points = []
    for _number, (_label, _scale, _hour) in enumerate(figure_points if _rows else [], start=1):
        _tiles = {}
        if "True data" in _rows:
            _tiles["True data"] = "not measured"
            if figure_measured_images is None:
                _tiles["True data"] = "not saved"
            else:
                _table, _arrays = figure_measured_images
                # Measured colonies at this point: same plating and KO time
                _here = _table[
                    np.isclose(_table["density"], _scale * _training, rtol=0.02)
                    & np.isclose(_table["knockout_hour"], _hour)
                ].sort_values("colony")
                if not _here.empty:
                    _index = int(_here["image"].iloc[min(int(figure_image_colony.value), len(_here)) - 1])
                    _values = {_m: _arrays[f"colony{_index}_{_m}"] for _m in figure_rules.markers}
                    _tiles["True data"] = colony_tile(
                        _values, _arrays[f"colony{_index}_mask"], figure_rules, figure_image_style.value, _resolve,
                    )
        if "NCA" in _rows:
            _tiles["NCA"] = "not rolled out"
            if figure_nca_states is not None:
                _samples = figure_nca_states["states"][_number - 1]
                _state = _samples[min(int(figure_image_sample.value), len(_samples)) - 1]
                _values = marker_values(_state, figure_nca_states["names"], figure_rules, axis=0)
                _tiles["NCA"] = colony_tile(
                    _values, figure_nca_states["mask"], figure_rules, figure_image_style.value, _resolve,
                )
        figure_image_points.append({
            "number": _number, "label": _label, "density": _scale * _training, "hour": _hour, "tiles": _tiles,
        })
    figure_image_row_names = _rows
    return figure_image_points, figure_image_row_names


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure
    """)
    return


@app.cell(hide_code=True)
def _(
    figure_cell_types,
    figure_colour_max,
    figure_image_points,
    figure_image_row_names,
    figure_image_style,
    figure_settings,
    figure_show_slices,
    figure_slice_density,
    figure_slice_ko,
    figure_summary,
    figure_uncertainty,
    measured_table,
):
    mo.stop(not figure_cell_types.value, mo.md("- Choose at least one cell type"))
    _cell_types = [_c for _c in figure_settings["cell_types"] if _c in figure_cell_types.value]
    _shown = figure_summary[figure_summary["cell_type"].isin(_cell_types)]
    _colour_max = 1.0 if figure_colour_max.value == "one" else float(max(
        _shown["mean"].max(), measured_table[measured_table["cell_type"].isin(_cell_types)]["share"].max(),
    ))
    with paper_style():
        density_ko_figure = draw_figure(
            figure_summary, measured_table, figure_settings, _cell_types,
            figure_slice_ko.value, figure_slice_density.value, _colour_max,
            figure_show_slices.value, figure_uncertainty.value != "none",
            figure_image_points, figure_image_row_names, figure_image_style.value,
        )
    density_ko_figure
    return (density_ko_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save
    """)
    return


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text("density_ko_heldout", label="File name")
    figure_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    figure_save = mo.ui.run_button(label="Save to figures/density_ko/")
    mo.hstack([figure_name, figure_svg, figure_save], justify="start")
    return figure_name, figure_save, figure_svg


@app.cell(hide_code=True)
def _(density_ko_figure, figure_name, figure_save, figure_svg):
    mo.stop(not figure_save.value)
    _paths = save_figure(density_ko_figure, figure_name.value.strip() or "density_ko_heldout", "density_ko",
                         svg=figure_svg.value)
    mo.md("Saved " + ", ".join(f"`{_p.name}`" for _p in _paths))
    return


if __name__ == "__main__":
    app.run()
