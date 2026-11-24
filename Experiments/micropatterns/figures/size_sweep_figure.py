"""Colony size figure: predicted cell-type prevalence against colony diameter at a
fixed cell density and at a fixed cell count, with numbered 48 h snapshots.

Run with:
    marimo edit Experiments/micropatterns/figures/size_sweep_figure.py

- Data: the results of a size × density sweep (``size_density_explorer.py``)
  run with the size sweeps switched on: a saved ``.npz`` file, or the folder
  of a remote job's parts.
- One figure per condition (baseline, NODAL KO from 0 h, from 24 h).
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Size sweep figure")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    from Common.dataloader.cell_type_shares import colony_label_shares
    from Common.dataloader.cell_types import OTHER, SEVERAL, label_names, load_cell_type_rules
    from Experiments.micropatterns.ko_evaluation import model_stain_counts, uncertainty_band
    from Experiments.micropatterns.size_density_sweep import load_size_density_results
    from Experiments.micropatterns.figures.drawing import (
        composite,
        draw_sectors,
        marker_legend_entries,
        rows_to_columns,
        sliced_rgba,
    )
    from Experiments.micropatterns.figures.style import (
        BAND_ALPHA,
        CONDITION_LABELS,
        FIGURE_CHANNELS,
        FONT,
        GROUP_LABELS,
        INK,
        LINE,
        MUTED,
        PAGE,
        REPOSITORY_ROOT,
        cell_type_colour,
        label_panel,
        legend_options,
        marker_colour,
        new_figure,
        paper_style,
        save_figure,
    )

    CELL_TYPE_DIRECTORY = REPOSITORY_ROOT / "Experiments/micropatterns/conf/cell_types"
    FIGURE_FOLDER = "size_sweep"
    # The two size sweeps (result "source"): column title and what is held fixed
    SWEEPS = {
        "measured_density": ("Fixed density", "measured cells per mm²"),
        "measured_cell_count": ("Fixed cell count", "measured cells per colony"),
    }
    # Sweep conditions and their keys in style.CONDITION_LABELS
    CONDITION_KEYS = {"baseline": "ctrl", "ko_0h": "sl0", "ko_24h": "sl24"}
    SHARE_LABELS = {"pattern_counts": "Share of colony pixels", "cell_pattern_counts": "Share of cells"}
    TRUE_DATA_MARKER = "*"


@app.function(hide_code=True)
def condition_label(condition):
    """Figure label of a sweep condition (``baseline``, ``ko_0h``, ...)."""
    return CONDITION_LABELS.get(CONDITION_KEYS.get(condition), condition)


@app.function(hide_code=True)
def matching_cell_type_files(thresholds):
    """Cell type files (oldest first) with the marker thresholds the saved
    pattern counts were made with; other files would misread the counts."""
    files = []
    for path in sorted(CELL_TYPE_DIRECTORY.glob("*.yaml")):
        rules = load_cell_type_rules(path)
        if set(rules.markers) == set(thresholds) and all(
            np.isclose(float(rules.thresholds[_m]), float(thresholds[_m])) for _m in rules.markers
        ):
            files.append(path)
    return files


@app.function(hide_code=True)
def share_rows(results, rules_by_projection, basis, resolve_several):
    """One row per run, size, condition and cell type of the size sweeps and the true data."""
    rows = []
    for result in results:
        counts = result[basis]
        if result["source"] not in (*SWEEPS, "true_data") or counts is None:
            continue
        rules = rules_by_projection[result["projection"]]
        for size, diameter in enumerate(result["diameters_um"]):
            for k, condition in enumerate(result["conditions"]):
                shares = colony_label_shares(
                    model_stain_counts(counts[size, 0, k], rules), rules, resolve_several=resolve_several,
                )
                for cell_type, share in zip(label_names(rules), shares):
                    rows.append({
                        "model_id": result["model_id"], "replicate": result["replicate"],
                        "run": result["run_index"], "source": result["source"],
                        "diameter_um": float(diameter), "condition": condition,
                        "cell_type": cell_type, "share": float(share),
                        "cells_per_mm2": float(result["cells_per_mm2"][size, 0]),
                        "training_diameter_um": float(result["training_diameter_um"]),
                    })
    rows = pd.DataFrame(rows)
    if not rows.empty:
        # The replicates' true colonies differ a little in size: pooled at their mean size
        true = rows["source"] == "true_data"
        rows.loc[true, "diameter_um"] = rows.loc[true, "diameter_um"].mean()
    return rows


@app.function(hide_code=True)
def summarise(rows, mode):
    """Mean and band (``uncertainty_band`` ``mode``) over models, replicates and runs."""
    summary = []
    for (source, condition, cell_type, diameter), values in rows.groupby(
        ["source", "condition", "cell_type", "diameter_um"], sort=True
    ):
        mean, lower, upper = uncertainty_band(values["share"], mode)
        summary.append({
            "source": source, "condition": condition, "cell_type": cell_type, "diameter_um": diameter,
            "mean": mean, "lower": lower, "upper": upper,
            "cells_per_mm2": float(values["cells_per_mm2"].mean()), "samples": len(values),
        })
    return pd.DataFrame(summary)


@app.function(hide_code=True)
def sector_rgba(states, names, groups, mask):
    """RGBA of a state ``[C, H, W]``, one sector per experiment group, transparent outside ``mask``."""
    rgbas = []
    for group in groups:
        colours = [marker_colour(_n) if _n in FIGURE_CHANNELS[group] else None for _n in names]
        rgb = composite(np.clip(np.asarray(states, dtype=np.float32), 0.0, 1.0), colours)
        rgbas.append(np.concatenate([rgb, mask[..., None].astype(float)], axis=-1))
    return sliced_rgba(rgbas)


@app.function(hide_code=True)
def number_badge(axis, number, xy, xycoords, offset=(0, 0)):
    """A snapshot's number in a small circle."""
    axis.annotate(
        str(number), xy=xy, xycoords=xycoords, xytext=offset, textcoords="offset points",
        ha="center", va="center", fontsize=FONT["annotation"], fontweight="bold", color=INK,
        bbox={"boxstyle": "circle,pad=0.2", "facecolor": PAGE, "edgecolor": INK, "linewidth": LINE["thin"]},
        annotation_clip=False, zorder=6,
    )


@app.function(hide_code=True)
def scale_bar(axis, half_um, y_um):
    """Scale bar (µm) about a quarter of the panel wide, centred at height ``y_um``."""
    lengths = [_l for _l in (25, 50, 100, 200, 250, 500, 1000) if _l <= 0.5 * half_um]
    length = lengths[-1] if lengths else 0.5 * half_um
    axis.plot([-length / 2, length / 2], [y_um, y_um], color=INK, linewidth=LINE["emphasis"], solid_capstyle="butt")
    axis.annotate(f"{length:g} µm", xy=(0, y_um), xytext=(0, -2), textcoords="offset points", ha="center",
                  va="top", fontsize=FONT["annotation"], color=INK, annotation_clip=False)


@app.function(hide_code=True)
def draw_figure(summary, snapshots, cell_types, condition, share_label, show_density, show_band, groups,
                names, to_scale, training_diameter, snapshot_hour, snapshot_note):
    """a: share against colony diameter for each size sweep (optionally with the cells
    per mm² below); b: numbered snapshots, their numbers also marked on a."""
    sweeps = [_s for _s in SWEEPS if _s in set(summary["source"])]
    a_height = 1.9 + (0.6 if show_density else 0.0)
    b_height = 1.45 if snapshots else 0.0
    # Empty band (inches) between a and b: a's tick labels and axis label, then b's heading and snapshot titles
    gap = 0.85 if snapshots else 0.0
    figure = new_figure("double", height=a_height + gap + b_height + 0.3)
    outer = figure.add_gridspec(
        3 if snapshots else 1, 1, height_ratios=[a_height, gap, b_height] if snapshots else [1.0], hspace=0.0,
    )
    top = outer[0].subgridspec(
        2 if show_density else 1, len(sweeps) + 1, width_ratios=[1.0] * len(sweeps) + [0.6],
        height_ratios=[3.0, 1.0] if show_density else [1.0], hspace=0.12, wspace=0.25,
    )
    true = summary[summary["source"] == "true_data"]

    # a: one column per size sweep
    share_axes = []
    for column, source in enumerate(sweeps):
        axis = figure.add_subplot(top[0, column], sharey=share_axes[0] if share_axes else None)
        share_axes.append(axis)
        sweep = summary[summary["source"] == source]
        for cell_type in cell_types:
            colour = cell_type_colour(cell_type)
            line = sweep[sweep["cell_type"] == cell_type].sort_values("diameter_um")
            axis.plot(line["diameter_um"], line["mean"], color=colour, linewidth=LINE["data"], marker="o",
                      markersize=2.5)
            if show_band:
                axis.fill_between(line["diameter_um"], line["lower"], line["upper"], color=colour,
                                  alpha=BAND_ALPHA, linewidth=0)
            point = true[true["cell_type"] == cell_type]
            axis.plot(point["diameter_um"], point["mean"], linestyle="none", marker=TRUE_DATA_MARKER,
                      markersize=7, markerfacecolor=colour, markeredgecolor=INK,
                      markeredgewidth=LINE["hairline"], zorder=4)
        axis.axvline(training_diameter, color=MUTED, linewidth=LINE["thin"], linestyle="--", zorder=0)
        axis.set_ylim(0.0, 1.0)
        # Snapshot numbers above the axis at their diameter, with a guide line through it
        for snapshot in snapshots:
            if snapshot["source"] == source:
                axis.axvline(snapshot["diameter"], color=MUTED, linewidth=LINE["hairline"], linestyle=":", zorder=0)
                number_badge(axis, snapshot["number"], (snapshot["diameter"], 1.0),
                             blended_transform_factory(axis.transData, axis.transAxes), offset=(0, 6))
        if column == 0:
            # Two lines when the cells per mm² row is below, so the labels do not meet
            axis.set_ylabel(share_label.replace(" of ", " of\n") if show_density else share_label)
        else:
            axis.tick_params(labelleft=False)
        title, subtitle = SWEEPS[source]
        label_panel(axis, "a" if column == 0 else None, title, subtitle, offset=(0, 14))
        bottom_axis = axis
        if show_density:
            axis.tick_params(labelbottom=False)
            bottom_axis = figure.add_subplot(top[1, column], sharex=axis)
            sizes = sweep[sweep["cell_type"] == sweep["cell_type"].iloc[0]].sort_values("diameter_um")
            bottom_axis.plot(sizes["diameter_um"], sizes["cells_per_mm2"], color=INK, linewidth=LINE["data"],
                             marker="o", markersize=2.5)
            bottom_axis.axvline(training_diameter, color=MUTED, linewidth=LINE["thin"], linestyle="--", zorder=0)
            bottom_axis.set_ylim(0.0, None)
            if column == 0:
                bottom_axis.set_ylabel("Cells per mm²")
        bottom_axis.set_xlabel("Colony diameter (µm)")

    # Legend of a
    legend_axis = figure.add_subplot(top[:, -1])
    legend_axis.axis("off")
    handles = [Line2D([], [], color=cell_type_colour(_c), linewidth=LINE["data"]) for _c in cell_types]
    labels = list(cell_types)
    if show_band:
        handles.append(Patch(facecolor=MUTED, alpha=BAND_ALPHA * 2, linewidth=0))
        labels.append("NCA mean ± spread")
    if not true.empty:
        handles.append(Line2D([], [], linestyle="none", marker=TRUE_DATA_MARKER, markersize=7,
                              markerfacecolor="white", markeredgecolor=INK, markeredgewidth=LINE["hairline"]))
        labels.append("NCA from true data")
    handles.append(Line2D([], [], color=MUTED, linewidth=LINE["thin"], linestyle="--"))
    labels.append("Training colony size")
    legend_axis.legend(handles, labels, loc="center left", title=condition_label(condition),
                       title_fontsize=FONT["legend"], **legend_options())
    figure.align_ylabels()

    # b: snapshots, all to one scale or each filling its panel
    if snapshots:
        bottom = outer[2].subgridspec(1, len(snapshots) + 1, width_ratios=[1.0] * len(snapshots) + [1.6],
                                      wspace=0.1)
        largest = max(_s["diameter"] for _s in snapshots) / 2.0
        for index, snapshot in enumerate(snapshots):
            axis = figure.add_subplot(bottom[0, index])
            height, width = snapshot["mask"].shape
            pixel_um = snapshot["pixel_um"]
            extent = (-width / 2 * pixel_um, width / 2 * pixel_um, -height / 2 * pixel_um, height / 2 * pixel_um)
            axis.imshow(sector_rgba(snapshot["states"], names, groups, snapshot["mask"]), extent=extent,
                        interpolation="nearest")
            radius = float(np.sqrt(snapshot["mask"].sum() / np.pi)) * pixel_um
            draw_sectors(axis, (0.0, 0.0), radius, [False] * len(groups))
            half = 1.05 * (largest if to_scale else snapshot["diameter"] / 2.0)
            axis.set_xlim(-half, half)
            axis.set_ylim(-1.3 * half, half)
            axis.set_aspect("equal")
            axis.axis("off")
            if not to_scale or index == 0:
                scale_bar(axis, half, -1.12 * half)
            number_badge(axis, snapshot["number"], (0.0, 1.0), "axes fraction", offset=(4, 0))
            axis.annotate(f"{SWEEPS[snapshot['source']][0]}\n{snapshot['diameter']:g} µm", xy=(0.5, 1.0),
                          xycoords="axes fraction", xytext=(0, 3), textcoords="offset points", ha="center",
                          va="bottom", fontsize=FONT["annotation"], color=INK, annotation_clip=False)
            if index == 0:
                label_panel(axis, "b", f"NCA at {snapshot_hour:g} h", snapshot_note, offset=(0, 24))
        channel_axis = figure.add_subplot(bottom[0, -1])
        channel_axis.axis("off")
        entries = marker_legend_entries([f"{_g}/{_m}" for _g in groups for _m in FIGURE_CHANNELS[_g]], groups,
                                        sectors=groups)
        channel_handles, channel_labels, columns = rows_to_columns(entries)
        channel_axis.legend(channel_handles, channel_labels, ncol=columns, loc="center left", **legend_options())
    return figure


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Colony size figure
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **a:** predicted prevalence against colony diameter, at the measured cell density (left) and with the measured number of cells per colony (right); stars: the NCA started from the true colonies; optional row of the cells per mm² each size got
    - **b:** chosen 48 h snapshots; their numbers sit above the matching sweep and diameter in **a**
    - One figure per condition; data from `size_density_explorer.py` with the size sweeps switched on
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
    results_path = mo.ui.text(
        "output/size_density/", label="Size × density results: file or folder of parts (relative to the repository)",
        full_width=True,
    )
    results_path
    return (results_path,)


@app.cell(hide_code=True)
def _(results_path):
    _path = REPOSITORY_ROOT / results_path.value.strip()
    mo.stop(
        not (_path.is_file() or any(_path.glob("part_*.npz"))),
        mo.callout(f"No results file, or folder of `part_*.npz`, at `{_path}`.", kind="warn"),
    )
    with mo.status.spinner("Loading results"):
        loaded_results = load_size_density_results(_path)
    _sources = pd.Series([_r["source"] for _r in loaded_results]).value_counts()
    mo.stop(
        not any(_s in _sources for _s in SWEEPS),
        mo.callout("These results have no size sweeps; run the sweep with them switched on.", kind="warn"),
    )
    mo.md("- Results: " + ", ".join(f"{_n} {_s}" for _s, _n in _sources.items()))
    return (loaded_results,)


@app.cell(hide_code=True)
def _(loaded_results):
    # Cell type files that match the thresholds the counts were saved with, per projection
    _thresholds = {}
    for _r in loaded_results:
        _thresholds.setdefault(_r["projection"], _r["cell_type_thresholds"])
    _matches = {_p: matching_cell_type_files(_t) for _p, _t in _thresholds.items()}
    _missing = [_p for _p, _files in _matches.items() if not _files]
    mo.stop(
        bool(_missing),
        mo.callout(f"No cell type file in `{CELL_TYPE_DIRECTORY.name}` matches the saved thresholds of: "
                   + ", ".join(_missing), kind="danger"),
    )
    cell_type_files = mo.ui.dictionary({
        _p: mo.ui.dropdown({_f.name: str(_f) for _f in _files}, value=_files[-1].name,
                           label=f"Cell type file ({_p})")
        for _p, _files in _matches.items()
    })
    cell_type_files
    return (cell_type_files,)


@app.cell(hide_code=True)
def _(cell_type_files):
    rules_by_projection = {_p: load_cell_type_rules(_path) for _p, _path in cell_type_files.value.items()}
    _label_sets = {tuple(label_names(_rules)) for _rules in rules_by_projection.values()}
    mo.stop(len(_label_sets) > 1, mo.callout("The cell type files must define the same cell types.", kind="danger"))
    cell_type_labels = next(iter(_label_sets))
    return cell_type_labels, rules_by_projection


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Content
    """)
    return


@app.cell(hide_code=True)
def _(cell_type_labels, loaded_results):
    _labels = {_r["model_id"]: _r["label"] for _r in loaded_results}
    figure_models = mo.ui.multiselect(
        {_label: _id for _id, _label in _labels.items()}, value=list(_labels.values()), label="Models (pooled)",
        full_width=True,
    )
    _conditions = list(loaded_results[0]["conditions"])
    figure_condition = mo.ui.dropdown({condition_label(_c): _c for _c in _conditions},
                                      value=condition_label(_conditions[0]), label="Condition")
    figure_cell_types = mo.ui.multiselect(
        list(cell_type_labels), value=[_c for _c in cell_type_labels if _c not in (SEVERAL, OTHER)],
        label="Cell types",
    )
    figure_basis = mo.ui.dropdown({_label: _key for _key, _label in SHARE_LABELS.items()},
                                  value=SHARE_LABELS["pattern_counts"], label="Prevalence as")
    figure_resolve = mo.ui.slider(0.0, 1.0, value=0.0, step=0.05, label="Resolve multiple matches")
    figure_uncertainty = mo.ui.dropdown(
        {"SD": "std", "SEM": "sem", "95% interval": "percentile", "None": "none"}, value="SD", label="Band",
    )
    figure_show_density = mo.ui.switch(value=False, label="Cells per mm² row")
    mo.vstack([
        figure_models,
        mo.hstack([figure_condition, figure_cell_types, figure_basis, figure_resolve], justify="start", wrap=True),
        mo.hstack([figure_uncertainty, figure_show_density], justify="start"),
        mo.md("- Share of cells has no true data (it has no cell counts)"),
    ])
    return (
        figure_basis,
        figure_cell_types,
        figure_condition,
        figure_models,
        figure_resolve,
        figure_show_density,
        figure_uncertainty,
    )


@app.cell(hide_code=True)
def _(
    figure_basis,
    figure_condition,
    figure_models,
    figure_resolve,
    figure_uncertainty,
    loaded_results,
    rules_by_projection,
):
    figure_results = [_r for _r in loaded_results if _r["model_id"] in figure_models.value]
    mo.stop(not figure_results, mo.md("- Choose at least one model"))
    figure_rows = share_rows(figure_results, rules_by_projection, figure_basis.value, figure_resolve.value)
    figure_summary = summarise(
        figure_rows[figure_rows["condition"] == figure_condition.value], figure_uncertainty.value,
    )
    return figure_results, figure_rows, figure_summary


@app.cell(hide_code=True)
def _(figure_results):
    # Snapshots: images are kept for the first run of each job part
    _with_images = [_r for _r in figure_results if _r["source"] in SWEEPS and _r["final_states"] is not None]
    _models = {_r["label"]: _r["model_id"] for _r in _with_images}
    snapshot_model = mo.ui.dropdown(_models, value=next(iter(_models), None), label="Snapshots of model")
    _sizes = {
        _source: sorted({float(_d) for _r in _with_images if _r["source"] == _source for _d in _r["diameters_um"]})
        for _source in SWEEPS
    }
    # Default: the smallest and largest colony of each sweep
    snapshot_sizes = mo.ui.dictionary({
        _source: mo.ui.multiselect(
            {f"{_d:g} µm": _d for _d in _sizes[_source]},
            value=[f"{_d:g} µm" for _d in dict.fromkeys(_sizes[_source][:1] + _sizes[_source][-1:])],
            label=f"{SWEEPS[_source][0]} snapshots",
        )
        for _source in SWEEPS if _sizes[_source]
    })
    _names = set(figure_results[0]["state_channel_names"])
    _groups = {GROUP_LABELS[_g]: _g for _g, _markers in FIGURE_CHANNELS.items() if _names & set(_markers)}
    snapshot_groups = mo.ui.multiselect(_groups, value=list(_groups), label="Sectors")
    snapshot_to_scale = mo.ui.dropdown(
        {"Real size (one scale)": True, "Same size (each fills its panel)": False},
        value="Real size (one scale)", label="Colonies in b",
    )
    (
        mo.vstack([
            mo.md("### Snapshots (b)\n\n- Numbered in order: fixed density first, then fixed cell count, small to large"),
            mo.hstack([snapshot_model, snapshot_groups, snapshot_to_scale], justify="start"),
            snapshot_sizes,
        ])
        if _with_images else mo.callout("No snapshot images in these results.", kind="warn")
    )
    return snapshot_groups, snapshot_model, snapshot_sizes, snapshot_to_scale


@app.cell(hide_code=True)
def _(figure_condition, figure_results, snapshot_model, snapshot_sizes):
    # The chosen colonies' end states: the first replicate with an image of each size
    snapshots = []
    _replicates = set()
    for _source in SWEEPS:
        _candidates = sorted(
            [
                _r for _r in figure_results
                if _r["source"] == _source and _r["model_id"] == snapshot_model.value and _r["final_states"] is not None
            ],
            key=lambda _r: _r["replicate"],
        )
        for _diameter in sorted(snapshot_sizes.value.get(_source, [])):
            for _r in _candidates:
                _sizes = np.asarray(_r["diameters_um"], dtype=float)
                if np.any(np.isclose(_sizes, _diameter)):
                    _s = int(np.flatnonzero(np.isclose(_sizes, _diameter))[0])
                    _k = list(_r["conditions"]).index(figure_condition.value)
                    snapshots.append({
                        "number": len(snapshots) + 1, "source": _source, "diameter": float(_diameter),
                        "states": _r["final_states"][_s, 0, _k],
                        "mask": np.asarray(_r["colony_masks"][_s], dtype=bool),
                        "pixel_um": float(_r["pixel_um"]),
                    })
                    _replicates.add(_r["replicate"])
                    break
    # No model name in the figure: the snapshot model is chosen above
    snapshot_note = "one NCA run" + (
        f", replicate {', '.join(map(str, sorted(_replicates)))}" if _replicates else ""
    )
    return snapshot_note, snapshots


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Figure
    """)
    return


@app.cell(hide_code=True)
def _(
    figure_basis,
    figure_cell_types,
    figure_condition,
    figure_results,
    figure_rows,
    figure_show_density,
    figure_summary,
    figure_uncertainty,
    snapshot_groups,
    snapshot_note,
    snapshot_to_scale,
    snapshots,
):
    mo.stop(not figure_cell_types.value, mo.md("- Choose at least one cell type"))
    mo.stop(figure_summary.empty, mo.md("- No size sweep results for these choices"))
    _cell_types = [_c for _c in figure_rows["cell_type"].unique() if _c in figure_cell_types.value]
    _groups = [_g for _g in FIGURE_CHANNELS if _g in snapshot_groups.value]
    with paper_style():
        size_sweep_figure = draw_figure(
            figure_summary, snapshots if _groups else [], _cell_types, figure_condition.value,
            SHARE_LABELS[figure_basis.value], figure_show_density.value, figure_uncertainty.value != "none",
            _groups, list(figure_results[0]["state_channel_names"]), snapshot_to_scale.value,
            float(figure_rows["training_diameter_um"].mean()), float(figure_results[0]["snapshot_hour"]),
            snapshot_note,
        )
    size_sweep_figure
    return (size_sweep_figure,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save
    """)
    return


@app.cell(hide_code=True)
def _(figure_condition):
    figure_name = mo.ui.text(f"size_sweep_{figure_condition.value}", label="File name")
    figure_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    figure_save = mo.ui.run_button(label=f"Save to figures/{FIGURE_FOLDER}/")
    mo.hstack([figure_name, figure_svg, figure_save], justify="start")
    return figure_name, figure_save, figure_svg


@app.cell(hide_code=True)
def _(figure_condition, figure_name, figure_save, figure_svg, size_sweep_figure):
    mo.stop(not figure_save.value)
    _paths = save_figure(size_sweep_figure, figure_name.value.strip() or f"size_sweep_{figure_condition.value}",
                         FIGURE_FOLDER, svg=figure_svg.value)
    mo.md("Saved " + ", ".join(f"`{_p.name}`" for _p in _paths))
    return


if __name__ == "__main__":
    app.run()
