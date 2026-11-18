"""Baseline NCA trajectories against their data: one model per projection, one per resolution,
a grid of both at one timepoint, and final losses per projection or resolution.

Run with:
    marimo edit Experiments/micropatterns/figures/projection_resolution_figure.py

Reads ``MODEL_STORE_ROOT`` and ``DATA_PATH_BASE``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="projection and resolution trajectories",
)

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from dotenv import load_dotenv

    from Experiments.micropatterns.figures.data import baseline_rollout, load_selection_records, model_final_losses
    from Experiments.micropatterns.figures.drawing import grid_figure, snapshot_loss_figure, trajectory_figure
    from Experiments.micropatterns.figures.style import (
        GROUP_LABELS,
        INK,
        PROJECTION_LABELS,
        SOURCE_COLOURS,
        SOURCE_LABELS,
        WIDTH,
        ZOOM_COLOURS,
        paper_style,
        save_figure,
    )

    load_dotenv()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NCA trajectories per projection and resolution

    - One baseline (no KO) model per projection, from a selection exported by `model_registry_explorer.py`
    - Each model starts from the 0h state of the chosen replicate, loaded and scaled as for its training
    - Rows: true data and NCA for each projection; one sector per experiment group
    - Saved to `figures/schematics/`
    """)
    return


@app.cell(hide_code=True)
def _():
    projection_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    store_root = mo.ui.text(
        value=os.getenv("MODEL_STORE_ROOT", str(Path(mo.notebook_dir()).parents[2] / "models")),
        label="Model store",
        full_width=True,
    )
    data_root = mo.ui.text(
        value=os.getenv("DATA_PATH_BASE", ""),
        label="Data root (contains 260726_nca_dataset)",
        full_width=True,
    )
    mo.vstack([projection_selection_path, store_root, data_root])
    return data_root, projection_selection_path, store_root


@app.cell(hide_code=True)
def _(projection_selection_path, store_root):
    _path = Path(projection_selection_path.value).expanduser()
    mo.stop(not projection_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    _records = load_selection_records(_path, store_root.value)
    _options = {
        _projection: {_r["option_label"]: _r["model_id"] for _r in _records if _r["projection"] == _projection}
        for _projection in PROJECTION_LABELS
    }
    mo.stop(not any(_options.values()), mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    # One model per projection; projections without a model are left out of the figure
    projection_models = mo.ui.dictionary({
        _projection: mo.ui.dropdown(options=_o, value=next(iter(_o)), label=PROJECTION_LABELS[_projection][0],
                                    full_width=True)
        for _projection, _o in _options.items() if _o
    })
    projection_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    projection_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_projection_rollouts = mo.ui.run_button(label="Roll out models")
    mo.vstack([
        projection_models,
        mo.hstack([projection_replicate, projection_seed, run_projection_rollouts], justify="start"),
    ])
    return (
        projection_models,
        projection_replicate,
        projection_seed,
        run_projection_rollouts,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    projection_models,
    projection_replicate,
    projection_seed,
    run_projection_rollouts,
    store_root,
):
    mo.stop(not run_projection_rollouts.value, mo.md("Choose the models, then roll them out."))
    mo.stop(not data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
    _replicate = int(projection_replicate.value)
    projection_rollouts = {}
    with mo.status.progress_bar(total=len(projection_models.value), title="Loading data and rolling out") as _bar:
        for _index, (_projection, _model_id) in enumerate(projection_models.value.items()):
            projection_rollouts[_projection] = baseline_rollout(
                store_root.value, _model_id, data_root.value, _replicate, int(projection_seed.value) + _index,
            )
            _bar.update()
    _in_training = [_p for _p, _r in projection_rollouts.items() if _replicate in _r["train_replicates"]]
    mo.md(
        f"Replicate {_replicate} is in the training split of: {', '.join(_in_training)}"
        if _in_training else f"Replicate {_replicate} is held out from every model's training split."
    )
    return (projection_rollouts,)


@app.cell(hide_code=True)
def _(projection_rollouts):
    _names = next(iter(projection_rollouts.values()))["channel_names"]
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n))
    projection_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    projection_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    projection_row_order = mo.ui.dropdown(
        options={"True and NCA per projection": "projection", "True data, then NCA": "source"},
        value="True and NCA per projection",
        label="Rows",
    )
    projection_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    projection_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    mo.hstack([projection_groups, projection_gamma, projection_row_order, projection_colony_size, projection_hatch],
              justify="start")
    return (
        projection_colony_size,
        projection_gamma,
        projection_groups,
        projection_hatch,
        projection_row_order,
    )


@app.cell(hide_code=True)
def _(
    projection_colony_size,
    projection_gamma,
    projection_groups,
    projection_hatch,
    projection_rollouts,
    projection_row_order,
):
    mo.stop(not projection_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(projection_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in projection_groups.value]
    # Blocks of rows, each with a bracketed label: (label, colour, rows), and a
    # row is (projection, source, row label, row label colour).
    _projections = [_p for _p in PROJECTION_LABELS if _p in projection_rollouts]
    if projection_row_order.value == "projection":
        _blocks = [
            (f"{PROJECTION_LABELS[_p][0]}\n{PROJECTION_LABELS[_p][1]}", INK,
             [(_p, _src, SOURCE_LABELS[_src], SOURCE_COLOURS[_src]) for _src in SOURCE_LABELS])
            for _p in _projections
        ]
    else:
        _blocks = [
            (SOURCE_LABELS[_src], SOURCE_COLOURS[_src],
             [(_p, _src, PROJECTION_LABELS[_p][0], INK) for _p in _projections])
            for _src in SOURCE_LABELS
        ]
    with paper_style():
        projection_trajectories = trajectory_figure(
            projection_rollouts, _blocks, _groups, projection_gamma.value, projection_colony_size.value,
            projection_hatch.value,
        )
    projection_trajectories
    return (projection_trajectories,)


@app.cell(hide_code=True)
def _():
    projection_figure_name = mo.ui.text(value="projection_trajectories", label="File name")
    save_projection_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_projection_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([projection_figure_name, save_projection_svg, save_projection_figure], justify="start")
    return projection_figure_name, save_projection_figure, save_projection_svg


@app.cell
def _(
    projection_figure_name,
    projection_trajectories,
    save_projection_figure,
    save_projection_svg,
):
    mo.stop(not save_projection_figure.value)
    _paths = save_figure(projection_trajectories, projection_figure_name.value, "schematics",
                         svg=save_projection_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NCA at different resolutions

    - Baseline models of one projection at several downsampling ratios, from an exported selection
    - Model store and data root are set in the section above
    - Zoom squares magnify the same part of each colony at one timepoint, drawn pixel by pixel
    """)
    return


@app.cell(hide_code=True)
def _():
    resolution_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    resolution_selection_path
    return (resolution_selection_path,)


@app.cell(hide_code=True)
def _(resolution_selection_path, store_root):
    _path = Path(resolution_selection_path.value).expanduser()
    mo.stop(not resolution_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    resolution_records = load_selection_records(_path, store_root.value)
    _present = [_p for _p in PROJECTION_LABELS if any(_r["projection"] == _p for _r in resolution_records)]
    mo.stop(not _present, mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    resolution_projection = mo.ui.dropdown(
        options={PROJECTION_LABELS[_p][0]: _p for _p in _present},
        value=PROJECTION_LABELS[_present[0]][0],
        label="Projection",
    )
    resolution_projection
    return resolution_projection, resolution_records


@app.cell(hide_code=True)
def _(resolution_projection, resolution_records):
    _records = [_r for _r in resolution_records if _r["projection"] == resolution_projection.value]
    # One model per downsampling ratio, finest first
    _options = {}
    for _downsample in sorted({_r["downsample"] for _r in _records}):
        _options[str(_downsample)] = {
            _r["option_label"]: _r["model_id"] for _r in _records if _r["downsample"] == _downsample
        }
    resolution_models = mo.ui.dictionary({
        _downsample: mo.ui.dropdown(options=_o, value=next(iter(_o)), label=f"Δ = {_downsample}",
                                    full_width=True)
        for _downsample, _o in _options.items()
    })
    resolution_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    resolution_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_resolution_rollouts = mo.ui.run_button(label="Roll out models")
    mo.vstack([
        resolution_models,
        mo.hstack([resolution_replicate, resolution_seed, run_resolution_rollouts], justify="start"),
    ])
    return (
        resolution_models,
        resolution_replicate,
        resolution_seed,
        run_resolution_rollouts,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    resolution_models,
    resolution_replicate,
    resolution_seed,
    run_resolution_rollouts,
    store_root,
):
    mo.stop(not run_resolution_rollouts.value, mo.md("Choose the models, then roll them out."))
    mo.stop(not data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
    _replicate = int(resolution_replicate.value)
    resolution_rollouts = {}
    with mo.status.progress_bar(total=len(resolution_models.value), title="Loading data and rolling out") as _bar:
        for _index, (_downsample, _model_id) in enumerate(resolution_models.value.items()):
            resolution_rollouts[int(_downsample)] = baseline_rollout(
                store_root.value, _model_id, data_root.value, _replicate, int(resolution_seed.value) + _index,
            )
            _bar.update()
    _in_training = [str(_d) for _d, _r in resolution_rollouts.items() if _replicate in _r["train_replicates"]]
    mo.md(
        f"Replicate {_replicate} is in the training split of downsample: {', '.join(_in_training)}"
        if _in_training else f"Replicate {_replicate} is held out from every model's training split."
    )
    return (resolution_rollouts,)


@app.cell(hide_code=True)
def _(resolution_rollouts):
    _first = next(iter(resolution_rollouts.values()))
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _first["channel_names"] if "/" in _n))
    resolution_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    resolution_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    resolution_row_order = mo.ui.dropdown(
        options={"True and NCA per resolution": "resolution", "True data, then NCA": "source"},
        value="True and NCA per resolution",
        label="Rows",
    )
    resolution_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    resolution_width = mo.ui.dropdown(options={"double column": WIDTH["double"], "single column": WIDTH["single"]},
                                      value="double column", label="Width")
    resolution_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    resolution_zoom_time = mo.ui.dropdown(
        options={_t: _i for _i, _t in enumerate(_first["time_labels"])},
        value=_first["time_labels"][-1],
        label="Zoom timepoint",
    )
    resolution_zoom_count = mo.ui.slider(0, len(ZOOM_COLOURS), step=1, value=2, show_value=True, label="Zooms")
    resolution_zoom_width = mo.ui.slider(0.05, 0.5, step=0.01, value=0.2, show_value=True,
                                         label="Zoom box (fraction of colony)")
    # Whole-pixel boxes: N x N pixels at the coarsest resolution, 2N x 2N at the next, ...
    resolution_zoom_lock = mo.ui.checkbox(value=True, label="Lock zoom boxes to whole pixels")
    resolution_zoom_pixels = mo.ui.slider(1, 32, step=1, value=6, show_value=True,
                                          label="Pixels per box side, coarsest resolution")
    mo.vstack([
        mo.hstack([resolution_groups, resolution_gamma, resolution_row_order], justify="start"),
        mo.hstack([resolution_colony_size, resolution_width, resolution_hatch], justify="start"),
        mo.hstack([resolution_zoom_time, resolution_zoom_count], justify="start"),
        mo.hstack([resolution_zoom_lock, resolution_zoom_pixels, resolution_zoom_width], justify="start"),
    ])
    return (
        resolution_colony_size,
        resolution_gamma,
        resolution_groups,
        resolution_hatch,
        resolution_row_order,
        resolution_width,
        resolution_zoom_count,
        resolution_zoom_lock,
        resolution_zoom_pixels,
        resolution_zoom_time,
        resolution_zoom_width,
    )


@app.cell(hide_code=True)
def _(resolution_zoom_count):
    # Zoom centres as fractions of the colony (x right, y down)
    _defaults = [(0.5, 0.5), (0.5, 0.12), (0.2, 0.5), (0.8, 0.5)]
    resolution_zoom_centres = mo.ui.array([
        mo.ui.dictionary({
            "x": mo.ui.slider(0.0, 1.0, step=0.005, value=_x, show_value=True, label="x"),
            "y": mo.ui.slider(0.0, 1.0, step=0.005, value=_y, show_value=True, label="y"),
        })
        for _x, _y in _defaults[:resolution_zoom_count.value]
    ])
    mo.hstack([
        mo.vstack([mo.md(f"Zoom {_k + 1}"), _centre])
        for _k, _centre in enumerate(resolution_zoom_centres)
    ], justify="start")
    return (resolution_zoom_centres,)


@app.cell(hide_code=True)
def _(
    resolution_colony_size,
    resolution_gamma,
    resolution_groups,
    resolution_hatch,
    resolution_rollouts,
    resolution_row_order,
    resolution_width,
    resolution_zoom_centres,
    resolution_zoom_lock,
    resolution_zoom_pixels,
    resolution_zoom_time,
    resolution_zoom_width,
):
    mo.stop(not resolution_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(resolution_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in resolution_groups.value]

    # Blocks of rows as in the projection figure, keyed by downsampling ratio
    _downsamples = sorted(resolution_rollouts)
    if resolution_row_order.value == "resolution":
        _blocks = [
            (f"Δ = {_d}", INK, [(_d, _src, SOURCE_LABELS[_src], SOURCE_COLOURS[_src]) for _src in SOURCE_LABELS])
            for _d in _downsamples
        ]
    else:
        _blocks = [
            (SOURCE_LABELS[_src], SOURCE_COLOURS[_src], [(_d, _src, f"Δ = {_d}", INK) for _d in _downsamples])
            for _src in SOURCE_LABELS
        ]
    _zoom = None
    if resolution_zoom_centres.value:
        _time_label = next(iter(resolution_rollouts.values()))["time_labels"][resolution_zoom_time.value]
        _zoom = {
            "time": resolution_zoom_time.value,
            "centres": [(_c["x"], _c["y"]) for _c in resolution_zoom_centres.value],
            "width": resolution_zoom_width.value,
            "title": f"Zoom ×{1 / resolution_zoom_width.value:.1f} at {_time_label}",
        }
        if resolution_zoom_lock.value:
            # Downsampling ratios are powers of 2, so each finer one has a whole multiple of the pixels
            _coarsest = max(_downsamples)
            _n = resolution_zoom_pixels.value
            _zoom["pixels"] = {_d: _n * _coarsest // _d for _d in _downsamples}
            _zoom["title"] = f"Zoom at {_time_label}, {_n}×{_n} px at Δ = {_coarsest}"
    with paper_style():
        resolution_trajectories = trajectory_figure(
            resolution_rollouts, _blocks, _groups, resolution_gamma.value, resolution_colony_size.value,
            resolution_hatch.value, zoom=_zoom, width=resolution_width.value,
        )
    resolution_trajectories
    return (resolution_trajectories,)


@app.cell(hide_code=True)
def _():
    resolution_figure_name = mo.ui.text(value="resolution_trajectories", label="File name")
    save_resolution_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_resolution_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([resolution_figure_name, save_resolution_svg, save_resolution_figure], justify="start")
    return resolution_figure_name, save_resolution_figure, save_resolution_svg


@app.cell
def _(
    resolution_figure_name,
    resolution_trajectories,
    save_resolution_figure,
    save_resolution_svg,
):
    mo.stop(not save_resolution_figure.value)
    _paths = save_figure(resolution_trajectories, resolution_figure_name.value, "schematics",
                         svg=save_resolution_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NCA across projection and resolution

    - Baseline models for each projection and downsampling ratio, from one exported selection
    - One timepoint (48h by default): columns are downsampling ratios, rows are true data and NCA per projection
    - Model store and data root are set in the first section
    """)
    return


@app.cell(hide_code=True)
def _():
    combined_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    combined_selection_path
    return (combined_selection_path,)


@app.cell(hide_code=True)
def _(combined_selection_path, store_root):
    _path = Path(combined_selection_path.value).expanduser()
    mo.stop(not combined_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    _records = [_r for _r in load_selection_records(_path, store_root.value) if _r["projection"] in PROJECTION_LABELS]
    mo.stop(not _records, mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    # One model per (projection, downsampling ratio) found in the selection
    _options = {}
    for _projection in PROJECTION_LABELS:
        for _downsample in sorted({_r["downsample"] for _r in _records}):
            _o = {_r["option_label"]: _r["model_id"] for _r in _records
                  if _r["projection"] == _projection and _r["downsample"] == _downsample}
            if _o:
                _options[f"{_projection}|{_downsample}"] = _o
    combined_models = mo.ui.dictionary({
        _key: mo.ui.dropdown(options=_o, value=next(iter(_o)), full_width=True,
                             label=f"{PROJECTION_LABELS[_key.split('|')[0]][0]}, Δ = {_key.split('|')[1]}")
        for _key, _o in _options.items()
    })
    combined_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    combined_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_combined_rollouts = mo.ui.run_button(label="Roll out models")
    mo.vstack([
        combined_models,
        mo.hstack([combined_replicate, combined_seed, run_combined_rollouts], justify="start"),
    ])
    return combined_models, combined_replicate, combined_seed, run_combined_rollouts


@app.cell(hide_code=True)
def _(
    combined_models,
    combined_replicate,
    combined_seed,
    data_root,
    run_combined_rollouts,
    store_root,
):
    mo.stop(not run_combined_rollouts.value, mo.md("Choose the models, then roll them out."))
    mo.stop(not data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
    _replicate = int(combined_replicate.value)
    # Keyed by (projection, downsampling ratio)
    combined_rollouts = {}
    with mo.status.progress_bar(total=len(combined_models.value), title="Loading data and rolling out") as _bar:
        for _index, (_key, _model_id) in enumerate(combined_models.value.items()):
            _projection, _downsample = _key.split("|")
            combined_rollouts[(_projection, int(_downsample))] = baseline_rollout(
                store_root.value, _model_id, data_root.value, _replicate, int(combined_seed.value) + _index,
            )
            _bar.update()
    _in_training = [f"{PROJECTION_LABELS[_p][0]} Δ = {_d}" for (_p, _d), _r in combined_rollouts.items()
                    if _replicate in _r["train_replicates"]]
    mo.md(
        f"Replicate {_replicate} is in the training split of: {', '.join(_in_training)}"
        if _in_training else f"Replicate {_replicate} is held out from every model's training split."
    )
    return (combined_rollouts,)


@app.cell(hide_code=True)
def _(combined_rollouts):
    _first = next(iter(combined_rollouts.values()))
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _first["channel_names"] if "/" in _n))
    combined_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    combined_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    _times = list(dict.fromkeys(_t for _r in combined_rollouts.values() for _t in _r["time_labels"]))
    combined_time = mo.ui.dropdown(options=_times, value="48h" if "48h" in _times else _times[-1],
                                   label="Timepoint")
    combined_row_order = mo.ui.dropdown(
        options={"True and NCA per projection": "projection", "True data, then NCA": "source"},
        value="True and NCA per projection",
        label="Rows",
    )
    combined_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size")
    combined_width = mo.ui.dropdown(options={"double column": WIDTH["double"], "single column": WIDTH["single"]},
                                    value="double column", label="Width")
    combined_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    mo.vstack([
        mo.hstack([combined_groups, combined_gamma, combined_time, combined_row_order], justify="start"),
        mo.hstack([combined_colony_size, combined_width, combined_hatch], justify="start"),
    ])
    return (
        combined_colony_size,
        combined_gamma,
        combined_groups,
        combined_hatch,
        combined_row_order,
        combined_time,
        combined_width,
    )


@app.cell(hide_code=True)
def _(
    combined_colony_size,
    combined_gamma,
    combined_groups,
    combined_hatch,
    combined_rollouts,
    combined_row_order,
    combined_time,
    combined_width,
):
    mo.stop(not combined_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(combined_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in combined_groups.value]
    _projections = [_p for _p in PROJECTION_LABELS if any(_k[0] == _p for _k in combined_rollouts)]
    _columns = [(_d, f"Δ = {_d}") for _d in sorted({_k[1] for _k in combined_rollouts})]
    if combined_row_order.value == "projection":
        _blocks = [
            (f"{PROJECTION_LABELS[_p][0]}\n{PROJECTION_LABELS[_p][1]}", INK,
             [(_p, _src, SOURCE_LABELS[_src], SOURCE_COLOURS[_src]) for _src in SOURCE_LABELS])
            for _p in _projections
        ]
    else:
        _blocks = [
            (SOURCE_LABELS[_src], SOURCE_COLOURS[_src],
             [(_p, _src, PROJECTION_LABELS[_p][0], INK) for _p in _projections])
            for _src in SOURCE_LABELS
        ]
    with paper_style():
        combined_grid = grid_figure(
            combined_rollouts, _blocks, _columns, _groups, combined_gamma.value, combined_colony_size.value,
            combined_hatch.value, combined_time.value, title=f"Downsampling ratio, at {combined_time.value}",
            width=combined_width.value,
        )
    combined_grid
    return (combined_grid,)


@app.cell(hide_code=True)
def _():
    combined_figure_name = mo.ui.text(value="projection_resolution_grid", label="File name")
    save_combined_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_combined_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([combined_figure_name, save_combined_svg, save_combined_figure], justify="start")
    return combined_figure_name, save_combined_figure, save_combined_svg


@app.cell
def _(combined_figure_name, combined_grid, save_combined_figure, save_combined_svg):
    mo.stop(not save_combined_figure.value)
    _paths = save_figure(combined_grid, combined_figure_name.value, "schematics", svg=save_combined_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Final losses across projection and resolution

    - Losses are recomputed from the saved models with the training loss and its final curriculum weights (`Experiments/micropatterns/final_losses.py`), the same seed for every model, and cached in `figures/cache`
    - Step 1: choose the models to evaluate (all models of the selection by default) and compute their losses here, or export them as a job for EIDF and load its results
    - Step 2: statistics per projection and Δ over the evaluated models (mean, standard deviation, count)
    - Each model is scored on its own training and validation replicates (from its config); the per-model table lists them
    - Step 3: figure with columns of projections (at one Δ) or of Δ (for one projection): true data and one chosen NCA at one timepoint above the losses; bar = mean over the evaluated models of that column, dot = one model
    - Model store and data root are set in the first section
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_selection_path = mo.ui.text(
        placeholder="/path/to/model_registry_selection.yaml",
        label="Exported registry selection",
        full_width=True,
    )
    loss_selection_path
    return (loss_selection_path,)


@app.cell(hide_code=True)
def _(loss_selection_path, store_root):
    _path = Path(loss_selection_path.value).expanduser()
    mo.stop(not loss_selection_path.value.strip(), mo.md("Choose an exported model selection YAML file."))
    mo.stop(not _path.is_file(), mo.callout(f"Selection file not found: `{_path}`", kind="danger"))
    loss_records = [_r for _r in load_selection_records(_path, store_root.value) if _r["projection"] in PROJECTION_LABELS]
    mo.stop(not loss_records, mo.callout("The selection has no slice, max or sum models.", kind="warn"))
    _options = {_r["option_label"]: _r["model_id"] for _r in loss_records}
    loss_included = mo.ui.multiselect(options=_options, value=list(_options), label="Models to evaluate",
                                      full_width=True)
    loss_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Loss seed")
    # Here: computed in this notebook (cached); remote: a job on EIDF, loaded back below
    loss_refresh = mo.ui.checkbox(value=False, label="Recompute cached losses")
    run_losses = mo.ui.run_button(label="Compute here")
    loss_job_name = mo.ui.text("final_losses", label="Job name")
    loss_job_models_per_part = mo.ui.number(start=1, stop=100, step=1, value=1, label="Models per part (pod)")
    loss_job_output = mo.ui.text("/workspace/writeable/output/final_losses",
                                 label="Results folder on the cluster (the job name is added)", full_width=True)
    export_loss_job = mo.ui.run_button(label="Export remote job")
    loss_results_path = mo.ui.text("output/final_losses/final_losses",
                                   label="Folder of a remote job's parts (relative to the repository root)",
                                   full_width=True)
    load_losses = mo.ui.run_button(label="Load remote results")
    mo.vstack([
        loss_included,
        loss_seed,
        mo.md("**Compute here** (slow at fine resolutions; needs a large GPU)"),
        mo.hstack([loss_refresh, run_losses], justify="start"),
        mo.md("**Or run on EIDF** (`docs/remote_notebook_jobs.md`)"),
        mo.hstack([loss_job_name, loss_job_models_per_part, export_loss_job], justify="start"),
        loss_job_output,
        mo.hstack([loss_results_path, load_losses], justify="start"),
    ])
    return (
        export_loss_job,
        load_losses,
        loss_included,
        loss_job_models_per_part,
        loss_job_name,
        loss_job_output,
        loss_records,
        loss_refresh,
        loss_results_path,
        loss_seed,
        run_losses,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    export_loss_job,
    loss_included,
    loss_job_models_per_part,
    loss_job_name,
    loss_job_output,
    loss_seed,
    store_root,
):
    mo.stop(not export_loss_job.value)
    mo.stop(not loss_included.value, mo.callout("Choose at least one model.", kind="warn"))
    from Experiments.micropatterns.final_losses import job_configs as _job_configs
    from Experiments.notebook_jobs import REPOSITORY_ROOT as _root
    from Experiments.notebook_jobs import write_job_manifest as _write_job_manifest

    _name = loss_job_name.value.strip() or "final_losses"
    _output = f"{loss_job_output.value.strip().rstrip('/')}/{_name}"
    _configs = _job_configs(loss_included.value, int(loss_seed.value), _output, store_root.value, data_root.value,
                            int(loss_job_models_per_part.value))
    # Next to the training manifests, so a git checkout on the cluster has it
    _manifest = _write_job_manifest(
        _root / "Experiments/micropatterns/conf/generated/final_losses" / _name / "manifest.yaml",
        f"final_losses_{_name}",
        _configs,
    )
    _relative = _manifest.relative_to(_root)
    mo.callout(
        mo.md(
            f"Wrote {_relative} ({len(_configs)} parts, {len(loss_included.value)} models). "
            f"On the cluster, from the repository root:\n\n"
            f"```bash\nbash launch/launch_batch_multi_job.sh "
            f"Experiments/micropatterns/final_losses.py {_relative} <workers> H200\n```\n\n"
            f"Then copy `{_output}/` back to `output/final_losses/{_name}/` and load that folder above."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(
    data_root,
    load_losses,
    loss_included,
    loss_refresh,
    loss_results_path,
    loss_seed,
    run_losses,
    store_root,
):
    mo.stop(not (run_losses.value or load_losses.value),
            mo.md("Choose the models, then compute their losses here or load a remote job's results."))
    mo.stop(not loss_included.value, mo.callout("Choose at least one model.", kind="warn"))
    # {model ID: {"train": {score: value}, "validation": {score: value}, "replicates": ...}}
    if load_losses.value:
        from Experiments.micropatterns.final_losses import load_final_losses as _load_final_losses

        _loaded = _load_final_losses(loss_results_path.value.strip())
        loss_values = {_m: _loaded[_m] for _m in loss_included.value if _m in _loaded}
        _missing = len(loss_included.value) - len(loss_values)
        _seeds = sorted({_v.get("seed") for _v in loss_values.values()})
        _message = (f"Loaded losses of {len(loss_values)} models (seed {', '.join(map(str, _seeds))})"
                    + (f"; {_missing} chosen models are not in the results" if _missing else "") + ".")
    else:
        mo.stop(not data_root.value.strip(), mo.callout("Set the data root first.", kind="warn"))
        loss_values = {}
        with mo.status.progress_bar(total=len(loss_included.value), title="Computing final losses") as _bar:
            for _model_id in loss_included.value:
                loss_values[_model_id] = model_final_losses(
                    store_root.value, _model_id, data_root.value, int(loss_seed.value), refresh=loss_refresh.value,
                )
                _bar.update()
        _message = f"Losses of {len(loss_values)} models, seed {int(loss_seed.value)}."
    mo.stop(not loss_values, mo.callout("None of the chosen models are in the results.", kind="warn"))
    mo.md(_message)
    return (loss_values,)


@app.cell(hide_code=True)
def _(loss_values):
    # Scores found in every model's results, e.g. "rollout/loss", "per_transition/loss_component_raw/texture"
    _scores = sorted(set.intersection(*(set(_losses[_split]) for _losses in loss_values.values()
                                        for _split in ("train", "validation") if _split in _losses)))
    _how = {"rollout": "rollout from 0h", "per_transition": "per 12 h step from data"}

    def _score_name(_name):
        # Total, components (weighted or raw) and per-group totals; None for the rest
        _parts = _name.split("/")
        if _parts == ["loss"]:
            return "Total"
        if _parts[0] == "loss_component":
            return f"{_parts[1]} (weighted)"
        if _parts[0] == "loss_component_raw":
            return f"{_parts[1]} (raw)"
        if _parts[0] == "loss_detail" and _parts[-1] == "total":
            return f"{GROUP_LABELS.get(_parts[1], _parts[1])} total"
        return None

    _score_labels = {
        f"{_score_name(_s.split('/', 1)[1])}, {_how[_s.split('/', 1)[0]]}": _s
        for _s in _scores if _score_name(_s.split("/", 1)[1])
    }
    _default = next((_l for _l, _s in _score_labels.items() if _s == "rollout/loss"), next(iter(_score_labels)))
    loss_score = mo.ui.dropdown(options=_score_labels, value=_default, label="Loss shown")
    loss_score
    return (loss_score,)


@app.cell(hide_code=True)
def _(loss_records, loss_score, loss_values):
    # One row per (projection, Δ) and one per model; tables can be downloaded as CSV
    _evaluated = [_r for _r in loss_records if _r["model_id"] in loss_values]
    _summary = []
    for _projection in PROJECTION_LABELS:
        for _downsample in sorted({_r["downsample"] for _r in _evaluated}):
            _models = [loss_values[_r["model_id"]] for _r in _evaluated
                       if _r["projection"] == _projection and _r["downsample"] == _downsample]
            if not _models:
                continue
            _row = {"projection": _projection, "Δ": _downsample, "models": len(_models)}
            for _split in ("train", "validation"):
                _v = np.array([_m[_split][loss_score.value] for _m in _models if _split in _m])
                _row[f"{_split} mean"] = float(_v.mean()) if _v.size else None
                _row[f"{_split} sd"] = float(_v.std(ddof=1)) if _v.size > 1 else None
            _summary.append(_row)
    _per_model = [
        {"projection": _r["projection"], "Δ": _r["downsample"], "model": _r["option_label"],
         # Each model's own split (missing from losses cached before the split was recorded)
         **{f"{_split} replicates": ", ".join(map(str, _ids)) for _split, _ids
            in loss_values[_r["model_id"]].get("replicates", {}).items()},
         **{_split: loss_values[_r["model_id"]][_split][loss_score.value]
            for _split in ("train", "validation") if _split in loss_values[_r["model_id"]]}}
        for _r in _evaluated
    ]
    mo.vstack([
        mo.ui.table(_summary, selection=None, label=f"Per projection and Δ ({loss_score.value}); sd over models"),
        mo.accordion({"Per model": mo.ui.table(_per_model, selection=None)}),
    ])
    return


@app.cell(hide_code=True)
def _(loss_records, loss_values):
    # The figure uses the evaluated models only
    loss_evaluated = [_r for _r in loss_records if _r["model_id"] in loss_values]
    loss_axis = mo.ui.dropdown(options={"Projection": "projection", "Resolution (Δ)": "downsample"},
                               value="Projection", label="Figure: compare")
    loss_axis
    return loss_axis, loss_evaluated


@app.cell(hide_code=True)
def _(loss_axis, loss_evaluated):
    # The setting held fixed: Δ when comparing projections, the projection when comparing Δ
    if loss_axis.value == "projection":
        _options = {f"Δ = {_d}": _d for _d in sorted({_r["downsample"] for _r in loss_evaluated})}
        _label = "At downsampling ratio"
    else:
        _options = {PROJECTION_LABELS[_p][0]: _p for _p in PROJECTION_LABELS
                    if any(_r["projection"] == _p for _r in loss_evaluated)}
        _label = "For projection"
    loss_fixed = mo.ui.dropdown(options=_options, value=next(iter(_options)), label=_label)
    loss_fixed
    return (loss_fixed,)


@app.cell(hide_code=True)
def _(loss_axis, loss_evaluated, loss_fixed):
    _fixed = "downsample" if loss_axis.value == "projection" else "projection"
    _records = [_r for _r in loss_evaluated if _r[_fixed] == loss_fixed.value]
    if loss_axis.value == "projection":
        _keys = [_p for _p in PROJECTION_LABELS if any(_r["projection"] == _p for _r in _records)]
        loss_columns = [(_p, PROJECTION_LABELS[_p][0]) for _p in _keys]
    else:
        _keys = sorted({_r["downsample"] for _r in _records})
        loss_columns = [(_d, f"Δ = {_d}") for _d in _keys]
    # Every evaluated model of a column goes into its bars; one is shown as the snapshot
    loss_column_models = {_k: [_r["model_id"] for _r in _records if _r[loss_axis.value] == _k] for _k in _keys}
    loss_snapshot_models = mo.ui.dictionary({
        str(_k): mo.ui.dropdown(
            options={_r["option_label"]: _r["model_id"] for _r in _records if _r[loss_axis.value] == _k},
            value=next(_r["option_label"] for _r in _records if _r[loss_axis.value] == _k),
            label=f"Snapshot model, {_label}", full_width=True,
        )
        for _k, _label in loss_columns
    })
    loss_replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Snapshot replicate (one-based)")
    loss_rollout_seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    run_loss_rollouts = mo.ui.run_button(label="Roll out snapshot models")
    mo.vstack([
        mo.md(", ".join(f"{_label}: {len(loss_column_models[_k])} models" for _k, _label in loss_columns)),
        loss_snapshot_models,
        mo.hstack([loss_replicate, loss_rollout_seed, run_loss_rollouts], justify="start"),
    ])
    return (
        loss_column_models,
        loss_columns,
        loss_replicate,
        loss_rollout_seed,
        loss_snapshot_models,
        run_loss_rollouts,
    )


@app.cell(hide_code=True)
def _(
    data_root,
    loss_columns,
    loss_replicate,
    loss_rollout_seed,
    loss_snapshot_models,
    run_loss_rollouts,
    store_root,
):
    mo.stop(not run_loss_rollouts.value, mo.md("Choose the snapshot models, then roll them out."))
    loss_rollouts = {}
    with mo.status.progress_bar(total=len(loss_columns), title="Rolling out snapshot models") as _bar:
        for _index, (_key, _) in enumerate(loss_columns):
            loss_rollouts[_key] = baseline_rollout(
                store_root.value, loss_snapshot_models.value[str(_key)], data_root.value,
                int(loss_replicate.value), int(loss_rollout_seed.value) + _index,
            )
            _bar.update()
    _in_training = [_label for _key, _label in loss_columns
                    if int(loss_replicate.value) in loss_rollouts[_key]["train_replicates"]]
    mo.md(
        f"Snapshot replicate {int(loss_replicate.value)} is in the training split of: {', '.join(_in_training)}"
        if _in_training else
        f"Snapshot replicate {int(loss_replicate.value)} is held out from every snapshot model's training split."
    )
    return (loss_rollouts,)


@app.cell(hide_code=True)
def _(loss_rollouts):
    _first = next(iter(loss_rollouts.values()))
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _first["channel_names"] if "/" in _n))
    loss_groups = mo.ui.multiselect(
        options=_groups,
        value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
        label="Experiment groups (one sector each)",
    )
    loss_gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    _times = list(dict.fromkeys(_t for _r in loss_rollouts.values() for _t in _r["time_labels"]))
    loss_time = mo.ui.dropdown(options=_times, value="48h" if "48h" in _times else _times[-1], label="Timepoint")
    loss_axis_label = mo.ui.text(value="Final loss", label="Loss axis label")
    loss_colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size")
    loss_width = mo.ui.dropdown(options={"double column": WIDTH["double"], "single column": WIDTH["single"]},
                                value="double column", label="Width")
    loss_bar_height = mo.ui.slider(0.6, 3.0, step=0.1, value=1.3, show_value=True, label="Bar chart height (in)")
    loss_hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured groups")
    mo.vstack([
        mo.hstack([loss_groups, loss_gamma, loss_time, loss_axis_label], justify="start"),
        mo.hstack([loss_colony_size, loss_width, loss_bar_height, loss_hatch], justify="start"),
    ])
    return (
        loss_axis_label,
        loss_bar_height,
        loss_colony_size,
        loss_gamma,
        loss_groups,
        loss_hatch,
        loss_time,
        loss_width,
    )


@app.cell(hide_code=True)
def _(
    loss_axis,
    loss_axis_label,
    loss_bar_height,
    loss_colony_size,
    loss_column_models,
    loss_columns,
    loss_fixed,
    loss_gamma,
    loss_groups,
    loss_hatch,
    loss_rollouts,
    loss_score,
    loss_time,
    loss_values,
    loss_width,
):
    mo.stop(not loss_groups.value, mo.callout("Select at least one experiment group.", kind="warn"))
    _names = next(iter(loss_rollouts.values()))["channel_names"]
    _groups = [_g for _g in dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n)
               if _g in loss_groups.value]
    # {column key: {"train": [one value per model], "validation": [...]}}
    _losses = {
        _key: {_split: [loss_values[_m][_split][loss_score.value] for _m in _models if _split in loss_values[_m]]
               for _split in ("train", "validation")}
        for _key, _models in loss_column_models.items()
    }
    _title = (f"Projection, at Δ = {loss_fixed.value}" if loss_axis.value == "projection"
              else f"Downsampling ratio, {PROJECTION_LABELS[loss_fixed.value][0].lower()}")
    with paper_style():
        loss_figure = snapshot_loss_figure(
            loss_rollouts, loss_columns, _groups, loss_gamma.value, loss_colony_size.value, loss_hatch.value,
            loss_time.value, _losses, loss_axis_label.value, title=_title, width=loss_width.value,
            bar_height=loss_bar_height.value,
        )
    loss_figure
    return (loss_figure,)


@app.cell(hide_code=True)
def _():
    loss_figure_name = mo.ui.text(value="projection_resolution_losses", label="File name")
    save_loss_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_loss_figure = mo.ui.run_button(label="Save figure")
    mo.hstack([loss_figure_name, save_loss_svg, save_loss_figure], justify="start")
    return loss_figure_name, save_loss_figure, save_loss_svg


@app.cell
def _(loss_figure, loss_figure_name, save_loss_figure, save_loss_svg):
    mo.stop(not save_loss_figure.value)
    _paths = save_figure(loss_figure, loss_figure_name.value, "schematics", svg=save_loss_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
