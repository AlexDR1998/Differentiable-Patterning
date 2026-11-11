"""Baseline NCA trajectories against their data: one model per projection, and one per resolution.

Run with:
    marimo edit Experiments/micropatterns/figures/projection_resolution_figure.py

Reads ``MODEL_STORE_ROOT`` and ``DATA_PATH_BASE``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="projection and resolution trajectories")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    from dotenv import load_dotenv

    from Experiments.micropatterns.figures.data import baseline_rollout, load_selection_records
    from Experiments.micropatterns.figures.drawing import trajectory_figure
    from Experiments.micropatterns.figures.style import (
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
def _(projection_figure_name, projection_trajectories, save_projection_figure, save_projection_svg):
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
            "x": mo.ui.slider(0.0, 1.0, step=0.01, value=_x, show_value=True, label="x"),
            "y": mo.ui.slider(0.0, 1.0, step=0.01, value=_y, show_value=True, label="y"),
        })
        for _x, _y in _defaults[:resolution_zoom_count.value]
    ])
    mo.hstack([
        mo.vstack([mo.md(f"<span style='color:{_c}'>■</span> Zoom {_k + 1}"), _centre])
        for _k, (_c, _centre) in enumerate(zip(ZOOM_COLOURS, resolution_zoom_centres))
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
            "colours": ZOOM_COLOURS,
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
def _(resolution_figure_name, resolution_trajectories, save_resolution_figure, save_resolution_svg):
    mo.stop(not save_resolution_figure.value)
    _paths = save_figure(resolution_trajectories, resolution_figure_name.value, "schematics",
                         svg=save_resolution_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
