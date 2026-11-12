"""True data and NCA trajectories under NODAL knockout, as rows of colony discs.

Run with:
    marimo edit Experiments/micropatterns/figures/nodal_ko_schematic.py

The drawing code is in ``ko_trajectory.py``. Rollouts are cached in ``figures/cache/``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="nodal knockout schematic")

with app.setup:
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv

    from Common.dataloader.cell_types import load_cell_type_rules
    from Experiments.micropatterns.figures.data import (
        CACHE_DIRECTORY,
        cache_name,
        cached,
        load_selection_records,
        model_rollouts,
    )
    from Experiments.micropatterns.figures.ko_trajectory import (
        CONDITIONS,
        KoTrajectoryOptions,
        ko_trajectory_figure,
    )
    from Experiments.micropatterns.figures.style import (
        ACCENT,
        DATA,
        MODEL,
        REPOSITORY_ROOT,
        SERIES,
        colour_or,
        paper_style,
        save_figure,
    )

    load_dotenv()

    # Cell type files exported by data_cleaning.py
    CELL_TYPE_DIRECTORY = REPOSITORY_ROOT / "Experiments/micropatterns/conf/cell_types"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # NODAL knockout trajectories

    - One NCA rolled out from a replicate's 0h state for baseline, KO 24h and KO 0h, beside the true data
    - Data scaled as in the model's training
    - Figures are saved to `figures/schematics/`
    """)
    return


@app.cell(hide_code=True)
def _():
    selection_path = mo.ui.text(placeholder="/path/to/model_registry_selection.yaml (optional)",
                                label="Exported registry selection", full_width=True)
    store_root = mo.ui.text(value=os.getenv("MODEL_STORE_ROOT", str(REPOSITORY_ROOT / "models")),
                            label="Model store", full_width=True)
    data_root = mo.ui.text(value=os.getenv("DATA_PATH_BASE", ""), label="Data root (contains 260726_nca_dataset)",
                           full_width=True)
    mo.vstack([mo.md("**Model and data**"), selection_path, store_root, data_root])
    return data_root, selection_path, store_root


@app.cell(hide_code=True)
def _(selection_path, store_root):
    _path = Path(selection_path.value).expanduser()
    _options = {}
    if selection_path.value.strip() and _path.is_file():
        _options = {_r["option_label"]: _r["model_id"] for _r in load_selection_records(_path, store_root.value)}
    # From the selection if there is one, else a typed model ID
    if _options:
        model_id = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="Model", full_width=True)
    else:
        model_id = mo.ui.text(placeholder="model ID", label="Model", full_width=True)
    replicate = mo.ui.number(start=1, stop=999, step=1, value=1, label="Replicate (one-based)")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Rollout seed")
    refresh = mo.ui.checkbox(value=False, label="Ignore cache")
    run_rollouts = mo.ui.run_button(label="Roll out baseline, KO 0h and KO 24h")
    mo.vstack([
        model_id,
        mo.hstack([replicate, seed, refresh, run_rollouts], justify="start"),
        mo.md("- Cached rollouts load without pressing the button"),
    ])
    return model_id, refresh, replicate, run_rollouts, seed


@app.cell(hide_code=True)
def _(data_root, model_id, refresh, replicate, run_rollouts, seed, store_root):
    _model = (model_id.value or "").strip()
    _name = cache_name("rollouts", _model, f"r{int(replicate.value)}", f"s{int(seed.value)}")
    _cached = bool(_model) and (CACHE_DIRECTORY / _name).is_file()
    mo.stop(not _model or not (run_rollouts.value or _cached), mo.md("Choose a model, then roll it out."))
    mo.stop(not data_root.value.strip() and not _cached, mo.callout("Set the data root first.", kind="warn"))
    with mo.status.spinner("Loading data and rolling out"):
        rollouts = cached(
            _name,
            lambda: model_rollouts(store_root.value, _model, data_root.value, int(replicate.value),
                                   int(seed.value), conditions=tuple(CONDITIONS)),
            refresh=run_rollouts.value and refresh.value,
        )
    _split = "in" if int(replicate.value) in rollouts["baseline"]["train_replicates"] else "held out from"
    mo.md(f"Rollouts ready. Replicate {int(replicate.value)} is {_split} the model's training split.")
    return (rollouts,)


@app.cell(hide_code=True)
def _(rollouts):
    _names = rollouts["baseline"]["channel_names"]
    _groups = list(dict.fromkeys(_n.split("/")[0] for _n in _names if "/" in _n))
    layout = mo.ui.dropdown(
        options={"Row per condition, sector per staining": "stainings",
                 "Row per staining, sector per condition (CMY)": "conditions"},
        value="Row per condition, sector per staining", label="Layout")
    groups = mo.ui.multiselect(options=_groups, value=[_g for _g in _groups if _g.startswith("cell_fate")] or _groups[:1],
                               label="Stainings")
    colour_by = mo.ui.dropdown(options={"Channels": "channels", "Cell types (file below)": "cell_types"},
                               value="Channels", label="NCA colours")
    _files = sorted(CELL_TYPE_DIRECTORY.glob("*.yaml"))
    cell_type_file = mo.ui.dropdown(options={_p.name: str(_p) for _p in _files},
                                    value=_files[-1].name if _files else None, label="Cell type file")
    show_true = mo.ui.checkbox(value=True, label="Show true data")
    show_model = mo.ui.checkbox(value=True, label="Show NCA")
    gamma = mo.ui.slider(0.3, 2.0, step=0.05, value=1.0, show_value=True, label="Gamma (1 = linear)")
    branch = mo.ui.dropdown(options={"One row each": False, "KO 24h branches off baseline": True},
                            value="KO 24h branches off baseline", label="Conditions (row per condition only)")
    panels = mo.ui.dropdown(
        options={"Rows, true then NCA": "rows", "Two panels, side by side": "panels",
                 "Two panels, true data on top": "stacked"},
        value="Two panels, true data on top", label="NCA and true data")
    hatch = mo.ui.checkbox(value=True, label="Hatch unmeasured stainings")
    lmbr = mo.ui.checkbox(value=False, label="LMBR in yellow (Fate S2)")
    legend_style = mo.ui.dropdown(options={"Black box": "black", "White, outlined colours": "white"},
                                  value="White, outlined colours", label="Legend style")
    connectors = mo.ui.dropdown(options={"Arrows": "arrows", "Backgrounds flow (Sankey)": "flow", "Both": "both"},
                                value="Backgrounds flow (Sankey)", label="Branch connectors")
    legend_font = mo.ui.slider(4.0, 12.0, step=0.5, value=6.5, show_value=True, label="Legend font size (pt)")
    legend_shape = mo.ui.dropdown(
        options={"Auto (by position)": "auto", "Row per staining": "rows", "Column per staining": "columns",
                 "Single column": "single"},
        value="Auto (by position)", label="Legend entries")
    track_gap = mo.ui.slider(1.0, 12.0, step=0.5, value=5.0, show_value=True, label="Gap between tracks (h)")
    # Branch view: 0 is level with KO 24h, 1 with the later baseline
    start_height = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, show_value=True,
                                label="Shared baseline start height")
    legend = mo.ui.dropdown(
        options={"Auto (corner of branch view, else right)": "auto", "Corner of branch view": "gap",
                 "Right": "right", "Top": "top", "Bottom": "bottom"},
        value="Top", label="Legend")
    font_size = mo.ui.slider(4.0, 12.0, step=0.5, value=7.0, show_value=True, label="Font size (pt)")
    # In hours of the time axis, whose ticks are 12h apart
    colony_size = mo.ui.slider(5.0, 11.5, step=0.5, value=9.0, show_value=True, label="Colony size (h)")
    # Any matplotlib colour; blank (or not a colour) draws no background
    true_colour = mo.ui.text(value=DATA, label="True data label colour")
    model_colour = mo.ui.text(value=MODEL, label="NCA label colour")
    baseline_band = mo.ui.text(value=SERIES[2], label="Baseline background")
    ko_band = mo.ui.text(value=ACCENT, label="KO background")
    band_alpha = mo.ui.slider(0.0, 1.0, step=0.02, value=0.12, show_value=True, label="Background opacity")
    mo.vstack([
        mo.md("**Display**"),
        mo.hstack([layout, groups, colour_by, cell_type_file], justify="start"),
        mo.hstack([branch, panels], justify="start"),
        mo.hstack([baseline_band, ko_band, band_alpha], justify="start"),
        mo.hstack([true_colour, model_colour], justify="start"),
        mo.hstack([font_size, colony_size, track_gap], justify="start"),
        mo.hstack([start_height, connectors], justify="start"),
        mo.hstack([legend, legend_shape, legend_style, legend_font], justify="start"),
        mo.hstack([show_true, show_model, hatch, lmbr, gamma], justify="start"),
    ])
    return (
        band_alpha,
        baseline_band,
        branch,
        cell_type_file,
        colony_size,
        colour_by,
        connectors,
        font_size,
        gamma,
        groups,
        hatch,
        ko_band,
        layout,
        legend,
        legend_font,
        legend_shape,
        legend_style,
        lmbr,
        model_colour,
        panels,
        show_model,
        show_true,
        start_height,
        track_gap,
        true_colour,
    )


@app.cell(hide_code=True)
def _(cell_type_file, colour_by):
    cell_type_rules, _note = None, None
    if colour_by.value == "cell_types":
        mo.stop(cell_type_file.value is None,
                mo.callout(f"No cell type files in `{CELL_TYPE_DIRECTORY}`.", kind="warn"))
        cell_type_rules = load_cell_type_rules(cell_type_file.value)
        _projection = cell_type_rules.preprocessing.get("projection", "max")
        _note = mo.md(f"- Cell types tuned on the {_projection} projection at {cell_type_rules.hour}h")
    _note
    return (cell_type_rules,)


@app.cell(hide_code=True)
def _(
    band_alpha,
    baseline_band,
    branch,
    cell_type_rules,
    colony_size,
    connectors,
    font_size,
    gamma,
    groups,
    hatch,
    ko_band,
    layout,
    legend,
    legend_font,
    legend_shape,
    legend_style,
    lmbr,
    model_colour,
    panels,
    rollouts,
    show_model,
    show_true,
    start_height,
    track_gap,
    true_colour,
):
    mo.stop(not groups.value, mo.callout("Select at least one staining.", kind="warn"))
    mo.stop(not (show_true.value or show_model.value), mo.callout("Show the true data, the NCA or both.", kind="warn"))

    trajectory_options = KoTrajectoryOptions(
        groups=tuple(groups.value), layout=layout.value, show_true=show_true.value, show_model=show_model.value,
        cell_type_rules=cell_type_rules, branch=branch.value, panels=panels.value, hatch=hatch.value,
        lmbr=lmbr.value, gamma=gamma.value, font_size=font_size.value, colony_size=colony_size.value,
        track_gap=track_gap.value, start_height=start_height.value, connectors=connectors.value,
        true_colour=colour_or(true_colour.value, DATA), model_colour=colour_or(model_colour.value, MODEL),
        baseline_band=colour_or(baseline_band.value, None), ko_band=colour_or(ko_band.value, None),
        band_alpha=band_alpha.value, legend=legend.value, legend_shape=legend_shape.value,
        legend_style=legend_style.value, legend_font=legend_font.value,
    )
    with paper_style():
        ko_trajectories = ko_trajectory_figure(rollouts, trajectory_options)
    ko_trajectories
    return (ko_trajectories,)


@app.cell(hide_code=True)
def _():
    figure_name = mo.ui.text(value="nodal_knockout_trajectories", label="File name")
    save_svg = mo.ui.checkbox(value=False, label="Also SVG (editable text)")
    save_button = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_svg, save_button], justify="start")
    return figure_name, save_button, save_svg


@app.cell
def _(figure_name, ko_trajectories, save_button, save_svg):
    mo.stop(not save_button.value)
    _paths = save_figure(ko_trajectories, figure_name.value, "schematics", svg=save_svg.value)
    mo.md("Saved " + ", ".join(f"`{p}`" for p in _paths))
    return


if __name__ == "__main__":
    app.run()
