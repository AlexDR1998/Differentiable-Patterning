# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

"""Compare the saved runs of impulse sweeps, to pick impulse settings.

Run from the repository root with:

    marimo edit Experiments/impulse/impulse_sweep_explorer.py

Reads the ``.json``/``.eqx``/``.npz`` files written by
``Experiments/impulse/optimise.py`` below the results folder (default
``IMPULSE_OUTPUT_PATH``). Needs no model bundles or data: everything shown
comes from the saved runs. For rollouts of a chosen impulse from new states
(switch matrices, robustness to scale, noise and position) use
``impulse_explorer.py``.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle

    from Experiments.impulse.analysis import (
        load_impulse_results,
        load_saved_intervention,
        summary_row,
    )
    from Experiments.impulse.records import impulse_size

    plt.style.use("default")

    # Categorical hues in fixed order (one per level of the colour factor)
    COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

    # Columns that describe an impulse setting. A setting's runs differ only in
    # model, transition and repeat.
    SETTINGS = (
        "sweep", "objective", "channels", "spatial", "width", "patch_size", "duration",
        "budget", "l2", "learn_rate", "rollout_steps", "loss_window", "iterations",
    )
    # Columns that can be plotted: column -> axis label
    METRICS = {
        "switch_rate": "switch rate after the evaluation rollout",
        "evaluation_loss": "target loss after the evaluation rollout",
        "impulse_l2": "impulse energy (l2 summed over steps)",
        "impulse_linf": "largest change of one value",
        "footprint": "perturbed (step, pixel) pairs",
        "best_loss": "best training objective",
    }

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def varying(df, columns):
        """The columns that take more than one value in ``df``."""
        return [name for name in columns if name in df and df[name].nunique(dropna=False) > 1]

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def factor_panels(df, y, x, hue=None, facet=None):
        """Mean of ``y`` per level of ``x`` (one line per ``hue`` level), one panel per ``facet`` level.

        Dots are single runs, shifted slightly per hue so they do not overlap.
        """
        facets = [None] if facet is None else levels(df[facet])
        fig, axes = plt.subplots(
            1, len(facets), figsize=(4.4 * len(facets), 3.6), squeeze=False, sharey=True, constrained_layout=True
        )
        for ax, level in zip(axes[0], facets):
            rows = df if facet is None else df[df[facet] == level]
            xs = levels(rows[x])
            hues = [None] if hue is None else levels(rows[hue])
            for k, value in enumerate(hues):
                sub = rows if hue is None else rows[rows[hue] == value]
                colour = COLOURS[k % len(COLOURS)]
                shift = (k - (len(hues) - 1) / 2) * 0.6 / len(hues)
                means = []
                for i, level_x in enumerate(xs):
                    values = sub.loc[sub[x] == level_x, y].dropna().to_numpy(float)
                    ax.scatter(np.full(len(values), i + shift), values, s=14, color=colour, alpha=0.4, lw=0)
                    means.append(values.mean() if len(values) else np.nan)
                ax.plot(np.arange(len(xs)) + shift, means, marker="o", ms=6, lw=1.5, color=colour,
                        label=None if value is None else str(value))
            ax.set_xticks(range(len(xs)), [str(v) for v in xs])
            ax.set_xlabel(x)
            if facet is not None:
                ax.set_title(f"{facet} = {level}", fontsize=9)
            style_axes(ax)
        axes[0, 0].set_ylabel(METRICS.get(y, y), fontsize=9)
        if y == "switch_rate":
            axes[0, 0].set_ylim(-0.05, 1.05)
        if hue is not None:
            axes[0, -1].legend(title=hue, frameon=False, fontsize=8)
        return fig

    def rgb(state):
        """[C, H, W] state -> [H, W, 3] image of its first three channels."""
        return np.clip(np.moveaxis(np.asarray(state[:3]), 0, -1), 0.0, 1.0)

    def show(ax, image, title, **kwargs):
        ax.imshow(image, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def mark_impulse(ax, intervention, shape):
        """Draw the centre (and the k x k square of a patch) of a local impulse on an image."""
        spatial = intervention.mode["spatial"]
        if spatial not in ("local", "patch"):
            return
        row, col = np.asarray(intervention.get_location()) * (np.asarray(shape) - 1)
        ax.plot(col, row, "x", color="#00e5ff", ms=7, mew=1.5)
        if spatial == "patch":
            half = intervention.PATCH_SIZE / 2
            ax.add_patch(Rectangle((col - half, row - half), 2 * half, 2 * half, fill=False, ec="#00e5ff", lw=1))


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Impulse sweep explorer
    - Rank impulse settings by switch rate over transitions and repeats
    - Plot any score against the sweep factors
    - Then one setting (where its impulses sit) and one run (states, impulse and rollout)
    """)
    return


@app.cell
def _():
    results_dir = mo.ui.text(
        value=os.environ.get("IMPULSE_OUTPUT_PATH", "impulse_runs"), label="Impulse results", full_width=True
    )
    results_dir
    return (results_dir,)


@app.cell
def _(results_dir):
    results = {}
    interventions = {}
    _rows = []
    for _result in mo.status.progress_bar(
        load_impulse_results(results_dir.value), title="Reading saved impulses", remove_on_exit=True
    ):
        if not _result["path"].with_suffix(".npz").is_file():
            continue
        _intervention, _initial = load_saved_intervention(_result)
        results[_result["name"]] = _result
        interventions[_result["name"]] = _intervention
        _rows.append({**summary_row(_result), **impulse_size(_intervention, _initial[:1])})
    runs = pd.DataFrame(_rows)
    mo.stop(runs.empty, mo.md(f"No impulse runs found below `{results_dir.value}`"))
    mo.md(f"{len(runs)} runs from {runs.sweep.nunique()} sweeps")
    return interventions, results, runs


@app.cell
def _(runs):
    sweep_filter = mo.ui.multiselect(levels(runs.sweep), value=levels(runs.sweep), label="Sweeps")
    model_filter = mo.ui.multiselect(levels(runs.model_id), value=levels(runs.model_id), label="Models")
    channel_filter = mo.ui.multiselect(levels(runs.channels), value=levels(runs.channels), label="Channels")
    mo.hstack([sweep_filter, model_filter, channel_filter], justify="start")
    return channel_filter, model_filter, sweep_filter


@app.cell
def _(channel_filter, model_filter, runs, sweep_filter):
    view = runs[
        runs.sweep.isin(sweep_filter.value)
        & runs.model_id.isin(model_filter.value)
        & runs.channels.isin(channel_filter.value)
    ].copy()
    mo.stop(view.empty, mo.md("No runs match the filters"))
    # The settings that differ between the shown runs name each setting
    setting_columns = varying(view, SETTINGS) or ["sweep"]
    view["setting"] = [
        ", ".join(f"{_name}={_value}" for _name, _value in zip(setting_columns, _values))
        for _values in view[setting_columns].itertuples(index=False)
    ]
    mo.md(f"{len(view)} runs, {view.setting.nunique()} settings, {view.model_id.nunique()} models")
    return setting_columns, view


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Settings ranked
    - worst transition: lowest mean switch rate of any (model, transition), so a setting must work for all of them
    - full switch: fraction of runs where every evaluation rollout switched
    - Select a row to look at that setting below
    """)
    return


@app.cell
def _(setting_columns, view):
    _per_transition = (
        view.groupby(["setting", "model_id", "transition"]).switch_rate.mean().groupby("setting")
        .agg(worst_transition="min", mean_switch="mean")
    )
    _per_setting = view.groupby("setting").agg(
        runs=("run", "size"),
        full_switch=("switch_rate", lambda _rates: float(np.mean(_rates >= 1.0))),
        impulse_l2=("impulse_l2", "median"),
        footprint=("footprint", "median"),
        **{_name: (_name, "first") for _name in setting_columns},
    )
    ranking = (
        _per_setting.join(_per_transition)
        .sort_values(["worst_transition", "mean_switch", "impulse_l2"], ascending=[False, False, True])
        .reset_index()
    )
    ranking_table = mo.ui.table(
        ranking[[*setting_columns, "worst_transition", "mean_switch", "full_switch", "impulse_l2", "footprint", "runs", "setting"]].round(4),
        selection="single",
        page_size=12,
    )
    ranking_table
    return ranking, ranking_table


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Scores against sweep factors
    - Lines: mean over runs; dots: single runs
    """)
    return


@app.cell
def _(setting_columns, view):
    _factors = [*setting_columns, "model_id", "transition", "repeat"]
    _factors = [_name for _name in dict.fromkeys(_factors) if _name in view]

    def _pick(*names):
        return next((_name for _name in names if _name in setting_columns), None)

    # Budget sweep: duration within each budget. Patch sweep: patch size.
    _budgets = "budget" in setting_columns and "duration" in setting_columns
    _x = _pick("duration") if _budgets else _pick("patch_size", "duration", "l2", "width") or _factors[0]
    _hue = _pick(*[_name for _name in ("l2", "channels", "learn_rate") if _name != _x])
    _facet = _pick("budget") if _budgets else _pick(*[_name for _name in ("channels", "spatial") if _name != _hue])
    metric = mo.ui.dropdown({METRICS[_name]: _name for _name in METRICS}, value=METRICS["switch_rate"], label="Score")
    x_factor = mo.ui.dropdown(_factors, value=_x, label="x")
    hue_factor = mo.ui.dropdown(_factors, value=_hue, label="Colour", allow_select_none=True)
    facet_factor = mo.ui.dropdown(_factors, value=_facet, label="Panels", allow_select_none=True)
    mo.hstack([metric, x_factor, hue_factor, facet_factor], justify="start")
    return facet_factor, hue_factor, metric, x_factor


@app.cell
def _(facet_factor, hue_factor, metric, view, x_factor):
    factor_panels(view, metric.value, x_factor.value, hue_factor.value, facet_factor.value)
    return


@app.cell
def _(hue_factor, view):
    # Switch rate against impulse energy: good settings sit top left
    _fig, _ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    _hues = [None] if hue_factor.value is None else levels(view[hue_factor.value])
    for _k, _level in enumerate(_hues):
        _rows = view if _level is None else view[view[hue_factor.value] == _level]
        _ax.scatter(_rows.impulse_l2, _rows.switch_rate, s=18, alpha=0.6, lw=0,
                    color=COLOURS[_k % len(COLOURS)], label=None if _level is None else str(_level))
    if (view.impulse_l2 > 0).all():
        _ax.set_xscale("log")
    _ax.set_xlabel(METRICS["impulse_l2"])
    _ax.set_ylabel(METRICS["switch_rate"])
    _ax.set_ylim(-0.05, 1.05)
    style_axes(_ax)
    if hue_factor.value is not None:
        _ax.legend(title=hue_factor.value, frameon=False, fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## One setting
    - Impulse centres of all its runs on the source pattern, coloured by switch rate
    - Select a run to look at it below
    """)
    return


@app.cell
def _(ranking, ranking_table, view):
    chosen_setting = (
        ranking_table.value.setting.iloc[0] if len(ranking_table.value) else ranking.setting.iloc[0]
    )
    setting_runs = view[view.setting == chosen_setting].sort_values(
        ["model_id", "transition", "switch_rate"], ascending=[True, True, False]
    )
    setting_runs_table = mo.ui.table(
        setting_runs[["run", "model_id", "transition", "repeat", "switch_rate", "evaluation_loss", "impulse_l2", "footprint"]].round(4),
        selection="single",
        page_size=8,
    )
    mo.vstack([mo.md(f"**{chosen_setting}**"), setting_runs_table])
    return setting_runs, setting_runs_table


@app.cell
def _(interventions, results, setting_runs):
    _groups = list(setting_runs.groupby(["model_id", "transition"], sort=False))
    _fig, _axes = plt.subplots(
        1, len(_groups), figsize=(3.2 * len(_groups), 3.4), squeeze=False, constrained_layout=True
    )
    for _ax, ((_model, _transition), _rows) in zip(_axes[0], _groups):
        with np.load(results[_rows.run.iloc[0]]["path"].with_suffix(".npz")) as _arrays:
            _source = _arrays["initial_states"][0]
        _ax.imshow(rgb(_source))
        for _run, _rate in zip(_rows.run, _rows.switch_rate):
            _intervention = interventions[_run]
            if _intervention.mode["spatial"] not in ("local", "patch"):
                continue
            _row, _col = np.asarray(_intervention.get_location()) * (np.asarray(_source.shape[-2:]) - 1)
            _ax.scatter(_col, _row, s=50, c=[plt.cm.RdYlGn(_rate)], edgecolors="black", lw=0.8)
        _ax.set_title(f"{_model[-8:]}  {_transition}", fontsize=9)
        _ax.axis("off")
    _fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn"), ax=_axes[0, -1], label="switch rate", shrink=0.8)
    _fig
    return


@app.cell
def _(results, setting_runs, setting_runs_table):
    chosen_run = (
        setting_runs_table.value.run.iloc[0] if len(setting_runs_table.value) else setting_runs.run.iloc[0]
    )
    _batch = results[chosen_run]["config"].impulse.batch_size
    element = mo.ui.slider(0, max(_batch - 1, 1), value=0, label="Batch element", show_value=True)
    mo.vstack([mo.md(f"## One run\n- `{chosen_run}`"), element])
    return chosen_run, element


@app.cell
def _(chosen_run, element, interventions, results):
    _result = results[chosen_run]
    with np.load(_result["path"].with_suffix(".npz")) as _arrays:
        saved = {_name: _arrays[_name] for _name in _arrays.files}
    _batch = len(saved["initial_states"])
    shown = min(int(element.value), _batch - 1)
    _intervention = interventions[chosen_run]
    _deltas = np.asarray(_intervention.deltas(saved["initial_states"][shown : shown + 1]))
    _shape = saved["initial_states"].shape[-2:]

    _fig, _axes = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)
    show(_axes[0], rgb(saved["initial_states"][shown]), "source")
    show(_axes[1], rgb(saved["perturbed_initial_states"][shown]), "source + first impulse step")
    # Largest change of any channel, summed over the impulse steps
    show(_axes[2], np.abs(_deltas).max(axis=(1, 2)).sum(axis=0), f"impulse ({len(_deltas)} steps)", cmap="magma")
    mark_impulse(_axes[2], _intervention, _shape)
    show(_axes[3], rgb(saved["final_states"][shown]), "end of evaluation rollout")
    show(_axes[4], rgb(saved["target_states"][shown]), "target")
    _out = [_fig]
    if len(_deltas) > 1:
        _energy = np.sqrt(np.mean(_deltas**2, axis=(1, 2, 3, 4)))
        _steps_fig, _steps_ax = plt.subplots(figsize=(6, 2.4), constrained_layout=True)
        _steps_ax.bar(np.arange(len(_energy)), _energy, color=COLOURS[0])
        _steps_ax.set_xlabel("impulse step")
        _steps_ax.set_ylabel("RMS change")
        style_axes(_steps_ax)
        _out.append(_steps_fig)
    mo.vstack([mo.md(f"Batch element {shown} of {_batch}"), *_out])
    return saved, shown


@app.cell
def _(chosen_run, results, saved, shown):
    # Distance (RMS over observed channels) to the target and to the source
    # along the saved evaluation rollouts, with and without the impulse
    _every = results[chosen_run]["config"].impulse.rollout.evaluation_frame_every
    _observed = saved["perturbed_trajectory"].shape[2]
    _frames = {"impulse": saved["perturbed_trajectory"][shown], "no impulse": saved["baseline_trajectory"][shown]}
    _references = {
        "target": saved["target_states"][shown, :_observed],
        "source": saved["initial_states"][shown, :_observed],
    }
    _times = _every * np.arange(1, len(_frames["impulse"]) + 1)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    for _k, (_reference_name, _reference) in enumerate(_references.items()):
        for _name, _trajectory in _frames.items():
            _distance = np.sqrt(np.mean((_trajectory - _reference) ** 2, axis=(1, 2, 3)))
            _ax.plot(_times, _distance, color=COLOURS[_k], ls="-" if _name == "impulse" else "--",
                     label=f"to {_reference_name}, {_name}")
    _ax.set_xlabel("NCA step")
    _ax.set_ylabel("RMS distance")
    _ax.legend(frameon=False, fontsize=8)
    style_axes(_ax)

    # Eight evenly spaced frames of each rollout
    _picks = np.linspace(0, len(_times) - 1, 8).round().astype(int)
    _strip, _strip_axes = plt.subplots(2, len(_picks), figsize=(1.9 * len(_picks), 4.0), constrained_layout=True)
    for _row, (_name, _trajectory) in enumerate(_frames.items()):
        for _column, _index in enumerate(_picks):
            show(_strip_axes[_row, _column], rgb(_trajectory[_index]), f"{_name}, t={_times[_index]}")
    mo.vstack([_fig, _strip])
    return


if __name__ == "__main__":
    app.run()
