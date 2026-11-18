# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///

"""When can a developing pattern no longer be switched to the other attractor?

Run from the repository root with:

    marimo edit Experiments/impulse/impulse_critical_time.py

Reads the impulse records of critical time sweeps
(``conf/experiments/emoji_attractor_switch_critical_time.yaml``) from the model
store (``MODEL_STORE_ROOT``, default ``models/``). Each run applies an impulse
to the source pattern after ``impulse_time`` of growth and is scored against
the mature source and target. Needs no model bundles or data. The critical
time helpers are in ``Experiments/impulse/analysis.py``.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from Experiments.impulse.analysis import critical_time, impulse_record_table, sigmoid_midpoint
    from Experiments.model_registry import ModelRegistry

    plt.style.use("default")

    # Categorical hues in fixed order (one per switch within a panel)
    COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
    # Columns that identify one curve: a model, a switch direction and an impulse setting
    CURVE = ["model", "switch", "setting"]

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def rgb(state):
        """[C, H, W] state -> [H, W, 3] image of its first three channels."""
        return np.clip(np.moveaxis(np.asarray(state[:3], dtype=float), 0, -1), 0.0, 1.0)

    def logistic(t, t0, width):
        return 1 / (1 + np.exp((np.asarray(t) - t0) / width))

    def model_panels(view, x, y, ylabel, critical=None, horizon=None):
        """One panel per model, one line per (switch, setting): mean of ``y`` per time, dots for runs.

        ``critical`` (one row per curve) adds the threshold crossing (solid
        vertical line) and the fitted sigmoid (thin curve, dotted line at its
        midpoint). ``horizon`` marks the training horizon per model.
        """
        models = list(dict.fromkeys(view.model))
        columns = min(len(models), 4)
        rows = int(np.ceil(len(models) / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.2 * columns, 3.3 * rows), squeeze=False, sharex=True, sharey=True,
            constrained_layout=True,
        )
        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        for ax, model in zip(axes.flat, models):
            panel = view[view.model == model]
            for k, ((switch, setting), runs) in enumerate(panel.groupby(["switch", "setting"], sort=False)):
                colour = COLOURS[k % len(COLOURS)]
                means = runs.groupby(x)[y].mean()
                ax.scatter(runs[x], runs[y], s=12, color=colour, alpha=0.4, lw=0)
                label = switch if panel.setting.nunique() == 1 else f"{switch} ({setting})"
                ax.plot(means.index, means.values, marker="o", ms=4, lw=1.5, color=colour, label=label)
                if y == "switch_rate":
                    baseline = runs.groupby(x).baseline_switch_rate.mean()
                    ax.plot(baseline.index, baseline.values, ls="--", lw=1, color=colour, alpha=0.6)
                if critical is not None:
                    row = critical[(critical.model == model) & (critical.switch == switch) & (critical.setting == setting)]
                    if len(row):
                        row = row.iloc[0]
                        if np.isfinite(row.threshold_time):
                            ax.axvline(row.threshold_time, color=colour, lw=1.2)
                        if y == "switch_rate" and np.isfinite(row.sigmoid_midpoint):
                            grid = np.linspace(runs[x].min(), runs[x].max(), 200)
                            ax.plot(grid, row.sigmoid_top * logistic(grid, row.sigmoid_midpoint, row.sigmoid_width),
                                    color=colour, lw=0.8, alpha=0.7)
                            ax.axvline(row.sigmoid_midpoint, color=colour, lw=1, ls=":")
            if horizon is not None:
                ax.axvline(horizon(panel), color="#888888", lw=1, ls=(0, (1, 3)))
            ax.set_title(model, fontsize=9)
            ax.legend(frameon=False, fontsize=7)
            style_axes(ax)
        for ax in axes[-1]:
            ax.set_xlabel(x.replace("_", " "))
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=9)
        if y == "switch_rate":
            axes[0, 0].set_ylim(-0.05, 1.05)
        return fig


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Critical time for switching attractors
    - Each run: an impulse on the source after `impulse_time` of growth, scored against the mature source and target
    - Switch rate against impulse time per model; the critical time is when switching stops
    - Two estimates: threshold crossing (solid line) and fitted sigmoid midpoint (dotted line)
    - Grey dotted line: the training horizon (run.t)
    """)
    return


@app.cell
def _():
    store_root = mo.ui.text(
        value=os.environ.get("MODEL_STORE_ROOT", "models"), label="Model store", full_width=True
    )
    store_root
    return (store_root,)


@app.cell
def _(store_root):
    # Impulse records grouped by experiment folder: <store>/impulses/<experiment>/<id>
    record_paths = {}
    for _path in ModelRegistry(store_root.value).impulse_paths():
        record_paths.setdefault(_path.parent.name, []).append(_path)
    mo.stop(not record_paths, mo.md(f"No impulse records below `{store_root.value}/impulses`"))
    _critical = sorted(_name for _name in record_paths if "critical-time" in _name)
    experiment_choice = mo.ui.multiselect(
        sorted(record_paths), value=_critical[-1:], label="Impulse experiments"
    )
    experiment_choice
    return experiment_choice, record_paths


@app.cell
def _(experiment_choice, record_paths):
    mo.stop(not experiment_choice.value, mo.md("Pick at least one experiment"))
    with mo.status.spinner("Reading impulse records"):
        runs = impulse_record_table(
            [_path for _name in experiment_choice.value for _path in record_paths[_name]]
        )
    runs = runs[runs.impulse_time.notna()].copy()
    mo.stop(runs.empty, mo.md("These experiments have no runs with an impulse time"))
    # Name each model by its patterns, plus its family and the end of its ID
    _patterns = {
        _id: " / ".join(sorted(set(_rows.source_pattern) | set(_rows.target_pattern)))
        for _id, _rows in runs.groupby("model_id")
    }
    runs["model"] = [
        f"{_patterns[_id]} ({_family}, {_id[-8:]})" for _id, _family in zip(runs.model_id, runs.family)
    ]
    runs["loss_ratio"] = runs.evaluation_loss / runs.baseline_evaluation_loss
    mo.md(
        f"{len(runs)} runs, {runs.model_id.nunique()} models, {runs.groupby(CURVE).ngroups} switches, "
        f"{runs.impulse_time.nunique()} impulse times (unit: {', '.join(runs.time_unit.unique())})"
    )
    return (runs,)


@app.cell
def _(runs):
    model_filter = mo.ui.multiselect(sorted(runs.model.unique()), value=sorted(runs.model.unique()), label="Models")
    setting_filter = mo.ui.multiselect(
        sorted(runs.setting.unique()), value=sorted(runs.setting.unique()), label="Impulse settings"
    )
    time_axis = mo.ui.radio(
        {"sweep unit (fraction of run.t)": "impulse_time", "NCA steps": "impulse_steps"},
        value="sweep unit (fraction of run.t)", label="Impulse time in",
    )
    threshold = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5, label="Switch rate threshold", show_value=True)
    mo.hstack([model_filter, setting_filter, time_axis, threshold], justify="start")
    return model_filter, setting_filter, threshold, time_axis


@app.cell
def _(model_filter, runs, setting_filter, threshold, time_axis):
    view = runs[runs.model.isin(model_filter.value) & runs.setting.isin(setting_filter.value)]
    mo.stop(view.empty, mo.md("No runs match the filters"))
    x = time_axis.value

    def horizon(panel):
        """run.t of a model's runs, in the chosen unit."""
        if x == "impulse_steps" or (panel.time_unit == "steps").all():
            return float(panel.training_t.iloc[0])
        return 1.0

    # Both critical time estimates for every curve
    _rows = []
    for (_model, _switch, _setting), _curve in view.groupby(CURVE, sort=False):
        _means = _curve.groupby(x).switch_rate.mean()
        _t0, _width, _top = sigmoid_midpoint(_means.index, _means.values)
        _rows.append({
            "model": _model,
            "switch": _switch,
            "setting": _setting,
            "threshold_time": critical_time(_means.index, _means.values, threshold.value),
            "sigmoid_midpoint": _t0,
            "sigmoid_width": _width,
            "sigmoid_top": _top,
            "rate_at_first_time": float(_means.iloc[0]),
            "rate_at_last_time": float(_means.iloc[-1]),
            "runs": len(_curve),
        })
    critical = pd.DataFrame(_rows).sort_values(["threshold_time", "sigmoid_midpoint"]).reset_index(drop=True)
    return critical, horizon, view, x


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Switch rate against impulse time
    - Lines: mean switch rate; dots: single runs; dashed: no impulse
    - Solid vertical line: threshold crossing; thin curve and dotted line: fitted sigmoid and its midpoint
    """)
    return


@app.cell
def _(critical, horizon, view, x):
    model_panels(view, x, "switch_rate", "switch rate", critical=critical, horizon=horizon)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Target loss relative to no impulse
    - Evaluation target loss divided by the loss without an impulse (below 1: closer to the target)
    - A finer measure than the switch rate, which comes from few rollouts
    """)
    return


@app.cell
def _(critical, horizon, view, x):
    _fig = model_panels(view, x, "loss_ratio", "target loss / no-impulse loss", critical=critical, horizon=horizon)
    if (view.loss_ratio > 0).all():
        _fig.axes[0].set_yscale("log")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Overview
    - Rows: switches, sorted by threshold crossing; colour: mean switch rate
    """)
    return


@app.cell
def _(critical, view, x):
    _labels = {
        _key: f"{_key[0]}: {_key[1]}" + ("" if view.setting.nunique() == 1 else f" ({_key[2]})")
        for _key in critical[CURVE].itertuples(index=False, name=None)
    }
    _table = view.groupby([*CURVE, x]).switch_rate.mean().unstack(x)
    _table = _table.loc[[_key for _key in _labels if _key in _table.index]]
    _fig, _ax = plt.subplots(figsize=(1.0 + 0.45 * _table.shape[1] + 3, 0.9 + 0.3 * len(_table)), constrained_layout=True)
    _image = _ax.imshow(_table.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    _ax.set_xticks(range(_table.shape[1]), [f"{_value:g}" for _value in _table.columns], rotation=90, fontsize=8)
    _ax.set_yticks(range(len(_table)), [_labels[_key] for _key in _table.index], fontsize=8)
    _ax.set_xlabel(x.replace("_", " "))
    _fig.colorbar(_image, ax=_ax, label="switch rate", shrink=0.8)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Critical time per switch
    - threshold_time: earliest time after which the mean switch rate stays below the threshold (inf: still switches at the last time)
    - sigmoid_midpoint: where a fitted decreasing sigmoid is half its height (sigmoid_top); sigmoid_width: how gradual the drop is
    """)
    return


@app.cell
def _(critical):
    mo.ui.table(critical.round(3), selection=None, page_size=20)
    return


@app.cell
def _(critical, view, x):
    # Both estimates per switch; switches still possible at the last time are drawn at the right edge
    _right = float(view[x].max())
    _fig, _ax = plt.subplots(figsize=(7, 0.8 + 0.32 * len(critical)), constrained_layout=True)
    _y = np.arange(len(critical))
    _threshold = critical.threshold_time.to_numpy(float)
    _finite = np.isfinite(_threshold)
    _ax.scatter(_threshold[_finite], _y[_finite], color=COLOURS[0], s=30, label="threshold crossing")
    _ax.scatter(np.full((~_finite).sum(), _right), _y[~_finite], color=COLOURS[0], marker=">", s=40,
                label="still switches at last time")
    _ax.scatter(critical.sigmoid_midpoint, _y, color=COLOURS[1], marker="D", s=22, label="sigmoid midpoint")
    _ax.set_yticks(_y, [f"{_model}: {_switch}" for _model, _switch in zip(critical.model, critical.switch)], fontsize=8)
    _ax.invert_yaxis()
    _ax.set_xlabel(x.replace("_", " "))
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(_ax)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Other scores against impulse time
    - switch_time: NCA steps after the impulse until the rollout stays nearer the target (switched rollouts only)
    - impulse_l2: size of the optimised impulse
    """)
    return


@app.cell
def _():
    other_score = mo.ui.dropdown(
        {
            "switch time (NCA steps)": "switch_time",
            "impulse energy (l2)": "impulse_l2",
            "perturbed (step, pixel) pairs": "footprint",
            "target loss after evaluation": "evaluation_loss",
        },
        value="switch time (NCA steps)",
        label="Score",
    )
    other_score
    return (other_score,)


@app.cell
def _(horizon, other_score, view, x):
    model_panels(view, x, other_score.value, other_score.selected_key, horizon=horizon)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Development strip for one switch
    - Columns: impulse times; rows: source when the impulse acts, just after the impulse, end of the evaluation rollout
    - Reads the saved states of every run of the switch (can take a few seconds)
    """)
    return


@app.cell
def _(critical):
    strip_choice = mo.ui.dropdown(
        {f"{_model}: {_switch} ({_setting})": (_model, _switch, _setting)
         for _model, _switch, _setting in critical[CURVE].itertuples(index=False, name=None)},
        label="Switch",
    )
    strip_element = mo.ui.slider(0, 7, value=0, label="Batch element", show_value=True)
    mo.hstack([strip_choice, strip_element], justify="start")
    return strip_choice, strip_element


@app.cell
def _(strip_choice, strip_element, view, x):
    mo.stop(strip_choice.value is None, mo.md("Pick a switch to see its runs"))
    _model, _switch, _setting = strip_choice.value
    _runs = view[(view.model == _model) & (view.switch == _switch) & (view.setting == _setting)]
    # One run per impulse time (the first, when a time was repeated)
    _runs = _runs.sort_values(x).drop_duplicates(x)
    _names = {"initial_states": "at impulse", "perturbed_initial_states": "after impulse", "final_states": "end"}
    _states, _curves = [], []
    for _path in _runs.path:
        with np.load(Path(_path) / "states.npz") as _arrays:
            _element = min(strip_element.value, len(_arrays["initial_states"]) - 1)
            _states.append({_name: _arrays[_name][_element, :4] for _name in _names})
            if "source_states" in _arrays.files:
                _states[-1]["mature source"] = _arrays["source_states"][_element, :4]
                _states[-1]["target"] = _arrays["target_states"][_element, :4]
            if "evaluation_loss_curve" in _arrays.files:
                _curves.append((_arrays["evaluation_loss_curve"], _arrays["baseline_loss_curve"]))

    _columns = len(_runs) + 1
    _fig, _axes = plt.subplots(3, _columns, figsize=(1.5 * _columns, 4.9), squeeze=False, constrained_layout=True)
    # First column: the mature source and target the run is scored against
    for _row, _name in enumerate(("mature source", "target")):
        if _name in _states[0]:
            _axes[_row, 0].imshow(rgb(_states[0][_name]))
            _axes[_row, 0].set_title(_name, fontsize=8)
    for _ax in _axes.flat:
        _ax.axis("off")
    for _column, (_state, _time, _rate) in enumerate(zip(_states, _runs[x], _runs.switch_rate), start=1):
        for _row, (_name, _label) in enumerate(_names.items()):
            _axes[_row, _column].imshow(rgb(_state[_name]))
            _title = f"t={_time:g}\nrate {_rate:.2f}" if _row == 0 else _label
            _axes[_row, _column].set_title(_title, fontsize=8)
    _out = [_fig]

    if _curves:
        # Target loss along the evaluation rollouts, coloured by impulse time
        _every = OmegaConf.load(Path(_runs.path.iloc[0]) / "config.yaml").impulse.rollout.evaluation_frame_every
        _loss_fig, _loss_ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
        _colours = plt.cm.viridis(np.linspace(0, 1, len(_curves)))
        for _colour, (_curve, _baseline), _time in zip(_colours, _curves, _runs[x]):
            _steps = _every * np.arange(1, len(_curve) + 1)
            _loss_ax.plot(_steps, _curve, color=_colour, lw=1.2, label=f"{_time:g}")
            _loss_ax.plot(_steps, _baseline, color=_colour, lw=0.8, ls="--", alpha=0.5)
        _loss_ax.set_yscale("log")
        _loss_ax.set_xlabel("NCA step after the impulse")
        _loss_ax.set_ylabel("target loss (dashed: no impulse)")
        _loss_ax.legend(title="impulse time", frameon=False, fontsize=7, ncol=2)
        style_axes(_loss_ax)
        _out.append(_loss_fig)
    mo.vstack(_out)
    return


if __name__ == "__main__":
    app.run()
