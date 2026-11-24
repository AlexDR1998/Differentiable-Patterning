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
    from Experiments.micropatterns.figures.style import FONT, GROUP_LINESTYLES, INK, LINE, MUTED, SERIES, WIDTH, paper_style, save_figure
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

    # --- Summary figure -----------------------------------------------------

    # Paper line colours first, then the notebook's own for more than five switches
    FIGURE_COLOURS = (*SERIES, *COLOURS[5:])

    def white_rgb(state):
        """[C, H, W] RGBA state -> [H, W, 3] image on a white background."""
        state = np.clip(np.asarray(state[:4], dtype=float), 0.0, 1.0)
        return np.moveaxis(state[:3] * state[3:4] + 1.0 - state[3:4], 0, -1)

    def curve_labels(keys, model_names):
        """Legend label for each (model, switch, setting) curve: the switch, then the
        setting and the model name only when they tell curves apart."""
        several_models = len({key[0] for key in keys}) > 1
        several_settings = len({key[2] for key in keys}) > 1
        return {
            key: key[1] + (f" ({key[2]})" if several_settings else "") + (f", {model_names[key[0]]}" if several_models else "")
            for key in keys
        }

    def load_rollout(path, element):
        """One batch element of a saved run: the state before and just after the impulse,
        the evaluation rollout after it (one frame every ``every`` steps), and the
        mature source and target."""
        with np.load(Path(path) / "states.npz") as arrays:
            element = min(element, len(arrays["initial_states"]) - 1)
            # The mature source is saved when the impulse acts during development
            source = arrays["source_states" if "source_states" in arrays.files else "initial_states"][element]
            rollout = {
                "before": arrays["initial_states"][element],
                "impulse": arrays["perturbed_initial_states"][element],
                "trajectory": arrays["perturbed_trajectory"][element],
                "references": {"source": source, "target": arrays["target_states"][element]},
            }
        rollout["every"] = OmegaConf.load(Path(path) / "config.yaml").impulse.rollout.evaluation_frame_every
        return rollout

    def arrange_frames(rows, frames, timing):
        """Pick the rollout frames of panel b: the same columns for every row.

        ``rows`` are (impulse step, rollout). Returns the column titles, the axis
        label of the frame columns and, per row, a list of images (None where the
        row has no frame for that column).
        - ``timing="after impulse"``: ``frames`` evenly spaced times after each impulse.
        - ``timing="since start"``: ``frames`` evenly spaced NCA steps since growth
          started, up to the end of the earliest impulse's rollout. Later impulses
          start later, so they have fewer frames.
        """
        every = rows[0][1]["every"]
        length = min(len(rollout["trajectory"]) for _, rollout in rows)
        if timing == "after impulse":
            picks = np.unique(np.linspace(0, length - 1, frames).round().astype(int))
            images = [[rollout["trajectory"][pick] for pick in picks] for _, rollout in rows]
            return [f"+{every * (pick + 1)}" for pick in picks], "+n: NCA steps after the impulse", images
        first = min(start for start, _ in rows)
        # Rounded to whole frame intervals, for readable column titles
        steps = np.unique(np.round(np.linspace(first + every, first + every * length, frames) / every).astype(int) * every)
        images = []
        for start, rollout in rows:
            # Nearest saved frame, if it is within half a frame interval of the column's step
            index = np.round((steps - start) / rollout["every"]).astype(int) - 1
            near = np.abs(start + rollout["every"] * (index + 1) - steps) <= rollout["every"] / 2
            valid = (index >= 0) & (index < len(rollout["trajectory"])) & near
            images.append([rollout["trajectory"][i] if ok else None for i, ok in zip(index, valid)])
        return [f"{step}" for step in steps], "NCA steps since growth started", images

    def number_labels(ax, points):
        """Write the row numbers next to their circles in panel a; numbers whose
        circles (nearly) coincide share one label, e.g. "1, 4"."""
        groups = []
        for number, time, rate in points:
            for group in groups:
                if np.isclose(group[0], time) and abs(group[1] - rate) < 0.06:
                    group[2].append(number)
                    break
            else:
                groups.append([time, rate, [number]])
        for time, rate, numbers in groups:
            ax.annotate(", ".join(map(str, numbers)), (time, rate), xytext=(5, 3), textcoords="offset points",
                        fontsize=FONT["annotation"], color=INK, fontweight="bold")

    def summary_figure(shown, x, colours, labels, blocks=(), columns=(), frame_label="", horizon=None,
                       critical=None, show_runs=True):
        """Panel a: switch rate against impulse time, one line per (model, switch, setting) curve.
        Panel b: snapshots of saved rollouts, one block of rows per switch.

        Each block is a dict with the curve ``key``, a row ``label``, the mature
        ``references`` (source and target) and ``rows`` of (impulse time, mean
        switch rate, images; None leaves a cell empty). Snapshot rows are
        numbered, and the numbers mark their impulse times on the curves in
        panel a. ``columns`` are the titles of the image columns, ``frame_label``
        explains the rollout columns. ``critical`` maps a curve to its threshold
        crossing; ``horizon`` is the training horizon. With several models, each
        model has its own line style.
        """
        width = WIDTH["double"]
        n_columns = len(columns) + 1  # the last column holds the source and target
        # Snapshot rows in order: (block, impulse time, rate, images); numbered from 1
        rows = [(block, *row) for block in blocks for row in block["rows"]]
        if rows:
            image = 0.6 * width / (n_columns + 0.8)
            grid_height = len(rows) + 0.2 + 0.25 * len(blocks) + 0.15 * (len(blocks) - 1)
            height = max(0.4 * width, grid_height * image * 1.1 + 0.3)
            fig = plt.figure(figsize=(width, height), layout="constrained")
            fig_a, fig_b = fig.subfigures(1, 2, width_ratios=[0.4, 0.6], wspace=0.03)
        else:
            fig = fig_a = plt.figure(figsize=(0.45 * width, 0.4 * width), layout="constrained")

        # a: switch rate against impulse time
        ax = fig_a.subplots()
        models = list(dict.fromkeys(key[0] for key in colours))
        for key, colour in colours.items():
            runs = shown[(shown.model == key[0]) & (shown.switch == key[1]) & (shown.setting == key[2])]
            means = runs.groupby(x).switch_rate.mean()
            if show_runs:
                ax.scatter(runs[x], runs.switch_rate, s=4, color=colour, alpha=0.35, lw=0, zorder=2)
            ax.plot(means.index, means.values, color=colour, lw=LINE["data"], marker="o", ms=2.5,
                    ls=GROUP_LINESTYLES[models.index(key[0]) % len(GROUP_LINESTYLES)], label=labels[key], zorder=3)
            if critical is not None and np.isfinite(critical.get(key, np.nan)):
                ax.axvline(critical[key], color=colour, lw=LINE["thin"], ls="--", zorder=1)
        if horizon is not None:
            ax.axvline(horizon, color=MUTED, lw=LINE["thin"], ls=":", zorder=1)
            ax.annotate("end of training rollout", (horizon, 1.04), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT["annotation"], color=MUTED, annotation_clip=False)
        # Circles of later blocks are larger, so circles at the same point stay visible
        block_index = {id(block): index for index, block in enumerate(blocks)}
        for block, time, rate, _ in rows:
            ax.plot(time, rate, marker="o", ms=5 + 2.5 * block_index[id(block)], mfc="none",
                    mec=colours[block["key"]], mew=LINE["data"], zorder=4)
        number_labels(ax, [(number, time, rate) for number, (_, time, rate, _) in enumerate(rows, start=1)])
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("impulse time (NCA steps)" if x == "impulse_steps" else "impulse time (fraction of training rollout)")
        ax.set_ylabel("switch rate")
        ax.spines[["top", "right"]].set_visible(False)
        fig_a.legend(frameon=False, loc="outside lower center")
        fig_a.text(0, 1, "a", ha="left", va="top", fontsize=FONT["letter"], fontweight="bold")
        if not rows:
            return fig

        # b: snapshots. Grid rows: column titles, then per block a header line and
        # its image rows, with a small gap between blocks
        heights = [0.2]
        for index, block in enumerate(blocks):
            heights += ([0.15] if index else []) + [0.25] + [1] * len(block["rows"])
        grid = fig_b.add_gridspec(len(heights), n_columns, height_ratios=heights, wspace=0.05, hspace=0.05)
        for column, column_title in enumerate(columns):
            axis = fig_b.add_subplot(grid[0, column])
            axis.axis("off")
            axis.text(0.5, 0, column_title, ha="center", va="bottom", fontsize=FONT["annotation"], in_layout=False)
        grid_row, number = 1, 1
        for index, block in enumerate(blocks):
            grid_row += 1 if index else 0
            colour = colours[block["key"]]
            grid_row += 1  # the header row stays empty; the header is written above the first image
            axes = [[fig_b.add_subplot(grid[grid_row + row, column]) for column in range(n_columns)]
                    for row in range(len(block["rows"]))]
            # Header: the switch, in the colour of its curve in panel a
            axes[0][0].annotate(block["label"], xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                                textcoords="offset points", ha="left", va="bottom", fontsize=FONT["label"],
                                color=colour, annotation_clip=False, in_layout=False)
            for axis in (axis for line in axes for axis in line):
                axis.set_xticks([])
                axis.set_yticks([])
                axis.spines[:].set_visible(False)
            for row, (time, rate, images) in enumerate(block["rows"]):
                for column, state in enumerate(images):
                    if state is not None:
                        axes[row][column].imshow(white_rgb(state))
                axes[row][0].set_ylabel(f"$\\bf{{{number}}}$  t = {time:g}\nrate {rate:.2f}", rotation=0,
                                        ha="right", va="center", fontsize=FONT["annotation"], color=colour)
                number += 1
            # Mature source and target of this switch, stacked in the last column, named on the right
            for row, name in enumerate(["target"] if len(block["rows"]) == 1 else ["source", "target"]):
                axis = axes[row][-1]
                axis.imshow(white_rgb(block["references"][name]))
                axis.spines[:].set_visible(True)
                axis.spines[:].set_color(MUTED)
                axis.yaxis.set_label_position("right")
                axis.set_ylabel(name, rotation=270, va="bottom", fontsize=FONT["annotation"], color=MUTED)
            grid_row += len(block["rows"])
        fig_b.supxlabel(frame_label, fontsize=FONT["annotation"], color=MUTED)
        fig_b.text(0, 1, "b", ha="left", va="top", fontsize=FONT["letter"], fontweight="bold")
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
        records = impulse_record_table(
            [_path for _name in experiment_choice.value for _path in record_paths[_name]]
        )
    records = records[records.impulse_time.notna()].copy()
    mo.stop(records.empty, mo.md("These experiments have no runs with an impulse time"))
    return (records,)


@app.cell
def _(records):
    # One short name per pattern image, used instead of its file name in every label and figure
    _files = sorted(set(records.source_pattern) | set(records.target_pattern))
    pattern_names = mo.ui.dictionary(
        {_file: mo.ui.text(value=Path(_file).stem.replace("_", " ")) for _file in _files},
        label="Pattern names",
    )
    pattern_names
    return (pattern_names,)


@app.cell
def _(pattern_names, records):
    _names = {_file: (_name.strip() or _file) for _file, _name in pattern_names.value.items()}
    runs = records.copy()
    runs["source_pattern"] = runs.source_pattern.map(_names)
    runs["target_pattern"] = runs.target_pattern.map(_names)
    runs["switch"] = runs.source_pattern + " → " + runs.target_pattern
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
    ## Summary figure
    - a: switch rate against impulse time for the chosen models (dots: single runs)
    - b: one block of rows per chosen switch (e.g. both directions of a model), one rollout per chosen impulse time
    - Numbered circles in a mark the rows of b; times between two critical times show the direction difference
    - Rollout frames at the same steps after each impulse, or at the same steps since growth started (later impulses: fewer frames)
    - Pattern names (top of the notebook) and model names (below) replace file names and IDs in the figure
    - Uses the filters above (settings, time unit, threshold); saves to `figures/impulse/`
    """)
    return


@app.cell
def _(view):
    _models = sorted(view.model.unique())
    figure_models = mo.ui.multiselect(_models, value=_models[:1], label="Models in the figure", full_width=True)
    figure_runs = mo.ui.checkbox(value=True, label="Show single runs")
    figure_critical = mo.ui.checkbox(value=False, label="Mark critical time (threshold crossing)")
    figure_horizon = mo.ui.checkbox(value=True, label="Mark end of training rollout")
    mo.vstack([figure_models, mo.hstack([figure_runs, figure_critical, figure_horizon], justify="start")])
    return figure_critical, figure_horizon, figure_models, figure_runs


@app.cell
def _(figure_models, view):
    # Short name per model, shown in the figure only when it has more than one model
    _families = {_model: view[view.model == _model].family.iloc[0] for _model in figure_models.value}
    _defaults = {
        _model: _family if list(_families.values()).count(_family) == 1
        else f"{_family} {[_m for _m in _families if _families[_m] == _family].index(_model) + 1}"
        for _model, _family in _families.items()
    }
    figure_model_names = mo.ui.dictionary(
        {_model: mo.ui.text(value=_defaults[_model]) for _model in figure_models.value},
        label="Model names in the figure (used with more than one model)",
    )
    figure_model_names
    return (figure_model_names,)


@app.cell
def _(critical, figure_models):
    _options = {
        f"{_model}: {_switch} ({_setting})": (_model, _switch, _setting)
        for _model, _switch, _setting in critical[CURVE].itertuples(index=False, name=None)
        if _model in figure_models.value
    }
    # Start with every switch of the first model, e.g. both directions between two patterns
    _first = figure_models.value[0] if figure_models.value else None
    snapshot_curves = mo.ui.multiselect(
        _options, value=[_label for _label, _key in _options.items() if _key[0] == _first],
        label="Panel b: rollouts of (one block of rows each)", full_width=True,
    )
    snapshot_curves
    return (snapshot_curves,)


@app.cell
def _(snapshot_curves, view, x):
    # Impulse times run for any of the chosen switches; the same times are shown in every block
    _chosen = view[[_key in snapshot_curves.value for _key in view[CURVE].itertuples(index=False, name=None)]]
    _times = sorted(_chosen[x].unique())
    # Four evenly spaced impulse times to start with
    _default = [_times[_i] for _i in np.unique(np.linspace(0, len(_times) - 1, 4).round().astype(int))] if _times else []
    snapshot_times = mo.ui.multiselect(
        {f"{_time:g}": _time for _time in _times}, value=[f"{_time:g}" for _time in _default], label="Impulse times"
    )
    snapshot_frames = mo.ui.slider(2, 10, value=5, label="Rollout frames", show_value=True)
    snapshot_element = mo.ui.slider(0, 7, value=0, label="Batch element", show_value=True)
    snapshot_timing = mo.ui.radio(
        {"steps after the impulse": "after impulse", "steps since growth started": "since start"},
        value="steps after the impulse", label="Rollout frames at",
    )
    mo.hstack([snapshot_times, snapshot_frames, snapshot_element, snapshot_timing], justify="start")
    return snapshot_element, snapshot_frames, snapshot_timing, snapshot_times


@app.cell
def _(
    critical,
    figure_critical,
    figure_horizon,
    figure_model_names,
    figure_models,
    figure_runs,
    horizon,
    snapshot_curves,
    snapshot_element,
    snapshot_frames,
    snapshot_times,
    snapshot_timing,
    view,
    x,
):
    mo.stop(not figure_models.value, mo.md("Pick at least one model for the figure"))
    _shown = view[view.model.isin(figure_models.value)]
    # Curves in the order of the model list, so colours stay put when models are added or removed
    _keys = sorted(_shown.groupby(CURVE, sort=False).groups, key=lambda _key: (figure_models.value.index(_key[0]), _key))
    _colours = {_key: FIGURE_COLOURS[_k % len(FIGURE_COLOURS)] for _k, _key in enumerate(_keys)}
    _critical = {
        (_model, _switch, _setting): _time
        for _model, _switch, _setting, _time in critical[[*CURVE, "threshold_time"]].itertuples(index=False, name=None)
    }

    _labels = curve_labels(_keys, figure_model_names.value)

    # One block per chosen switch, in the order of panel a; one saved run per chosen
    # impulse time (the first, when a time was repeated)
    _blocks, _rollouts = [], []
    for _key in [_key for _key in _keys if _key in snapshot_curves.value]:
        _curve = _shown[(_shown.model == _key[0]) & (_shown.switch == _key[1]) & (_shown.setting == _key[2])]
        _block = {"key": _key, "label": _labels[_key], "rows": [], "references": None}
        for _time in sorted(snapshot_times.value):
            _runs = _curve[_curve[x] == _time]
            if _runs.empty:
                continue
            _rollout = load_rollout(_runs.path.iloc[0], snapshot_element.value)
            _block["references"] = _rollout["references"]
            _block["rows"].append([_time, float(_runs.switch_rate.mean()), _rollout])
            _rollouts.append((int(_runs.impulse_steps.iloc[0]), _rollout))
        if _block["rows"]:
            _blocks.append(_block)

    # The same frame columns for every row; rows become (time, rate, images)
    _columns, _frame_label = [], ""
    if _rollouts:
        _frame_titles, _frame_label, _frames = arrange_frames(_rollouts, snapshot_frames.value, snapshot_timing.value)
        _columns = ["before", "impulse", *_frame_titles]
        _frames = iter(_frames)
        for _block in _blocks:
            _block["rows"] = [
                (_time, _rate, [_rollout["before"], _rollout["impulse"], *next(_frames)])
                for _time, _rate, _rollout in _block["rows"]
            ]

    with paper_style():
        summary_fig = summary_figure(
            _shown, x, _colours, _labels, blocks=_blocks, columns=_columns, frame_label=_frame_label,
            horizon=horizon(_shown) if figure_horizon.value else None,
            critical=_critical if figure_critical.value else None,
            show_runs=figure_runs.value,
        )
    summary_fig
    return (summary_fig,)


@app.cell
def _():
    figure_name = mo.ui.text(value="impulse_critical_time", label="File name")
    save_button = mo.ui.run_button(label="Save figure")
    mo.hstack([figure_name, save_button], justify="start")
    return figure_name, save_button


@app.cell
def _(figure_name, save_button, summary_fig):
    mo.stop(not save_button.value)
    _paths = save_figure(summary_fig, figure_name.value, folder="impulse")
    mo.md("Saved " + ", ".join(f"`{_path}`" for _path in _paths))
    return


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
