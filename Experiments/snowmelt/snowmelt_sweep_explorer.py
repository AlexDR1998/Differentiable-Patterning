# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

"""Compare the snowmelt NCA models trained by each experiment sweep.

Run from the repository root with:

    marimo edit Experiments/snowmelt/snowmelt_sweep_explorer.py

The sweeps are listed in ``Experiments/snowmelt/sweeps.py:SWEEPS``: resolution,
input channels, model architecture, update rule and held-out dates. Model bundles are read from
the model store (``MODEL_STORE_ROOT``, default ``models/``) and the data from
``SNOWMELT_DATA_ROOT`` (default ``~/PhD/Data/snowmelt``, holding ``v1/``, ``v2/``);
each model is scored on the dataset version it was trained on. Training metrics come
from each bundle's manifest; inference scores come from rolling every model out
with ``Experiments/snowmelt/evaluation.py``. Maps and single-model views are in
``snowmelt_explorer.py``.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    import zlib
    from pathlib import Path

    import jax.random as jr
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    from Common.dataloader.snowmelt import load_snowmelt, resolve_snowmelt_root
    from Experiments.snowmelt import sweeps

    # Light figures regardless of the marimo theme, so annotation ink stays legible.
    plt.style.use("default")

    DATA_ROOT = Path(
        os.environ.get("SNOWMELT_DATA_ROOT", Path.home() / "PhD" / "Data" / "snowmelt")
    )

    # Snow is white: dark rock (no snow / low NDSI) -> white (snow / high NDSI)
    SNOW_CMAP = LinearSegmentedColormap.from_list("snow", ["#2b2622", "#7d7266", "#c9c3bb", "#ffffff"])
    if "snow" not in matplotlib.colormaps:
        matplotlib.colormaps.register(SNOW_CMAP)

    # Categorical hues in fixed order (one per level of the colour factor)
    COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

    # Summary columns that can be plotted: column -> axis label
    METRICS = {
        "snow_csi": "snow CSI, mean over dates (higher is better)",
        "final_snow_csi": "snow CSI, final date (higher is better)",
        "skill": "MSE skill vs persistence, mean over dates (higher is better)",
        "final_skill": "MSE skill vs persistence, final date (higher is better)",
        "rmse": "RMSE, mean over dates (lower is better)",
        "held_out_snow_csi": "snow CSI, held-out dates (higher is better)",
        "held_out_skill": "MSE skill vs persistence, held-out dates (higher is better)",
        "held_out_rmse": "RMSE, held-out dates (lower is better)",
        "snow_area_error": "snow-covered fraction, predicted − observed",
        "rollout_spread": "std across rollouts",
        "best_loss": "best training loss (lower is better)",
        "best_at_fraction": "best checkpoint iteration / iterations",
    }
    # Metrics centred on zero get a diverging colour scale
    DIVERGING = {"snow_area_error", "skill", "final_skill", "held_out_skill"}

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def hue_width(n):
        """Space given to each of ``n`` hue levels within one x position (0.8 in all)."""
        return 0.8 / n

    def hue_offset(k, n):
        """Horizontal shift of hue level ``k`` of ``n`` within one x position."""
        return (k - (n - 1) / 2) * hue_width(n)

    def factor_plot(ax, df, x, y, hue=None, x_order=None, hue_order=None):
        """The mean over repeats per level, side by side for each hue, with one dot per run.

        Numeric ``x``: means are markers joined by a line. Categorical ``x``: means are bars,
        with the std across repeats as error bars.
        """
        x_order = levels(df[x]) if x_order is None else [v for v in x_order if v in set(df[x])]
        hue_order = [None] if hue is None else (levels(df[hue]) if hue_order is None else hue_order)
        numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in x_order)
        for k, level in enumerate(hue_order):
            rows = df if hue is None else df[df[hue] == level]
            offset = hue_offset(k, len(hue_order))
            colour = COLOURS[k % len(COLOURS)]
            label = None if level is None else str(level)
            runs = [rows.loc[rows[x] == value, y].dropna().to_numpy(float) for value in x_order]
            means = [values.mean() if len(values) else np.nan for values in runs]
            positions = np.arange(len(x_order)) + offset
            if numeric:
                for i, values in enumerate(runs):
                    ax.scatter(np.full(len(values), i + offset), values, s=18, color=colour, alpha=0.45, lw=0)
                ax.plot(positions, means, marker="o", ms=7, lw=1.5, color=colour, label=label)
            else:
                spread = [values.std(ddof=1) if len(values) > 1 else np.nan for values in runs]
                ax.bar(positions, means, width=0.9 * hue_width(len(hue_order)), color=colour, alpha=0.85, label=label)
                ax.errorbar(positions, means, yerr=spread, fmt="none", ecolor="#333", capsize=2, lw=1)
                for i, values in enumerate(runs):
                    ax.scatter(np.full(len(values), i + offset), values, s=10, color="#222", alpha=0.5, lw=0, zorder=3)
        ax.set_xticks(range(len(x_order)), [str(v) for v in x_order], rotation=0 if numeric else 20,
                      ha="center" if numeric else "right")
        ax.set_xlabel(x)
        ax.set_ylabel(METRICS.get(y, y), fontsize=9)
        style_axes(ax)
        if hue is not None:
            ax.legend(title=hue, frameon=False, fontsize=8)

    def heatmap(ax, df, rows, cols, value, row_order, col_order):
        """Mean of ``value`` per (row, column) cell, annotated with the std across repeats and the run count."""
        def table(aggfunc):
            return df.pivot_table(index=rows, columns=cols, values=value, aggfunc=aggfunc).reindex(
                index=row_order, columns=col_order
            ).to_numpy(float)

        mean, std, count = table("mean"), table("std"), table("count")
        if value in DIVERGING:
            limit = np.nanmax(np.abs(mean)) if np.isfinite(mean).any() else 1.0
            image = ax.imshow(mean, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        else:
            image = ax.imshow(mean, cmap="cividis", aspect="auto")
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                if np.isfinite(mean[i, j]):
                    spread = f" ± {std[i, j]:.3g}" if np.isfinite(std[i, j]) else ""
                    ax.text(j, i, f"{mean[i, j]:.3g}{spread}\n(n={int(count[i, j])})", ha="center", va="center",
                            fontsize=7, color="white" if image.norm(mean[i, j]) < 0.5 else "black")
        ax.set_xticks(range(len(col_order)), col_order, rotation=20, ha="right")
        ax.set_yticks(range(len(row_order)), row_order)
        ax.set_xlabel(cols)
        ax.set_ylabel(rows)
        return image

    def snow_area_plot(ax, scores, hue, hue_order):
        """Snow-covered fraction by date: observed, and the mean prediction of each hue level."""
        observed = scores.groupby("days")["observed_snow_fraction"].mean()
        ax.plot(observed.index, 100 * observed.to_numpy(), color="#222", lw=2.5, marker="o", ms=7,
                label="observed", zorder=3)
        for k, level in enumerate(hue_order):
            rows = scores[scores[hue] == level]
            if rows.empty:
                continue
            per_model = rows.pivot_table(index="days", columns="model_id", values="predicted_snow_fraction")
            mean, spread = per_model.mean(axis=1), per_model.std(axis=1).fillna(0.0)
            colour = COLOURS[k % len(COLOURS)]
            ax.plot(mean.index, 100 * mean.to_numpy(), color=colour, lw=2, marker="o", ms=4, label=str(level))
            ax.fill_between(mean.index, 100 * (mean - spread).to_numpy(), 100 * (mean + spread).to_numpy(),
                            color=colour, alpha=0.12, lw=0)
        ax.set_xlabel("days since first acquisition")
        ax.set_ylabel("snow-covered area (% of catchment)")
        ax.set_ylim(0, 105)
        style_axes(ax)
        ax.legend(title=hue, frameon=False, fontsize=8)

    def condition_label(row, sweep):
        """Short name of a run's condition: its factor values, without the repeat."""
        return " · ".join(f"{name}={row[name]}" for name in sweep.factors)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Snowmelt NCA sweeps: model comparison

    Compares the models trained by each snowmelt experiment sweep
    (`Experiments/snowmelt/conf/experiments/`):

    | Sweep | Question | Factors |
    |---|---|---|
    | Resolution | How do spatial and temporal resolution affect the model? | downsample and t, together (convective scaling) |
    | Input channels | Which target channels and terrain (static) channels help? | target set × static set |
    | Architecture | Gated vs plain update, perception kernels, width | family × kernels × channels |
    | Update rule | Stochastic vs deterministic updates, activation | fire rate (with t, fire rate × t fixed) × activation |
    | Held-out dates | Does the model interpolate to a date it was not trained on? | held-out date |

    **Training metrics** (best loss, and when the best checkpoint came) are read from
    each bundle's manifest. The training loss is only comparable between runs with the
    same target channels and resolution: it is not comparable across the resolution
    sweep or across target sets in the input sweep.

    **Inference scores** come from rolling each model out from the first acquisition
    (as in `snowmelt_explorer.py`) and comparing it with the later acquisitions.
    Every model is scored on one channel: **NDSI** when it is a target, otherwise
    **SCA**. *Snow CSI* (hits / (hits + misses + false alarms) of the snow map,
    NDSI > 0.4 or SCA > 0.5) is the one score that is comparable across every run,
    including the SCA-only models. *Skill* is 1 − MSE / MSE<sub>persistence</sub>,
    where persistence repeats the first image. Scoring on the *10 m grid* compares
    models trained at different resolutions on the same pixels.
    """)
    return


@app.cell(hide_code=True)
def _():
    _default_store = os.environ.get("MODEL_STORE_ROOT", str(Path(__file__).resolve().parents[2] / "models"))
    store_root = mo.ui.text(_default_store, label="Model store", full_width=True)
    reload_button = mo.ui.button(label="Reload models", value=0, on_click=lambda count: count + 1)
    mo.vstack([
        store_root,
        mo.hstack([reload_button, mo.md("after copying new bundles into the store")], justify="start"),
    ])
    return reload_button, store_root


@app.cell(hide_code=True)
def _(reload_button, store_root):
    # One row per completed bundle of each sweep, reruns included (they are dropped after the
    # group filter below). Bundles are read from their folders, so the registry index does not
    # need rebuilding.
    reload_button.value
    _store = Path(store_root.value).expanduser()

    bundles = {}
    expected_by_sweep = {}
    all_runs = []
    for _name, _sweep in sweeps.SWEEPS.items():
        expected_by_sweep[_name] = sweeps.expected_runs(_sweep)
        for _bundle in sweeps.find_bundles(_store, _sweep):
            bundles[_bundle.id] = _bundle
            _row = {"sweep": _name, **sweeps.training_row(_bundle, _sweep)}
            _row["wandb_group"] = _row["wandb_group"] or "(no group)"
            all_runs.append(_row)
    mo.md(f"**{len(all_runs)}** sweep bundles in `{_store}`.")
    return all_runs, bundles, expected_by_sweep


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model selection

    - Pick the W&B groups to use for each sweep (newest group first, model count in brackets)
    - Within a group, only the newest bundle of each run is kept; runs from different groups are pooled as repeats
    - Untick rows in the table to drop single models; evaluation and every plot below use the ticked models
    """)
    return


@app.cell(hide_code=True)
def _(all_runs):
    _selects = {}
    for _name, _sweep in sweeps.SWEEPS.items():
        _runs = [_row for _row in all_runs if _row["sweep"] == _name]
        _newest = {}
        for _row in sorted(_runs, key=lambda _r: _r["created_at"]):
            _newest[_row["wandb_group"]] = _row["created_at"]
        _counts = pd.Series([_row["wandb_group"] for _row in _runs], dtype=object).value_counts()
        _options = {f"{_group} ({_counts[_group]})": _group
                    for _group in sorted(_newest, key=_newest.get, reverse=True)}
        _selects[_name] = mo.ui.multiselect(_options, value=list(_options), label=_sweep.title, full_width=True)
    group_select = mo.ui.dictionary(_selects)
    mo.vstack([group_select[_name] for _name in _selects])
    return (group_select,)


@app.cell(hide_code=True)
def _(all_runs, expected_by_sweep, group_select):
    _kept_rows = []
    _status_rows = []
    _missing = {}
    for _name, _sweep in sweeps.SWEEPS.items():
        _groups = set(group_select.value[_name])
        _rows = [_row for _row in all_runs if _row["sweep"] == _name and _row["wandb_group"] in _groups]
        _kept, _superseded = sweeps.latest_per_run(_rows, _sweep)
        _expected = expected_by_sweep[_name]
        _missing[_name] = sweeps.missing_runs(_expected, _kept, _sweep)
        _kept_rows += _kept
        _status_rows.append({
            "sweep": _sweep.title,
            "experiment": _sweep.experiment,
            "groups": len(_groups),
            "expected runs": len(_expected) or None,
            "found": len(_kept),
            "missing": len(_missing[_name]) if _expected else None,
            "superseded (older reruns)": len(_superseded),
        })
    # Models in the selected groups: the rows the table below can pick from
    candidates = pd.DataFrame(_kept_rows)

    _missing_tables = {
        f"{sweeps.SWEEPS[_name].title}: {len(_rows)} missing": mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=10)
        for _name, _rows in _missing.items() if _rows
    }
    model_table = mo.ui.table(
        candidates.dropna(axis=1, how="all").round(4) if len(candidates) else pd.DataFrame({"model_id": []}),
        initial_selection=list(range(len(candidates))),
        page_size=10,
        label="Models",
    )
    mo.vstack([
        mo.ui.table(pd.DataFrame(_status_rows), selection=None),
        mo.accordion(_missing_tables) if _missing_tables else mo.md(""),
        model_table,
    ])
    return candidates, model_table


@app.cell(hide_code=True)
def _(candidates, model_table):
    # The models used for evaluation and for every plot below
    _ticked = set(model_table.value["model_id"]) if len(model_table.value) else set()
    selected = candidates[candidates.model_id.isin(_ticked)] if len(candidates) else candidates
    mo.md(f"**{len(selected)}** of {len(candidates)} models selected.")
    return (selected,)


@app.cell(hide_code=True)
def _(bundles):
    # Each bundle is scored on the dataset version it was trained on; versions load on first use.
    _raw_by_version = {}

    def raw_for(version):
        if version not in _raw_by_version:
            _raw_by_version[version] = load_snowmelt(resolve_snowmelt_root(DATA_ROOT, version))
        return _raw_by_version[version]

    _needed = sorted({_b.config.data.snowmelt.version for _b in bundles.values()})
    _missing = []
    for _version in _needed:
        try:
            resolve_snowmelt_root(DATA_ROOT, _version)
        except (FileNotFoundError, ValueError) as _error:
            _missing.append(f"{_version}: {_error}")
    if _missing:
        _out = mo.callout(mo.md("Some dataset versions the models need are missing; their models will fail:\n\n"
                                + "\n".join(f"- {_m}" for _m in _missing)
                                + "\n\nSet SNOWMELT_DATA_ROOT to the folder holding `v1/`, `v2/`. "
                                  "Training metrics are still shown."), kind="warn")
    else:
        _out = mo.md(f"Data: `{DATA_ROOT}`; dataset versions used by these models: {', '.join(_needed) or 'none'}.")
    _out
    return (raw_for,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_mode = mo.ui.radio(
        {"Free run from first date": "free", "One interval ahead": "interval"},
        value="Free run from first date", label="Rollout", inline=True,
    )
    eval_rollouts = mo.ui.number(1, 32, value=4, step=1, label="Rollouts per model")
    eval_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    eval_grid = mo.ui.radio({"Model grid": "model", "10 m grid": "full"}, value="10 m grid", label="Score on", inline=True)
    eval_verify = mo.ui.checkbox(value=True, label="Require input to match the bundle fingerprint")
    eval_run = mo.ui.run_button(label="Evaluate models")
    mo.hstack([eval_mode, eval_rollouts, eval_seed, eval_grid, eval_verify, eval_run],
              justify="start", gap=1.5, wrap=True)
    return eval_grid, eval_mode, eval_rollouts, eval_run, eval_seed, eval_verify


@app.cell
def _():
    # Scores of every model evaluated in this session, keyed by (model_id, evaluation settings),
    # so re-running the evaluation only rolls out new models.
    score_cache = {}
    return (score_cache,)


@app.cell(hide_code=True)
def _(
    bundles,
    eval_grid,
    eval_mode,
    eval_rollouts,
    eval_run,
    eval_seed,
    eval_verify,
    raw_for,
    score_cache,
    selected,
):
    # Models already scored with these settings are shown straight away; the button scores the rest
    _settings = (eval_mode.value, int(eval_rollouts.value), int(eval_seed.value), eval_grid.value, eval_verify.value)
    _todo = list(selected.model_id) if len(selected) else []
    _failures = []
    if eval_run.value:
        _sequences, _references = {}, {}
        _new = [_model_id for _model_id in _todo if (_model_id, *_settings) not in score_cache]
        for _model_id in mo.status.progress_bar(_new, title="Evaluating models", remove_on_exit=True):
            # A fixed key per model, so results do not depend on which models are selected
            _key = jr.fold_in(jr.PRNGKey(_settings[2]), zlib.crc32(_model_id.encode()) & 0x7FFFFFFF)
            try:
                _bundle = bundles[_model_id]
                score_cache[(_model_id, *_settings)] = sweeps.evaluate_bundle(
                    _bundle, raw_for(_bundle.config.data.snowmelt.version), _key, n_rollouts=_settings[1], mode=_settings[0],
                    grid=_settings[3], verify=_settings[4], sequences=_sequences, references=_references,
                )
            except Exception as _error:  # keep going: one bad bundle should not stop the comparison
                _failures.append({"model_id": _model_id, "error": f"{type(_error).__name__}: {_error}"})

    _score_rows = []
    _summary_rows = []
    eval_maps = {}
    for _model_id in _todo:
        if (_model_id, *_settings) not in score_cache:
            continue
        _rows, _maps = score_cache[(_model_id, *_settings)]
        _score_rows += [{"model_id": _model_id, **_row} for _row in _rows if _row["channel"] == _maps["channel"]]
        _summary_rows.append({"model_id": _model_id, **sweeps.summarise_scores(_rows, _maps["channel"])})
        eval_maps[_model_id] = _maps

    _remaining = len(_todo) - len(_summary_rows)
    _status = mo.vstack([
        mo.md(f"Scored **{len(_summary_rows)}** of {len(_todo)} selected model(s): "
              f"{ {'free': 'free run', 'interval': 'one interval ahead'}[_settings[0]]}, {_settings[1]} rollout(s) each, "
              f"on the { {'model': 'model grid', 'full': '10 m grid'}[_settings[3]]}"
              + ("" if _settings[4] else " (**input fingerprints not checked**)") + "."
              + (f" Click **Evaluate models** to score the other {_remaining}; training metrics are shown below without it."
                 if _remaining and not _failures else "")),
        mo.callout(mo.vstack([mo.md(f"**{len(_failures)}** model(s) failed:"),
                              mo.ui.table(pd.DataFrame(_failures), selection=None)]), kind="danger")
        if _failures else mo.md(""),
    ])

    # Training metrics for every selected model, with the inference summary where it has been computed
    results = selected.merge(pd.DataFrame(_summary_rows), on="model_id", how="left") if _summary_rows else selected
    # Scores by date on each model's score channel, with the model's factors
    scores = selected.merge(pd.DataFrame(_score_rows), on="model_id") if _score_rows else pd.DataFrame()
    _status
    return eval_maps, results, scores


@app.cell(hide_code=True)
def _(results, scores):
    if results.empty:
        _out = mo.md("")
    else:
        _out = mo.vstack([
            mo.md("### All models (one row per run)"),
            mo.ui.table(results.dropna(axis=1, how="all").round(4), selection=None, page_size=12),
            mo.hstack([
                mo.download(results.to_csv(index=False).encode(), filename="snowmelt_sweep_models.csv",
                            mimetype="text/csv", label="Models CSV"),
                mo.download(scores.to_csv(index=False).encode(), filename="snowmelt_sweep_scores_by_date.csv",
                            mimetype="text/csv", label="Scores by date CSV"),
            ], justify="start") if not scores.empty else mo.md(""),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(results):
    _available = [_name for _name in METRICS if _name in results and results[_name].notna().any()]
    metric = mo.ui.dropdown(
        {METRICS[_name]: _name for _name in _available},
        value=METRICS["snow_csi" if "snow_csi" in _available else "best_loss"] if _available else None,
        label="Metric for the plots below",
    )
    metric if _available else mo.md("")
    return (metric,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spatiotemporal resolution

    Resolution and step count change together (t ∝ 1/h), so each NCA step spans the same
    fraction of a pixel. The training loss falls at coarser resolution partly because the
    targets are smoother, so use the inference scores (on the 10 m grid) to compare.
    """)
    return


@app.cell(hide_code=True)
def _(metric, results):
    _df = results[results.sweep == "resolution"].copy() if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No resolution-sweep models found.")
    else:
        _df["resolution"] = [f"{_r} m, t={_t}" for _r, _t in zip(_df.resolution_m, _df.t)]
        _order = [f"{_r} m, t={_t}" for _r, _t in sorted(set(zip(_df.resolution_m, _df.t)))]
        _panels = list(dict.fromkeys([metric.value, "best_loss", "best_at_fraction"]))
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(5 * len(_panels), 3.8), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            factor_plot(_ax, _df, "resolution", _y, x_order=_order)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(scores):
    _df = scores[scores.sweep == "resolution"].copy() if len(scores) else scores
    if _df.empty or "observed_snow_fraction" not in _df:
        _out = mo.md("")
    else:
        _df["resolution"] = [f"{_r} m, t={_t}" for _r, _t in zip(_df.resolution_m, _df.t)]
        _order = [f"{_r} m, t={_t}" for _r, _t in sorted(set(zip(_df.resolution_m, _df.t)))]
        _fig, _ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        snow_area_plot(_ax, _df, "resolution", _order)
        _ax.set_title("Snow-covered area (band = ±1 std across repeats)", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Input channels and terrain

    Hidden channels are fixed at 28, so the target and static sets change only what the
    model sees and predicts. Compare target sets with *snow CSI* (the only score shared
    with the SCA-only models); NDSI skill compares the NDSI-based sets. The loss map is
    only comparable along a row (same targets).
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, metric, results):
    _df = results[results.sweep == "input"] if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No input-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["input"]) if expected_by_sweep["input"] else _df
        _targets, _static = levels(_expected.targets), levels(_expected.static)
        _panels = list(dict.fromkeys([metric.value, "best_loss"]))
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(6.5 * len(_panels), 4.5), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            _image = heatmap(_ax, _df, "targets", "static", _y, _targets, _static)
            _fig.colorbar(_image, ax=_ax, shrink=0.8, label=METRICS.get(_y, _y))
            _ax.set_title(f"{_y}: mean ± std over repeats", fontsize=10)
        _fig2, _ax2 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        factor_plot(_ax2, _df, "targets", metric.value, hue="static", x_order=_targets, hue_order=_static)
        _out = mo.vstack([_fig, _fig2])
    _out
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, scores):
    _df = scores[scores.sweep == "input"] if len(scores) else scores
    if _df.empty or "observed_snow_fraction" not in _df:
        _out = mo.md("")
    else:
        _expected = pd.DataFrame(expected_by_sweep["input"]) if expected_by_sweep["input"] else _df
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, sharey=True)
        snow_area_plot(_axes[0], _df, "static", levels(_expected.static))
        _axes[0].set_title("By static set (all target sets)", fontsize=10)
        snow_area_plot(_axes[1], _df, "targets", levels(_expected.targets))
        _axes[1].set_title("By target set (all static sets)", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model architecture

    DIFF is the gradient magnitude (isotropic), GRAD the full gradient (magnitude and
    direction). The second figure shows GRAD minus DIFF for each family and width, with
    and without LAP: above zero means the gradient direction helps (for scores where
    higher is better).
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    _source = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
    _families = levels(_source.family) if len(_source) else []
    arch_family = mo.ui.radio({str(_f): _f for _f in _families},
                              value="gNCA" if "gNCA" in _families else (str(_families[-1]) if _families else None),
                              label="Model family for the architecture figures", inline=True)
    arch_family if _families else mo.md("")
    return (arch_family,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Width and kernels

    - Colour: static (terrain) set
    - Left: score against channel count, pooled over kernels
    - Right: score by kernel set at one channel count
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    _source = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
    _widths = levels(_source.channels) if len(_source) else []
    arch_width = mo.ui.dropdown({str(_w): _w for _w in _widths}, value=str(_widths[-1]) if _widths else None,
                                label="Channels for the kernel panels")
    arch_width if _widths else mo.md("")
    return (arch_width,)


@app.cell(hide_code=True)
def _(arch_family, arch_width, expected_by_sweep, metric, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or metric.value is None or arch_width.value is None or arch_family.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _statics = levels(_expected.static)
        _df = _df[_df.family == arch_family.value]
        _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, sharey=True)
        factor_plot(_left, _df, "channels", metric.value, hue="static",
                    x_order=levels(_expected.channels), hue_order=_statics)
        _left.set_title(f"{arch_family.value}: more channels", fontsize=10)
        factor_plot(_right, _df[_df.channels == arch_width.value], "kernels", metric.value,
                    hue="static", x_order=levels(_expected.kernels), hue_order=_statics)
        _right.set_title(f"{arch_family.value}: kernels at {arch_width.value} channels", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Predictions and training loss

    - Left: best training loss by kernel set and static set, for the chosen family and one width; letters mark the models on the right
    - Right: observed and predicted score channel on one date, one model (repeat) per condition
    - Predictions need the models to be evaluated above
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    _source = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
    _has = len(_source) > 0
    _widths = levels(_source.channels) if _has else []
    _repeats = levels(_source.repeat) if _has else []
    _conditions = {f"{_k} · {_s}": (_k, _s) for _k in (levels(_source.kernels) if _has else [])
                   for _s in levels(_source.static)}
    # Default: the isotropic vs directional gradient (with LAP) under each static set
    _default = [_c for _c, (_k, _) in _conditions.items() if _k in ("ID+LAP+DIFF", "ID+LAP+GRAD")] or list(_conditions)[:4]
    pred_width = mo.ui.dropdown({str(_w): _w for _w in _widths}, value=str(_widths[-1]) if _widths else None,
                                label="Channels")
    pred_repeat = mo.ui.dropdown({str(_r): _r for _r in _repeats}, value=str(_repeats[0]) if _repeats else None,
                                 label="Repeat")
    pred_conditions = mo.ui.multiselect(_conditions, value=_default, label="Conditions shown (kernels · static)",
                                        full_width=True)
    pred_view = mo.ui.radio({"Prediction": "mean", "Prediction − observed": "error"}, value="Prediction",
                            label="Show", inline=True)
    mo.vstack([
        mo.hstack([pred_width, pred_repeat, pred_view], justify="start", gap=1.5),
        pred_conditions,
    ]) if _has else mo.md("")
    return pred_conditions, pred_repeat, pred_view, pred_width


@app.cell(hide_code=True)
def _(eval_maps):
    # Date 0 is the initial condition, so the slider starts at the first predicted date
    _n_dates = max((len(_m["dates"]) for _m in eval_maps.values()), default=2)
    pred_date = mo.ui.slider(1, max(_n_dates - 1, 1), value=max(_n_dates - 1, 1), label="Date index", show_value=True)
    pred_date if eval_maps else mo.md("")
    return (pred_date,)


@app.cell(hide_code=True)
def _(
    arch_family,
    eval_maps,
    expected_by_sweep,
    pred_conditions,
    pred_date,
    pred_repeat,
    pred_view,
    pred_width,
    results,
):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or arch_family.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _kernels, _statics = levels(_expected.kernels), levels(_expected.static)
        _df = _df[(_df.family == arch_family.value) & (_df.channels == pred_width.value)]

        # One model per condition: the chosen repeat (the newest bundle if several groups have it)
        _shown = []
        for _k, (_kernel, _static) in enumerate(pred_conditions.value):
            _rows = _df[(_df.kernels == _kernel) & (_df.static == _static) & (_df.repeat == pred_repeat.value)]
            _model_id = _rows.sort_values("created_at").model_id.iloc[-1] if len(_rows) else None
            _shown.append((chr(ord("A") + _k), _kernel, _static, _model_id))

        _fig = plt.figure(figsize=(15, 5.5), layout="constrained")
        _left_fig, _right_fig = _fig.subfigures(1, 2, width_ratios=[1, 1.5])

        # Training loss, with the letter of each shown condition inside its bar
        _ax = _left_fig.subplots()
        factor_plot(_ax, _df, "kernels", "best_loss", hue="static", x_order=_kernels, hue_order=_statics)
        _present = [_v for _v in _kernels if _v in set(_df.kernels)]
        for _letter, _kernel, _static, _model_id in _shown:
            if _model_id is None or _kernel not in _present:
                continue
            _x = _present.index(_kernel) + hue_offset(_statics.index(_static), len(_statics))
            _mean = _df.loc[(_df.kernels == _kernel) & (_df.static == _static), "best_loss"].mean()
            _ax.text(_x, _mean / 2, _letter, ha="center", va="center", fontsize=10, weight="bold",
                     color="white", zorder=4)
        _ax.set_title(f"{arch_family.value}, {pred_width.value} channels: training loss", fontsize=10)

        # Observed map, then one prediction per shown condition
        _evaluated = [_m for *_, _m in _shown if _m in eval_maps]
        if not _evaluated:
            _right_fig.text(0.5, 0.5, "Evaluate these models to show their predictions",
                            ha="center", va="center", fontsize=10, color="#555")
        else:
            _first = eval_maps[_evaluated[0]]
            _t = min(int(pred_date.value), len(_first["dates"]) - 1)
            _observed = np.where(_first["catchment"], _first["observed"][_t], np.nan)
            _panels = [("observed", _observed)]
            for _letter, _kernel, _static, _model_id in _shown:
                _title = f"{_letter}: {_kernel}\n{_static}"
                if _model_id not in eval_maps:
                    _panels.append((f"{_title}\n(not evaluated)", None))
                    continue
                _maps = eval_maps[_model_id]
                _value = _maps["mean"][_t] - (_maps["observed"][_t] if pred_view.value == "error" else 0)
                _panels.append((_title, np.where(_maps["catchment"], _value, np.nan)))

            _snow_range = np.nanpercentile(_observed, (2, 98))
            if pred_view.value == "error":
                _errors = np.concatenate([_p.ravel() for _, _p in _panels[1:] if _p is not None])
                _limit = np.nanpercentile(np.abs(_errors), 98) or 1.0
            _ncols = len(_panels) if len(_panels) <= 3 else -(-len(_panels) // 2)
            _nrows = -(-len(_panels) // _ncols)
            _axes = _right_fig.subplots(_nrows, _ncols, squeeze=False)
            for _a in _axes.ravel():
                _a.axis("off")
            _images = []
            for _i, (_a, (_title, _image)) in enumerate(zip(_axes.ravel(), _panels)):
                _a.set_title(_title, fontsize=7)
                if _image is None:
                    continue
                if _i == 0 or pred_view.value == "mean":
                    _images.append(_a.imshow(_image, cmap="snow", vmin=_snow_range[0], vmax=_snow_range[1],
                                             interpolation="nearest"))
                else:
                    _images.append(_a.imshow(_image, cmap="RdBu_r", vmin=-_limit, vmax=_limit,
                                             interpolation="nearest"))
            _channel = _first["channel"]
            _right_fig.colorbar(_images[0], ax=list(_axes.ravel()), shrink=0.6, label=_channel)
            if pred_view.value == "error" and len(_images) > 1:
                _right_fig.colorbar(_images[-1], ax=list(_axes.ravel()), shrink=0.6,
                                    label=f"{_channel}, predicted − observed")
            _right_fig.suptitle(f"{_channel} on {_first['dates'][_t]}", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(arch_family, expected_by_sweep, metric, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or metric.value is None or arch_family.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _kernels, _channels = levels(_expected.kernels), levels(_expected.channels)
        _statics = levels(_expected.static)
        _df = _df[_df.family == arch_family.value]
        _outputs = []
        for _y in dict.fromkeys([metric.value, "best_loss"]):
            _fig, _axes = plt.subplots(1, len(_statics), figsize=(6.5 * len(_statics), 4),
                                       constrained_layout=True, sharey=True, squeeze=False)
            for _ax, _static in zip(_axes[0], _statics):
                factor_plot(_ax, _df[_df.static == _static], "kernels", _y,
                            hue="channels", x_order=_kernels, hue_order=_channels)
                _ax.set_title(f"{arch_family.value}, static {_static}", fontsize=10)
            _outputs.append(_fig)

        # Direction effect: mean(GRAD) - mean(DIFF), with the std of that difference from the repeats
        _pairs = {"without LAP": ("ID+DIFF", "ID+GRAD"), "with LAP": ("ID+LAP+DIFF", "ID+LAP+GRAD")}
        _rows = []
        for (_width, _static), _group in _df.groupby(["channels", "static"]):
            for _pair, (_isotropic, _directional) in _pairs.items():
                _a = _group.loc[_group.kernels == _isotropic, metric.value].dropna()
                _b = _group.loc[_group.kernels == _directional, metric.value].dropna()
                if len(_a) and len(_b):
                    _se = np.sqrt(_a.var(ddof=1) / len(_a) + _b.var(ddof=1) / len(_b)) if min(len(_a), len(_b)) > 1 else np.nan
                    _rows.append({"model": f"{_width} ch", "static": _static, "pair": _pair,
                                  "GRAD − DIFF": _b.mean() - _a.mean(), "std error": _se})
        if _rows:
            _diff = pd.DataFrame(_rows)
            _names = list(dict.fromkeys(_diff.model))
            _fig, _axes = plt.subplots(1, len(_pairs), figsize=(13, 3.8), constrained_layout=True, sharey=True)
            for _ax, _pair in zip(_axes, _pairs):
                for _k, _static in enumerate(_statics):
                    _sub = _diff[(_diff.pair == _pair) & (_diff.static == _static)].set_index("model").reindex(_names)
                    _x = np.arange(len(_names)) + hue_offset(_k, len(_statics))
                    _ax.bar(_x, _sub["GRAD − DIFF"], width=0.9 * hue_width(len(_statics)), color=COLOURS[_k % len(COLOURS)], label=_static)
                    _ax.errorbar(_x, _sub["GRAD − DIFF"], yerr=_sub["std error"], fmt="none", ecolor="#333", capsize=3)
                _ax.axhline(0, color="#777", lw=1)
                _ax.set_xticks(range(len(_names)), _names, rotation=20, ha="right")
                _ax.set_title(f"{arch_family.value}: effect of gradient direction, {_pair}", fontsize=10)
                style_axes(_ax)
            _axes[0].set_ylabel(f"GRAD − DIFF\n{METRICS.get(metric.value, metric.value)}", fontsize=9)
            _axes[0].legend(title="static", frameon=False, fontsize=8)
            _fig.suptitle("Error bars: std error from repeats", fontsize=9)
            _outputs.append(_fig)
        _out = mo.vstack(_outputs)
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Update rule

    Fire rate and step count are paired (fire rate × t fixed), so every model makes the
    same expected number of updates per pixel and only the randomness of the update
    changes. Fire rate 1 is deterministic, so its spread across rollouts is zero.
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, metric, results):
    _df = results[results.sweep == "update_rule"] if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No update-rule-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["update_rule"]) if expected_by_sweep["update_rule"] else _df
        _activations = levels(_expected.activation)
        _panels = list(dict.fromkeys(
            [metric.value, "best_loss"] + (["rollout_spread"] if "rollout_spread" in _df else [])
        ))
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(5 * len(_panels), 3.8), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            factor_plot(_ax, _df, "fire_rate", _y, hue="activation", hue_order=_activations)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(scores):
    _df = scores[scores.sweep == "update_rule"] if len(scores) else scores
    if _df.empty or "observed_snow_fraction" not in _df:
        _out = mo.md("")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
        snow_area_plot(_ax, _df, "fire_rate", levels(_df.fire_rate))
        _ax.set_title("Snow-covered area by fire rate (all activations)", fontsize=10)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Held-out dates

    - Each run leaves one date (index into the v2 dates) out of training; `none` is the control
    - Left: score by date, hollow circle = the held-out date
    - Right: score on each held-out date, model that did not see it vs the control
    - Date 9 is the last date, so it tests extrapolation rather than interpolation
    """)
    return


@app.cell(hide_code=True)
def _(metric, scores):
    _df = scores[scores.sweep == "hold_out"] if len(scores) else scores
    _y = metric.value if metric.value in ("snow_csi", "skill", "rmse") else "snow_csi"
    if _df.empty or _y not in _df:
        _out = mo.md("No evaluated held-out-date models.")
    else:
        _conditions = levels(_df.held_out_dates)
        _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
        for _k, _condition in enumerate(_conditions):
            _rows = _df[_df.held_out_dates == _condition]
            _colour = "#222" if _condition == "none" else COLOURS[_k % len(COLOURS)]
            _mean = _rows.groupby("days")[_y].mean()
            _left.plot(_mean.index, _mean.to_numpy(), color=_colour, marker="o", ms=3, lw=1.5, label=_condition)
            _held = _rows[_rows.held_out].groupby("days")[_y].mean()
            _left.scatter(_held.index, _held.to_numpy(), s=110, facecolor="none", edgecolor=_colour, lw=2, zorder=3)
        _left.set_xlabel("days since first acquisition")
        _left.set_ylabel(_y)
        _left.legend(title="held out", frameon=False, fontsize=8)
        style_axes(_left)

        _held = _df[_df.held_out]
        _control = _df[_df.held_out_dates == "none"]
        _dates = sorted(set(zip(_held.days, _held.date)))
        for _i, (_day, _) in enumerate(_dates):
            for _offset, _rows, _colour, _label in ((-0.15, _held, COLOURS[0], "held out"),
                                                     (0.15, _control, "#999", "control (trained on it)")):
                _values = _rows.loc[_rows.days == _day, _y].dropna().to_numpy(float)
                if len(_values):
                    _right.bar(_i + _offset, _values.mean(), width=0.27, color=_colour, alpha=0.85,
                               yerr=_values.std(ddof=1) if len(_values) > 1 else None,
                               error_kw={"ecolor": "#333", "capsize": 2, "lw": 1},
                               label=_label if _i == 0 else None)
                _right.scatter(np.full(len(_values), _i + _offset), _values, s=10, color="#222", alpha=0.5, lw=0, zorder=3)
        _right.set_xticks(range(len(_dates)), [_date for _, _date in _dates], rotation=20, ha="right")
        _right.set_ylabel(_y)
        _right.legend(frameon=False, fontsize=8)
        style_axes(_right)
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Maps by condition

    The score channel on one date for every condition of a sweep, averaged over its
    repeats (and over rollouts), on each model's own grid.
    """)
    return


@app.cell(hide_code=True)
def _(eval_maps, results):
    _evaluated = results[results.model_id.isin(eval_maps)] if len(results) else results
    _names = [_name for _name in sweeps.SWEEPS if len(_evaluated) and (_evaluated.sweep == _name).any()]
    map_sweep = mo.ui.dropdown({sweeps.SWEEPS[_name].title: _name for _name in _names},
                               value=sweeps.SWEEPS[_names[0]].title if _names else None, label="Sweep")
    map_view = mo.ui.radio({"Prediction": "mean", "Prediction − observed": "error", "Std across rollouts": "std"},
                           value="Prediction − observed", label="Show", inline=True)
    map_date = mo.ui.slider(1, 4, value=4, label="Date index", show_value=True)
    mo.hstack([map_sweep, map_view, map_date], justify="start", gap=1.5) if _names else mo.md("")
    return map_date, map_sweep, map_view


@app.cell(hide_code=True)
def _(eval_maps, map_date, map_sweep, map_view, results):
    if map_sweep.value is None:
        _out = mo.md("Evaluate models to compare maps.")
    else:
        _sweep = sweeps.SWEEPS[map_sweep.value]
        _df = results[(results.sweep == map_sweep.value) & results.model_id.isin(eval_maps)]
        _conditions = {}
        for _row in _df.to_dict("records"):
            _conditions.setdefault(condition_label(_row, _sweep), []).append(eval_maps[_row["model_id"]])
        _first = next(iter(_conditions.values()))[0]
        _t = min(int(map_date.value), len(_first["dates"]) - 1)
        _channel = _first["channel"]

        def _field(maps):
            _inside = maps["catchment"]
            if map_view.value == "error":
                _value = maps["mean"][_t] - maps["observed"][_t]
            else:
                _value = maps[map_view.value][_t]
            return np.where(_inside, _value, np.nan)

        _panels = [("observed", np.where(_first["catchment"], _first["observed"][_t], np.nan))] if map_view.value == "mean" else []
        # Repeats of one condition share a grid, so their maps can be averaged
        _panels += [(_label, np.nanmean([_field(_m) for _m in _maps], axis=0)) for _label, _maps in _conditions.items()]
        _stack = np.concatenate([_p[1].ravel() for _p in _panels])
        if map_view.value == "error":
            _limit = np.nanpercentile(np.abs(_stack), 98) or 1.0
            _cmap, _vmin, _vmax = "RdBu_r", -_limit, _limit
        elif map_view.value == "std":
            _cmap, _vmin, _vmax = "Purples", 0.0, np.nanpercentile(_stack, 99) or 1.0
        else:
            _cmap, (_vmin, _vmax) = "snow", np.nanpercentile(_stack, (2, 98))
        _ncols = min(len(_panels), 6)
        _nrows = -(-len(_panels) // _ncols)
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(2.9 * _ncols, 2.9 * _nrows),
                                   constrained_layout=True, squeeze=False)
        for _ax in _axes.ravel():
            _ax.axis("off")
        for _ax, (_label, _image) in zip(_axes.ravel(), _panels):
            _im = _ax.imshow(_image, cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation="nearest")
            _ax.set_facecolor("#e6e6e6")
            _ax.set_title(_label.replace(" · ", "\n"), fontsize=7)
        _fig.colorbar(_im, ax=list(_axes.ravel()), shrink=0.6, label=f"{_channel} ({map_view.value})")
        _fig.suptitle(f"{_sweep.title}: {_channel} on {_first['dates'][_t]}", fontsize=10)
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
