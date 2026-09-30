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
input channels, model architecture and update rule. Model bundles are read from
the model store (``MODEL_STORE_ROOT``, default ``models/``) and the data from
``SNOWMELT_DATA_ROOT`` (default ``~/PhD/Data/snowmelt``). Training metrics come
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

    from Common.dataloader.snowmelt import load_snowmelt
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
        "snow_area_error": "snow-covered fraction, predicted − observed",
        "rollout_spread": "std across rollouts",
        "best_loss": "best training loss (lower is better)",
        "best_at_fraction": "best checkpoint iteration / iterations",
    }
    # Metrics centred on zero get a diverging colour scale
    DIVERGING = {"snow_area_error", "skill", "final_skill"}

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def factor_plot(ax, df, x, y, hue=None, x_order=None, hue_order=None):
        """One dot per run and the mean over repeats per level, side by side for each hue.

        Means are joined by a line when ``x`` is numeric.
        """
        x_order = levels(df[x]) if x_order is None else [v for v in x_order if v in set(df[x])]
        hue_order = [None] if hue is None else (levels(df[hue]) if hue_order is None else hue_order)
        numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in x_order)
        for k, level in enumerate(hue_order):
            rows = df if hue is None else df[df[hue] == level]
            offset = 0.0 if len(hue_order) == 1 else 0.6 * (k / (len(hue_order) - 1) - 0.5)
            colour = COLOURS[k % len(COLOURS)]
            means = []
            for i, value in enumerate(x_order):
                values = rows.loc[rows[x] == value, y].dropna().to_numpy(float)
                ax.scatter(np.full(len(values), i + offset), values, s=18, color=colour, alpha=0.45, lw=0)
                means.append(values.mean() if len(values) else np.nan)
            ax.plot(np.arange(len(x_order)) + offset, means, marker="o", ms=7, lw=1.5 if numeric else 0,
                    color=colour, label=None if level is None else str(level))
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
    # One row per sweep run found in the store (the newest bundle when a run was repeated).
    # Bundles are read from their folders, so the registry index does not need rebuilding.
    reload_button.value
    _store = Path(store_root.value).expanduser()

    bundles = {}
    expected_by_sweep = {}
    _training_rows = []
    _status_rows = []
    _missing = {}
    for _name, _sweep in sweeps.SWEEPS.items():
        _rows = []
        for _bundle in sweeps.find_bundles(_store, _sweep):
            bundles[_bundle.id] = _bundle
            _rows.append({"sweep": _name, **sweeps.training_row(_bundle, _sweep)})
        _kept, _superseded = sweeps.latest_per_run(_rows, _sweep)
        expected_by_sweep[_name] = sweeps.expected_runs(_sweep)
        _missing[_name] = sweeps.missing_runs(expected_by_sweep[_name], _kept, _sweep)
        _training_rows += _kept
        _status_rows.append({
            "sweep": _sweep.title,
            "experiment": _sweep.experiment,
            "expected runs": len(expected_by_sweep[_name]) or None,
            "found": len(_kept),
            "missing": len(_missing[_name]) if expected_by_sweep[_name] else None,
            "superseded (older reruns)": len(_superseded),
        })
    training = pd.DataFrame(_training_rows)

    _missing_tables = {
        f"{sweeps.SWEEPS[_name].title}: {len(_rows)} missing": mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=10)
        for _name, _rows in _missing.items() if _rows
    }
    mo.vstack([
        mo.md(f"**{len(training)}** sweep models in `{_store}`."),
        mo.ui.table(pd.DataFrame(_status_rows), selection=None),
        mo.accordion(_missing_tables) if _missing_tables else mo.md(""),
    ])
    return bundles, expected_by_sweep, training


@app.cell(hide_code=True)
def _():
    try:
        raw = load_snowmelt(DATA_ROOT)
        _out = mo.md(f"Data: `{DATA_ROOT}` ({len(raw['dates'])} acquisitions, {', '.join(raw['dates'])}).")
    except (FileNotFoundError, OSError, ValueError) as _error:
        raw = None
        _out = mo.callout(f"Could not load the snowmelt data from `{DATA_ROOT}` ({_error}). "
                          "Set SNOWMELT_DATA_ROOT to evaluate models; training metrics are still shown.", kind="warn")
    _out
    return (raw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_sweeps = mo.ui.multiselect(
        {_sweep.title: _name for _name, _sweep in sweeps.SWEEPS.items()},
        value=[_sweep.title for _sweep in sweeps.SWEEPS.values()],
        label="Sweeps",
    )
    eval_mode = mo.ui.radio(
        {"Free run from first date": "free", "One interval ahead": "interval"},
        value="Free run from first date", label="Rollout", inline=True,
    )
    eval_rollouts = mo.ui.number(1, 32, value=4, step=1, label="Rollouts per model")
    eval_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    eval_grid = mo.ui.radio({"Model grid": "model", "10 m grid": "full"}, value="10 m grid", label="Score on", inline=True)
    eval_verify = mo.ui.checkbox(value=True, label="Require input to match the bundle fingerprint")
    eval_run = mo.ui.run_button(label="Evaluate models")
    mo.vstack([
        eval_sweeps,
        mo.hstack([eval_mode, eval_rollouts, eval_seed, eval_grid, eval_verify, eval_run],
                  justify="start", gap=1.5, wrap=True),
    ])
    return eval_grid, eval_mode, eval_rollouts, eval_run, eval_seed, eval_sweeps, eval_verify


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
    eval_sweeps,
    eval_verify,
    raw,
    score_cache,
    training,
):
    _settings = (eval_mode.value, int(eval_rollouts.value), int(eval_seed.value), eval_grid.value, eval_verify.value)
    _score_rows = []
    _summary_rows = []
    eval_maps = {}
    _failures = []
    if not eval_run.value:
        _status = mo.md("Choose sweeps and settings, then click **Evaluate models**. Training metrics are shown below without it.")
    elif raw is None:
        _status = mo.callout("The snowmelt data is not loaded.", kind="danger")
    else:
        _todo = list(training.loc[training.sweep.isin(eval_sweeps.value), "model_id"]) if len(training) else []
        _sequences, _references = {}, {}
        for _model_id in mo.status.progress_bar(_todo, title="Evaluating models", remove_on_exit=True):
            _cache_key = (_model_id, *_settings)
            if _cache_key not in score_cache:
                # A fixed key per model, so results do not depend on which models are selected
                _key = jr.fold_in(jr.PRNGKey(_settings[2]), zlib.crc32(_model_id.encode()) & 0x7FFFFFFF)
                try:
                    score_cache[_cache_key] = sweeps.evaluate_bundle(
                        bundles[_model_id], raw, _key, n_rollouts=_settings[1], mode=_settings[0],
                        grid=_settings[3], verify=_settings[4], sequences=_sequences, references=_references,
                    )
                except Exception as _error:  # keep going: one bad bundle should not stop the comparison
                    _failures.append({"model_id": _model_id, "error": f"{type(_error).__name__}: {_error}"})
                    continue
            _rows, _maps = score_cache[_cache_key]
            _score_rows += [{"model_id": _model_id, **_row} for _row in _rows if _row["channel"] == _maps["channel"]]
            _summary_rows.append({"model_id": _model_id, **sweeps.summarise_scores(_rows, _maps["channel"])})
            eval_maps[_model_id] = _maps
        _status = mo.vstack([
            mo.md(f"Evaluated **{len(_summary_rows)}** model(s): "
                  f"{ {'free': 'free run', 'interval': 'one interval ahead'}[_settings[0]]}, {_settings[1]} rollout(s) each, "
                  f"scored on the { {'model': 'model grid', 'full': '10 m grid'}[_settings[3]]}"
                  + ("" if _settings[4] else " (**input fingerprints not checked**)") + "."),
            mo.callout(mo.vstack([mo.md(f"**{len(_failures)}** model(s) failed:"),
                                  mo.ui.table(pd.DataFrame(_failures), selection=None)]), kind="danger")
            if _failures else mo.md(""),
        ])

    # Training metrics for every model, with the inference summary where it has been computed
    results = training.merge(pd.DataFrame(_summary_rows), on="model_id", how="left") if _summary_rows else training
    # Scores by date on each model's score channel, with the model's factors
    scores = training.merge(pd.DataFrame(_score_rows), on="model_id") if _score_rows else pd.DataFrame()
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
def _(expected_by_sweep, metric, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _kernels, _channels = levels(_expected.kernels), levels(_expected.channels)
        _families = levels(_expected.family)
        _outputs = []
        for _y in dict.fromkeys([metric.value, "best_loss"]):
            _fig, _axes = plt.subplots(1, len(_families), figsize=(6.5 * len(_families), 4),
                                       constrained_layout=True, sharey=True, squeeze=False)
            for _ax, _family in zip(_axes[0], _families):
                factor_plot(_ax, _df[_df.family == _family], "kernels", _y, hue="channels",
                            x_order=_kernels, hue_order=_channels)
                _ax.set_title(_family, fontsize=10)
            _outputs.append(_fig)

        # Direction effect: mean(GRAD) - mean(DIFF), with the std of that difference from the repeats
        _pairs = {"without LAP": ("ID+DIFF", "ID+GRAD"), "with LAP": ("ID+LAP+DIFF", "ID+LAP+GRAD")}
        _rows = []
        for (_family, _width), _group in _df.groupby(["family", "channels"]):
            for _pair, (_isotropic, _directional) in _pairs.items():
                _a = _group.loc[_group.kernels == _isotropic, metric.value].dropna()
                _b = _group.loc[_group.kernels == _directional, metric.value].dropna()
                if len(_a) and len(_b):
                    _se = np.sqrt(_a.var(ddof=1) / len(_a) + _b.var(ddof=1) / len(_b)) if min(len(_a), len(_b)) > 1 else np.nan
                    _rows.append({"model": f"{_family}, {_width} ch", "pair": _pair,
                                  "GRAD − DIFF": _b.mean() - _a.mean(), "std error": _se})
        if _rows:
            _diff = pd.DataFrame(_rows)
            _names = list(dict.fromkeys(_diff.model))
            _fig, _ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
            for _k, _pair in enumerate(_pairs):
                _sub = _diff[_diff.pair == _pair].set_index("model").reindex(_names)
                _x = np.arange(len(_names)) + (_k - 0.5) * 0.35
                _ax.bar(_x, _sub["GRAD − DIFF"], width=0.33, color=COLOURS[_k], label=_pair)
                _ax.errorbar(_x, _sub["GRAD − DIFF"], yerr=_sub["std error"], fmt="none", ecolor="#333", capsize=3)
            _ax.axhline(0, color="#777", lw=1)
            _ax.set_xticks(range(len(_names)), _names, rotation=20, ha="right")
            _ax.set_ylabel(f"GRAD − DIFF\n{METRICS.get(metric.value, metric.value)}", fontsize=9)
            _ax.set_title("Effect of gradient direction (error bars: std error from repeats)", fontsize=10)
            style_axes(_ax)
            _ax.legend(frameon=False, fontsize=8)
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
