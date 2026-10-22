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
input channels, single added bands, model architecture, update rule and held-out dates. Model bundles are read from
the model store (``MODEL_STORE_ROOT``, default ``models/``) and the data from
``SNOWMELT_DATA_ROOT`` (default ``~/PhD/Data/snowmelt``, holding ``v1/``, ``v2/``);
each model is scored on the dataset version it was trained on. Training metrics come
from each bundle's manifest; inference scores come from rolling every model out
with ``Experiments/snowmelt/evaluation.py``. Maps and single-model views are in
``snowmelt_explorer.py``.
"""

import marimo

__generated_with = "0.25.1"
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
        "snow_accuracy": "snow map accuracy, mean over dates (higher is better)",
        "final_snow_accuracy": "snow map accuracy, final date (higher is better)",
        "skill": "MSE skill vs persistence, mean over dates (higher is better)",
        "final_skill": "MSE skill vs persistence, final date (higher is better)",
        "rmse": "RMSE, mean over dates (lower is better)",
        "ndsi_rmse": "NDSI RMSE, mean over dates (lower is better)",
        "final_ndsi_rmse": "NDSI RMSE, final date (lower is better)",
        "held_out_snow_accuracy": "snow map accuracy, held-out dates (higher is better)",
        "held_out_skill": "MSE skill vs persistence, held-out dates (higher is better)",
        "held_out_rmse": "RMSE, held-out dates (lower is better)",
        "snow_area_error": "snow-covered fraction, predicted − observed",
        "rollout_spread": "std across rollouts",
        "best_loss": "best training loss (lower is better)",
        "best_at_fraction": "best checkpoint iteration / iterations",
    }
    # Metrics centred on zero get a diverging colour scale
    DIVERGING = {"snow_area_error", "skill", "final_skill", "held_out_skill"}
    # Fractions shown on [0, 1] unless the plots are set to fit the data
    ACCURACIES = {"snow_accuracy", "final_snow_accuracy", "held_out_snow_accuracy"}

    def value_range(name, values, fit):
        """Axis or colour limits for ``name``: None (matplotlib default) except for accuracies,
        which get [0, 1], or with ``fit`` the range of ``values`` plus a small margin."""
        if name not in ACCURACIES:
            return None
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if not fit or not len(values):
            return 0.0, 1.0
        margin = 0.05 * (values.max() - values.min()) or 0.01
        return max(values.min() - margin, 0.0), min(values.max() + margin, 1.0)

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def available(df, names):
        """The columns of ``names`` (duplicates dropped) that ``df`` has values for."""
        return [name for name in dict.fromkeys(names) if name in df and df[name].notna().any()]

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def hue_width(n):
        """Space given to each of ``n`` hue levels within one x position (0.8 in all)."""
        return 0.8 / n

    def hue_offset(k, n):
        """Horizontal shift of hue level ``k`` of ``n`` within one x position."""
        return (k - (n - 1) / 2) * hue_width(n)

    def factor_plot(ax, df, x, y, hue=None, x_order=None, hue_order=None, fit=False):
        """The mean over repeats per level, side by side for each hue, with one dot per run.

        Numeric ``x``: means are markers joined by a line. Categorical ``x``: means are bars,
        with the std across repeats as error bars. ``fit``: see ``value_range``.
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
        limits = value_range(y, df[y], fit)
        if limits:
            ax.set_ylim(*limits)
        style_axes(ax)
        if hue is not None:
            ax.legend(title=hue, frameon=False, fontsize=8)

    def heatmap(ax, df, rows, cols, value, row_order, col_order, diverging=None, fit=False):
        """Mean of ``value`` per (row, column) cell, annotated with the std across repeats and the run count.

        ``diverging`` (default: ``value in DIVERGING``) puts zero at the centre of the colour scale.
        ``fit``: see ``value_range``.
        """
        def table(aggfunc):
            return df.pivot_table(index=rows, columns=cols, values=value, aggfunc=aggfunc).reindex(
                index=row_order, columns=col_order
            ).to_numpy(float)

        mean, std, count = table("mean"), table("std"), table("count")
        centred = (value in DIVERGING) if diverging is None else diverging
        if centred:
            limit = np.nanmax(np.abs(mean)) if np.isfinite(mean).any() else 1.0
            image = ax.imshow(mean, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        else:
            vmin, vmax = value_range(value, mean, fit) or (None, None)
            image = ax.imshow(mean, cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                if np.isfinite(mean[i, j]):
                    spread = f" ± {std[i, j]:.3g}" if np.isfinite(std[i, j]) else ""
                    # White text only on dark cells: both ends of RdBu, the low end of cividis
                    shade = image.norm(mean[i, j])
                    dark = abs(shade - 0.5) > 0.35 if centred else shade < 0.5
                    ax.text(j, i, f"{mean[i, j]:.3g}{spread}\n(n={int(count[i, j])})", ha="center", va="center",
                            fontsize=7, color="white" if dark else "black")
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

    # Plain names of the terrain (static) channels; northness and eastness together are the aspect
    TERRAIN_NAMES = {"DEM": "elevation", "SLOPE": "slope", "NORTHNESS": "aspect", "EASTNESS": "aspect"}

    def static_label(static):
        """Short name of a static channel set, e.g. ``DEM+SLOPE`` -> ``elevation + slope``."""
        if static == "none":
            return "no terrain"
        names = [TERRAIN_NAMES.get(name, name.lower()) for name in str(static).split("+")]
        return " + ".join(dict.fromkeys(names))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Snowmelt NCA sweeps: model comparison

    Compares the models trained by each snowmelt experiment sweep
    (`Experiments/snowmelt/conf/experiments/`):

    | Sweep | Question | Factors |
    |---|---|---|
    | Resolution | How do spatial and temporal resolution affect the model? | downsample and t, together (convective scaling) |
    | Input channels | Which target channels and terrain (static) channels help? | target set × static set × kernels |
    | Single added band | Which one band, added to NDSI, helps most? | added band × static set |
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
    **SCA**. *Snow map accuracy* (the fraction of catchment pixels correctly classified as
    snow or no snow, NDSI > 0.4 or SCA > 0.5) is the one score that is comparable across every run,
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

    - Pick one W&B group (one sweep) to evaluate and plot; newest group first, model count in brackets
    - Within the group, only the newest bundle of each run is kept
    - Untick rows in the table to drop single models; evaluation and every plot below use the ticked models
    """)
    return


@app.cell(hide_code=True)
def _(all_runs):
    # One option per (sweep, W&B group), newest group first
    _newest, _counts = {}, {}
    for _row in sorted(all_runs, key=lambda _r: _r["created_at"]):
        _key = (_row["sweep"], _row["wandb_group"])
        _newest[_key] = _row["created_at"]
        _counts[_key] = _counts.get(_key, 0) + 1
    _options = {f"{sweeps.SWEEPS[_sweep].title} · {_group} ({_counts[(_sweep, _group)]})": (_sweep, _group)
                for _sweep, _group in sorted(_newest, key=_newest.get, reverse=True)}
    group_select = mo.ui.dropdown(_options, value=next(iter(_options), None), label="W&B group", full_width=True)
    group_select if _options else mo.md("No sweep bundles found.")
    return (group_select,)


@app.cell(hide_code=True)
def _(all_runs, expected_by_sweep, group_select):
    if group_select.value is None:
        candidates = pd.DataFrame({"model_id": []})
        _status, _missing = [], []
    else:
        _name, _group = group_select.value
        _sweep = sweeps.SWEEPS[_name]
        _rows = [_row for _row in all_runs if _row["sweep"] == _name and _row["wandb_group"] == _group]
        _kept, _superseded = sweeps.latest_per_run(_rows, _sweep)
        _expected = expected_by_sweep[_name]
        _missing = sweeps.missing_runs(_expected, _kept, _sweep)
        _status = [{
            "sweep": _sweep.title,
            "experiment": _sweep.experiment,
            "group": _group,
            "expected runs": len(_expected) or None,
            "found": len(_kept),
            "missing": len(_missing) if _expected else None,
            "superseded (older reruns)": len(_superseded),
        }]
        # Models in the chosen group: the rows the table below can pick from
        candidates = pd.DataFrame(_kept) if _kept else pd.DataFrame({"model_id": []})

    model_table = mo.ui.table(
        candidates.dropna(axis=1, how="all").round(4),
        initial_selection=list(range(len(candidates))),
        page_size=10,
        label="Models",
    )
    mo.vstack([
        mo.ui.table(pd.DataFrame(_status), selection=None) if _status else mo.md(""),
        mo.accordion({f"{len(_missing)} missing": mo.ui.table(pd.DataFrame(_missing), selection=None, page_size=10)})
        if _missing else mo.md(""),
        model_table,
    ])
    return candidates, model_table


@app.cell(hide_code=True)
def _(candidates, model_table):
    # The models used for evaluation and for every plot below
    _ticked = set(model_table.value["model_id"]) if len(model_table.value) else set()
    selected = candidates[candidates.model_id.isin(_ticked)]
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
    return (
        eval_grid,
        eval_mode,
        eval_rollouts,
        eval_run,
        eval_seed,
        eval_verify,
    )


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
    if _summary_rows:
        # Error on the NDSI prediction alone; empty for models that do not predict NDSI (SCA only)
        _is_ndsi = results.score_channel == "NDSI"
        results = results.assign(ndsi_rmse=results.rmse.where(_is_ndsi), final_ndsi_rmse=results.final_rmse.where(_is_ndsi))
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
        value=METRICS["snow_accuracy" if "snow_accuracy" in _available else "best_loss"] if _available else None,
        label="Metric for the plots below",
    )
    metric if _available else mo.md("")
    return (metric,)


@app.cell(hide_code=True)
def _():
    fit_accuracy = mo.ui.checkbox(value=False, label="Fit accuracy axes to the data (default 0–1)")
    fit_accuracy
    return (fit_accuracy,)


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
def _(fit_accuracy, metric, results):
    _df = results[results.sweep == "resolution"].copy() if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No resolution-sweep models found.")
    else:
        _df["resolution"] = [f"{_r} m, t={_t}" for _r, _t in zip(_df.resolution_m, _df.t)]
        _order = [f"{_r} m, t={_t}" for _r, _t in sorted(set(zip(_df.resolution_m, _df.t)))]
        _panels = available(_df, [metric.value, "best_loss", "ndsi_rmse", "best_at_fraction"])
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(5 * len(_panels), 3.8), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            factor_plot(_ax, _df, "resolution", _y, x_order=_order, fit=fit_accuracy.value)
        _out = _fig
    _out
    return


@app.cell
def _(metric, results):
    # Barebones copy of the figure above: mean ± std over repeats, one dot per run
    _df = results[results.sweep == "resolution"] if len(results) else results
    mo.stop(_df.empty)
    _df = _df.assign(resolution=[f"{_r} m, t={_t}" for _r, _t in zip(_df.resolution_m, _df.t)])
    _xs = [f"{_r} m, t={_t}" for _r, _t in sorted(set(zip(_df.resolution_m, _df.t)))]
    _ys = available(_df, [metric.value, "best_loss", "ndsi_rmse", "best_at_fraction"])

    _fig, _axes = plt.subplots(1, len(_ys), figsize=(5 * len(_ys), 3.8), squeeze=False, constrained_layout=True)
    for _ax, _y in zip(_axes[0], _ys):
        _stats = _df.groupby("resolution")[_y].agg(["mean", "std"]).reindex(_xs)
        _ax.bar(range(len(_xs)), _stats["mean"], yerr=_stats["std"], capsize=3)
        _ax.scatter(_df.resolution.map(_xs.index), _df[_y], s=10, color="k", alpha=0.5)
        _ax.set_xticks(range(len(_xs)), _xs, rotation=20, ha="right")
        _ax.set_ylabel(_y)
    _fig
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


@app.cell
def _(scores):
    # Barebones copy of the figure above: observed snow area, and mean ± std over repeats of the prediction
    _df = scores[scores.sweep == "resolution"] if len(scores) else scores
    mo.stop(_df.empty or "observed_snow_fraction" not in _df)
    _df = _df.assign(resolution=[f"{_r} m, t={_t}" for _r, _t in zip(_df.resolution_m, _df.t)])
    _hues = [f"{_r} m, t={_t}" for _r, _t in sorted(set(zip(_df.resolution_m, _df.t)))]

    _fig, _ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    _observed = 100 * _df.groupby("days").observed_snow_fraction.mean()
    _ax.plot(_observed.index, _observed, "ko-", lw=2.5, label="observed")
    for _hue in _hues:
        _per_model = 100 * _df[_df.resolution == _hue].pivot_table(
            index="days", columns="model_id", values="predicted_snow_fraction")
        _mean, _std = _per_model.mean(axis=1), _per_model.std(axis=1).fillna(0)
        _line, = _ax.plot(_mean.index, _mean, "o-", ms=4, label=_hue)
        _ax.fill_between(_mean.index, _mean - _std, _mean + _std, color=_line.get_color(), alpha=0.15)
    _ax.set_xlabel("days since first acquisition")
    _ax.set_ylabel("snow-covered area (%)")
    _ax.set_ylim(0, 105)
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Input channels and terrain

    Hidden channels are fixed at 28, so the target and static sets change only what the
    model sees and predicts. Compare target sets with *snow map accuracy* (the only score shared
    with the SCA-only models); NDSI skill compares the NDSI-based sets. The loss map is
    only comparable along a row (same targets).
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, fit_accuracy, metric, results):
    _df = results[results.sweep == "input"] if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No input-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["input"]) if expected_by_sweep["input"] else _df
        _targets, _static = levels(_expected.targets), levels(_expected.static)
        _panels = available(_df, [metric.value, "best_loss", "ndsi_rmse"])
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(6.5 * len(_panels), 4.5), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            _image = heatmap(_ax, _df, "targets", "static", _y, _targets, _static, fit=fit_accuracy.value)
            _fig.colorbar(_image, ax=_ax, shrink=0.8, label=METRICS.get(_y, _y))
            _ax.set_title(f"{_y}: mean ± std over repeats", fontsize=10)
        _fig2, _ax2 = plt.subplots(figsize=(9, 4), constrained_layout=True)
        factor_plot(_ax2, _df, "targets", metric.value, hue="static", x_order=_targets, hue_order=_static, fit=fit_accuracy.value)
        _out = mo.vstack([_fig, _fig2])
    _out
    return


@app.cell
def _(metric, results):
    # Barebones copy of the figures above: heatmaps of mean ± std over repeats, then bars by target set
    _df = results[results.sweep == "input"] if len(results) else results
    mo.stop(_df.empty or metric.value is None)
    _targets, _statics = levels(_df.targets), levels(_df.static)
    _ys = available(_df, [metric.value, "best_loss", "ndsi_rmse"])

    _fig, _axes = plt.subplots(1, len(_ys), figsize=(6.5 * len(_ys), 4.5), squeeze=False, constrained_layout=True)
    for _ax, _y in zip(_axes[0], _ys):
        _mean = _df.pivot_table(index="targets", columns="static", values=_y, aggfunc="mean").reindex(
            index=_targets, columns=_statics).to_numpy(float)
        _std = _df.pivot_table(index="targets", columns="static", values=_y, aggfunc="std").reindex(
            index=_targets, columns=_statics).to_numpy(float)
        _image = _ax.imshow(_mean, cmap="cividis", aspect="auto")
        for _i, _j in np.ndindex(_mean.shape):
            _ax.text(_j, _i, f"{_mean[_i, _j]:.3g}\n± {_std[_i, _j]:.2g}", ha="center", va="center", fontsize=7,
                     color="w" if _image.norm(_mean[_i, _j]) < 0.5 else "k")
        _ax.set_xticks(range(len(_statics)), _statics, rotation=20, ha="right")
        _ax.set_yticks(range(len(_targets)), _targets)
        _ax.set_title(_y)
        _fig.colorbar(_image, ax=_ax)

    _y = metric.value
    _fig2, _ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    _width = 0.8 / len(_statics)
    for _k, _static in enumerate(_statics):
        _sub = _df[_df.static == _static]
        _stats = _sub.groupby("targets")[_y].agg(["mean", "std"]).reindex(_targets)
        _x = np.arange(len(_targets)) + (_k - (len(_statics) - 1) / 2) * _width
        _ax.bar(_x, _stats["mean"], width=_width, yerr=_stats["std"], capsize=2, label=_static)
        _ax.scatter(_x[_sub.targets.map(_targets.index).to_numpy(int)], _sub[_y], s=8, color="k", alpha=0.5)
    _ax.set_xticks(range(len(_targets)), _targets, rotation=20, ha="right")
    _ax.set_ylabel(_y)
    _ax.legend(title="static")
    mo.vstack([_fig, _fig2])
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


@app.cell
def _(scores):
    # Barebones copy of the figure above: snow area by static set (left) and by target set (right)
    _df = scores[scores.sweep == "input"] if len(scores) else scores
    mo.stop(_df.empty or "observed_snow_fraction" not in _df)
    _observed = 100 * _df.groupby("days").observed_snow_fraction.mean()

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    for _ax, _factor in zip(_axes, ["static", "targets"]):
        _ax.plot(_observed.index, _observed, "ko-", lw=2.5, label="observed")
        for _hue in levels(_df[_factor]):
            _per_model = 100 * _df[_df[_factor] == _hue].pivot_table(
                index="days", columns="model_id", values="predicted_snow_fraction")
            _mean, _std = _per_model.mean(axis=1), _per_model.std(axis=1).fillna(0)
            _line, = _ax.plot(_mean.index, _mean, "o-", ms=4, label=_hue)
            _ax.fill_between(_mean.index, _mean - _std, _mean + _std, color=_line.get_color(), alpha=0.15)
        _ax.set_xlabel("days since first acquisition")
        _ax.set_title(f"by {_factor}")
        _ax.legend()
    _axes[0].set_ylabel("snow-covered area (%)")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Single added band

    - Each run predicts NDSI plus one more band (or NDSI only, `none`), at 32 channels and ID+LAP+GRAD
    - Top: NDSI RMSE by added band, colour = terrain (static) channels
    - Bottom: change from the NDSI-only model with the same static set (mean over repeats)
    - The training loss is left out: NDSI-only models have one target fewer; NDSI RMSE needs the models evaluated
    """)
    return


@app.cell(hide_code=True)
def _(expected_by_sweep, results):
    _df = results[results.sweep == "input_single_band"].copy() if len(results) else results
    if _df.empty or "ndsi_rmse" not in _df or _df.ndsi_rmse.isna().all():
        _out = mo.md("No evaluated single-band-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["input_single_band"]) if expected_by_sweep["input_single_band"] else _df

        def _band(targets):
            return "none" if targets == "NDSI" else targets.removeprefix("NDSI+")

        _df["band"] = _df.targets.map(_band)
        _df["static"] = _df.static.map(static_label)
        _bands = [_band(_t) for _t in levels(_expected.targets)]
        _statics = [static_label(_s) for _s in levels(_expected.static)]

        _fig, _ax = plt.subplots(figsize=(14, 4.2), constrained_layout=True)
        factor_plot(_ax, _df, "band", "ndsi_rmse", hue="static", x_order=_bands, hue_order=_statics)
        _ax.set_xlabel("band added to NDSI")
        _ax.set_title("NDSI RMSE by added band", fontsize=10)
        _ax.legend(title="terrain channels", frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

        # Each run minus the mean NDSI-only RMSE with the same static set
        _fig2, _ax2 = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        _baseline = _df[_df.band == "none"].groupby("static").ndsi_rmse.mean()
        _change = _df.assign(change=_df.ndsi_rmse - _df.static.map(_baseline))
        _image = heatmap(_ax2, _change[_change.band != "none"], "band", "static", "change",
                         [_b for _b in _bands if _b != "none"], _statics, diverging=True)
        _ax2.set_xlabel("terrain channels")
        _ax2.set_ylabel("band added to NDSI")
        _fig2.colorbar(_image, ax=_ax2, shrink=0.8, label="change in NDSI RMSE from NDSI only")
        _ax2.set_title("Change in NDSI RMSE from adding the band", fontsize=10)
        _out = mo.vstack([_fig, _fig2])
    _out
    return


@app.cell
def _(metric, results):
    # Barebones copy of the figures above: score by added band (colour = static set), then change from NDSI only
    _df = results[results.sweep == "input_single_band"] if len(results) else results
    mo.stop(_df.empty or metric.value is None)
    _df = _df.assign(band=["none" if _t == "NDSI" else _t.removeprefix("NDSI+") for _t in _df.targets])
    _bands, _statics = levels(_df.band), levels(_df.static)
    _ys = available(_df, [metric.value, "ndsi_rmse"])
    # print(_ys)
    _fig, _axes = plt.subplots(len(_ys[1:]), 1, figsize=(14, 4.2 * len(_ys[1:])), squeeze=False, constrained_layout=True)
    _width = 0.8 / len(_statics)
    for _ax, _y in zip(_axes[:, 0], _ys[1:]):
        for _k, _static in enumerate(_statics):
            _sub = _df[_df.static == _static]
            _stats = _sub.groupby("band")[_y].agg(["mean", "std"]).reindex(_bands)
            _x = np.arange(len(_bands)) + (_k - (len(_statics) - 1) / 2) * _width
            _ax.bar(_x, _stats["mean"], width=_width, yerr=_stats["std"], capsize=2, label=_static)
            _ax.scatter(_x[_sub.band.map(_bands.index).to_numpy(int)], _sub[_y], s=8, color="k", alpha=0.5)
        _ax.set_xticks(range(len(_bands)), _bands, rotation=20, ha="right")
        _ax.set_xlabel("band added to NDSI")
        _ax.set_ylabel("NDSI RMSE")
    
        _ax.legend(title="static")

    # Mean per (band, static set) minus the mean NDSI-only score with the same static set
    _added = [_b for _b in _bands if _b != "none"]
    _fig2, _axes2 = plt.subplots(1, len(_ys), figsize=(7 * len(_ys), 5.5), squeeze=False, constrained_layout=True)
    for _ax, _y in zip(_axes2[0], _ys):
        _mean = _df.pivot_table(index="band", columns="static", values=_y, aggfunc="mean").reindex(
            index=["none", *_added], columns=_statics)
        _change = (_mean.loc[_added] - _mean.loc["none"]).to_numpy(float)
        _limit = np.nanmax(np.abs(_change))
        _image = _ax.imshow(_change, cmap="RdBu", vmin=-_limit, vmax=_limit, aspect="auto")
        for _i, _j in np.ndindex(_change.shape):
            _ax.text(_j, _i, f"{_change[_i, _j]:.3g}", ha="center", va="center", fontsize=7)
        _ax.set_xticks(range(len(_statics)), _statics, rotation=20, ha="right")
        _ax.set_yticks(range(len(_added)), _added)
        _ax.set_title(f"change in {_y} from adding the band")
        _fig2.colorbar(_image, ax=_ax, label=f"{_y} − NDSI only")
    mo.vstack([_fig, _fig2])
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
def _(
    arch_family,
    arch_width,
    expected_by_sweep,
    fit_accuracy,
    metric,
    results,
):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or metric.value is None or arch_width.value is None or arch_family.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _statics = levels(_expected.static)
        _df = _df[_df.family == arch_family.value]
        _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, sharey=True)
        factor_plot(_left, _df, "channels", metric.value, hue="static",
                    x_order=levels(_expected.channels), hue_order=_statics, fit=fit_accuracy.value)
        _left.set_title(f"{arch_family.value}: more channels", fontsize=10)
        factor_plot(_right, _df[_df.channels == arch_width.value], "kernels", metric.value,
                    hue="static", x_order=levels(_expected.kernels), hue_order=_statics, fit=fit_accuracy.value)
        _right.set_title(f"{arch_family.value}: kernels at {arch_width.value} channels", fontsize=10)
        _out = _fig
    _out
    return


@app.cell
def _(arch_family, arch_width, metric, results):
    # Barebones copy of the figure above: score against channels (left), and by kernels at one width (right)
    _df = results[results.sweep == "architecture"] if len(results) else results
    _df = _df[_df.family == arch_family.value] if len(_df) else _df
    mo.stop(_df.empty or metric.value is None)
    _y = metric.value
    _statics = levels(_df.static)

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    for _static in _statics:
        _sub = _df[_df.static == _static]
        _stats = _sub.groupby("channels")[_y].agg(["mean", "std"])
        _line = _left.errorbar(_stats.index, _stats["mean"], yerr=_stats["std"], marker="o", capsize=3, label=_static)
        _left.scatter(_sub.channels, _sub[_y], s=8, alpha=0.4, color=_line[0].get_color())
    _left.set_xlabel("channels")
    _left.set_ylabel(_y)
    _left.set_title(f"{arch_family.value}: more channels")
    _left.legend(title="static")

    _at_width = _df[_df.channels == arch_width.value]
    _kernels = levels(_at_width.kernels)
    _width = 0.8 / len(_statics)
    for _k, _static in enumerate(_statics):
        _sub = _at_width[_at_width.static == _static]
        _stats = _sub.groupby("kernels")[_y].agg(["mean", "std"]).reindex(_kernels)
        _x = np.arange(len(_kernels)) + (_k - (len(_statics) - 1) / 2) * _width
        _right.bar(_x, _stats["mean"], width=_width, yerr=_stats["std"], capsize=2, label=_static)
        _right.scatter(_x[_sub.kernels.map(_kernels.index).to_numpy(int)], _sub[_y], s=8, color="k", alpha=0.5)
    _right.set_xticks(range(len(_kernels)), _kernels, rotation=20, ha="right")
    _right.set_title(f"{arch_family.value}: kernels at {arch_width.value} channels")
    _fig
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


@app.cell
def _(
    arch_family,
    eval_maps,
    pred_conditions,
    pred_date,
    pred_repeat,
    pred_view,
    pred_width,
    results,
):
    # Barebones copy of the figure above: training loss by kernels (left), observed and predicted maps (right)
    _df = results[results.sweep == "architecture"] if len(results) else results
    _df = _df[(_df.family == arch_family.value) & (_df.channels == pred_width.value)] if len(_df) else _df
    mo.stop(_df.empty)
    _kernels, _statics = levels(_df.kernels), levels(_df.static)

    _fig = plt.figure(figsize=(15, 5.5), layout="constrained")
    _left_fig, _right_fig = _fig.subfigures(1, 2, width_ratios=[1, 1.5])
    _ax = _left_fig.subplots()
    _width = 0.8 / len(_statics)
    for _k, _static in enumerate(_statics):
        _sub = _df[_df.static == _static]
        _stats = _sub.groupby("kernels")["best_loss"].agg(["mean", "std"]).reindex(_kernels)
        _x = np.arange(len(_kernels)) + (_k - (len(_statics) - 1) / 2) * _width
        _ax.bar(_x, _stats["mean"], width=_width, yerr=_stats["std"], capsize=2, label=_static)
        _ax.scatter(_x[_sub.kernels.map(_kernels.index).to_numpy(int)], _sub.best_loss, s=8, color="k", alpha=0.5)
    _ax.set_xticks(range(len(_kernels)), _kernels, rotation=20, ha="right")
    _ax.set_ylabel("best_loss")
    _ax.set_title(f"{arch_family.value}, {pred_width.value} channels: training loss")
    _ax.legend(title="static")

    # One evaluated model per chosen condition: the chosen repeat, newest bundle
    _fields = []
    for _kernel, _static in pred_conditions.value:
        _ids = _df[(_df.kernels == _kernel) & (_df.static == _static) & (_df.repeat == pred_repeat.value)
                   & _df.model_id.isin(eval_maps)].sort_values("created_at").model_id
        if len(_ids):
            _maps = eval_maps[_ids.iloc[-1]]
            _t = min(int(pred_date.value), len(_maps["dates"]) - 1)
            _value = _maps["mean"][_t] - (_maps["observed"][_t] if pred_view.value == "error" else 0)
            _fields.append((f"{_kernel}\n{_static}", np.where(_maps["catchment"], _value, np.nan)))
    if _fields:
        _observed = np.where(_maps["catchment"], _maps["observed"][_t], np.nan)
        _snow = np.nanpercentile(_observed, (2, 98))
        _limit = np.nanpercentile(np.abs(np.concatenate([_f.ravel() for _, _f in _fields])), 98)
        _axes = _right_fig.subplots(2, -(-(len(_fields) + 1) // 2), squeeze=False).ravel()
        for _a in _axes:
            _a.axis("off")
        _im = _axes[0].imshow(_observed, cmap="snow", vmin=_snow[0], vmax=_snow[1], interpolation="nearest")
        _axes[0].set_title("observed", fontsize=7)
        _right_fig.colorbar(_im, ax=list(_axes), shrink=0.6, label=_maps["channel"])
        for _a, (_title, _field) in zip(_axes[1:], _fields):
            if pred_view.value == "error":
                _im = _a.imshow(_field, cmap="RdBu_r", vmin=-_limit, vmax=_limit, interpolation="nearest")
            else:
                _im = _a.imshow(_field, cmap="snow", vmin=_snow[0], vmax=_snow[1], interpolation="nearest")
            _a.set_title(_title, fontsize=7)
        if pred_view.value == "error":
            _right_fig.colorbar(_im, ax=list(_axes), shrink=0.6, label="predicted − observed")
        _right_fig.suptitle(f"{_maps['channel']} on {_maps['dates'][_t]}")
    _fig
    return


@app.cell(hide_code=True)
def _(arch_family, expected_by_sweep, fit_accuracy, metric, results):
    _df = results[results.sweep == "architecture"] if len(results) else results
    if _df.empty or metric.value is None or arch_family.value is None:
        _out = mo.md("No architecture-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["architecture"]) if expected_by_sweep["architecture"] else _df
        _kernels, _channels = levels(_expected.kernels), levels(_expected.channels)
        _statics = levels(_expected.static)
        _df = _df[_df.family == arch_family.value]
        _outputs = []
        for _y in available(_df, [metric.value, "best_loss", "ndsi_rmse"]):
            _fig, _axes = plt.subplots(1, len(_statics), figsize=(6.5 * len(_statics), 4),
                                       constrained_layout=True, sharey=True, squeeze=False)
            for _ax, _static in zip(_axes[0], _statics):
                factor_plot(_ax, _df[_df.static == _static], "kernels", _y,
                            hue="channels", x_order=_kernels, hue_order=_channels, fit=fit_accuracy.value)
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


@app.cell
def _(arch_family, metric, results):
    # Barebones copy of the figures above: score by kernels (colour = channels), one row per score and
    # one column per static set; then GRAD − DIFF
    _df = results[results.sweep == "architecture"] if len(results) else results
    _df = _df[_df.family == arch_family.value] if len(_df) else _df
    mo.stop(_df.empty or metric.value is None)
    _kernels, _channels, _statics = levels(_df.kernels), levels(_df.channels), levels(_df.static)
    _ys = available(_df, [metric.value, "best_loss", "ndsi_rmse"])

    _fig, _axes = plt.subplots(len(_ys), len(_statics), figsize=(6.5 * len(_statics), 4 * len(_ys)),
                               sharey="row", squeeze=False, constrained_layout=True)
    _width = 0.8 / len(_channels)
    for _row, _y in zip(_axes, _ys):
        for _ax, _static in zip(_row, _statics):
            for _k, _c in enumerate(_channels):
                _sub = _df[(_df.static == _static) & (_df.channels == _c)]
                _stats = _sub.groupby("kernels")[_y].agg(["mean", "std"]).reindex(_kernels)
                _x = np.arange(len(_kernels)) + (_k - (len(_channels) - 1) / 2) * _width
                _ax.bar(_x, _stats["mean"], width=_width, yerr=_stats["std"], capsize=2, label=_c)
                _ax.scatter(_x[_sub.kernels.map(_kernels.index).to_numpy(int)], _sub[_y], s=8, color="k", alpha=0.5)
            _ax.set_xticks(range(len(_kernels)), _kernels, rotation=20, ha="right")
            _ax.set_title(f"{arch_family.value}, static {_static}")
        _row[0].set_ylabel(_y)
    _axes[0, 0].legend(title="channels")

    # mean(GRAD) − mean(DIFF) for each width and static set; error bar = std error from the repeats
    _y = metric.value
    _stats = _df.groupby(["kernels", "channels", "static"])[_y].agg(["mean", "var", "count"])
    _pairs = {"without LAP": ("ID+DIFF", "ID+GRAD"), "with LAP": ("ID+LAP+DIFF", "ID+LAP+GRAD")}
    _fig2, _axes2 = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True, constrained_layout=True)
    _width = 0.8 / len(_statics)
    for _ax, (_pair, (_isotropic, _directional)) in zip(_axes2, _pairs.items()):
        if not {_isotropic, _directional} <= set(_kernels):
            continue
        _a, _b = _stats.loc[_isotropic], _stats.loc[_directional]
        _diff = (_b["mean"] - _a["mean"]).unstack("static").reindex(index=_channels, columns=_statics)
        _se = np.sqrt(_a["var"] / _a["count"] + _b["var"] / _b["count"]).unstack("static").reindex(
            index=_channels, columns=_statics)
        for _k, _static in enumerate(_statics):
            _x = np.arange(len(_channels)) + (_k - (len(_statics) - 1) / 2) * _width
            _ax.bar(_x, _diff[_static], width=_width, yerr=_se[_static], capsize=3, label=_static)
        _ax.axhline(0, color="grey", lw=1)
        _ax.set_xticks(range(len(_channels)), [f"{_c} ch" for _c in _channels])
        _ax.set_title(f"{arch_family.value}: GRAD − DIFF, {_pair}")
    _axes2[0].set_ylabel(f"GRAD − DIFF\n{_y}")
    _axes2[0].legend(title="static")
    mo.vstack([_fig, _fig2])
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
def _(expected_by_sweep, fit_accuracy, metric, results):
    _df = results[results.sweep == "update_rule"] if len(results) else results
    if _df.empty or metric.value is None:
        _out = mo.md("No update-rule-sweep models found.")
    else:
        _expected = pd.DataFrame(expected_by_sweep["update_rule"]) if expected_by_sweep["update_rule"] else _df
        _activations = levels(_expected.activation)
        _panels = available(_df, [metric.value, "best_loss", "ndsi_rmse", "rollout_spread"])
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(5 * len(_panels), 3.8), constrained_layout=True)
        for _ax, _y in zip(np.atleast_1d(_axes), _panels):
            factor_plot(_ax, _df, "fire_rate", _y, hue="activation", hue_order=_activations, fit=fit_accuracy.value)
        _out = _fig
    _out
    return


@app.cell
def _(metric, results):
    # Barebones copy of the figure above: scores against fire rate, colour = activation
    _df = results[results.sweep == "update_rule"] if len(results) else results
    mo.stop(_df.empty or metric.value is None)
    _ys = available(_df, [metric.value, "best_loss", "ndsi_rmse", "rollout_spread"])

    _fig, _axes = plt.subplots(1, len(_ys), figsize=(5 * len(_ys), 3.8), squeeze=False, constrained_layout=True)
    for _ax, _y in zip(_axes[0], _ys):
        for _activation in levels(_df.activation):
            _sub = _df[_df.activation == _activation]
            _stats = _sub.groupby("fire_rate")[_y].agg(["mean", "std"])
            _line = _ax.errorbar(_stats.index, _stats["mean"], yerr=_stats["std"], marker="o", capsize=3,
                                 label=_activation)
            _ax.scatter(_sub.fire_rate, _sub[_y], s=8, alpha=0.4, color=_line[0].get_color())
        _ax.set_xlabel("fire rate")
        _ax.set_ylabel(_y)
        _ax.legend(title="activation")
    _fig
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


@app.cell
def _(scores):
    # Barebones copy of the figure above: observed snow area, and mean ± std over repeats per fire rate
    _df = scores[scores.sweep == "update_rule"] if len(scores) else scores
    mo.stop(_df.empty or "observed_snow_fraction" not in _df)

    _fig, _ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    _observed = 100 * _df.groupby("days").observed_snow_fraction.mean()
    _ax.plot(_observed.index, _observed, "ko-", lw=2.5, label="observed")
    for _hue in levels(_df.fire_rate):
        _per_model = 100 * _df[_df.fire_rate == _hue].pivot_table(
            index="days", columns="model_id", values="predicted_snow_fraction")
        _mean, _std = _per_model.mean(axis=1), _per_model.std(axis=1).fillna(0)
        _line, = _ax.plot(_mean.index, _mean, "o-", ms=4, label=_hue)
        _ax.fill_between(_mean.index, _mean - _std, _mean + _std, color=_line.get_color(), alpha=0.15)
    _ax.set_xlabel("days since first acquisition")
    _ax.set_ylabel("snow-covered area (%)")
    _ax.set_ylim(0, 105)
    _ax.legend(title="fire rate")
    _fig
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
def _(fit_accuracy, metric, scores):
    _df = scores[scores.sweep == "hold_out"] if len(scores) else scores
    _y = metric.value if metric.value in ("snow_accuracy", "skill", "rmse") else "snow_accuracy"
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
        _limits = value_range(_y, _df[_y], fit_accuracy.value)
        if _limits:
            _left.set_ylim(*_limits)
            _right.set_ylim(*_limits)
        _right.legend(frameon=False, fontsize=8)
        style_axes(_right)
        _out = _fig
    _out
    return


@app.cell
def _(metric, scores):
    # Barebones copy of the figure above: score by date per condition (circle = held-out date), then
    # held out vs control on each held-out date
    _df = scores[scores.sweep == "hold_out"] if len(scores) else scores
    _y = metric.value if metric.value in ("snow_accuracy", "skill", "rmse") else "snow_accuracy"
    mo.stop(_df.empty or _y not in _df)

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for _condition in levels(_df.held_out_dates):
        _sub = _df[_df.held_out_dates == _condition]
        _mean = _sub.groupby("days")[_y].mean()
        _line, = _left.plot(_mean.index, _mean, "o-", ms=3, color="k" if _condition == "none" else None,
                            label=_condition)
        _held = _sub[_sub.held_out].groupby("days")[_y].mean()
        _left.scatter(_held.index, _held, s=110, facecolor="none", edgecolor=_line.get_color(), lw=2)
    _left.set_xlabel("days since first acquisition")
    _left.set_ylabel(_y)
    _left.legend(title="held out")

    _dates = sorted(set(zip(_df[_df.held_out].days, _df[_df.held_out].date)))
    _days = [_d for _d, _ in _dates]
    for _offset, _sub, _label in ((-0.15, _df[_df.held_out], "held out"),
                                  (0.15, _df[_df.held_out_dates == "none"], "control (trained on it)")):
        _sub = _sub[_sub.days.isin(_days)]
        _stats = _sub.groupby("days")[_y].agg(["mean", "std"]).reindex(_days)
        _x = np.arange(len(_days)) + _offset
        _right.bar(_x, _stats["mean"], width=0.27, yerr=_stats["std"], capsize=2, label=_label)
        _right.scatter(_x[_sub.days.map(_days.index).to_numpy(int)], _sub[_y], s=8, color="k", alpha=0.5)
    _right.set_xticks(range(len(_dates)), [_date for _, _date in _dates], rotation=20, ha="right")
    _right.set_ylabel(_y)
    _right.legend()
    _fig
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


@app.cell
def _(eval_maps, map_date, map_sweep, map_view, results):
    # Barebones copy of the figure above: one map per condition, mean over its repeats
    mo.stop(map_sweep.value is None)
    _sweep = sweeps.SWEEPS[map_sweep.value]
    _df = results[(results.sweep == map_sweep.value) & results.model_id.isin(eval_maps)]
    _by_condition = {}
    for _row in _df.to_dict("records"):
        _by_condition.setdefault(condition_label(_row, _sweep), []).append(eval_maps[_row["model_id"]])
    _first = next(iter(_by_condition.values()))[0]
    _t = min(int(map_date.value), len(_first["dates"]) - 1)

    _panels = [("observed", np.where(_first["catchment"], _first["observed"][_t], np.nan))] if map_view.value == "mean" else []
    for _label, _maps in _by_condition.items():
        if map_view.value == "error":
            _fields = [_m["mean"][_t] - _m["observed"][_t] for _m in _maps]
        else:
            _fields = [_m[map_view.value][_t] for _m in _maps]
        _panels.append((_label, np.where(_maps[0]["catchment"], np.mean(_fields, axis=0), np.nan)))

    _stack = np.concatenate([_p.ravel() for _, _p in _panels])
    if map_view.value == "error":
        _limit = np.nanpercentile(np.abs(_stack), 98)
        _cmap, _vmin, _vmax = "RdBu_r", -_limit, _limit
    elif map_view.value == "std":
        _cmap, _vmin, _vmax = "Purples", 0, np.nanpercentile(_stack, 99)
    else:
        _cmap, (_vmin, _vmax) = "snow", np.nanpercentile(_stack, (2, 98))

    _ncols = min(len(_panels), 6)
    _nrows = -(-len(_panels) // _ncols)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(2.9 * _ncols, 2.9 * _nrows), squeeze=False,
                               constrained_layout=True)
    for _ax in _axes.ravel():
        _ax.axis("off")
    for _ax, (_label, _image) in zip(_axes.ravel(), _panels):
        _im = _ax.imshow(_image, cmap=_cmap, vmin=_vmin, vmax=_vmax, interpolation="nearest")
        _ax.set_title(_label.replace(" · ", "\n"), fontsize=7)
    _fig.colorbar(_im, ax=list(_axes.ravel()), shrink=0.6, label=f"{_first['channel']} ({map_view.value})")
    _fig.suptitle(f"{_sweep.title}: {_first['channel']} on {_first['dates'][_t]}")
    _fig
    return


if __name__ == "__main__":
    app.run()
