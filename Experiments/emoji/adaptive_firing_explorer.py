# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

"""Explore emoji NCAs with a learned per-cell firing rate (aNCA, gaNCA).

Run from the repository root with:

    marimo edit Experiments/emoji/adaptive_firing_explorer.py

Lists the multi-attractor emoji bundles of the NCA, gNCA, aNCA and gaNCA
families in the model store (``MODEL_STORE_ROOT``, default ``models/``), e.g.
from the ``emoji_adaptive_firing_*`` sweeps. Emoji images are read from
``DATA_PATH_BASE/Emojis``. The rollout helpers are in
``Experiments/emoji/adaptive_firing.py``.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    import zlib
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    from Experiments.emoji import adaptive_firing as af
    from Experiments.emoji import multi_attractor as ma
    from Experiments.model_registry import ModelRegistry, open_model_bundle, verify_evaluation_input
    from NCA.model.factory import portable_family

    _repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(_repository_root / ".env", override=False)
    DEFAULT_STORE = os.environ.get("MODEL_STORE_ROOT", str(_repository_root / "models"))
    DEFAULT_DATA = os.environ.get("DATA_PATH_BASE", "")

    # Light figures regardless of the marimo theme
    plt.style.use("default")

    COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#8c564b")

    # Factors that sweeps vary, used for plot axes and colours
    FACTORS = ("family", "firing", "firing_energy", "energy_per_update", "t", "fire_rate", "channels",
               "repeat", "experiment", "wandb_group", "targets")

    def style_axes(ax):
        ax.grid(color="#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    def show_image(ax, rgba, title=None):
        ax.imshow(ma.to_image(rgba), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=8)

    def levels(values):
        """Plot order of a factor: numbers ascending, anything else in order of first appearance."""
        unique = list(dict.fromkeys(v for v in values if not pd.isna(v)))
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in unique):
            return sorted(unique)
        return unique

    def strip_plot(ax, df, x, y, hue):
        """One dot per model and the mean of each (x, hue) level, side by side for each hue."""
        df = df.copy()
        for column in filter(None, (x, hue)):
            df[column] = df[column].where(df[column].notna(), "none")
        x_order = levels(df[x])
        hue_order = levels(df[hue]) if hue else [None]
        width = 0.8 / len(hue_order)
        for k, level in enumerate(hue_order):
            rows = df if level is None else df[df[hue] == level]
            offset = (k - (len(hue_order) - 1) / 2) * width
            colour = COLOURS[k % len(COLOURS)]
            means = []
            for i, value in enumerate(x_order):
                values = rows.loc[rows[x] == value, y].dropna().to_numpy(float)
                jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(values)) * width
                ax.scatter(i + offset + jitter, values, s=16, color=colour, alpha=0.5, lw=0)
                means.append(values.mean() if len(values) else np.nan)
            ax.scatter(np.arange(len(x_order)) + offset, means, marker="_", s=400, lw=2.5, color=colour,
                       label=None if level is None else str(level))
        ax.set_xticks(range(len(x_order)), [str(v) for v in x_order])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        style_axes(ax)
        if hue:
            ax.legend(title=hue, frameon=False, fontsize=8)

    def model_label(row):
        """Short text for one model in drop-downs."""
        energy = f" · λ {row.firing_energy:g}" if row.family in af.ADAPTIVE_FAMILIES else ""
        return f"{row.family} · {row.firing}{energy} · t {row.t} · rep {row.repeat} · {row.model_id[:8]}"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Adaptive firing emoji NCAs

    - aNCA / gaNCA learn a firing probability per cell, stored in the last state channel; NCA / gNCA fire at a fixed `fire_rate`
    - **Training**: best loss from each bundle, against the sweep settings
    - **Rollouts**: activity (mean firing probability), expected updates per cell, when the NCA goes quiet, and error to the target
    - **Single model**: frames, firing maps and optional damage
    """)
    return


@app.cell(hide_code=True)
def _():
    _devices = ", ".join(f"{_device.platform}: {_device.device_kind}" for _device in jax.devices())
    mo.callout(
        f"JAX backend: **{jax.default_backend()}** ({_devices}). The first rollout of each model size includes compilation.",
        kind="success" if jax.default_backend() == "gpu" else "warn",
    )
    return


@app.cell(hide_code=True)
def _():
    store_root = mo.ui.text(DEFAULT_STORE, label="Model store", full_width=True)
    data_root = mo.ui.text(DEFAULT_DATA, label="Data root (holds Emojis/)", full_width=True)
    reload_button = mo.ui.button(label="Reload models", value=0, on_click=lambda count: count + 1)
    mo.vstack([store_root, data_root, mo.hstack([reload_button, mo.md("after copying new bundles into the store")],
                                                justify="start")])
    return data_root, reload_button, store_root


@app.cell(hide_code=True)
def _(reload_button, store_root):
    # Bundles are read from their folders, so the registry index does not need rebuilding
    reload_button.value
    _registry = ModelRegistry(Path(store_root.value).expanduser())
    bundles = {}
    _skipped = 0
    for _path in _registry.bundle_paths():
        try:
            _bundle = open_model_bundle(_path)
        except (ValueError, TypeError, KeyError):
            _skipped += 1
            continue
        if ma.is_multi_attractor(_bundle.config) and portable_family(_bundle.config.model.family) in af.FAMILIES:
            bundles[_bundle.id] = _bundle
    runs = pd.DataFrame([af.summary_row(_bundle) for _bundle in bundles.values()])
    if len(runs):
        runs = runs.sort_values("created_at", ascending=False).reset_index(drop=True)
    _skipped_text = f" ({_skipped} bundles could not be read)" if _skipped else ""
    mo.md(f"**{len(runs)}** multi-attractor NCA / gNCA / aNCA / gaNCA bundles in `{_registry.root}`{_skipped_text}.")
    return bundles, runs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Models

    - Experiments default to the `adaptive_firing` sweeps when there are any
    """)
    return


@app.cell(hide_code=True)
def _(runs):
    mo.stop(len(runs) == 0, mo.callout("No matching bundles found in this model store.", kind="warn"))

    def _choice(column, label, default=None):
        _options = sorted(set(runs[column].astype(str)))
        return mo.ui.multiselect(_options, value=default or _options, label=label)

    _experiments = sorted(set(runs.experiment.astype(str)))
    _adaptive = [_name for _name in _experiments if "adaptive_firing" in _name]
    model_filters = mo.ui.dictionary({
        "experiment": _choice("experiment", "Experiment", _adaptive or None),
        "wandb_group": _choice("wandb_group", "W&B group"),
        "family": _choice("family", "Family"),
        "firing": _choice("firing", "Firing"),
        "firing_energy": _choice("firing_energy", "Firing energy"),
        "t": _choice("t", "run.t"),
        "targets": _choice("targets", "Targets"),
    })
    mo.vstack([
        mo.hstack([model_filters[_name] for _name in ("experiment", "wandb_group", "targets")], justify="start", wrap=True),
        mo.hstack([model_filters[_name] for _name in ("family", "firing", "firing_energy", "t")], justify="start", wrap=True),
    ])
    return (model_filters,)


@app.cell(hide_code=True)
def _(model_filters, runs):
    _keep = np.ones(len(runs), dtype=bool)
    for _column, _values in model_filters.value.items():
        _keep &= runs[_column].astype(str).isin(_values).to_numpy()
    candidates = runs[_keep].reset_index(drop=True)
    model_table = mo.ui.table(
        candidates.dropna(axis=1, how="all").round(5),
        initial_selection=list(range(len(candidates))),
        page_size=10,
        label="Untick rows to drop models from the plots and rollouts",
    )
    model_table
    return candidates, model_table


@app.cell(hide_code=True)
def _(candidates, model_table):
    _ticked = set(model_table.value["model_id"]) if len(model_table.value) else set()
    selected = candidates[candidates.model_id.isin(_ticked)].reset_index(drop=True)
    mo.md(f"**{len(selected)}** of {len(candidates)} models selected.")
    return (selected,)


@app.cell(hide_code=True)
def _(bundles, data_root):
    # Models and training seeds, loaded once per session
    _models = {}
    _seeds = {}

    def get_model(model_id):
        if model_id not in _models:
            _models[model_id] = bundles[model_id].load_model()
        return _models[model_id]

    def get_seeds(model_id):
        """(seeds, targets, pair_target, fingerprint warning or None) of a model."""
        if model_id not in _seeds:
            _bundle = bundles[model_id]
            _channels = get_model(model_id).N_CHANNELS
            _impath = Path(data_root.value).expanduser() / "Emojis"
            _seed, _target, _pair_target, _data = ma.load_seeds_and_targets(_bundle.config, _channels, _impath)
            _warning = None
            _provenance = _bundle.manifest.get("evaluation_input")
            if _provenance is not None:
                try:
                    verify_evaluation_input(_data, _provenance)
                except ValueError as _error:
                    _warning = str(_error)
            _seeds[model_id] = (_seed, _target, _pair_target, _warning)
        return _seeds[model_id]

    return get_model, get_seeds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    - `best_loss` and `best_iteration` as recorded in each bundle (no rollouts needed)
    - `energy_per_update` = `firing_energy` / `run.t`: the cost of one update of every cell
    """)
    return


@app.cell(hide_code=True)
def _(selected):
    _factors = [_f for _f in FACTORS if _f in selected]
    train_x = mo.ui.dropdown(_factors, value="firing_energy" if "firing_energy" in _factors else _factors[0], label="x")
    train_hue = mo.ui.dropdown(["(none)"] + _factors, value="firing" if "firing" in _factors else "(none)",
                               label="colour")
    train_y = mo.ui.dropdown(["best_loss", "best_iteration"], value="best_loss", label="y")
    mo.hstack([train_x, train_hue, train_y], justify="start")
    return train_hue, train_x, train_y


@app.cell(hide_code=True)
def _(selected, train_hue, train_x, train_y):
    mo.stop(len(selected) == 0, mo.md("Select some models."))
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    strip_plot(_ax, selected, train_x.value, train_y.value, None if train_hue.value == "(none)" else train_hue.value)
    if train_y.value == "best_loss":
        _ax.set_yscale("log")
    _ax.set_title("one dot per model, bar = mean", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Rollouts

    - Every training seed is rolled out *rollouts* times for *length* × `run.t` steps
    - Activity is the firing probability averaged over the whole grid (padding included); fixed-rate models have activity `fire_rate`
    - `updates_per_cell`: activity summed over the rollout; `updates_to_t`: the same over the first `run.t` steps
    - `quiet_step`: first step after which activity stays below the *quiet threshold*; NaN if it never does. `quiet_fraction`: share of rollouts that go quiet (`quiet_step` is averaged over those only)
    - `mse_*`: RGBA error to the seed's own target at `run.t`, at the end, at the quiet step and at its minimum
    - *Damage*: a disc at the grid centre is zeroed at *damage at* × `run.t`; adds `damage_peak_activity`, `damage_updates_per_cell` and `mse_after_damage`
    """)
    return


@app.cell(hide_code=True)
def _():
    eval_length = mo.ui.slider(1, 8, value=4, step=1, label="Length (× run.t)", show_value=True)
    eval_repeats = mo.ui.number(1, 64, value=4, step=1, label="Rollouts per seed")
    quiet_threshold = mo.ui.number(0.0, 1.0, value=0.05, step=0.01, label="Quiet threshold")
    eval_damage = mo.ui.checkbox(value=False, label="Damage")
    damage_at = mo.ui.slider(0.25, 7.75, value=2.0, step=0.25, label="Damage at (× run.t)", show_value=True)
    damage_radius = mo.ui.slider(1, 32, value=8, label="Damage radius", show_value=True)
    eval_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Seed")
    # Seeds are rolled out in chunks that fit this budget; it changes speed only, not results
    memory_budget = mo.ui.number(0.125, 64, value=1.0, step=0.125, label="Rollout memory (GB)")
    eval_run = mo.ui.run_button(label="Roll out selected models")
    mo.vstack([
        mo.hstack([eval_length, eval_repeats, quiet_threshold, eval_seed, memory_budget], justify="start", gap=1.5, wrap=True),
        mo.hstack([eval_damage, damage_at, damage_radius, eval_run], justify="start", gap=1.5, wrap=True),
    ])
    return (
        damage_at,
        damage_radius,
        eval_damage,
        eval_length,
        eval_repeats,
        eval_run,
        eval_seed,
        memory_budget,
        quiet_threshold,
    )


@app.cell
def _():
    # Results of every model rolled out in this session, keyed by (model_id, settings),
    # so pressing the button again only rolls out new models
    trace_cache = {}
    return (trace_cache,)


@app.cell(hide_code=True)
def _(
    bundles,
    damage_at,
    damage_radius,
    eval_damage,
    eval_length,
    eval_repeats,
    eval_run,
    eval_seed,
    get_model,
    get_seeds,
    memory_budget,
    quiet_threshold,
    selected,
    trace_cache,
):
    _settings = (eval_length.value, eval_repeats.value, quiet_threshold.value, eval_damage.value,
                 damage_at.value, damage_radius.value, eval_seed.value)
    _todo = [_id for _id in selected.model_id if (_id, _settings) not in trace_cache]
    _failures = []
    if eval_run.value and _todo:
        for _id in mo.status.progress_bar(_todo, title="Rolling out models"):
            _t = bundles[_id].config.run.t
            _steps = eval_length.value * _t
            _damage_step = min(int(damage_at.value * _t), _steps - 1) if eval_damage.value else -1
            try:
                _seed, _target, _pair_target, _ = get_seeds(_id)
                _rows, _activity, _distance = af.evaluate_seeds(
                    get_model(_id), _seed, _target, _pair_target, _t, _steps, eval_repeats.value,
                    jr.fold_in(jr.PRNGKey(eval_seed.value), zlib.crc32(_id.encode())), quiet_threshold.value,
                    _damage_step, damage_radius.value, memory_budget.value,
                )
                trace_cache[_id, _settings] = {
                    "rows": _rows,
                    # Mean over seeds and rollouts, [steps + 1]
                    "activity": _activity.mean(axis=(0, 1)),
                    "distance": _distance.mean(axis=(0, 1)),
                    "t": _t,
                    "damage_step": _damage_step,
                }
            except (FileNotFoundError, ValueError) as _error:
                _failures.append(f"- `{_id}`: {_error}")

    _rows = []
    # model_id -> mean activity and distance traces, run.t and damage step
    traces = {}
    for _id in selected.model_id:
        if (_id, _settings) not in trace_cache:
            continue
        _entry = trace_cache[_id, _settings]
        _rows.extend({"model_id": _id, **_row} for _row in _entry["rows"])
        traces[_id] = _entry
    rollout_scores = pd.DataFrame(_rows)
    mo.stop(
        not len(rollout_scores),
        mo.callout(mo.md("Press **Roll out selected models**.\n\n" + "\n".join(_failures)), kind="neutral"),
    )
    _metrics = [_c for _c in rollout_scores.columns if _c not in ("model_id", "pair", "target", "rollout")]
    _grouped = rollout_scores.groupby("model_id")
    _summary = _grouped[_metrics].mean()
    _summary["quiet_fraction"] = _grouped.quiet_step.apply(lambda _v: float(np.isfinite(_v).mean()))
    model_scores = selected.merge(_summary.reset_index(), on="model_id")
    _missing = len(selected) - len(model_scores)
    _notes = []
    if _missing:
        _notes.append(f"{_missing} selected models have no rollouts for these settings; press the button to add them.")
    if _failures:
        _notes.append("Failed:\n\n" + "\n".join(_failures))
    mo.callout(mo.md("\n\n".join(_notes)), kind="warn") if _notes else mo.md(
        f"Rollouts for **{len(model_scores)}** models."
    )
    return model_scores, rollout_scores, traces


@app.cell(hide_code=True)
def _(model_scores, rollout_scores):
    _factors = [_f for _f in FACTORS if _f in model_scores]
    _metrics = [_c for _c in rollout_scores.columns if _c not in ("model_id", "pair", "target", "rollout")]
    _metrics += ["quiet_fraction", "best_loss"]
    score_x = mo.ui.dropdown(_factors, value="t" if "t" in _factors else _factors[0], label="x")
    score_hue = mo.ui.dropdown(["(none)"] + _factors, value="family", label="colour")
    score_y = mo.ui.dropdown(_metrics, value="updates_per_cell", label="y")
    score_log = mo.ui.checkbox(value=False, label="log y")
    mo.hstack([score_x, score_hue, score_y, score_log], justify="start")
    return score_hue, score_log, score_x, score_y


@app.cell(hide_code=True)
def _(model_scores, rollout_scores, score_hue, score_log, score_x, score_y):
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    strip_plot(_ax, model_scores, score_x.value, score_y.value, None if score_hue.value == "(none)" else score_hue.value)
    if score_log.value:
        _ax.set_yscale("log")
    _ax.set_title("one dot per model (mean over seeds and rollouts), bar = mean", fontsize=9)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.ui.table(model_scores.round(4), selection=None, page_size=10, label="Per model"),
        mo.accordion({"Per rollout": mo.ui.table(rollout_scores.round(4), selection=None, page_size=10)}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Traces

    - One line per model: activity (top) and RGBA error to the own target (bottom), averaged over seeds and rollouts
    """)
    return


@app.cell(hide_code=True)
def _(model_scores):
    _factors = [_f for _f in FACTORS if _f in model_scores]
    trace_colour = mo.ui.dropdown(_factors, value="family", label="colour")
    trace_scale = mo.ui.radio({"step": False, "step / run.t": True}, value="step / run.t", label="x axis", inline=True)
    mo.hstack([trace_colour, trace_scale], justify="start", gap=1.5)
    return trace_colour, trace_scale


@app.cell(hide_code=True)
def _(model_scores, trace_colour, trace_scale, traces):
    _values = model_scores[trace_colour.value].where(model_scores[trace_colour.value].notna(), "none")
    _order = levels(_values)
    _fig, (_ax_activity, _ax_error) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for _id, _value in zip(model_scores.model_id, _values):
        _entry = traces[_id]
        _x = np.arange(len(_entry["activity"])) / (_entry["t"] if trace_scale.value else 1)
        _colour = COLOURS[_order.index(_value) % len(COLOURS)]
        _ax_activity.plot(_x, _entry["activity"], color=_colour, lw=1, alpha=0.6)
        _ax_error.plot(_x, _entry["distance"], color=_colour, lw=1, alpha=0.6)
        if _entry["damage_step"] >= 0:
            _ax_activity.axvline(_entry["damage_step"] / (_entry["t"] if trace_scale.value else 1),
                                 color="#b8001f", ls="--", lw=0.8, alpha=0.4)
    for _k, _value in enumerate(_order):
        _ax_activity.plot([], [], color=COLOURS[_k % len(COLOURS)], label=str(_value))
    if trace_scale.value:
        for _ax in (_ax_activity, _ax_error):
            _ax.axvline(1, color="#999", ls=":", lw=1)
    _ax_activity.set_ylim(0, 1)
    _ax_activity.set_ylabel("mean firing probability")
    _ax_activity.legend(title=trace_colour.value, frameon=False, fontsize=8)
    _ax_error.set_yscale("log")
    _ax_error.set_ylabel("RGBA MSE to own target")
    _ax_error.set_xlabel("step / run.t" if trace_scale.value else "step")
    style_axes(_ax_activity)
    style_axes(_ax_error)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Single model

    - RGBA (top) and firing probability (middle, 0 to 1) of one rollout, with activity and error underneath
    """)
    return


@app.cell(hide_code=True)
def _(candidates, selected):
    _pool = selected if len(selected) else candidates
    _options = {model_label(_row): _row.model_id for _row in _pool.itertuples()}
    single_model = mo.ui.dropdown(_options, value=next(iter(_options), None), label="Model", full_width=True)
    single_model
    return (single_model,)


@app.cell(hide_code=True)
def _(bundles, get_seeds, single_model):
    mo.stop(single_model.value is None, mo.md("Select a model."))
    single_id = single_model.value
    single_cfg = bundles[single_id].config
    try:
        single_seeds, single_targets, single_pair_target, _warning = get_seeds(single_id)
    except FileNotFoundError as _error:
        mo.stop(True, mo.callout(f"Could not load the emoji images: {_error}", kind="danger"))
    _names = ma.target_names(single_cfg)
    single_pair = mo.ui.dropdown({f"seed {_p} → {_names[single_pair_target[_p]]}": _p for _p in range(len(single_seeds))},
                                 value=f"seed 0 → {_names[single_pair_target[0]]}", label="Seed")
    single_length = mo.ui.slider(1, 8, value=4, step=1, label="Length (× run.t)", show_value=True)
    single_damage = mo.ui.checkbox(value=False, label="Damage")
    single_damage_at = mo.ui.slider(0.25, 7.75, value=2.0, step=0.25, label="Damage at (× run.t)", show_value=True)
    single_radius = mo.ui.slider(1, 32, value=8, label="Damage radius", show_value=True)
    single_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="Rollout seed")
    _items = [mo.hstack([single_pair, single_length, single_seed], justify="start", gap=1.5, wrap=True),
              mo.hstack([single_damage, single_damage_at, single_radius], justify="start", gap=1.5, wrap=True)]
    if _warning:
        _items.insert(0, mo.callout(f"The reloaded seeds differ from training: {_warning}", kind="warn"))
    mo.vstack(_items)
    return (
        single_cfg,
        single_damage,
        single_damage_at,
        single_id,
        single_length,
        single_pair,
        single_pair_target,
        single_radius,
        single_seed,
        single_seeds,
        single_targets,
    )


@app.cell(hide_code=True)
def _(
    get_model,
    single_cfg,
    single_damage,
    single_damage_at,
    single_id,
    single_length,
    single_pair,
    single_pair_target,
    single_radius,
    single_seed,
    single_seeds,
    single_targets,
):
    _t = single_cfg.run.t
    _steps = single_length.value * _t
    _damage_step = min(int(single_damage_at.value * _t), _steps - 1) if single_damage.value else -1
    _rgba, _firing, _distances = (np.asarray(_v) for _v in af.activity_rollout(
        get_model(single_id), jnp.asarray(single_seeds[single_pair.value]), jnp.asarray(single_targets), _steps,
        jr.PRNGKey(single_seed.value), _damage_step, float(single_radius.value)))
    _own = int(single_pair_target[single_pair.value])

    _frames = np.linspace(0, _steps, 9).astype(int)
    _fig = plt.figure(figsize=(1.5 * len(_frames), 6.4))
    _grid = _fig.add_gridspec(3, len(_frames), height_ratios=[1, 1, 1.6])
    for _i, _frame in enumerate(_frames):
        show_image(_fig.add_subplot(_grid[0, _i]), _rgba[_frame], f"step {_frame}")
        _ax = _fig.add_subplot(_grid[1, _i])
        _ax.imshow(_firing[_frame], vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        _ax.set_axis_off()
    _ax = _fig.add_subplot(_grid[2, :])
    _ax.plot(_firing.mean(axis=(1, 2)), color="black", lw=1.5, label="mean firing probability")
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("step")
    _ax.set_ylabel("mean firing probability")
    _ax.axvline(_t, color="#999", ls=":", lw=1, label="run.t")
    if _damage_step >= 0:
        _ax.axvline(_damage_step, color="#b8001f", ls="--", lw=1, label="damage")
    _twin = _ax.twinx()
    _twin.plot(_distances[:, _own], color=COLOURS[0], lw=1, label="RGBA MSE to own target")
    _twin.set_yscale("log")
    _twin.set_ylabel("RGBA MSE to own target")
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _twin.legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(_ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
