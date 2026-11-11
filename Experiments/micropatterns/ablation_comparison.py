# /// script
# dependencies = [
#   "marimo",
#   "pandas",
#   "matplotlib",
#   "wandb",
#   "python-dotenv",
# ]
# ///

"""Compare the runs of the micropattern ablation sweep
(conf/experiments/nca_micropattern_ablations.yaml) using their W&B validation logs.

Run from the repository root with:

    marimo edit Experiments/micropatterns/ablation_comparison.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from Experiments.model_registry import ModelRegistry, open_model_bundle
    from NCA.trainer.intervention import rollout_model_sampled

    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env", override=False)

    # One row per labels.ablation value, in sweep order: (label, category, comparability).
    # Comparability with the baseline:
    #   same       same objective and validation data
    #   objective  different loss weights or curriculum, same validation data
    #   data       different targets (resolution or projection)
    #   validation a different (smaller) validation set
    #   none       no validation set
    ABLATIONS = {
        "baseline": ("Baseline", "Reference", "same"),
        "loss_texture_only": ("Texture only", "Loss terms", "objective"),
        "loss_radial_only": ("Radial only", "Loss terms", "objective"),
        "loss_l2_only": ("L2 only", "Loss terms", "objective"),
        "loss_l2_not_radial": ("L2 + texture", "Loss terms", "objective"),
        "loss_add_l2": ("+ L2", "Loss terms", "objective"),
        "loss_add_summary": ("+ mean & corr.", "Loss terms", "objective"),
        "curriculum_static": ("Static weights", "Curriculum", "objective"),
        "curriculum_macro_first": ("Macro first", "Curriculum", "objective"),
        "curriculum_reinjection_decay": ("Reinjection decay", "Curriculum", "same"),
        "model_nca": ("NCA (no gate)", "Model", "same"),
        "downsample_16": ("Downsample 16", "Resolution", "data"),
        "downsample_32": ("Downsample 32", "Resolution", "data"),
        "mask_uniform": ("Uniform mask", "Adhesion mask", "same"),
        "no_intermediate_state_reg": ("No state reg.", "Regularisers", "same"),
        "no_boundary_reg": ("No boundary reg.", "Regularisers", "same"),
        "no_regularisers": ("No regularisers", "Regularisers", "same"),
        "train_replicates_1": ("1 train replicate", "Replicates", "same"),
        "train_replicates_3": ("3 train replicates", "Replicates", "validation"),
        "train_replicates_4": ("4 train replicates", "Replicates", "none"),
        "projection_max": ("Max projection", "Projection", "data"),
        "projection_sum": ("Sum projection", "Projection", "data"),
    }

    def ablation_label(ablation):
        return ABLATIONS.get(ablation, (ablation,))[0]

    # Multi-target loss terms each ablation trains on (non-zero weight)
    TRAINED_TERMS = {
        "baseline": {"texture", "radial"},
        "loss_texture_only": {"texture"},
        "loss_radial_only": {"radial"},
        "loss_l2_only": {"l2"},
        "loss_l2_not_radial": {"l2", "texture"},
        "loss_add_l2": {"l2", "texture", "radial"},
        "loss_add_summary": {"texture", "channel_mean", "correlation", "radial"},
        "curriculum_static": {"texture", "radial"},
        "curriculum_macro_first": {"texture", "radial"},
    }

    # Validation replicates tag -> train/validation split (matched to the baseline split)
    SPLITS = {"3-4": "A", "4": "A", "1-2": "B", "2": "B"}
    SPLIT_MARKERS = {"A": "o", "B": "^", "none": "s"}
    SPLIT_LABELS = {
        "A": "train 1,2 · val 3,4",
        "B": "train 3,4 · val 1,2",
        "none": "no validation",
    }

    ROLLOUT = "ValidationRollout"
    RAW = f"{ROLLOUT}/loss_component_raw"
    # column: (W&B key, label, map to an error where lower is better)
    ERROR_METRICS = {
        "l2": (f"{RAW}/l2", "Pixel L2", None),
        "radial": (f"{RAW}/radial", "Radial profile", None),
        "channel_mean": (f"{RAW}/channel_mean", "Channel mean", None),
        "correlation": (f"{RAW}/correlation", "Channel corr.", None),
        "texture": (f"{RAW}/texture", "Texture", None),
        "energy_distance": (
            f"{ROLLOUT}/variation/energy_distance", "Energy distance", None,
        ),
        "dispersion": (
            f"{ROLLOUT}/variation/dispersion_ratio",
            "|log dispersion ratio|",
            lambda value: np.abs(np.log(value)),
        ),
        "distance_correlation": (
            f"{ROLLOUT}/variation/pairwise_distance_correlation",
            "1 − distance corr.",
            lambda value: 1.0 - value,
        ),
    }
    LOSS_COLUMNS = ["l2", "radial", "channel_mean", "correlation", "texture"]
    VARIATION_COLUMNS = ["energy_distance", "dispersion", "distance_correlation"]

    # Colours: baseline in grey, ablations from a fixed categorical order
    BASELINE_COLOUR = "#8a8985"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
    DIVERGING = LinearSegmentedColormap.from_list(
        "better_worse", ["#2a78d6", "#e8e7e3", "#e34948"]
    )

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

    def parse_tags(tags):
        """W&B ``key:value`` tags as a dict."""
        return dict(tag.split(":", 1) for tag in tags if ":" in tag)

    def fetch_wandb_runs(entity, project, group):
        """Run settings (from tags) and validation/training histories of a sweep."""
        import wandb
        from tqdm.auto import tqdm

        api = wandb.Api(timeout=120)
        path = f"{entity}/{project}" if entity else project
        filters = {"group": group} if group else {}
        runs, histories = [], []
        for run in tqdm(api.runs(path, filters=filters)):
            tags = parse_tags(run.tags)
            if "labels.ablation" not in tags:
                continue
            runs.append({
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "ablation": tags["labels.ablation"],
                "repeat": int(tags.get("run.repeat", -1)),
                "train_replicates": tags.get("data.micropattern.train_replicates", ""),
                "validation_replicates": tags.get(
                    "data.micropattern.validation_replicates", ""
                ),
                "runtime_hours": float(run.summary.get("_runtime", np.nan)) / 3600,
            })
            # Validation keys are all logged at the same steps
            keys = [
                key for key in run.summary.keys()
                if key.startswith(f"{ROLLOUT}/") or key == "Validation/loss"
            ]
            if keys:
                frame = pd.DataFrame(list(run.scan_history(keys=["_step", *keys], page_size=1000)))
                histories.append(frame.assign(run_id=run.id))
            # Training loss is logged every step, so only a sample is fetched
            train = run.history(keys=["Train/loss"], samples=500, pandas=True)
            if not train.empty:
                histories.append(train[["_step", "Train/loss"]].assign(run_id=run.id))
        history = (
            pd.concat(histories, ignore_index=True)
            .melt(id_vars=["run_id", "_step"], var_name="metric", value_name="value")
            .dropna(subset=["value"])
            if histories else
            pd.DataFrame(columns=["run_id", "_step", "metric", "value"])
        )
        return pd.DataFrame(runs), history

    def final_values(history, fraction):
        """Mean of each metric over the last ``fraction`` of its logged iterations, per run."""
        last = history.groupby(["run_id", "metric"])["_step"].transform("max")
        window = history[history["_step"] >= last * (1.0 - fraction)]
        return window.groupby(["run_id", "metric"])["value"].mean().unstack("metric")

    def error_table(final):
        """Final values of ERROR_METRICS as errors (lower is better), one row per run."""
        table = pd.DataFrame(index=final.index)
        for column, (key, _, transform) in ERROR_METRICS.items():
            values = final[key] if key in final.columns else pd.Series(np.nan, index=final.index)
            table[column] = values if transform is None else transform(values)
        # Texture is logged as 0 when a run does not compute it
        table.loc[table["texture"] == 0, "texture"] = np.nan
        return table

    def relative_to_baseline(errors, runs, columns):
        """Mean log2(error / baseline error) per ablation, against the baseline of the same split."""
        merged = errors[columns].join(runs.set_index("run_id")[["ablation", "split"]])
        baseline = merged[merged["ablation"] == "baseline"].groupby("split")[columns].mean()
        rows = {}
        for ablation, group in merged.groupby("ablation"):
            ratios = [
                np.log2(split_rows[columns] / baseline.loc[split])
                for split, split_rows in group.groupby("split")
                if split in baseline.index
            ]
            rows[ablation] = (
                pd.concat(ratios).mean() if ratios
                else pd.Series(np.nan, index=columns)
            )
        return pd.DataFrame(rows).T

    def ablation_colours(ablations):
        """Grey for the baseline, then the categorical order for the rest."""
        colours, index = [], 0
        for ablation in ablations:
            if ablation == "baseline":
                colours.append(BASELINE_COLOUR)
            else:
                colours.append(SERIES[index % len(SERIES)])
                index += 1
        return colours

    def run_ids(runs, ablation):
        return runs.loc[runs["ablation"] == ablation, "run_id"]

    def draw_curves(ax, history, runs, metric, ablations, colours=None):
        """Mean over runs (line) and min–max range (band) of a metric against iteration."""
        colours = colours or ablation_colours(ablations)
        for ablation, colour in zip(ablations, colours):
            rows = history[
                (history["metric"] == metric)
                & history["run_id"].isin(run_ids(runs, ablation))
            ]
            if rows.empty:
                continue
            stats = rows.groupby("_step")["value"].agg(["mean", "min", "max"])
            ax.fill_between(
                stats.index, stats["min"], stats["max"], color=colour, alpha=0.15, lw=0
            )
            ax.plot(
                stats.index, stats["mean"], color=colour, lw=1.5,
                label=ablation_label(ablation),
            )
        ax.set_xlabel("Iteration")

    def draw_strip(ax, values, runs, ablations, colours=None, filled=None, positions=None):
        """One point per run (marker = split), with a bar at the mean of each ablation."""
        colours = colours or ablation_colours(ablations)
        positions = list(range(len(ablations))) if positions is None else positions
        for x, ablation, colour in zip(positions, ablations, colours):
            rows = runs[runs["ablation"] == ablation]
            points = values.reindex(rows["run_id"])
            face = colour if filled is None or filled(ablation) else "none"
            for (_, run), value in zip(rows.iterrows(), points):
                offset = 0.08 * (run["repeat"] - 0.5) + (0.12 if run["split"] == "B" else -0.12)
                ax.scatter(
                    x + offset, value, s=24, marker=SPLIT_MARKERS[run["split"]],
                    facecolor=face, edgecolor=colour, lw=1.2, zorder=3,
                )
            if points.notna().any():
                ax.hlines(points.mean(), x - 0.3, x + 0.3, color=colour, lw=2, zorder=2)
        ax.set_xticks(positions, [ablation_label(a) for a in ablations], rotation=35, ha="right")
        ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)

    # ---- Model inference (section 5) ----

    # Marker triplets drawn as cyan, magenta, yellow (as in rendering.render_cmy_composite)
    TRIPLETS = {
        "Cell fate (SOX2 · TBXT · SOX17)": ("SOX2", "TBXT", "SOX17"),
        "Nodal pathway (CER1 · LEFTY · NODAL)": ("CER1", "LEFTY", "NODAL"),
        "Signalling (FOXA2 · LEF1 · SMAD23)": ("FOXA2", "LEF1", "SMAD23"),
    }

    # Loaded data and rollouts, kept between cell runs
    DATA_CACHE = {}
    ROLLOUT_CACHE = {}

    def bundle_table(registry, experiment):
        """One row per complete bundle of the sweep, keeping the newest per (ablation, repeat, split)."""
        models = registry.models_df()
        models = models[(models["experiment"] == experiment) & (models["status"] == "complete")]
        rows = []
        for _, model in models.iterrows():
            bundle = open_model_bundle(model["path"])
            validation = bundle.config.data.micropattern.validation_replicates
            rows.append({
                "model_id": model["model_id"],
                "created_at": model["created_at"],
                "ablation": bundle.config.labels.ablation,
                "repeat": bundle.config.run.repeat,
                "split": SPLITS.get("-".join(map(str, validation or ())), "none"),
                "path": model["path"],
            })
        if not rows:
            return pd.DataFrame(columns=["model_id", "created_at", "ablation", "repeat", "split", "path"])
        return (
            pd.DataFrame(rows)
            .sort_values("created_at")
            .drop_duplicates(["ablation", "repeat", "split"], keep="last")
            .reset_index(drop=True)
        )

    def rollout_bundle(path, data_path, seed=0):
        """Roll a bundle's model out from 0h on its validation replicates (training replicates if it has none).

        Returns predictions and targets ``[replicate, time, state channel, H, W]``,
        the colony masks ``[replicate, H, W]`` and the replicate IDs.
        """
        from Experiments.micropatterns.config_helpers import load_train_validation_data

        bundle = open_model_bundle(path)
        cfg = bundle.config
        # Same splits and scaling as during training
        data_key = (repr(cfg.data), data_path)
        if data_key not in DATA_CACHE:
            DATA_CACHE[data_key] = load_train_validation_data(cfg.data, impath=data_path)
        train, validation = DATA_CACHE[data_key]
        if validation is not None:
            data, aux, _, boundary, _, _ = validation
            source, replicates = "validation", cfg.data.micropattern.validation_replicates
        else:
            data, aux, _, boundary, _, _ = train
            source = "training"
            replicates = cfg.data.micropattern.train_replicates or tuple(range(1, cfg.data.batches + 1))
        schema = aux["channel_schema"]
        primary = np.asarray(schema.primary_measurements)
        data = np.asarray(data)[:, :, primary]
        boundary = np.asarray(boundary)

        model = bundle.load_model()
        channels = int(model.N_CHANNELS)
        observation_steps = jnp.arange(data.shape[1], dtype=jnp.int32) * int(cfg.run.t)
        predictions = []
        for index in range(data.shape[0]):
            state = jnp.pad(
                jnp.asarray(data[index, 0]),
                ((0, channels - data.shape[2]), (0, 0), (0, 0)),
            )
            state = model.prepare_pool_state(state[None])[0]
            frames = rollout_model_sampled(
                model,
                state,
                jnp.asarray(boundary[index]),
                cfg.trainer.boundary_mode,
                jr.PRNGKey(seed),
                int(observation_steps[-1]),
                observation_steps,
            )
            predictions.append(np.asarray(frames)[:, : data.shape[2]])
        return {
            "prediction": np.stack(predictions),
            "target": data,
            "mask": boundary.reshape(boundary.shape[0], -1, *boundary.shape[-2:])[:, 0],
            "names": tuple(schema.state_channels),
            "replicates": tuple(replicates),
            "source": source,
            "model_id": bundle.id,
        }

    def cmy_image(states, names, triplet, mask=None):
        """RGB ``[H, W, 3]`` of three state channels ``states [C, H, W]`` as cyan, magenta, yellow."""
        cyan, magenta, yellow = (
            np.clip(states[names.index(name)], 0.0, 1.0) if name in names
            else np.zeros(states.shape[-2:])
            for name in triplet
        )
        rgb = np.stack((0.5 * (magenta + yellow), 0.5 * (cyan + yellow), 0.5 * (cyan + magenta)), axis=-1)
        return rgb if mask is None else rgb * (mask[..., None] > 0)

    def replicate_index(result, replicate):
        """Index of a physical replicate in a result, or its first replicate if it lacks that one."""
        return result["replicates"].index(replicate) if replicate in result["replicates"] else 0

    def output_tile(result, kind, replicate, time, triplet, masked):
        """Image of the data ("target") or the model ("prediction") at one timepoint."""
        index = replicate_index(result, replicate)
        mask = result["mask"][index] if masked else None
        return cmy_image(result[kind][index, time], result["names"], triplet, mask)

    def own_data_needed(ablation, result, replicate):
        """Whether an ablation needs its own data panel (targets, replicate or split differ)."""
        return (
            ABLATIONS.get(ablation, ("", "", "data"))[2] in ("data", "validation", "none")
            or replicate not in result["replicates"]
        )

    def show_tile(ax, image, title):
        ax.set_axis_off()
        if image is not None:
            ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=7)

    def split_legend(ax, **kwargs):
        handles = [
            Line2D([], [], ls="", marker=SPLIT_MARKERS[split], color="#52514e",
                   markerfacecolor="none", label=SPLIT_LABELS[split])
            for split in ("A", "B")
        ]
        ax.legend(handles=handles, **kwargs)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Micropattern NCA ablations

    - Sweep: `conf/experiments/nca_micropattern_ablations.yaml` (23 conditions × 2 splits × 2 repeats)
    - Metrics come from W&B validation rollouts (from 0h, on the held-out replicates)
    - Raw loss components (unweighted) compare runs trained with different losses; the weighted validation loss is only compared between runs with the same objective
    - Resolution and projection runs have different targets, so their errors are indicative only
    """)
    return


@app.cell(hide_code=True)
def _():
    entity = mo.ui.text(os.environ.get("WANDB_ENTITY", ""), label="W&B entity")
    project = mo.ui.text("nca_micropattern_ablations", label="W&B project")
    group = mo.ui.text("nca_micropattern_ablations_v1", label="W&B group (empty: all)")
    fetch = mo.ui.run_button(label="Fetch from W&B")
    finished_only = mo.ui.checkbox(True, label="Finished runs only")
    final_fraction = mo.ui.slider(
        0.02, 0.3, step=0.02, value=0.1, show_value=True,
        label="Final value: mean over last fraction of training",
    )
    mo.vstack([
        mo.hstack([entity, project, group, fetch], justify="start"),
        mo.hstack([finished_only, final_fraction], justify="start"),
    ])
    return entity, fetch, final_fraction, finished_only, group, project


@app.cell(hide_code=True)
def _(entity, fetch, group, project):
    _cache_path = (
        repository_root / "output" / "ablation_comparison"
        / f"{project.value}__{group.value or 'all'}.pkl"
    )
    if fetch.value:
        with mo.status.spinner("Fetching runs from W&B"):
            _runs, _history = fetch_wandb_runs(entity.value, project.value, group.value)
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"runs": _runs, "history": _history}, _cache_path)
    else:
        mo.stop(
            not _cache_path.exists(),
            mo.callout("No cached runs yet: press **Fetch from W&B**.", kind="warn"),
        )
        _cached = pd.read_pickle(_cache_path)
        _runs, _history = _cached["runs"], _cached["history"]
    runs_all = _runs
    history_all = _history
    mo.md(f"Cache: `{_cache_path.relative_to(repository_root)}` · {len(runs_all)} runs")
    return history_all, runs_all


@app.cell(hide_code=True)
def _(finished_only, history_all, runs_all):
    runs = runs_all.copy()
    if finished_only.value:
        runs = runs[runs["state"] == "finished"]
    runs["split"] = runs["validation_replicates"].map(SPLITS).fillna("none")
    # Re-run conditions: keep only the newest run of each (ablation, repeat, split)
    if "created_at" in runs.columns:
        runs = runs.sort_values("created_at").drop_duplicates(
            ["ablation", "repeat", "split"], keep="last"
        )
    runs["label"] = runs["ablation"].map(ablation_label)
    runs["category"] = runs["ablation"].map(lambda a: ABLATIONS.get(a, ("", "Other"))[1])
    runs = runs.reset_index(drop=True)
    history = history_all[history_all["run_id"].isin(runs["run_id"])]

    # Runs per ablation and state, to spot missing or crashed runs
    _status = (
        runs_all.assign(ablation=runs_all["ablation"].map(ablation_label))
        .pivot_table(index="ablation", columns="state", values="run_id", aggfunc="count", fill_value=0)
        .reindex([ablation_label(a) for a in ABLATIONS])
        .fillna(0).astype(int)
    )
    _unknown = sorted(set(runs_all["ablation"]) - set(ABLATIONS))
    mo.vstack([
        mo.md("### Runs per ablation (4 expected; only the newest run of a re-run condition is used)"),
        mo.ui.table(_status.reset_index(), selection=None, page_size=25),
        mo.callout(f"Ablations not listed in ABLATIONS: {_unknown}", kind="warn") if _unknown else mo.md(""),
    ])
    return history, runs


@app.cell(hide_code=True)
def _(final_fraction, history, runs):
    final = final_values(history, final_fraction.value).reindex(runs["run_id"])
    errors = error_table(final)
    return errors, final


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1 · Overview

    - log2(error / baseline error), baseline taken from the same train/validation split; blue = better, red = worse
    - Columns: raw loss terms (left), replicate variation (right; needs 2 validation replicates)
    - Hatched rows: different targets or validation set; grey cells: not available
    """)
    return


@app.cell(hide_code=True)
def _(errors, runs):
    _columns = LOSS_COLUMNS + VARIATION_COLUMNS
    _ablations = [a for a in ABLATIONS if a != "baseline"]
    _relative = relative_to_baseline(errors, runs, _columns).reindex(_ablations)
    # Texture losses are not comparable across resolutions
    _relative.loc[["downsample_16", "downsample_32"], "texture"] = np.nan

    _values = _relative.to_numpy(dtype=float)
    _limit = max(0.5, float(np.nanmax(np.abs(_values)))) if np.isfinite(_values).any() else 1.0
    _limit = min(_limit, 3.0)
    _cmap = DIVERGING.copy()
    _cmap.set_bad("#f1f0ed")

    fig_overview, _ax = plt.subplots(figsize=(7.5, 7.5))
    _image = _ax.imshow(
        np.ma.masked_invalid(_values), cmap=_cmap, vmin=-_limit, vmax=_limit, aspect="auto"
    )
    for (_row, _column), _value in np.ndenumerate(_values):
        if np.isfinite(_value):
            _ax.text(
                _column, _row, f"{_value:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(_value) > 0.6 * _limit else "#0b0b0b",
            )
    for _row, _ablation in enumerate(_ablations):
        if ABLATIONS[_ablation][2] in ("data", "validation"):
            _ax.add_patch(Rectangle(
                (-0.5, _row - 0.5), len(_columns), 1, fill=False,
                hatch="////", edgecolor="#b5b4ae", lw=0,
            ))
    # Category separators, with category names on the right
    _categories = [ABLATIONS[a][1] for a in _ablations]
    for _row in range(1, len(_ablations)):
        if _categories[_row] != _categories[_row - 1]:
            _ax.axhline(_row - 0.5, color="white", lw=2)
    for _start, _category in enumerate(_categories):
        if _start > 0 and _category == _categories[_start - 1]:
            continue
        _ax.text(
            len(_columns) - 0.35, _start - 0.5 + _categories.count(_category) / 2, _category,
            ha="left", va="center", fontsize=7, color="#52514e", fontweight="bold",
        )
    # Raw loss terms | replicate variation
    _ax.axvline(len(LOSS_COLUMNS) - 0.5, color="white", lw=3)
    _ax.set_xticks(
        range(len(_columns)), [ERROR_METRICS[c][1] for c in _columns], rotation=35, ha="left"
    )
    _ax.xaxis.tick_top()
    _ax.set_yticks(range(len(_ablations)), [ablation_label(a) for a in _ablations])
    _ax.tick_params(length=0)
    for _spine in _ax.spines.values():
        _spine.set_visible(False)
    _bar = fig_overview.colorbar(
        _image, ax=_ax, orientation="horizontal", shrink=0.5, pad=0.03, aspect=30
    )
    _bar.set_label("log2 error ratio to baseline")
    fig_overview
    return (fig_overview,)


@app.cell(hide_code=True)
def _(errors, final, runs):
    def _mean_sd(values):
        values = values.dropna()
        if values.empty:
            return ""
        return f"{values.mean():.3g} ± {values.std():.2g}" if len(values) > 1 else f"{values.mean():.3g}"

    _table = (
        errors.join(final.reindex(columns=[f"{ROLLOUT}/loss"]).rename(columns={f"{ROLLOUT}/loss": "val_loss"}))
        .join(runs.set_index("run_id")[["ablation", "runtime_hours"]])
    )
    _rows = []
    for _ablation, (_label, _category, _comparability) in ABLATIONS.items():
        _group = _table[_table["ablation"] == _ablation]
        _row = {
            "category": _category,
            "ablation": _label,
            "runs": len(_group),
            "comparable": _comparability,
            # The weighted loss only means the same thing for the same objective
            "validation loss": _mean_sd(_group["val_loss"]) if _comparability == "same" else "",
        }
        for _column in LOSS_COLUMNS + VARIATION_COLUMNS:
            _row[ERROR_METRICS[_column][1]] = _mean_sd(_group[_column])
        _row["runtime (h)"] = _mean_sd(_group["runtime_hours"])
        _rows.append(_row)
    summary_table = pd.DataFrame(_rows)
    mo.vstack([
        mo.md("### Final values (mean ± sd over runs)"),
        mo.ui.table(summary_table, selection=None, page_size=25),
    ])
    return (summary_table,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2 · Model and training setup

    - Same objective and validation data as the baseline, so the weighted validation rollout loss is compared directly
    - Top: mean over runs (line) and range (band); bottom: final value per run
    """)
    return


@app.cell(hide_code=True)
def _(final, history, runs):
    _panels = {
        "Model": ["baseline", "model_nca"],
        "Adhesion mask": ["baseline", "mask_uniform"],
        "Regularisers": ["baseline", "no_intermediate_state_reg", "no_boundary_reg", "no_regularisers"],
        "Reinjection curriculum": ["baseline", "curriculum_reinjection_decay"],
    }
    _metric = f"{ROLLOUT}/loss"
    fig_training, _axes = plt.subplots(
        2, len(_panels), figsize=(11, 5.5), sharey="row",
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    _values = final[_metric] if _metric in final.columns else pd.Series(dtype=float)
    for _column, (_title, _ablations) in enumerate(_panels.items()):
        _top, _bottom = _axes[:, _column]
        draw_curves(_top, history, runs, _metric, _ablations)
        _top.set_yscale("log")
        _top.set_title(_title)
        _top.legend(loc="upper right")
        draw_strip(_bottom, _values, runs, _ablations)
    _axes[0, 0].set_ylabel("Validation rollout loss")
    _axes[1, 0].set_ylabel("Final validation rollout loss")
    split_legend(_axes[1, -1], loc="upper right")
    fig_training.tight_layout()
    fig_training
    return (fig_training,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 3 · Loss terms and loss-weight curricula

    - Each panel is one raw (unweighted) loss term on the validation rollouts
    - Filled points: the run trains on that term; open points: it does not
    - Texture is not computed for runs without a texture term
    - Bottom: raw terms during training; the shaded window is where curriculum weights change
    """)
    return


@app.cell(hide_code=True)
def _(errors, history, runs):
    _loss_ablations = [
        "baseline", "loss_texture_only", "loss_radial_only", "loss_l2_only",
        "loss_l2_not_radial", "loss_add_l2", "loss_add_summary",
    ]
    _curriculum_ablations = ["curriculum_static", "curriculum_macro_first"]
    _ablations = _loss_ablations + _curriculum_ablations
    _positions = list(range(len(_loss_ablations))) + [
        len(_loss_ablations) + 0.6 + i for i in range(len(_curriculum_ablations))
    ]
    _colours = [BASELINE_COLOUR if a == "baseline" else SERIES[0] for a in _ablations]

    fig_objective = plt.figure(figsize=(15, 8))
    _top, _bottom = fig_objective.subfigures(2, 1, height_ratios=[1.2, 1])
    _top_axes = _top.subplots(1, len(LOSS_COLUMNS))
    for _ax, _column in zip(_top_axes, LOSS_COLUMNS):
        draw_strip(
            _ax, errors[_column], runs, _ablations, colours=_colours,
            filled=lambda a, c=_column: c in TRAINED_TERMS[a], positions=_positions,
        )
        _ax.set_title(ERROR_METRICS[_column][1])
        _ax.axvline(len(_loss_ablations) - 0.2, color="#d4d3cf", lw=1)
    _top_axes[0].set_ylabel("Final raw validation loss")
    _top.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color="#52514e", label="trained on term"),
            Line2D([], [], ls="", marker="o", color="#52514e", markerfacecolor="none",
                   label="not trained on term"),
            *[
                Line2D([], [], ls="", marker=SPLIT_MARKERS[_split], color="#52514e",
                       markerfacecolor="none", label=SPLIT_LABELS[_split])
                for _split in ("A", "B")
            ],
        ],
        loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
    )

    _curricula = ["baseline", *_curriculum_ablations]
    _curriculum_colours = [BASELINE_COLOUR, SERIES[1], SERIES[2]]
    _bottom_axes = _bottom.subplots(1, 3)
    _last_step = history["_step"].max() if not history.empty else 1
    for _ax, _column in zip(_bottom_axes, ["texture", "radial", "l2"]):
        draw_curves(
            _ax, history, runs, ERROR_METRICS[_column][0], _curricula,
            colours=_curriculum_colours,
        )
        _ax.axvspan(0.30 * _last_step, 0.75 * _last_step, color="#8a8985", alpha=0.08, lw=0)
        _ax.set_yscale("log")
        _ax.set_title(f"{ERROR_METRICS[_column][1]} during training")
    _bottom_axes[0].set_ylabel("Raw validation loss")
    _bottom_axes[0].legend(
        handles=[
            Line2D([], [], color=c, lw=1.5, label=l)
            for c, l in zip(_curriculum_colours, ["Texture first (baseline)", "Static weights", "Macro first"])
        ],
        loc="upper right",
    )
    fig_objective
    return (fig_objective,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 4 · Data: replicates, resolution and projection

    - Replicates: 3 training replicates leave one replicate for validation, 4 leave none (training loss only)
    - Resolution: spatially averaged errors relative to downsample 8 (texture and pixel L2 depend on resolution)
    - Projection: targets differ, so only the scale-free replicate variation metrics are shown
    """)
    return


@app.cell(hide_code=True)
def _(errors, final, runs):
    fig_data, _axes = plt.subplots(2, 3, figsize=(11, 6.5))

    # Replicates
    _counts = {"train_replicates_1": 1, "baseline": 2, "train_replicates_3": 3, "train_replicates_4": 4}
    _replicate_ablations = list(_counts)
    _positions = [_counts[a] for a in _replicate_ablations]
    for _ax, _metric, _label in [
        (_axes[0, 0], f"{ROLLOUT}/loss", "Final validation rollout loss"),
        (_axes[1, 0], "Train/loss", "Final training loss"),
    ]:
        _values = final[_metric] if _metric in final.columns else pd.Series(dtype=float)
        draw_strip(_ax, _values, runs, _replicate_ablations, positions=_positions)
        _ax.set_xticks(_positions, [str(c) for c in _positions], rotation=0, ha="center")
        _ax.set_ylabel(_label)
    _axes[0, 0].set_title("Training replicates")
    _axes[1, 0].set_xlabel("Number of training replicates")
    _axes[0, 0].annotate("1 val. replicate", (3, 0.02), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=6.5, color="#52514e")
    _axes[0, 0].annotate("no validation", (4, 0.02), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=6.5, color="#52514e")

    # Resolution
    _downsample = {"baseline": 8, "downsample_16": 16, "downsample_32": 32}
    _table = errors.join(runs.set_index("run_id")[["ablation", "runtime_hours"]])
    _table = _table[_table["ablation"].isin(_downsample)].assign(
        downsample=lambda t: t["ablation"].map(_downsample)
    )
    _means = _table.groupby("downsample")[["radial", "channel_mean", "correlation"]].mean()
    _ax = _axes[0, 1]
    for _column, _colour in zip(["radial", "channel_mean", "correlation"], SERIES):
        if 8 not in _means.index:
            break
        _relative = _means[_column] / _means.loc[8, _column]
        _ax.plot(_relative.index, _relative, color=_colour, lw=1.5, marker="o",
                 label=ERROR_METRICS[_column][1])
        _ax.scatter(_table["downsample"], _table[_column] / _means.loc[8, _column],
                    s=10, color=_colour, alpha=0.5)
    _ax.axhline(1, color="#d4d3cf", lw=1, ls="--")
    _ax.set_xscale("log", base=2)
    _ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    _ax.set_ylabel("Error relative to downsample 8")
    _ax.set_title("Resolution")
    _ax.legend(loc="upper left")
    _ax = _axes[1, 1]
    for _ablation, _colour in zip(_downsample, ablation_colours(list(_downsample))):
        _rows = _table[_table["ablation"] == _ablation]
        _ax.scatter(_rows["downsample"], _rows["runtime_hours"], color=_colour, s=20)
    _ax.set_xscale("log", base=2)
    _ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    _ax.set_xlabel("Downsample factor")
    _ax.set_ylabel("Wall-clock time (h)")
    _ax.set_ylim(bottom=0)

    # Projection
    _projections = ["baseline", "projection_max", "projection_sum"]
    _labels = ["Slice (baseline)", "Max", "Sum"]
    _variation = final.reindex(columns=[
        ERROR_METRICS["dispersion"][0], ERROR_METRICS["distance_correlation"][0]
    ])
    for _ax, _key, _ylabel, _ideal in [
        (_axes[0, 2], ERROR_METRICS["dispersion"][0], "Dispersion ratio (ideal 1)", 1.0),
        (_axes[1, 2], ERROR_METRICS["distance_correlation"][0], "Distance correlation (ideal 1)", 1.0),
    ]:
        draw_strip(_ax, _variation[_key], runs, _projections)
        _ax.set_xticks(range(3), _labels, rotation=0, ha="center")
        _ax.axhline(_ideal, color="#d4d3cf", lw=1, ls="--")
        _ax.set_ylabel(_ylabel)
    _axes[0, 2].set_title("Projection")
    split_legend(_axes[1, 0], loc="upper right")
    fig_data.tight_layout()
    fig_data
    return (fig_data,)


@app.cell(hide_code=True)
def _():
    save = mo.ui.run_button(label="Save figures and table")
    save
    return (save,)


@app.cell(hide_code=True)
def _(fig_data, fig_objective, fig_overview, fig_training, save, summary_table):
    mo.stop(not save.value)
    _directory = repository_root / "figures" / "micropattern_ablations"
    _directory.mkdir(parents=True, exist_ok=True)
    for _name, _figure in {
        "overview": fig_overview,
        "training_setup": fig_training,
        "objective": fig_objective,
        "data": fig_data,
    }.items():
        _figure.savefig(_directory / f"{_name}.pdf")
        _figure.savefig(_directory / f"{_name}.png")
    summary_table.to_csv(_directory / "summary.csv", index=False)
    mo.md(f"Saved to `{_directory.relative_to(repository_root)}`")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 5 · Model outputs

    - Loads one bundle per ablation from the model registry (the best checkpoint of each run) and rolls it out from 0h on its validation replicates, with the same data splits and scaling as in training
    - 4 training replicates: no validation replicates, so it is rolled out on training data
    - Colours: cyan, magenta, yellow for the three markers of the chosen group
    - Run `python -m Experiments.model_registry reindex` (or press Reindex) after copying bundles locally
    """)
    return


@app.cell(hide_code=True)
def _():
    _backend = jax.default_backend()
    mo.callout(
        f"JAX backend: **{_backend}** · the first rollout of each model shape includes compilation",
        kind="success" if _backend == "gpu" else "warn",
    )
    return


@app.cell(hide_code=True)
def _():
    store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(repository_root / "models")),
        label="Model store", full_width=True,
    )
    data_root = mo.ui.text(
        os.environ.get("DATA_PATH_BASE", ""),
        label="Data root (contains 260726_nca_dataset)", full_width=True,
    )
    experiment_name = mo.ui.text("nca_micropattern_ablations_v1", label="Experiment name")
    reindex = mo.ui.run_button(label="Reindex registry")
    mo.vstack([store_root, data_root, mo.hstack([experiment_name, reindex], justify="start")])
    return data_root, experiment_name, reindex, store_root


@app.cell(hide_code=True)
def _(experiment_name, reindex, store_root):
    _registry = ModelRegistry(Path(store_root.value).expanduser())
    if reindex.value:
        with mo.status.spinner("Reindexing"):
            _registry.reindex()
    with mo.status.spinner("Reading bundle configs"):
        bundles = bundle_table(_registry, experiment_name.value)
    _counts = bundles.groupby("ablation").size().reindex(list(ABLATIONS)).fillna(0).astype(int)
    mo.vstack([
        mo.md(f"{len(bundles)} bundles found for `{experiment_name.value}`"),
        mo.callout(
            "Missing: " + ", ".join(ablation_label(a) for a in _counts.index[_counts == 0]),
            kind="warn",
        ) if (_counts == 0).any() else mo.md(""),
    ])
    return (bundles,)


@app.cell(hide_code=True)
def _():
    repeat_choice = mo.ui.dropdown({"0": 0, "1": 1}, value="0", label="Repeat")
    split_choice = mo.ui.dropdown(
        {SPLIT_LABELS["A"]: "A", SPLIT_LABELS["B"]: "B"},
        value=SPLIT_LABELS["A"], label="Split",
    )
    run_inference = mo.ui.run_button(label="Run inference for missing models")
    mo.hstack([repeat_choice, split_choice, run_inference], justify="start")
    return repeat_choice, run_inference, split_choice


@app.cell(hide_code=True)
def _(bundles, data_root, repeat_choice, run_inference, split_choice):
    # One bundle per ablation; 4 training replicates has no split
    _chosen = bundles[
        (bundles["repeat"] == repeat_choice.value)
        & bundles["split"].isin([split_choice.value, "none"])
    ].set_index("ablation")
    _data_path = (
        str(Path(data_root.value).expanduser() / "260726_nca_dataset")
        if data_root.value.strip() else None
    )
    _missing = [a for a in _chosen.index if _chosen.at[a, "model_id"] not in ROLLOUT_CACHE]
    if run_inference.value and _missing:
        for _ablation in mo.status.progress_bar(_missing, title="Rolling out models"):
            ROLLOUT_CACHE[_chosen.at[_ablation, "model_id"]] = rollout_bundle(
                _chosen.at[_ablation, "path"], _data_path
            )
        _missing = []
    outputs = {
        _ablation: ROLLOUT_CACHE[_chosen.at[_ablation, "model_id"]]
        for _ablation in ABLATIONS
        if _ablation in _chosen.index and _chosen.at[_ablation, "model_id"] in ROLLOUT_CACHE
    }
    mo.md(
        f"{len(outputs)} of {len(_chosen)} selected models rolled out"
        + (f" · press **Run inference** for the other {len(_missing)}" if _missing else "")
    )
    return (outputs,)


@app.cell(hide_code=True)
def _(split_choice):
    _validation = {"A": [3, 4], "B": [1, 2]}[split_choice.value]
    replicate_choice = mo.ui.dropdown(
        {str(r): r for r in _validation}, value=str(_validation[0]), label="Replicate shown",
    )
    triplet_choice = mo.ui.dropdown(
        list(TRIPLETS), value=list(TRIPLETS)[0], label="Markers",
    )
    time_choice = mo.ui.dropdown(
        {"0h": 0, "12h": 1, "24h": 2, "36h": 3, "48h": 4}, value="48h", label="Timepoint",
    )
    masked = mo.ui.checkbox(True, label="Hide outside colony")
    mo.hstack([replicate_choice, triplet_choice, time_choice, masked], justify="start")
    return masked, replicate_choice, time_choice, triplet_choice


@app.cell(hide_code=True)
def _(masked, outputs, replicate_choice, time_choice, triplet_choice):
    mo.stop(not outputs, mo.md("Run inference to see model outputs."))
    _triplet = TRIPLETS[triplet_choice.value]
    _replicate = replicate_choice.value

    def _tile(ablation, kind):
        result = outputs.get(ablation)
        if result is None:
            return None, f"{ablation_label(ablation)}\n(not run)"
        image = output_tile(result, kind, _replicate, time_choice.value, _triplet, masked.value)
        title = ablation_label(ablation) if kind == "prediction" else f"Data · {ablation_label(ablation)}"
        shown = result["replicates"][replicate_index(result, _replicate)]
        if shown != _replicate or result["source"] == "training":
            title += f"\n({result['source']} rep. {shown})"
        return image, title

    # One row per category: data, baseline, then each ablation
    # (preceded by its own data when targets, split or replicate differ)
    _rows = []
    for _category in dict.fromkeys(c for _, c, _ in ABLATIONS.values()):
        if _category == "Reference":
            continue
        _baseline_data, _ = _tile("baseline", "target")
        _tiles = [(_baseline_data, "Data"), _tile("baseline", "prediction")]
        for _ablation, (_, _ablation_category, _) in ABLATIONS.items():
            if _ablation_category != _category:
                continue
            if _ablation in outputs and own_data_needed(_ablation, outputs[_ablation], _replicate):
                _tiles.append(_tile(_ablation, "target"))
            _tiles.append(_tile(_ablation, "prediction"))
        _rows.append((_category, _tiles))

    _columns = max(len(t) for _, t in _rows)
    fig_outputs, _axes = plt.subplots(
        len(_rows), _columns, figsize=(1.55 * _columns, 1.8 * len(_rows)), squeeze=False,
    )
    for _r, (_category, _tiles) in enumerate(_rows):
        for _c in range(_columns):
            _ax = _axes[_r, _c]
            if _c < len(_tiles):
                show_tile(_ax, *_tiles[_c])
            else:
                _ax.set_axis_off()
        _axes[_r, 0].text(
            -0.12, 0.5, _category, transform=_axes[_r, 0].transAxes, rotation=90,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )
    fig_outputs.suptitle(
        f"{triplet_choice.value} · {time_choice.selected_key} · replicate {_replicate}", fontsize=9
    )
    fig_outputs.tight_layout(rect=(0, 0, 1, 0.98))
    fig_outputs
    return (fig_outputs,)


@app.cell(hide_code=True)
def _():
    category_choice = mo.ui.dropdown(
        [c for c in dict.fromkeys(c for _, c, _ in ABLATIONS.values()) if c != "Reference"],
        value="Regularisers", label="Category over time",
    )
    category_choice
    return (category_choice,)


@app.cell(hide_code=True)
def _(category_choice, masked, outputs, replicate_choice, triplet_choice):
    mo.stop(not outputs)
    _triplet = TRIPLETS[triplet_choice.value]
    _replicate = replicate_choice.value
    _times = ["0h", "12h", "24h", "36h", "48h"]

    # Rows: data, baseline, then each ablation of the category (own data first if needed)
    _rows = []
    if "baseline" in outputs:
        _rows += [("Data", "baseline", "target"), ("Baseline", "baseline", "prediction")]
    for _ablation, (_label, _category, _) in ABLATIONS.items():
        if _category != category_choice.value or _ablation not in outputs:
            continue
        if own_data_needed(_ablation, outputs[_ablation], _replicate):
            _rows.append((f"Data · {_label}", _ablation, "target"))
        _rows.append((_label, _ablation, "prediction"))
    mo.stop(not _rows, mo.md("No rolled-out models in this category."))

    fig_category, _axes = plt.subplots(
        len(_rows), len(_times), figsize=(1.6 * len(_times) + 1, 1.6 * len(_rows)), squeeze=False,
    )
    for _r, (_label, _ablation, _kind) in enumerate(_rows):
        for _t, _time in enumerate(_times):
            _image = output_tile(outputs[_ablation], _kind, _replicate, _t, _triplet, masked.value)
            show_tile(_axes[_r, _t], _image, _time if _r == 0 else "")
        _axes[_r, 0].text(
            -0.08, 0.5, _label, transform=_axes[_r, 0].transAxes,
            ha="right", va="center", fontsize=7,
        )
    fig_category.suptitle(f"{category_choice.value} · {triplet_choice.value}", fontsize=9)
    fig_category.tight_layout(rect=(0, 0, 1, 0.97))
    fig_category
    return (fig_category,)


@app.cell(hide_code=True)
def _():
    save_outputs = mo.ui.run_button(label="Save output figures")
    save_outputs
    return (save_outputs,)


@app.cell(hide_code=True)
def _(
    category_choice,
    fig_category,
    fig_outputs,
    save_outputs,
    time_choice,
    triplet_choice,
):
    mo.stop(not save_outputs.value)
    _directory = repository_root / "figures" / "micropattern_ablations"
    _directory.mkdir(parents=True, exist_ok=True)
    _markers = "-".join(TRIPLETS[triplet_choice.value])
    for _name, _figure in {
        f"outputs_{_markers}_{time_choice.selected_key}": fig_outputs,
        f"over_time_{category_choice.value.replace(' ', '_')}_{_markers}": fig_category,
    }.items():
        _figure.savefig(_directory / f"{_name}.pdf")
        _figure.savefig(_directory / f"{_name}.png")
    mo.md(f"Saved to `{_directory.relative_to(repository_root)}`")
    return


if __name__ == "__main__":
    app.run()
