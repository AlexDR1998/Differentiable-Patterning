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
        "loss_l2_not_radial": ("Texture + L2", "Loss terms", "objective"),
        "loss_add_l2": ("Texture + radial + L2", "Loss terms", "objective"),
        "loss_add_summary": ("Texture + radial + mean + corr.", "Loss terms", "objective"),
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

    # Shared paper style: baseline in grey, ablations in the fixed line order,
    # better/worse on the diverging map (blue better, red worse)
    from Experiments.micropatterns.figures.style import (
        BAD_COLOUR,
        BASELINE,
        CMAP,
        FIGURE_CHANNELS,
        FONT,
        GROUP_LABELS,
        INK,
        MUTED,
        SERIES,
        STYLE_FILE,
        UNMEASURED,
        legend_options,
        marker_colour,
        save_figure,
    )
    from Experiments.micropatterns.figures.drawing import (
        colony_rgba,
        draw_sectors,
        marker_legend_entries,
        rows_to_columns,
        sliced_rgba,
    )

    BASELINE_COLOUR = BASELINE
    DIVERGING = plt.get_cmap(CMAP["diverging"])
    plt.style.use(["default", STYLE_FILE])

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

    # Output tiles are colony discs with one sector per experiment group,
    # showing that group's markers (FIGURE_CHANNELS) in MARKER_COLOURS
    SECTOR_GROUPS = {GROUP_LABELS[g]: g for g in FIGURE_CHANNELS}

    # Ablations that change the boundary: the toggle in the over-time figure
    # shows them uncropped, to see what they grow outside the colony
    BOUNDARY_ABLATIONS = ("mask_uniform", "no_boundary_reg", "no_regularisers")

    # Loaded data and rollouts, kept between cell runs
    DATA_CACHE = {}
    ROLLOUT_CACHE = {}
    # Rollouts are also saved to disk, one file per model ID (bundles never change).
    # Bump the version after changing rollout_bundle, so old files are not reused.
    ROLLOUT_CACHE_VERSION = 1

    LOSS_NAMES = {"l2": "L2", "texture": "texture", "radial": "radial",
                  "channel_mean": "mean", "correlation": "corr."}
    REGULARISER_NAMES = {"intermediate_state": "state", "boundary": "boundary"}

    def loss_description(loss):
        """Loss terms and regularisers of a typed loss config, e.g. ``("texture + radial", "state, boundary")``."""
        terms = []
        for term in loss.terms:
            weights = getattr(term, "multi_target_weights", None)
            if weights:
                terms += [LOSS_NAMES.get(k, k) for k, w in weights.items() if w > 0]
            elif term.weight > 0:
                terms.append(LOSS_NAMES.get(term.type, term.type))
        regularisers = [REGULARISER_NAMES.get(k, k) for k, w in loss.regularisers.items() if w > 0]
        return " + ".join(terms), ", ".join(regularisers) or "none"

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
                "losses": loss_description(bundle.config.loss),
                "path": model["path"],
            })
        if not rows:
            return pd.DataFrame(
                columns=["model_id", "created_at", "ablation", "repeat", "split", "losses", "path"]
            )
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

    def sector_image(states, names, groups, mask):
        """RGBA disc of state channels ``states [C, H, W]`` (bare marker names), one sector per group."""
        rgbas = [
            colony_rgba(states, mask, [marker_colour(n) if n in FIGURE_CHANNELS[g] else None for n in names])
            for g in groups
        ]
        return sliced_rgba(rgbas)

    def rollout_file(cache_dir, model_id, data_path, seed=0):
        """Disk cache file of one model's rollout (None when caching is off)."""
        if not cache_dir:
            return None
        from Common.dataloader.disk_cache import cache_folder

        settings = {"data_path": data_path, "seed": seed, "version": ROLLOUT_CACHE_VERSION}
        return cache_folder(cache_dir, "ablation_rollouts", settings) / f"{model_id}.npz"

    def save_rollout(file, result):
        temporary = file.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in result.items()})
        temporary.replace(file)

    def load_rollout(file):
        with np.load(file) as saved:
            return {
                "prediction": saved["prediction"],
                "target": saved["target"],
                "mask": saved["mask"],
                "names": tuple(str(n) for n in saved["names"]),
                "replicates": tuple(int(r) for r in saved["replicates"]),
                "source": str(saved["source"]),
                "model_id": str(saved["model_id"]),
            }

    def replicate_index(result, replicate):
        """Index of a physical replicate in a result, or its first replicate if it lacks that one."""
        return result["replicates"].index(replicate) if replicate in result["replicates"] else 0

    def output_tile(result, kind, replicate, time, groups, masked):
        """Sector disc of the data ("target") or the model ("prediction") at one timepoint."""
        index = replicate_index(result, replicate)
        states = result[kind][index, time]
        mask = result["mask"][index] if masked else np.ones(states.shape[-2:])
        return sector_image(states, result["names"], groups, mask)

    def sector_legend(fig, groups, font_scale=1.0):
        """Legend right of the figure: a row per group with its sector and marker colours."""
        names = [f"{g}/{m}" for g in groups for m in FIGURE_CHANNELS[g]]
        handles, labels, ncol = rows_to_columns(marker_legend_entries(names, groups, sectors=groups))
        options = legend_options()
        options["fontsize"] *= font_scale
        fig.legend(handles, labels, ncol=ncol, loc="center left", bbox_to_anchor=(1.0, 0.5), **options)

    def own_data_needed(ablation, result, replicate):
        """Whether an ablation needs its own data panel (targets, replicate or split differ)."""
        return (
            ABLATIONS.get(ablation, ("", "", "data"))[2] in ("data", "validation", "none")
            or replicate not in result["replicates"]
        )

    def show_tile(ax, image, title, sectors=1, outline=None):
        """Draw an image (a sector disc when ``sectors`` > 1, with white lines between sectors).

        ``outline`` is a colony mask drawn as a dashed edge (for uncropped tiles).
        """
        ax.set_axis_off()
        if image is not None:
            ax.imshow(image, interpolation="nearest", extent=(-0.5, 0.5, -0.5, 0.5))
            draw_sectors(ax, (0.0, 0.0), 0.5, [False] * sectors)
            if outline is not None:
                ax.contour(outline, levels=[0.5], extent=(-0.5, 0.5, -0.5, 0.5), origin="upper",
                           colors="white", linewidths=0.6, linestyles="--")
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)

    def split_legend(ax, **kwargs):
        handles = [
            Line2D([], [], ls="", marker=SPLIT_MARKERS[split], color=INK,
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
    _cmap.set_bad(BAD_COLOUR)

    fig_overview, _ax = plt.subplots(figsize=(7.5, 7.5))
    _image = _ax.imshow(
        np.ma.masked_invalid(_values), cmap=_cmap, vmin=-_limit, vmax=_limit, aspect="auto"
    )
    for (_row, _column), _value in np.ndenumerate(_values):
        if np.isfinite(_value):
            _ax.text(
                _column, _row, f"{_value:+.2f}", ha="center", va="center", fontsize=FONT["annotation"],
                color="white" if abs(_value) > 0.6 * _limit else INK,
            )
    for _row, _ablation in enumerate(_ablations):
        if ABLATIONS[_ablation][2] in ("data", "validation"):
            _ax.add_patch(Rectangle(
                (-0.5, _row - 0.5), len(_columns), 1, fill=False,
                hatch="////", edgecolor=UNMEASURED["edge"], lw=0,
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
            ha="left", va="center", fontsize=FONT["label"], color=INK, fontweight="bold",
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
        _ax.axvline(len(_loss_ablations) - 0.2, color=MUTED, lw=1)
    _top_axes[0].set_ylabel("Final raw validation loss")
    _top.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color=INK, label="trained on term"),
            Line2D([], [], ls="", marker="o", color=INK, markerfacecolor="none",
                   label="not trained on term"),
            *[
                Line2D([], [], ls="", marker=SPLIT_MARKERS[_split], color=INK,
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
        _ax.axvspan(0.30 * _last_step, 0.75 * _last_step, color=BASELINE, alpha=0.08, lw=0)
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
                         ha="center", fontsize=FONT["annotation"], color=INK)
    _axes[0, 0].annotate("no validation", (4, 0.02), xycoords=("data", "axes fraction"),
                         ha="center", fontsize=FONT["annotation"], color=INK)

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
    _ax.axhline(1, color=MUTED, lw=1, ls="--")
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
        _ax.axhline(_ideal, color=MUTED, lw=1, ls="--")
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
        save_figure(_figure, _name, "micropattern_ablations")
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
    rollout_cache_dir = mo.ui.text(
        os.environ.get("DATA_CACHE_DIR") or str(repository_root / "output" / "cache"),
        label="Rollout cache (empty: off)", full_width=True,
    )
    experiment_name = mo.ui.text("nca_micropattern_ablations_v1", label="Experiment name")
    reindex = mo.ui.run_button(label="Reindex registry")
    mo.vstack([
        store_root, data_root, rollout_cache_dir,
        mo.hstack([experiment_name, reindex], justify="start"),
    ])
    return data_root, experiment_name, reindex, rollout_cache_dir, store_root


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
def _(
    bundles,
    data_root,
    repeat_choice,
    rollout_cache_dir,
    run_inference,
    split_choice,
):
    # One bundle per ablation; 4 training replicates has no split
    _chosen = bundles[
        (bundles["repeat"] == repeat_choice.value)
        & bundles["split"].isin([split_choice.value, "none"])
    ].set_index("ablation")
    _data_path = (
        str(Path(data_root.value).expanduser() / "260726_nca_dataset")
        if data_root.value.strip() else None
    )
    _cache_dir = rollout_cache_dir.value.strip() or None

    # Rollouts already on disk are loaded without pressing the button
    _from_disk = 0
    for _ablation in _chosen.index:
        _id = _chosen.at[_ablation, "model_id"]
        _file = rollout_file(_cache_dir, _id, _data_path)
        if _id not in ROLLOUT_CACHE and _file is not None and _file.exists():
            ROLLOUT_CACHE[_id] = load_rollout(_file)
            _from_disk += 1

    _missing = [a for a in _chosen.index if _chosen.at[a, "model_id"] not in ROLLOUT_CACHE]
    if run_inference.value and _missing:
        for _ablation in mo.status.progress_bar(_missing, title="Rolling out models"):
            _id = _chosen.at[_ablation, "model_id"]
            ROLLOUT_CACHE[_id] = rollout_bundle(_chosen.at[_ablation, "path"], _data_path)
            _file = rollout_file(_cache_dir, _id, _data_path)
            if _file is not None:
                save_rollout(_file, ROLLOUT_CACHE[_id])
        _missing = []
    outputs = {
        _ablation: ROLLOUT_CACHE[_chosen.at[_ablation, "model_id"]]
        for _ablation in ABLATIONS
        if _ablation in _chosen.index and _chosen.at[_ablation, "model_id"] in ROLLOUT_CACHE
    }
    # (loss terms, regularisers) of each shown model, from its bundle config
    losses = {_ablation: _chosen.at[_ablation, "losses"] for _ablation in outputs}
    mo.md(
        f"{len(outputs)} of {len(_chosen)} selected models rolled out"
        + (f" ({_from_disk} loaded from the rollout cache)" if _from_disk else "")
        + (f" · press **Run inference** for the other {len(_missing)}" if _missing else "")
    )
    return losses, outputs


@app.cell(hide_code=True)
def _(split_choice):
    _validation = {"A": [3, 4], "B": [1, 2]}[split_choice.value]
    replicate_choice = mo.ui.dropdown(
        {str(r): r for r in _validation}, value=str(_validation[0]), label="Replicate shown",
    )
    sector_choice = mo.ui.multiselect(
        SECTOR_GROUPS, value=list(SECTOR_GROUPS), label="Sectors",
    )
    time_choice = mo.ui.dropdown(
        {"0h": 0, "12h": 1, "24h": 2, "36h": 3, "48h": 4}, value="48h", label="Timepoint",
    )
    masked = mo.ui.checkbox(True, label="Hide outside colony")
    mo.hstack([replicate_choice, sector_choice, time_choice, masked], justify="start")
    return masked, replicate_choice, sector_choice, time_choice


@app.cell(hide_code=True)
def _(masked, outputs, replicate_choice, sector_choice, time_choice):
    mo.stop(not outputs, mo.md("Run inference to see model outputs."))
    mo.stop(not sector_choice.value, mo.md("Select at least one sector."))
    _groups = [g for g in SECTOR_GROUPS.values() if g in sector_choice.value]
    _replicate = replicate_choice.value

    def _tile(ablation, kind):
        result = outputs.get(ablation)
        if result is None:
            return None, f"{ablation_label(ablation)}\n(not run)"
        image = output_tile(result, kind, _replicate, time_choice.value, _groups, masked.value)
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
                show_tile(_ax, *_tiles[_c], sectors=len(_groups))
            else:
                _ax.set_axis_off()
        _axes[_r, 0].text(
            -0.12, 0.5, _category, transform=_axes[_r, 0].transAxes, rotation=90,
            ha="right", va="center", fontsize=FONT["label"], fontweight="bold",
        )
    fig_outputs.suptitle(
        f"{time_choice.selected_key} · replicate {_replicate}", fontsize=FONT["title"], x=0.02, ha="left",
    )
    sector_legend(fig_outputs, _groups)
    fig_outputs.tight_layout(rect=(0, 0, 1, 0.98))
    fig_outputs
    return (fig_outputs,)


@app.cell(hide_code=True)
def _():
    category_choice = mo.ui.dropdown(
        [c for c in dict.fromkeys(c for _, c, _ in ABLATIONS.values()) if c != "Reference"],
        value="Regularisers", label="Preset (category)",
    )
    times_choice = mo.ui.multiselect(
        {"0h": 0, "12h": 1, "24h": 2, "36h": 3, "48h": 4},
        value=["48h"], label="Timepoints",
    )
    crop_boundary = mo.ui.checkbox(True, label="Crop boundary ablations to colony")
    show_loss_terms = mo.ui.switch(True, label="Show loss terms")
    show_regularisers = mo.ui.switch(False, label="Show regularisers")
    transpose = mo.ui.switch(False, label="Transpose (conditions as columns)")
    # Layout fine tuning
    font_scale = mo.ui.slider(0.5, 2.0, step=0.05, value=1.0, label="Font scale", show_value=True)
    colony_size = mo.ui.slider(0.6, 3.0, step=0.1, value=1.6, label="Colony size (in)", show_value=True)
    colony_spacing = mo.ui.slider(0.0, 4.0, step=0.1, value=1.0, label="Spacing (font sizes)", show_value=True)
    mo.vstack([
        mo.hstack(
            [category_choice, times_choice, crop_boundary, show_loss_terms, show_regularisers, transpose],
            justify="start", wrap=True,
        ),
        mo.hstack([font_scale, colony_size, colony_spacing], justify="start", wrap=True),
    ])
    return (
        category_choice,
        colony_size,
        colony_spacing,
        crop_boundary,
        font_scale,
        show_loss_terms,
        show_regularisers,
        times_choice,
        transpose,
    )


@app.cell(hide_code=True)
def _(category_choice):
    # A new preset resets the selection to the baseline and that category
    ablation_choice = mo.ui.multiselect(
        {f"{label} ({category})": a for a, (label, category, _) in ABLATIONS.items()},
        value=[
            f"{label} ({category})" for a, (label, category, _) in ABLATIONS.items()
            if a == "baseline" or category == category_choice.value
        ],
        label="Ablations over time", full_width=True,
    )
    ablation_choice
    return (ablation_choice,)


@app.cell(hide_code=True)
def _(
    ablation_choice,
    crop_boundary,
    losses,
    masked,
    outputs,
    replicate_choice,
    sector_choice,
    show_loss_terms,
    show_regularisers,
    times_choice,
    transpose,
    font_scale,
    colony_size,
    colony_spacing,
):
    mo.stop(not outputs or not sector_choice.value)
    _groups = [g for g in SECTOR_GROUPS.values() if g in sector_choice.value]
    _replicate = replicate_choice.value
    _times = [(f"{12 * t}h", t) for t in sorted(times_choice.value)]
    _selected = [a for a in ABLATIONS if a in ablation_choice.value and a in outputs]
    mo.stop(not _selected or not _times, mo.md("Select at least one rolled-out ablation and timepoint."))

    def _details(ablation):
        """Grey lines under a row name: the baseline's loss terms and regularisers,
        and for other rows only what differs from the baseline (and is not already in the name)."""
        terms, regularisers = losses[ablation]
        base_terms, base_regularisers = losses.get("baseline", (None, None))
        category = ABLATIONS[ablation][1]
        lines = []
        if show_loss_terms.value and category != "Loss terms" and terms != base_terms:
            lines.append(terms)
        if show_regularisers.value and category != "Regularisers" and regularisers != base_regularisers:
            lines.append(f"reg: {regularisers}")
        if ablation == "baseline":
            lines = [terms] * show_loss_terms.value + [f"reg: {regularisers}"] * show_regularisers.value
        return "\n".join(lines)

    # Rows: data (from the first selected model), then each selected ablation in sweep order
    # (preceded by its own data when targets, split or replicate differ)
    _rows = [("Data", "", _selected[0], "target")]
    for _ablation in _selected:
        _label = ablation_label(_ablation)
        if _ablation != _selected[0] and own_data_needed(_ablation, outputs[_ablation], _replicate):
            _rows.append((f"Data · {_label}", "", _ablation, "target"))
        _rows.append((_label, _details(_ablation), _ablation, "prediction"))

    # Conditions are rows and timepoints columns, or the other way round when transposed
    _flip = transpose.value
    _font = {_k: _v * font_scale.value for _k, _v in FONT.items()}
    _tile = colony_size.value
    _shape = (len(_times), len(_rows)) if _flip else (len(_rows), len(_times))
    _size = (
        _tile * _shape[1] + (0.6 if _flip else 2.2) * font_scale.value,
        _tile * _shape[0] + (0.6 if _flip else 0.3) * font_scale.value,
    )
    fig_category, _axes = plt.subplots(*_shape, figsize=_size, squeeze=False)
    for _r, (_label, _losses, _ablation, _kind) in enumerate(_rows):
        _result = outputs[_ablation]
        # Boundary ablations follow the crop toggle (with the colony edge dashed when uncropped)
        _uncropped = _ablation in BOUNDARY_ABLATIONS and _kind == "prediction" and not crop_boundary.value
        _crop = masked.value and not _uncropped
        _outline = _result["mask"][replicate_index(_result, _replicate)] if _uncropped else None
        for _c, (_time, _t) in enumerate(_times):
            _ax = _axes[_c, _r] if _flip else _axes[_r, _c]
            _image = output_tile(_result, _kind, _replicate, _t, _groups, _crop)
            show_tile(_ax, _image, "", sectors=len(_groups), outline=_outline)
        if _flip:
            # Condition name above its column, with the grey details between name and image
            _top = _axes[0, _r]
            _lines = _losses.count("\n") + 1 if _losses else 0
            if _losses:
                _top.text(0.5, 1.03, _losses, transform=_top.transAxes, ha="center", va="bottom",
                          fontsize=_font["annotation"], color=MUTED)
            _top.annotate(_label, xy=(0.5, 1.03), xycoords="axes fraction",
                          xytext=(0, _lines * 1.25 * _font["annotation"] + (2 if _lines else 0)),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=_font["label"], fontweight="bold")
        else:
            _axes[_r, 0].text(
                -0.08, 0.5, _label, transform=_axes[_r, 0].transAxes,
                ha="right", va="bottom" if _losses else "center", fontsize=_font["label"], fontweight="bold",
            )
            if _losses:
                _axes[_r, 0].text(
                    -0.08, 0.48, _losses, transform=_axes[_r, 0].transAxes,
                    ha="right", va="top", fontsize=_font["annotation"], color=MUTED,
                )
    # Timepoint labels: above the first row, or left of each row when transposed
    # (left out for a single timepoint, which goes in the title instead)
    for _c, (_time, _) in enumerate(_times if len(_times) > 1 else []):
        if _flip:
            _axes[_c, 0].text(-0.08, 0.5, _time, transform=_axes[_c, 0].transAxes,
                              ha="right", va="center", fontsize=_font["label"])
        else:
            _axes[0, _c].set_title(_time, fontsize=_font["label"])
    fig_category.suptitle(
        f"Replicate {_replicate}"
        + (f" · {_times[0][0]}" if len(_times) == 1 else "")
        + ("" if crop_boundary.value else " · boundary ablations uncropped"),
        fontsize=_font["title"], x=0.02, ha="left",
    )
    sector_legend(fig_category, _groups, font_scale.value)
    fig_category.tight_layout(rect=(0, 0, 1, 0.97), w_pad=colony_spacing.value, h_pad=colony_spacing.value)
    fig_category
    return (fig_category,)


@app.cell(hide_code=True)
def _():
    save_outputs = mo.ui.run_button(label="Save output figures")
    save_outputs
    return (save_outputs,)


@app.cell(hide_code=True)
def _(
    ablation_choice,
    fig_category,
    fig_outputs,
    save_outputs,
    sector_choice,
    time_choice,
):
    mo.stop(not save_outputs.value)
    _directory = repository_root / "figures" / "micropattern_ablations"
    _directory.mkdir(parents=True, exist_ok=True)
    _markers = "-".join(g for g in SECTOR_GROUPS.values() if g in sector_choice.value) or "none"
    _picked = [a for a in ABLATIONS if a in ablation_choice.value]
    _picked = "-".join(_picked) if len(_picked) <= 3 else f"{len(_picked)}_ablations"
    for _name, _figure in {
        f"outputs_{_markers}_{time_choice.selected_key}": fig_outputs,
        f"over_time_{_picked}_{_markers}": fig_category,
    }.items():
        save_figure(_figure, _name, "micropattern_ablations")
    mo.md(f"Saved to `{_directory.relative_to(repository_root)}`")
    return


if __name__ == "__main__":
    app.run()
