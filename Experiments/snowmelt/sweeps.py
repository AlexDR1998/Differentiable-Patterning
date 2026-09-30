"""The snowmelt NCA sweeps, and helpers to compare the models they trained.

Used by ``Experiments/snowmelt/snowmelt_sweep_explorer.py``. Each sweep's models
are found in the model store by its experiment name. Its factors (the config values
that tell its runs apart) are read from each bundle's saved config, and from the
generated manifest for the runs that are expected but missing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Experiments.model_registry import ModelRegistry, _slug, open_model_bundle
from Experiments.snowmelt import evaluation

GENERATED_ROOT = Path(__file__).resolve().parent / "conf" / "generated"


@dataclass(frozen=True)
class Sweep:
    title: str
    # experiment_name of the sweep file, recorded in every bundle it trains
    experiment: str
    # folder of the generated manifest under conf/generated
    manifest: str
    # column name -> dotted config key of each swept factor
    factors: dict


SWEEPS = {
    "resolution": Sweep(
        "Spatiotemporal resolution",
        "nca_snowmelt_resolution_sweep_v2",
        "nca_snowmelt_resolution_sweep",
        {"downsample": "data.downsample", "t": "run.t"},
    ),
    "input": Sweep(
        "Input channels",
        "nca_snowmelt_input_sweep",
        "nca_snowmelt_input_sweep",
        {"targets": "data.snowmelt.target_channels", "static": "data.snowmelt.static_channels"},
    ),
    "architecture": Sweep(
        "Model architecture",
        "nca_snowmelt_architecture_sweep",
        "nca_snowmelt_architecture_sweep",
        {"family": "model.family", "kernels": "model.kernel_str", "channels": "model.channels"},
    ),
    "update_rule": Sweep(
        "Update rule",
        "nca_snowmelt_update_rule_sweep",
        "nca_snowmelt_update_rule_sweep",
        {"fire_rate": "model.fire_rate", "activation": "model.activation"},
    ),
}


def format_value(value):
    """Readable, hashable form of a config value: lists become ``A+B``, an empty list ``none``."""
    if isinstance(value, (list, tuple)):
        return "+".join(map(str, value)) if value else "none"
    return value


def config_value(cfg, dotted_key):
    """Value of a dotted key in a typed config or a plain nested dict."""
    value = cfg
    for part in dotted_key.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    return format_value(value)


def run_factors(cfg, sweep):
    """The sweep factors of one run, plus its repeat and grid resolution in metres."""
    row = {name: config_value(cfg, key) for name, key in sweep.factors.items()}
    row["repeat"] = config_value(cfg, "run.repeat")
    row["resolution_m"] = 10 * config_value(cfg, "data.downsample")
    return row


def run_key(row, sweep):
    """What identifies a run within its sweep: the factor values and the repeat."""
    return tuple(row[name] for name in (*sweep.factors, "repeat"))


def expected_runs(sweep, generated_root=GENERATED_ROOT):
    """One row per entry of the sweep's generated manifest; empty if it has not been generated."""
    path = Path(generated_root) / sweep.manifest / "manifest.yaml"
    if not path.is_file():
        return []
    manifest = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if manifest["experiment_name"] != sweep.experiment:
        raise ValueError(
            f"{path} is for experiment {manifest['experiment_name']!r}, expected {sweep.experiment!r}"
        )
    return [{"index": entry["index"], **run_factors(entry["config"], sweep)} for entry in manifest["configs"]]


def find_bundles(store_root, sweep):
    """Completed bundles of a sweep in a model store.

    Bundles are stored under ``bundles/<collection>/<experiment>/``, so only
    that sweep's folders are opened. This reads the bundles directly instead of
    the registry index, which can be out of date or fail to rebuild because of
    unrelated bundles.
    """
    folder = _slug(sweep.experiment)
    bundles = []
    for path in ModelRegistry(store_root).bundle_paths():
        if path.parent.name != folder:
            continue
        bundle = open_model_bundle(path)
        if bundle.manifest.get("experiment") == sweep.experiment and bundle.manifest.status == "complete":
            bundles.append(bundle)
    return bundles


def training_row(bundle, sweep):
    """Factors and training metrics of one bundle, from its manifest and saved config."""
    training = bundle.manifest.training
    best_iteration = training.get("best_iteration")
    return {
        "model_id": bundle.id,
        **run_factors(bundle.config, sweep),
        "best_loss": training.get("best_loss"),
        "best_iteration": best_iteration,
        # Near 1: the best checkpoint came at the end, so longer training may still help.
        "best_at_fraction": None if best_iteration is None else best_iteration / bundle.config.run.iterations,
        "created_at": str(bundle.manifest.created_at),
        "wandb_run_id": bundle.manifest.provenance.wandb.get("run_id"),
    }


def latest_per_run(rows, sweep):
    """Keep the newest bundle of each run, so a rerun replaces the earlier attempt.

    Returns ``(kept, superseded)``; both are lists of rows.
    """
    newest = {}
    for row in sorted(rows, key=lambda row: row["created_at"]):
        newest[run_key(row, sweep)] = row
    kept = list(newest.values())
    kept_ids = {row["model_id"] for row in kept}
    return kept, [row for row in rows if row["model_id"] not in kept_ids]


def missing_runs(expected, found, sweep):
    """Rows of ``expected`` with no matching run in ``found``."""
    have = {run_key(row, sweep) for row in found}
    return [row for row in expected if run_key(row, sweep) not in have]


def score_channel(channel_names):
    """The channel models are compared on: NDSI when it is a target, else SCA, else the first target."""
    for name in ("NDSI", "SCA"):
        if name in channel_names:
            return name
    return channel_names[0]


def evaluate_bundle(bundle, raw, key, n_rollouts=4, mode="free", grid="full", verify=True,
                    sequences=None, references=None):
    """Roll out one bundle from its own training input and score it against the observations.

    ``raw`` is a ``load_snowmelt`` result. ``grid="full"`` scores on the 10 m
    grid, so models trained at different resolutions are scored on the same
    pixels; ``grid="model"`` scores on the model's own grid. ``sequences`` and
    ``references`` (dicts) share inputs between bundles with the same recipe.

    Returns ``(rows, maps)``. ``rows`` are ``evaluation.score`` rows, one per
    later date and target channel. ``maps`` holds the score channel on the model
    grid, border removed, in physical units: ``observed`` and the rollout
    ``mean`` and ``std`` (each ``[T, h, w]``), and the ``catchment`` mask.
    """
    if grid not in ("full", "model"):
        raise ValueError(f"grid must be 'full' or 'model', got {grid!r}")
    cfg = bundle.config
    recipe = evaluation.input_recipe(cfg)
    names, factor, pad = recipe["target_channels"], recipe["downsample"], recipe["pad"]
    sequence = evaluation.load_bundle_sequence(bundle, raw, cache=sequences, verify=verify)
    raw_prediction = evaluation.predict(
        bundle.load_model(), cfg, sequence, key, n_rollouts=n_rollouts, mode=mode
    )
    prediction = evaluation.to_physical(evaluation.strip_border(raw_prediction, pad), names)
    observed = evaluation.to_physical(evaluation.strip_border(sequence.data[0], pad), names)
    catchment = evaluation.strip_border(sequence.boundary_mask[0, 0], pad) > 0.5

    if grid == "full":
        references = {} if references is None else references
        if (names, factor) not in references:
            full, full_mask = evaluation.full_resolution_reference(raw, names, factor)
            references[(names, factor)] = (evaluation.to_physical(full, names), full_mask)
        score_observed, score_mask = references[(names, factor)]
        score_prediction = evaluation.upsample_blocks(prediction, factor)
    else:
        score_observed, score_mask, score_prediction = observed, catchment, prediction

    rows = evaluation.score(
        score_prediction, score_observed, score_mask, names,
        sequence.dates, sequence.observation_times, mode=mode,
    )
    c = names.index(score_channel(names))
    maps = {
        "channel": names[c],
        "factor": factor,
        "dates": sequence.dates,
        "observed": observed[:, c],
        "mean": prediction[:, :, c].mean(axis=0),
        "std": prediction[:, :, c].std(axis=0),
        "catchment": catchment,
    }
    return rows, maps


def summarise_scores(rows, channel):
    """One model's scores on ``channel``: means over the predicted dates, and the final date.

    ``snow_area_error`` is predicted minus observed snow-covered fraction of the catchment.
    """
    rows = [row for row in rows if row["channel"] == channel]
    if not rows:
        raise ValueError(f"No scores for channel {channel!r}")

    def mean(name):
        return float(np.nanmean([row[name] for row in rows]))

    summary = {
        "score_channel": channel,
        "skill": mean("skill"),
        "rmse": mean("rmse"),
        "bias": mean("bias"),
        "rollout_spread": mean("rollout_spread"),
        "final_skill": rows[-1]["skill"],
        "final_rmse": rows[-1]["rmse"],
    }
    if "snow_csi" in rows[0]:
        errors = [row["predicted_snow_fraction"] - row["observed_snow_fraction"] for row in rows]
        summary |= {
            "snow_csi": mean("snow_csi"),
            "final_snow_csi": rows[-1]["snow_csi"],
            "snow_area_error": float(np.mean(errors)),
            "final_snow_area_error": float(errors[-1]),
        }
    return summary


__all__ = [
    "GENERATED_ROOT",
    "SWEEPS",
    "Sweep",
    "config_value",
    "evaluate_bundle",
    "expected_runs",
    "find_bundles",
    "format_value",
    "latest_per_run",
    "missing_runs",
    "run_factors",
    "run_key",
    "score_channel",
    "summarise_scores",
    "training_row",
]
