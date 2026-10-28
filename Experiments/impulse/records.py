"""Impulse records: optimised interventions saved in the model store.

Each run of ``Experiments/impulse/optimise.py`` writes one record next to the
model bundles it perturbs (see ``docs/model_registry.md``):

    <store>/impulses/<experiment>/<impulse-id>/
        intervention.eqx   best intervention
        config.yaml        resolved impulse config
        manifest.yaml      model link, conditions, results and provenance
        states.npz         start, target and final states, saved rollouts

Records are never edited once written. ``ModelRegistry.reindex`` adds them to
``registry.sqlite`` (tables ``impulses`` and ``impulse_wandb_tags``).
"""

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from Experiments.config import config_to_dict, load_experiment_config
from Experiments.config_helpers import build_wandb_tags
from Experiments.impulse.config_helpers import intervention_from_config, pixel_steps
from Experiments.model_registry import (
    CONFIG_ID_LENGTH,
    _config_digest,
    _git_provenance,
    _identifier,
    _package_versions,
    _sha256,
    _slug,
    _utc_now,
    _write_yaml,
)
from NCA.trainer.impulse.regularisers import intervention_metrics

IMPULSE_SCHEMA_VERSION = 1
INTERVENTION_FILE = "intervention.eqx"
STATES_FILE = "states.npz"
INTERVENTION_BUILDER = "Experiments.impulse.config_helpers:intervention_from_config"


def _impulse_digest(config):
    """Config hash of an impulse run; the store location is not part of its identity."""

    identity = dict(config)
    if isinstance(identity.get("checkpoint"), dict):
        identity["checkpoint"] = {
            key: value for key, value in identity["checkpoint"].items() if key != "store_root"
        }
    return _config_digest(identity)


def create_impulse_id(cfg):
    """ID for one impulse run: timestamp, config hash and a random part."""

    return _identifier(_impulse_digest(config_to_dict(cfg)))


def _mean_switch_time(switch_steps):
    """Mean of the finite switch steps, or None when no rollout switched."""
    if switch_steps is None or not np.isfinite(switch_steps).any():
        return None
    return float(np.nanmean(switch_steps))


def impulse_size(intervention, states):
    """Size of an intervention applied to ``states``: energy, peak and footprint.

    ``footprint`` counts the (step, pixel) pairs whose largest channel change is
    over 1% of the peak, so a Gaussian impulse and a patch can be compared.
    """

    deltas = intervention.deltas(states)
    metrics = intervention_metrics(states, deltas)
    strength = jnp.max(jnp.abs(deltas), axis=(1, 2))
    return {
        "impulse_l2": float(metrics["l2"]),
        "impulse_linf": float(metrics["linf"]),
        "footprint": int(jnp.sum(strength > 0.01 * jnp.max(strength))),
    }


def _pattern_name(data_config, index):
    """Target image of ``data.emoji.pairs[index]`` (None for data without pairs)."""

    emoji = getattr(data_config, "emoji", None)
    pairs = getattr(emoji, "pairs", ()) or ()
    return str(pairs[index].target) if 0 <= index < len(pairs) else None


def _relative_to(path, root):
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class ImpulseRecord:
    path: Path
    manifest: DictConfig
    config: Any

    @property
    def id(self):
        return str(self.manifest.id)

    @property
    def model_id(self):
        return str(self.manifest.model.id)

    @property
    def intervention_path(self):
        return self.path / str(self.manifest.artifact.intervention)

    def verify(self):
        """Check the intervention file against its recorded checksum."""

        expected = str(self.manifest.artifact.sha256)
        actual = _sha256(self.intervention_path)
        if actual != expected:
            raise ValueError(f"Intervention checksum mismatch for {self.id}: {actual} != {expected}")

    def load_intervention(self):
        """Rebuild the intervention from its config and saved shapes, then load its values."""

        self.verify()
        artifact = self.manifest.artifact
        skeleton = intervention_from_config(
            self.config.impulse.intervention,
            int(artifact.observed_channels),
            tuple(int(size) for size in artifact.state_shape),
            jax.random.PRNGKey(0),
        )
        return eqx.tree_deserialise_leaves(self.intervention_path, skeleton)

    def load_states(self):
        """The saved states and rollouts (``states.npz``) as a dictionary of arrays."""

        with np.load(self.path / STATES_FILE) as arrays:
            return {name: arrays[name] for name in arrays.files}

    def model_bundle(self, registry=None):
        """The model bundle this impulse was optimised for (by default from the record's own store)."""

        from Experiments.model_registry import ModelRegistry

        # <store>/impulses/<experiment>/<impulse-id>
        registry = registry or ModelRegistry(self.path.parents[2])
        return registry.get(self.model_id)

    def verify_model(self, registry=None):
        """Return the linked bundle after checking it holds the exact weights the impulse used."""

        bundle = self.model_bundle(registry)
        bundle.verify()
        expected = str(self.manifest.model.checkpoint_sha256)
        actual = str(bundle.manifest.artifact.sha256)
        if actual != expected:
            raise ValueError(
                f"Impulse {self.id} was optimised for checkpoint {expected}, "
                f"but bundle {bundle.id} holds {actual}"
            )
        return bundle


def open_impulse_record(path):
    record_path = Path(path).expanduser().resolve()
    manifest = OmegaConf.load(record_path / "manifest.yaml")
    version = int(manifest.get("schema_version", 0))
    if version != IMPULSE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported impulse record schema version {version}")
    config = load_experiment_config(OmegaConf.load(record_path / "config.yaml"))
    return ImpulseRecord(record_path, manifest, config)


def publish_impulse_record(
    *,
    store_root,
    impulse_id,
    cfg,
    bundle,
    data_config,
    result,
    observed_channels,
    wandb_run_id=None,
    repository_root=None,
):
    """Write a finished impulse run as a new record in ``<store_root>/impulses/``.

    The files are written to a hidden folder that is renamed into place at the
    end, so a record is either complete or absent.
    """

    store_root = Path(store_root).expanduser().resolve()
    config = config_to_dict(cfg)
    config_sha256 = _impulse_digest(config)
    if f"cfg{config_sha256[:CONFIG_ID_LENGTH]}" not in impulse_id:
        raise ValueError("impulse_id must be created from this configuration by create_impulse_id()")
    experiment = cfg.experiment.name
    parent = store_root / "impulses" / _slug(str(experiment))
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / impulse_id
    repo = Path(repository_root or Path(__file__).resolve().parents[2])
    impulse = cfg.impulse
    initial_states = jnp.asarray(result.initial_states)

    staging = Path(tempfile.mkdtemp(prefix=f".{impulse_id}.", dir=parent))
    try:
        eqx.tree_serialise_leaves(staging / INTERVENTION_FILE, result.best_intervention)
        np.savez_compressed(
            staging / STATES_FILE,
            initial_states=np.asarray(result.initial_states),
            target_states=np.asarray(result.target_states),
            perturbed_initial_states=np.asarray(result.perturbed_initial_states),
            final_states=np.asarray(result.final_states),
            baseline_trajectory=np.asarray(result.baseline_trajectory),
            perturbed_trajectory=np.asarray(result.perturbed_trajectory),
            # Centre (row, column, 0 to 1) at each optimisation step; local and patch only
            **({} if result.location_history is None
               else {"location_history": np.asarray(result.location_history)}),
            # Per rollout: step from which it stays switched (NaN: did not switch)
            **({} if result.switch_steps is None else {"switch_steps": np.asarray(result.switch_steps)}),
            # Mean target loss of every saved evaluation frame
            **({} if result.evaluation_loss_curve is None else {
                "evaluation_loss_curve": np.asarray(result.evaluation_loss_curve),
                "baseline_loss_curve": np.asarray(result.baseline_loss_curve),
            }),
        )
        OmegaConf.save(OmegaConf.create(config), staging / "config.yaml")
        manifest = {
            "schema_version": IMPULSE_SCHEMA_VERSION,
            "id": impulse_id,
            "config_id": config_sha256[:CONFIG_ID_LENGTH],
            "experiment": experiment,
            "created_at": _utc_now(),
            "status": "complete",
            "artifact": {
                "intervention": INTERVENTION_FILE,
                "sha256": _sha256(staging / INTERVENTION_FILE),
                "format": "equinox-leaves",
                "builder": INTERVENTION_BUILDER,
                "state_shape": [int(size) for size in initial_states.shape[1:]],
                "observed_channels": int(observed_channels),
                "states": STATES_FILE,
            },
            "model": {
                "id": bundle.id,
                "config_id": str(bundle.manifest.config_id),
                "collection": str(bundle.manifest.collection),
                "experiment": bundle.manifest.get("experiment"),
                "family": bundle.manifest.model.get("family"),
                "checkpoint_sha256": str(bundle.manifest.artifact.sha256),
                "path": _relative_to(bundle.path, store_root),
            },
            "conditions": {
                "pair_source": impulse.pair_source.type,
                "source_index": impulse.pair_source.source_index,
                "target_index": impulse.pair_source.target_index,
                "source_pattern": _pattern_name(data_config, impulse.pair_source.source_index),
                "target_pattern": _pattern_name(data_config, impulse.pair_source.target_index),
                # bundle: the data the model was trained on; override: cfg.data
                "data": "bundle" if cfg.data is None else "override",
            },
            "results": {
                "best_step": int(result.best_step),
                "best_loss": float(result.best_loss),
                "evaluation_steps": impulse.rollout.evaluation_steps,
                "evaluation_loss": float(result.evaluation_loss),
                "baseline_evaluation_loss": float(result.baseline_evaluation_loss),
                "switch_rate": float(result.evaluation_switch_rate),
                "baseline_switch_rate": float(result.baseline_switch_rate),
                # Mean NCA step of switching, over the rollouts that switched
                "switch_time": _mean_switch_time(result.switch_steps),
                **impulse_size(result.best_intervention, initial_states[:1]),
                "pixel_steps": pixel_steps(impulse.intervention),
            },
            "provenance": {
                "config_sha256": config_sha256,
                "git": _git_provenance(repo),
                "python": sys.version.split()[0],
                "packages": _package_versions(("jax", "equinox", "optax")),
                "wandb": {
                    "project": cfg.logging.wandb.project,
                    "group": cfg.logging.wandb.group,
                    "run_id": wandb_run_id,
                    "tags": build_wandb_tags(cfg),
                },
            },
        }
        _write_yaml(staging / "manifest.yaml", manifest)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return open_impulse_record(destination)
