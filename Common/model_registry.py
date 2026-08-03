"""Immutable model bundles and a rebuildable, dataframe-friendly catalogue.

Training processes publish independent directories.  ``registry.sqlite`` is a
derived index and can always be recreated by scanning those directories, which
makes publication safe on shared filesystems and convenient from marimo.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from omegaconf import DictConfig, OmegaConf

from Common.trainer.training_result import TrainingResult


BUNDLE_SCHEMA_VERSION = 1
DEFAULT_MODEL_FACTORY = "Experiments.config_helpers:build_model"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:12]}"


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part)[:80] or "model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_fingerprint(value: Any) -> Dict[str, Any]:
    """Describe an array without persisting its potentially large contents."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return {
        "sha256": digest,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def evaluation_input_provenance(
    data: Any,
    *,
    boundary_mask: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fingerprint the canonical data-derived input used for local evaluation.

    Bundles already retain the resolved configuration that tells an evaluator
    how to reload its source dataset.  Persisting compact fingerprints instead
    of the arrays themselves lets a later evaluator verify that the reloaded
    initial state (and, where relevant, boundary mask) is exactly the one seen
    during training.
    """
    import numpy as np

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError(
            "Evaluation input data must have batch and time axes; "
            f"got shape {array.shape}"
        )
    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "data_t0",
        "initial_state": _array_fingerprint(array[:, 0]),
    }
    if boundary_mask is not None:
        provenance["boundary_mask"] = _array_fingerprint(boundary_mask)
    return provenance


def verify_evaluation_input(
    data: Any,
    provenance: Mapping[str, Any],
    *,
    boundary_mask: Optional[Any] = None,
) -> None:
    """Require reloaded evaluation inputs to match a bundle's provenance."""
    if provenance.get("schema_version") != 1 or provenance.get("kind") != "data_t0":
        raise ValueError("Unsupported evaluation input provenance")
    actual = evaluation_input_provenance(data, boundary_mask=boundary_mask)
    for key in ("initial_state", "boundary_mask"):
        if provenance.get(key) != actual.get(key):
            raise ValueError(
                f"Reconstructed {key.replace('_', ' ')} does not match the model bundle"
            )


def _config_container(cfg: Any) -> Dict[str, Any]:
    if OmegaConf.is_config(cfg):
        value = OmegaConf.to_container(cfg, resolve=True)
    else:
        value = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Model bundle configuration must be a mapping")
    return value


def _config_digest(config: Mapping[str, Any]) -> str:
    # Storage location is machine-specific provenance, not model identity.
    identity = json.loads(json.dumps(config, default=str))
    if isinstance(identity.get("model_store"), dict):
        identity["model_store"].pop("root", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_provenance(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    OmegaConf.save(OmegaConf.create(value), path)


@dataclass(frozen=True)
class ModelBundle:
    path: Path
    manifest: DictConfig
    config: DictConfig

    @property
    def id(self) -> str:
        return str(self.manifest.id)

    @property
    def checkpoint_path(self) -> Path:
        return self.path / str(self.manifest.artifact.checkpoint)

    def verify(self) -> None:
        expected = str(self.manifest.artifact.sha256)
        actual = _sha256(self.checkpoint_path)
        if actual != expected:
            raise ValueError(
                f"Checkpoint checksum mismatch for {self.id}: {actual} != {expected}"
            )

    def load_model(self, key=None):
        """Reconstruct the model with its saved factory and load Equinox leaves."""
        self.verify()
        if key is None:
            import jax.random as jr

            key = jr.PRNGKey(0)
        module_name, separator, attribute = str(self.manifest.model.factory).partition(":")
        if not separator:
            raise ValueError("model.factory must use the 'module:function' form")
        factory = getattr(importlib.import_module(module_name), attribute)
        model = factory(self.config, key=key)
        if isinstance(model, tuple):
            model = model[0]
        return model.load(self.checkpoint_path)


def open_model_bundle(path: Union[str, Path]) -> ModelBundle:
    bundle_path = Path(path).expanduser().resolve()
    manifest_path = bundle_path / "manifest.yaml"
    config_path = bundle_path / "config.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Not a model bundle: {bundle_path}")
    manifest = OmegaConf.load(manifest_path)
    if int(manifest.schema_version) != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported bundle schema version {manifest.schema_version}")
    return ModelBundle(bundle_path, manifest, OmegaConf.load(config_path))


def publish_model_bundle(
    *,
    store_root: Union[str, Path],
    collection: str,
    run_name: str,
    checkpoint_path: Union[str, Path],
    cfg: Any,
    training_result: TrainingResult,
    model_factory: str = DEFAULT_MODEL_FACTORY,
    repository_root: Optional[Union[str, Path]] = None,
    evaluation_input: Optional[Mapping[str, Any]] = None,
) -> ModelBundle:
    """Atomically publish one completed checkpoint as an immutable bundle."""
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    config = _config_container(cfg)
    model_id = _identifier()
    slug = _slug(run_name)
    collection_slug = _slug(collection)
    experiment_name = (
        config.get("experiment", {}).get("name")
        or config.get("logging", {}).get("wandb", {}).get("group")
        or "ungrouped"
    )
    experiment_slug = _slug(str(experiment_name))
    parent = (
        Path(store_root).expanduser().resolve()
        / "bundles"
        / collection_slug
        / experiment_slug
    )
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{slug}--{model_id}"
    repo = Path(repository_root or Path(__file__).resolve().parents[1])

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        bundled_checkpoint = staging / "model.eqx"
        shutil.copy2(source, bundled_checkpoint)
        OmegaConf.save(OmegaConf.create(config), staging / "config.yaml")
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "id": model_id,
            "slug": slug,
            "collection": collection,
            "experiment": experiment_name,
            "created_at": _utc_now(),
            "status": "complete" if training_result.completed else "failed",
            "artifact": {
                "checkpoint": bundled_checkpoint.name,
                "sha256": _sha256(bundled_checkpoint),
                "format": "equinox-leaves",
                "format_version": 1,
            },
            "model": {
                "factory": model_factory,
                "family": config.get("model", {}).get("family"),
            },
            "training": {
                **asdict(training_result),
                "checkpoint_path": str(source),
            },
            "data": {
                "dataset": config.get("data", {}).get("dataset"),
                "task": config.get("data", {}).get("task"),
            },
            "provenance": {
                "config_sha256": _config_digest(config),
                "git": _git_provenance(repo),
                "python": sys.version.split()[0],
                "packages": _package_versions(("jax", "equinox", "optax", "diffrax")),
                "wandb": {
                    "project": config.get("logging", {}).get("wandb", {}).get("project"),
                    "group": config.get("logging", {}).get("wandb", {}).get("group"),
                    "run_id": training_result.wandb_run_id,
                },
            },
        }
        if evaluation_input is not None:
            manifest["evaluation_input"] = dict(evaluation_input)
        _write_yaml(staging / "manifest.yaml", manifest)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return open_model_bundle(destination)


class ModelRegistry:
    """Read model bundles and maintain a disposable SQLite search index."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        self.database_path = self.root / "registry.sqlite"
        self.annotations_path = self.root / "annotations.yaml"

    @classmethod
    def from_env(cls, variable: str = "MODEL_STORE_ROOT") -> "ModelRegistry":
        root = os.environ.get(variable)
        if not root:
            raise ValueError(f"{variable} must be set")
        return cls(root)

    def bundle_paths(self) -> Iterable[Path]:
        bundles = self.root / "bundles"
        return sorted(path.parent for path in bundles.glob("*/*/*/manifest.yaml"))

    def reindex(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.database_path.with_suffix(f".sqlite.{uuid.uuid4().hex}.tmp")
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                CREATE TABLE models (
                    model_id TEXT PRIMARY KEY, slug TEXT, collection TEXT,
                    experiment TEXT,
                    path TEXT, created_at TEXT, status TEXT, family TEXT,
                    dataset TEXT, task TEXT, seed INTEGER, best_loss REAL,
                    best_iteration INTEGER, config_sha256 TEXT,
                    checkpoint_sha256 TEXT, git_commit TEXT, git_dirty INTEGER,
                    wandb_project TEXT, wandb_group TEXT, wandb_run_id TEXT
                );
                CREATE TABLE evaluations (
                    evaluation_id TEXT, model_id TEXT, evaluator TEXT,
                    dataset TEXT, metric TEXT, value REAL, seed INTEGER,
                    created_at TEXT, path TEXT
                );
                CREATE TABLE model_annotations (
                    model_id TEXT PRIMARY KEY, alias TEXT, notes TEXT
                );
                CREATE TABLE model_tags (model_id TEXT, tag TEXT);
                CREATE INDEX models_family_idx ON models(family);
                CREATE INDEX models_dataset_idx ON models(dataset);
                CREATE INDEX evaluations_model_idx ON evaluations(model_id);
                CREATE INDEX model_tags_tag_idx ON model_tags(tag);
                """
            )
            for path in self.bundle_paths():
                bundle = open_model_bundle(path)
                manifest = bundle.manifest
                git = manifest.provenance.git
                wandb = manifest.provenance.wandb
                connection.execute(
                    "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(manifest.id), str(manifest.slug), str(manifest.collection),
                        manifest.get("experiment"), str(path),
                        str(manifest.created_at), str(manifest.status),
                        manifest.model.get("family"), manifest.data.get("dataset"),
                        manifest.data.get("task"), bundle.config.get("seed"),
                        manifest.training.get("best_loss"),
                        manifest.training.get("best_iteration"),
                        str(manifest.provenance.config_sha256),
                        str(manifest.artifact.sha256), git.get("commit"),
                        git.get("dirty"), wandb.get("project"), wandb.get("group"),
                        wandb.get("run_id"),
                    ),
                )
            self._index_evaluations(connection)
            self._index_annotations(connection)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, self.database_path)
        return self.database_path

    def _index_evaluations(self, connection: sqlite3.Connection) -> None:
        for path in sorted((self.root / "evaluations").glob("*/*/manifest.yaml")):
            manifest = OmegaConf.load(path)
            for metric, value in manifest.get("metrics", {}).items():
                connection.execute(
                    "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        manifest.id, manifest.model_id, manifest.evaluator,
                        manifest.get("dataset"), metric, float(value),
                        manifest.get("seed"), manifest.created_at, str(path.parent),
                    ),
                )

    def _read_annotations(self) -> DictConfig:
        if not self.annotations_path.is_file():
            return OmegaConf.create({"models": {}})
        annotations = OmegaConf.load(self.annotations_path)
        if "models" not in annotations:
            annotations.models = {}
        return annotations

    def _index_annotations(self, connection: sqlite3.Connection) -> None:
        annotations = self._read_annotations()
        for model_id, values in annotations.models.items():
            connection.execute(
                "INSERT INTO model_annotations VALUES (?,?,?)",
                (model_id, values.get("alias"), values.get("notes")),
            )
            for tag in values.get("tags", []):
                connection.execute(
                    "INSERT INTO model_tags VALUES (?,?)", (model_id, str(tag))
                )

    def _ensure_index(self) -> None:
        if not self.database_path.is_file():
            self.reindex()

    def models_df(self):
        import pandas as pd

        self._ensure_index()
        with sqlite3.connect(self.database_path) as connection:
            return pd.read_sql_query("SELECT * FROM models ORDER BY created_at DESC", connection)

    def evaluations_df(self):
        import pandas as pd

        self._ensure_index()
        with sqlite3.connect(self.database_path) as connection:
            return pd.read_sql_query("SELECT * FROM evaluations", connection)

    def annotations_df(self):
        import pandas as pd

        self._ensure_index()
        with sqlite3.connect(self.database_path) as connection:
            return pd.read_sql_query("SELECT * FROM model_annotations", connection)

    def tags_df(self):
        import pandas as pd

        self._ensure_index()
        with sqlite3.connect(self.database_path) as connection:
            return pd.read_sql_query("SELECT * FROM model_tags", connection)

    def annotate(
        self,
        model_id: str,
        *,
        alias: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """Atomically update local, mutable notebook annotations for a bundle."""
        self.get(model_id)
        annotations = self._read_annotations()
        current = annotations.models.get(model_id, {})
        value = {
            "alias": alias if alias is not None else current.get("alias"),
            "tags": sorted(set(tags if tags is not None else current.get("tags", []))),
            "notes": notes if notes is not None else current.get("notes"),
        }
        annotations.models[model_id] = value
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.annotations_path.with_suffix(f".yaml.{uuid.uuid4().hex}.tmp")
        OmegaConf.save(annotations, temporary)
        os.replace(temporary, self.annotations_path)
        self.reindex()
        return self.annotations_path

    def get(self, identifier: str) -> ModelBundle:
        aliases = {
            str(values.get("alias")): str(model_id)
            for model_id, values in self._read_annotations().models.items()
            if values.get("alias")
        }
        identifier = aliases.get(identifier, identifier)
        matches = []
        for path in self.bundle_paths():
            bundle = open_model_bundle(path)
            if identifier in {bundle.id, str(bundle.manifest.slug), path.name}:
                matches.append(bundle)
        if not matches:
            raise KeyError(f"Unknown model {identifier!r}")
        if len(matches) > 1:
            raise ValueError(f"Model name {identifier!r} is ambiguous; use its ID")
        return matches[0]

    def load(self, identifier: str, key=None):
        return self.get(identifier).load_model(key=key)


def record_evaluation(
    *,
    store_root: Union[str, Path],
    model_id: str,
    evaluator: str,
    metrics: Mapping[str, float],
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write an immutable, indexable evaluation summary."""
    evaluation_id = _identifier()
    parent = Path(store_root).expanduser().resolve() / "evaluations" / _slug(evaluator)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / evaluation_id
    staging = Path(tempfile.mkdtemp(prefix=f".{evaluation_id}.", dir=parent))
    manifest = {
        "schema_version": 1,
        "id": evaluation_id,
        "model_id": model_id,
        "evaluator": evaluator,
        "dataset": dataset,
        "seed": seed,
        "parameters": dict(parameters or {}),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "created_at": _utc_now(),
    }
    try:
        _write_yaml(staging / "manifest.yaml", manifest)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
