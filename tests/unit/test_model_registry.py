import pytest
from types import SimpleNamespace

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from NCA.registry import (
    ModelRegistry,
    create_model_id,
    evaluation_input_provenance,
    open_model_bundle,
    publish_model_bundle,
    record_evaluation,
    verify_evaluation_input,
)
from Common.trainer.training_result import TrainingResult
from Experiments.config import experiment_config_from_mapping


def _config():
    value = OmegaConf.to_container(
        OmegaConf.load("Experiments/emoji/conf/base_config.yaml"),
        resolve=True,
    )
    value["seed"] = 7
    value["experiment"]["name"] = "baseline-sweep"
    value["model"]["channels"] = 4
    value["logging"]["wandb"] = {
        "project": "test-project",
        "group": "test-group",
        "tags": ["paper-candidate"],
    }
    return experiment_config_from_mapping(value)


def _sycl_config(family):
    value = OmegaConf.to_container(
        OmegaConf.load("Experiments/emoji/conf/base_config.yaml"),
        resolve=True,
    )
    value["model"]["family"] = family
    value["model"]["channels"] = 4
    value["trainer"]["backend"]["type"] = "sycl"
    return experiment_config_from_mapping(value)


def test_publish_verify_and_index_bundle(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"test-equinox-leaves")
    result = TrainingResult(
        checkpoint_path=checkpoint,
        best_iteration=12,
        best_loss=0.25,
        completed=True,
        wandb_run_id="run-123",
    )

    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="NCA Emoji",
        model_id=create_model_id(_config()),
        display_name="Baseline run",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=result,
        repository_root=tmp_path,
    )

    bundle.verify()
    assert bundle.path.name == bundle.id
    assert bundle.manifest.display_name == "Baseline run"
    assert bundle.id.startswith(f"{bundle.manifest.created_at[:10].replace('-', '')[:8]}")
    assert f"cfg{bundle.manifest.config_id}" in bundle.id
    assert bundle.path.parent.name == "baseline-sweep"
    assert bundle.path.parent.parent.name == "nca-emoji"
    assert open_model_bundle(bundle.path).id == bundle.id
    assert bundle.config.model.family == "NCA"

    registry = ModelRegistry(tmp_path)
    registry.reindex()
    models = registry.models_df()
    assert models.loc[0, "model_id"] == bundle.id
    assert models.loc[0, "experiment"] == "baseline-sweep"
    assert models.loc[0, "best_loss"] == 0.25
    assert models.loc[0, "wandb_run_id"] == "run-123"
    assert "paper-candidate" in set(registry.wandb_tags_df()["tag"])
    assert registry.get(bundle.id).path == bundle.path


def test_get_ignores_unrelated_bundle_with_unsupported_schema(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="z-current",
        model_id=create_model_id(_config()),
        display_name="current-model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
    )
    legacy_path = tmp_path / "bundles" / "a-legacy" / "experiment" / "legacy-model"
    legacy_path.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"schema_version": 1, "id": "legacy-id", "slug": "legacy"}),
        legacy_path / "manifest.yaml",
    )
    OmegaConf.save(OmegaConf.create({}), legacy_path / "config.yaml")

    registry = ModelRegistry(tmp_path)

    assert registry.get(bundle.id).path == bundle.path
    with pytest.raises(ValueError, match="Unsupported bundle schema version 1"):
        registry.get("legacy-id")
    with pytest.warns(UserWarning, match="Skipped 1 legacy model bundle"):
        registry.reindex()
    assert set(registry.models_df()["model_id"]) == {bundle.id}


def test_evaluation_input_provenance_is_compact_and_data_sensitive(tmp_path):
    import numpy as np

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    boundary = np.ones((2, 4), dtype=np.float32)
    provenance = evaluation_input_provenance(data, boundary_mask=boundary)

    assert provenance["kind"] == "data_t0"
    assert provenance["initial_state"]["shape"] == [2, 4]
    assert provenance["boundary_mask"]["shape"] == [2, 4]
    assert len(provenance["initial_state"]["sha256"]) == 64
    assert provenance != evaluation_input_provenance(data + 1, boundary_mask=boundary)
    verify_evaluation_input(data, provenance, boundary_mask=boundary)
    with pytest.raises(ValueError, match="initial state"):
        verify_evaluation_input(data + 1, provenance, boundary_mask=boundary)

    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
        evaluation_input=provenance,
    )
    assert bundle.manifest.evaluation_input.initial_state.sha256 == provenance["initial_state"]["sha256"]


def test_catalogue_is_rebuildable_and_indexes_evaluations(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    result = TrainingResult(checkpoint, 2, 1.5, True)
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=result,
        repository_root=tmp_path,
    )
    evaluation = record_evaluation(
        store_root=tmp_path,
        model_id=bundle.id,
        evaluator="damage recovery",
        dataset="held-out",
        seed=3,
        metrics={"final_l2": 0.125, "recovery_time": 42},
    )

    registry = ModelRegistry(tmp_path)
    first_database = registry.reindex()
    first_database.unlink()
    registry.reindex()

    evaluations = registry.evaluations_df()
    assert evaluation.is_dir()
    assert set(evaluations["metric"]) == {"final_l2", "recovery_time"}
    assert set(evaluations["model_id"]) == {bundle.id}


def test_verify_detects_checkpoint_changes(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
    )
    bundle.checkpoint_path.write_bytes(b"changed")

    try:
        bundle.verify()
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("Expected modified checkpoint to fail verification")


def test_load_model_reconstructs_from_saved_factory(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        model_factory="example.factory:build",
        repository_root=tmp_path,
    )
    calls = {}

    class FakeModel:
        def load(self, path):
            calls["checkpoint"] = path
            return "loaded-model"

    def build(cfg, key=None):
        calls["family"] = cfg.family
        calls["key"] = key
        return FakeModel(), "description"

    monkeypatch.setattr(
        "NCA.registry.importlib.import_module",
        lambda name: SimpleNamespace(build=build),
    )
    key = object()

    assert bundle.load_model(key=key) == "loaded-model"
    assert calls == {
        "family": "NCA",
        "key": key,
        "checkpoint": bundle.checkpoint_path,
    }


@pytest.mark.parametrize(
    ("recorded_family", "portable_module", "portable_class"),
    [
        ("NCA_sycl", "NCA.model.NCA_model_fast", "NCA"),
        ("gNCA_sycl", "NCA.model.NCA_gated_model", "gNCA"),
    ],
)
def test_load_model_portably_reconstructs_sycl_checkpoints(
    tmp_path, recorded_family, portable_module, portable_class
):
    eqx = pytest.importorskip("equinox")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jr = pytest.importorskip("jax.random")
    from Experiments.config_helpers import build_model

    cfg = _sycl_config(recorded_family)
    source, _ = build_model(cfg.model, key=jr.PRNGKey(3))
    checkpoint = tmp_path / "source.eqx"
    source.save(checkpoint)
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="portable-tests",
        model_id=create_model_id(cfg),
        display_name=recorded_family,
        checkpoint_path=checkpoint,
        cfg=cfg,
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
    )

    loaded = bundle.load_model(key=jr.PRNGKey(9), implementation="portable")
    source_arrays = [
        leaf for leaf in jax.tree_util.tree_leaves(source) if eqx.is_array(leaf)
    ]
    loaded_arrays = [
        leaf for leaf in jax.tree_util.tree_leaves(loaded) if eqx.is_array(leaf)
    ]

    assert type(loaded).__module__ == portable_module
    assert type(loaded).__name__ == portable_class
    assert len(source_arrays) == len(loaded_arrays)
    assert all(
        jnp.array_equal(source_leaf, loaded_leaf)
        for source_leaf, loaded_leaf in zip(source_arrays, loaded_arrays)
    )
    state = jr.normal(jr.PRNGKey(10), (cfg.model.channels, 8, 9))
    updated = loaded(state, key=jr.PRNGKey(11))
    assert updated.shape == state.shape
    assert jnp.all(jnp.isfinite(updated))


def test_load_model_rejects_unknown_implementation(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
    )

    with pytest.raises(ValueError, match="implementation"):
        bundle.load_model(implementation="unknown")


def test_annotations_are_separate_and_queryable(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        model_id=create_model_id(_config()),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=TrainingResult(checkpoint, 2, 1.5, True),
        repository_root=tmp_path,
    )
    registry = ModelRegistry(tmp_path)

    registry.annotate(
        bundle.id, alias="baseline", tags=["stable", "paper"], notes="Useful"
    )

    assert registry.get("baseline").id == bundle.id
    assert set(registry.tags_df()["tag"]) == {"stable", "paper"}
    annotation = registry.annotations_df().iloc[0]
    assert annotation["alias"] == "baseline"
    assert annotation["notes"] == "Useful"
    assert not (bundle.path / "annotations.yaml").exists()
