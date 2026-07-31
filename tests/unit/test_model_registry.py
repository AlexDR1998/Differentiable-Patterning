import pytest
from types import SimpleNamespace

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from Common.model_registry import (
    ModelRegistry,
    open_model_bundle,
    publish_model_bundle,
    record_evaluation,
)
from Common.trainer.training_result import TrainingResult


def _config():
    return OmegaConf.create(
        {
            "seed": 7,
            "experiment": {"name": "baseline-sweep"},
            "model": {"family": "NCA", "channels": 4},
            "data": {"dataset": "emojis", "task": "sequence"},
            "logging": {
                "wandb": {"project": "test-project", "group": "test-group"}
            },
        }
    )


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
        run_name="Baseline run",
        checkpoint_path=checkpoint,
        cfg=_config(),
        training_result=result,
        repository_root=tmp_path,
    )

    bundle.verify()
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
    assert registry.get(bundle.id).path == bundle.path


def test_catalogue_is_rebuildable_and_indexes_evaluations(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    result = TrainingResult(checkpoint, 2, 1.5, True)
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        run_name="model",
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
        run_name="model",
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
        run_name="model",
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
        calls["family"] = cfg.model.family
        calls["key"] = key
        return FakeModel(), "description"

    monkeypatch.setattr(
        "Common.model_registry.importlib.import_module",
        lambda name: SimpleNamespace(build=build),
    )
    key = object()

    assert bundle.load_model(key=key) == "loaded-model"
    assert calls == {
        "family": "NCA",
        "key": key,
        "checkpoint": bundle.checkpoint_path,
    }


def test_annotations_are_separate_and_queryable(tmp_path):
    checkpoint = tmp_path / "source.eqx"
    checkpoint.write_bytes(b"checkpoint")
    bundle = publish_model_bundle(
        store_root=tmp_path,
        collection="tests",
        run_name="model",
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
