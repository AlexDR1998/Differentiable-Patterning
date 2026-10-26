import shutil
import sqlite3
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
import yaml
from omegaconf import OmegaConf

from Common.trainer.training_result import TrainingResult
from Experiments.config import experiment_config_from_mapping, impulse_experiment_config_from_mapping
from Experiments.config_helpers import build_model
from Experiments.impulse.config_helpers import build_intervention
from Experiments.impulse.records import create_impulse_id, publish_impulse_record
from Experiments.model_registry import ModelRegistry, create_model_id, publish_model_bundle
from NCA.trainer.impulse import ExternalTargetPairSource, NCAImpulseOptimiser, TargetedObjective


def _publish_bundle(tmp_path, store):
    """A tiny 6-channel NCA bundle trained (nominally) on two emoji patterns."""

    value = yaml.safe_load(Path("Experiments/emoji/conf/base_config.yaml").read_text())
    value["model"]["channels"] = 6
    value["data"]["emoji"]["task"] = "multi_attractor"
    value["data"]["emoji"]["pairs"] = [
        {"initial": "crab.png", "target": "crab.png"},
        {"initial": "lizard.png", "target": "lizard.png"},
    ]
    train_cfg = experiment_config_from_mapping(value)
    model, _ = build_model(train_cfg.model, key=jax.random.PRNGKey(0))
    checkpoint = tmp_path / "trained.eqx"
    eqx.tree_serialise_leaves(checkpoint, model)
    return publish_model_bundle(
        store_root=store,
        collection="tests",
        model_id=create_model_id(train_cfg),
        display_name="model",
        checkpoint_path=checkpoint,
        cfg=train_cfg,
        training_result=TrainingResult(checkpoint, 1, 0.5, True),
        repository_root=tmp_path,
    )


def _impulse_cfg(model_id, experiment="test_impulses"):
    value = yaml.safe_load(Path("Experiments/impulse/conf/base_config.yaml").read_text())
    value["experiment"]["name"] = experiment
    value["checkpoint"]["model_id"] = model_id
    value["impulse"]["intervention"].update(spatial="patch", patch_size=3, duration=2)
    value["impulse"]["rollout"].update(steps=4, evaluation_steps=8, evaluation_frame_every=4)
    return impulse_experiment_config_from_mapping(value)


def _optimise(bundle, cfg):
    """A short impulse optimisation for the bundle's model on random 8 x 8 states."""

    model = bundle.load_model()
    states = jax.random.uniform(jax.random.PRNGKey(1), (2, 6, 8, 8))
    intervention = build_intervention(
        cfg.impulse.intervention, 4, model, states[:, None], jax.random.PRNGKey(2)
    )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=ExternalTargetPairSource(states[:1], states[1:]),
        intervention=intervention,
        objective=TargetedObjective(),
        optimiser=optax.adam(0.05),
        observed_channels=4,
        rollout_steps=4,
        loss_functions=[lambda x, y, **_: jnp.mean((x - y) ** 2, axis=(-3, -2, -1))],
        evaluation_frame_every=4,
    )
    return optimiser.train(iterations=2, batch_size=2, key=jax.random.PRNGKey(3), evaluation_steps=8)


def _publish(store, bundle, cfg, result, wandb_run_id="run-1"):
    return publish_impulse_record(
        store_root=store,
        impulse_id=create_impulse_id(cfg),
        cfg=cfg,
        bundle=bundle,
        data_config=bundle.config.data,
        result=result,
        observed_channels=4,
        wandb_run_id=wandb_run_id,
    )


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("store")
    store = tmp_path / "store"
    bundle = _publish_bundle(tmp_path, store)
    cfg = _impulse_cfg(bundle.id)
    result = _optimise(bundle, cfg)
    record = _publish(store, bundle, cfg, result)
    return store, bundle, cfg, result, record


def test_record_links_its_model_conditions_and_provenance(published):
    store, bundle, cfg, result, record = published

    # The experiment name is made into a slug, as for bundle folders
    assert record.path.parent == store.resolve() / "impulses" / "test-impulses"
    assert sorted(item.name for item in record.path.iterdir()) == [
        "config.yaml", "intervention.eqx", "manifest.yaml", "states.npz",
    ]
    # Only finished records, no leftover staging folders
    assert [item.name for item in record.path.parent.iterdir()] == [record.id]
    manifest = record.manifest
    assert manifest.model.id == bundle.id
    assert manifest.model.checkpoint_sha256 == bundle.manifest.artifact.sha256
    assert (store / manifest.model.path).resolve() == bundle.path
    assert (manifest.conditions.source_pattern, manifest.conditions.target_pattern) == ("crab.png", "lizard.png")
    assert manifest.conditions.data == "bundle"
    assert manifest.results.switch_rate == result.evaluation_switch_rate
    assert manifest.results.pixel_steps == 18
    assert list(manifest.artifact.state_shape) == [6, 8, 8]
    assert set(manifest.provenance.git) == {"commit", "dirty"}
    assert manifest.provenance.wandb.run_id == "run-1"
    assert record.config == cfg
    assert record.verify_model().id == bundle.id


def test_record_rebuilds_its_intervention_and_states(published):
    _, _, _, result, record = published
    states = jnp.asarray(result.initial_states)

    loaded = record.load_intervention()
    saved = record.load_states()

    assert loaded.duration == 2
    assert jnp.array_equal(loaded.deltas(states), result.best_intervention.deltas(states))
    assert jnp.array_equal(saved["final_states"], result.final_states)


def test_each_run_gets_its_own_record(published, tmp_path):
    store, bundle, cfg, result, record = published
    other_store = tmp_path / "other"

    first = _publish(other_store, bundle, cfg, result)
    second = _publish(other_store, bundle, cfg, result)

    assert first.id != second.id
    assert first.manifest.config_id == second.manifest.config_id == record.manifest.config_id


def test_impulse_id_must_come_from_the_config(published, tmp_path):
    _, bundle, cfg, result, _ = published

    with pytest.raises(ValueError, match="create_impulse_id"):
        publish_impulse_record(
            store_root=tmp_path, impulse_id="20260101T000000Z-cfg000000000000-abcdef12", cfg=cfg,
            bundle=bundle, data_config=bundle.config.data, result=result, observed_channels=4,
        )


def test_corrupted_intervention_fails_its_checksum(published, tmp_path):
    record_copy = tmp_path / "record"
    shutil.copytree(published[4].path, record_copy)
    with (record_copy / "intervention.eqx").open("ab") as handle:
        handle.write(b"0")
    from Experiments.impulse.records import open_impulse_record

    with pytest.raises(ValueError, match="checksum"):
        open_impulse_record(record_copy).load_intervention()


def test_registry_indexes_impulses_with_their_tags(published):
    store, bundle, _, _, record = published
    registry = ModelRegistry(store)
    registry.reindex()

    impulses = registry.impulses_df()
    tags = registry.impulse_wandb_tags_df()
    joined = impulses.merge(registry.models_df(), on="model_id", suffixes=("", "_model"))

    assert impulses.impulse_id.tolist() == [record.id]
    assert impulses.iloc[0][["spatial", "patch_size", "duration", "target_pattern"]].tolist() == [
        "patch", 3, 2, "lizard.png",
    ]
    assert "impulse.intervention.patch_size:3" in set(tags.tag)
    assert joined.experiment_model.tolist() == [bundle.manifest.experiment]
    # Filter impulses by a tag of their model and a tag of their own
    with sqlite3.connect(registry.database_path) as connection:
        rows = connection.execute(
            "SELECT i.impulse_id FROM impulses i WHERE "
            "EXISTS (SELECT 1 FROM model_wandb_tags t WHERE t.model_id = i.model_id "
            "AND t.tag = 'data.emoji.task:multi_attractor') AND "
            "EXISTS (SELECT 1 FROM impulse_wandb_tags t WHERE t.impulse_id = i.impulse_id "
            "AND t.tag = 'impulse.intervention.duration:2')"
        ).fetchall()
    assert rows == [(record.id,)]
    assert registry.get_impulse(record.id).id == record.id


def test_registry_keeps_impulses_without_their_model_and_skips_unknown_schemas(published, tmp_path):
    record = published[4]
    store = tmp_path / "store"
    shutil.copytree(record.path, store / "impulses" / "test_impulses" / record.id)
    future = store / "impulses" / "test_impulses" / "future-record"
    shutil.copytree(record.path, future)
    manifest = OmegaConf.load(future / "manifest.yaml")
    manifest.schema_version = 99
    OmegaConf.save(manifest, future / "manifest.yaml")
    registry = ModelRegistry(store)

    with pytest.warns(UserWarning, match="impulse record"):
        registry.reindex()

    assert registry.impulses_df().impulse_id.tolist() == [record.id]
    assert registry.models_df().empty


def test_index_written_before_impulse_records_is_rebuilt(published, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with sqlite3.connect(store / "registry.sqlite") as connection:
        connection.execute("CREATE TABLE models (model_id TEXT)")

    assert ModelRegistry(store).impulses_df().empty
