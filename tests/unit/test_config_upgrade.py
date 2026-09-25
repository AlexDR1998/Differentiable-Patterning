"""Schema-version-1 configs (old sweeps, manifests and saved bundles) still load."""

from pathlib import Path

import pytest
import yaml

from Experiments.config import (
    CONFIG_SCHEMA_VERSION,
    config_to_dict,
    experiment_config_from_mapping,
    upgrade_legacy_config,
)


def _current_micropattern_config():
    return yaml.safe_load(Path("Experiments/micropatterns/conf/base_config.yaml").read_text())


def _old_sweep_layout():
    """A micropattern config written in the version-1 sweep/manifest layout."""
    value = _current_micropattern_config()
    value["schema_version"] = 1
    value["knockout"] = value["data"].pop("knockout")
    pool = value["trainer"].pop("pool_admission")
    for key, item in pool.items():
        value["trainer"][f"pool_admission_{key}"] = item
    value["run"]["warmup"] = value["run"].pop("checkpoint_warmup")
    return value


def _old_bundle_layout():
    """The same config as a version-1 model bundle stored it."""
    value = _current_micropattern_config()
    data = value["data"]
    return {
        "schema_version": 1,
        "seed": value["seed"],
        "experiment": value["experiment"],
        "runtime": value["system"],
        "labels": value["labels"],
        "data": {
            "dataset": data["dataset"],
            "batches": data["batches"],
            "preprocessing": {"steps": [], "downsample": data["downsample"]},
            "augmentation": data["micropattern"],
            "intervention": data["knockout"],
        },
        "model": value["model"],
        "training": {
            "loop": {k: v for k, v in value["run"].items() if k != "checkpoint_warmup"},
            "trainer": value["trainer"],
            "optimizer": value["optimiser"],
            "loss": value["loss"],
            "checkpoint": {"warmup": value["run"]["checkpoint_warmup"]},
        },
        "logging": value["logging"],
        "model_store": value["model_store"],
        "initialization": value["initialization"],
    }


@pytest.mark.parametrize("old_layout", [_old_sweep_layout, _old_bundle_layout])
def test_version_1_layouts_give_the_same_config_as_the_current_layout(old_layout):
    current = experiment_config_from_mapping(_current_micropattern_config())

    assert experiment_config_from_mapping(old_layout()) == current


def test_upgraded_config_is_written_in_the_current_layout():
    upgraded = upgrade_legacy_config(_old_bundle_layout())

    assert upgraded["schema_version"] == CONFIG_SCHEMA_VERSION
    assert {"system", "run", "trainer", "optimiser", "loss"} <= set(upgraded)
    assert not {"runtime", "training", "knockout"} & set(upgraded)
    assert {"micropattern", "knockout", "downsample"} <= set(upgraded["data"])


def test_null_pool_admission_warmup_follows_checkpoint_warmup():
    value = _old_sweep_layout()
    value["run"]["warmup"] = 17
    value["trainer"]["pool_admission_warmup"] = None

    config = experiment_config_from_mapping(value)

    assert config.run.checkpoint_warmup == 17
    assert config.trainer.pool_admission.warmup == 17


def test_old_filename_mode_is_dropped():
    value = _old_sweep_layout()
    value["run"]["filename_mode"] = "hydra"

    assert experiment_config_from_mapping(value).run.t == value["run"]["t"]


def test_old_emoji_regenerate_only_applies_without_explicit_regeneration():
    value = yaml.safe_load(Path("Experiments/emoji/conf/base_config.yaml").read_text())
    value["schema_version"] = 1
    value["data"]["emoji"]["regenerate"] = False

    # As in old manifests: regeneration.enabled was already filled in, so wins.
    assert experiment_config_from_mapping(value).data.emoji.regeneration.enabled is True

    del value["data"]["emoji"]["regeneration"]["enabled"]
    assert experiment_config_from_mapping(value).data.emoji.regeneration.enabled is False


def test_current_layout_rejects_old_keys():
    value = _current_micropattern_config()
    value["knockout"] = value["data"].pop("knockout")

    with pytest.raises(ValueError, match="knockout"):
        experiment_config_from_mapping(value)


def test_current_config_round_trips():
    config = experiment_config_from_mapping(_current_micropattern_config())

    assert experiment_config_from_mapping(config_to_dict(config)) == config
