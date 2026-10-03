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


@pytest.mark.parametrize("version", [2, 3])
def test_removed_trainer_options_are_dropped_from_older_versions(version):
    value = _current_micropattern_config()
    value["schema_version"] = version
    value["trainer"]["sharding"] = None
    value["trainer"]["backend"] = "modular"

    upgraded = experiment_config_from_mapping(value)

    assert upgraded == experiment_config_from_mapping(_current_micropattern_config())


def test_current_version_rejects_sharding():
    value = _current_micropattern_config()
    value["trainer"]["sharding"] = None

    with pytest.raises(ValueError, match="sharding"):
        experiment_config_from_mapping(value)


def test_current_config_round_trips():
    config = experiment_config_from_mapping(_current_micropattern_config())

    assert experiment_config_from_mapping(config_to_dict(config)) == config


def test_current_micropattern_config_uses_new_260726_normalisation():
    raw = _current_micropattern_config()["data"]["micropattern"]
    micropattern = experiment_config_from_mapping(
        _current_micropattern_config()
    ).data.micropattern

    # The values are tuned in the data cleaning notebook, so compare with the
    # file rather than fixed numbers.
    assert micropattern.histogram_percentiles == tuple(float(v) for v in raw["histogram_percentiles"])
    assert micropattern.intensity_factors == {
        name: {int(hour): float(value) for hour, value in by_hour.items()}
        for name, by_hour in raw["intensity_factors"].items()
    }
    assert all(isinstance(hour, int) for by_hour in micropattern.intensity_factors.values() for hour in by_hour)
    assert micropattern.cleaning.enabled
    assert micropattern.cleaning.background_radii == raw["cleaning"]["background_radii"]
    assert micropattern.quality_flags_file == raw["quality_flags_file"]


def _without_version_6_fields(micropattern):
    """Remove what schema version 6 added, as an older config would."""
    del micropattern["intensity_factors"]
    del micropattern["cleaning"]


@pytest.mark.parametrize("version", [2, 3, 4])
def test_older_micropattern_configs_keep_their_260726_normalisation(version):
    value = _current_micropattern_config()
    value["schema_version"] = version
    del value["data"]["micropattern"]["histogram_percentiles"]
    _without_version_6_fields(value["data"]["micropattern"])

    micropattern = experiment_config_from_mapping(value).data.micropattern

    assert micropattern.histogram_percentiles == (0.5, 99.95)
    assert micropattern.intensity_factors == {"cell_fate_s2/FOXA2": {0: 0.075}}
    assert not micropattern.cleaning.enabled


def test_version_5_config_moves_0h_scales_and_keeps_cleaning_off():
    value = _current_micropattern_config()
    value["schema_version"] = 5
    _without_version_6_fields(value["data"]["micropattern"])
    value["data"]["micropattern"]["initial_intensity_scales"] = {"cell_fate_s2/FOXA2": 0.02}

    micropattern = experiment_config_from_mapping(value).data.micropattern

    assert micropattern.intensity_factors == {"cell_fate_s2/FOXA2": {0: 0.02}}
    assert not micropattern.cleaning.enabled


def test_version_1_bundle_keeps_its_260726_normalisation():
    value = _old_bundle_layout()
    del value["data"]["augmentation"]["histogram_percentiles"]
    _without_version_6_fields(value["data"]["augmentation"])

    micropattern = experiment_config_from_mapping(value).data.micropattern

    assert micropattern.histogram_percentiles == (0.5, 99.95)
    assert micropattern.intensity_factors == {"cell_fate_s2/FOXA2": {0: 0.075}}
    assert not micropattern.cleaning.enabled
