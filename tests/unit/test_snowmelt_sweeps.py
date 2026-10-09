from types import SimpleNamespace

import jax.random as jr
import numpy as np
import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from Common.dataloader.snowmelt import build_snowmelt_sequence
from Common.trainer.training_result import TrainingResult
from Experiments.config import experiment_config_from_mapping
from Experiments.model_registry import (
    create_model_id,
    evaluation_input_provenance,
    open_model_bundle,
    publish_model_bundle,
)
from Experiments.snowmelt import sweeps
from NCA.model.factory import build_model
from tests.unit.test_snowmelt_data import _synthetic_raw


def test_config_value_reads_typed_configs_and_dicts_alike():
    typed = SimpleNamespace(data=SimpleNamespace(snowmelt=SimpleNamespace(static_channels=())))
    plain = {"model": {"kernel_str": ["ID", "LAP"]}}

    assert sweeps.config_value(typed, "data.snowmelt.static_channels") == "none"
    assert sweeps.config_value(plain, "model.kernel_str") == "ID+LAP"


@pytest.mark.parametrize("name, count", [("resolution", 15), ("input", 40), ("architecture", 60), ("update_rule", 27), ("hold_out", 24)])
def test_generated_manifests_have_one_run_per_factor_combination(name, count):
    sweep = sweeps.SWEEPS[name]

    expected = sweeps.expected_runs(sweep)

    assert len(expected) == count
    assert len({sweeps.run_key(row, sweep) for row in expected}) == count


def test_latest_per_run_and_missing_runs():
    sweep = sweeps.SWEEPS["update_rule"]

    def row(model_id, repeat, created_at):
        return {"model_id": model_id, "fire_rate": 0.5, "activation": "relu", "repeat": repeat, "created_at": created_at}

    kept, superseded = sweeps.latest_per_run([row("old", 0, "1"), row("new", 0, "2"), row("other", 1, "1")], sweep)
    expected = [row(None, repeat, None) for repeat in (0, 1, 2)]

    assert sorted(r["model_id"] for r in kept) == ["new", "other"]
    assert [r["model_id"] for r in superseded] == ["old"]
    assert [r["repeat"] for r in sweeps.missing_runs(expected, kept, sweep)] == [2]


def test_score_channel_prefers_ndsi_then_sca():
    assert sweeps.score_channel(("B3", "NDSI")) == "NDSI"
    assert sweeps.score_channel(("SCA",)) == "SCA"
    assert sweeps.score_channel(("B3", "B11")) == "B3"


def _publish_small_bundle(tmp_path, raw, targets):
    """A snowmelt bundle built from an input-sweep entry, shrunk to the synthetic grid."""
    value = OmegaConf.to_container(OmegaConf.load(
        sweeps.GENERATED_ROOT / "nca_snowmelt_input_sweep" / "manifest.yaml"
    ), resolve=True)["configs"][5]["config"]  # [NDSI] with static [DEM]
    value["data"]["downsample"] = 2
    value["data"]["snowmelt"]["pad"] = 1
    value["data"]["snowmelt"]["exclude_dates"] = []  # the synthetic data has only 3 dates
    value["data"]["snowmelt"]["target_channels"] = list(targets)
    value["model"].update(channels=len(targets) + 5, kernel_str=["ID", "LAP"])
    value["run"]["t"] = 4
    cfg = experiment_config_from_mapping(value)

    sequence = build_snowmelt_sequence(
        target_channels=targets, static_channels=("DEM",), downsample=2, pad=1, raw=raw
    )
    model, _ = build_model(cfg.model, key=jr.PRNGKey(0))
    checkpoint = tmp_path / "model.eqx"
    model.save(checkpoint)
    bundle = publish_model_bundle(
        store_root=tmp_path / "store",
        collection="tests",
        model_id=create_model_id(cfg),
        display_name="snowmelt",
        checkpoint_path=checkpoint,
        cfg=cfg,
        training_result=TrainingResult(checkpoint, 30, 0.01, True),
        repository_root=tmp_path,
        evaluation_input=evaluation_input_provenance(sequence.data, boundary_mask=sequence.boundary_mask),
    )
    return open_model_bundle(bundle.path)


def test_find_bundles_reads_only_the_sweep_folder(tmp_path):
    bundle = _publish_small_bundle(tmp_path, _synthetic_raw(T=3), ("NDSI",))

    found = sweeps.find_bundles(tmp_path / "store", sweeps.SWEEPS["input"])

    assert [b.id for b in found] == [bundle.id]
    assert sweeps.find_bundles(tmp_path / "store", sweeps.SWEEPS["architecture"]) == []


@pytest.mark.parametrize("grid", ["full", "model"])
def test_evaluate_bundle_scores_the_ndsi_channel(tmp_path, grid):
    raw = _synthetic_raw(T=3)
    bundle = _publish_small_bundle(tmp_path, raw, ("NDSI", "B3"))
    sweep = sweeps.SWEEPS["input"]

    rows, maps = sweeps.evaluate_bundle(bundle, raw, jr.PRNGKey(1), n_rollouts=2, grid=grid)
    summary = sweeps.summarise_scores(rows, maps["channel"])
    training = sweeps.training_row(bundle, sweep)

    assert {row["channel"] for row in rows} == {"NDSI", "B3"}
    assert len(rows) == 2 * 2  # two later dates, two channels
    assert maps["channel"] == "NDSI"
    assert maps["observed"].shape == maps["mean"].shape == (3, 4, 4)
    np.testing.assert_allclose(maps["mean"][0], maps["observed"][0], atol=1e-6)  # rollouts start from the data
    assert np.isfinite(summary["skill"]) and 0.0 <= summary["snow_csi"] <= 1.0
    assert training["targets"] == "NDSI+B3" and training["static"] == "DEM"
    assert training["best_loss"] == 0.01 and training["best_at_fraction"] == 30 / bundle.config.run.iterations
