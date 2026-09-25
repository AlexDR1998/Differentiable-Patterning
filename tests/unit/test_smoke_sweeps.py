"""The pipeline smoke sweeps must always expand into valid typed configs."""

from pathlib import Path

import pytest

from Experiments.config import load_experiment_config
from Experiments.config_workflow import generate_manifest, load_config_from_entry, load_yaml
from Experiments.pipeline_smoke_test import PARENT_PLACEHOLDER, SMOKE_COLLECTION, STAGES

REPO_ROOT = Path(__file__).resolve().parents[2]


def _typed_configs(domain, sweep_name, tmp_path):
    conf = REPO_ROOT / "Experiments" / domain / "conf"
    sweep = load_yaml(conf / "experiments" / f"{sweep_name}.yaml")
    if PARENT_PLACEHOLDER in sweep["grid"].get("initialization.model_id", []):
        sweep["grid"]["initialization.model_id"] = ["some-parent-id"]
    manifest = generate_manifest(load_yaml(conf / "base_config.yaml"), sweep, tmp_path / sweep_name)
    return [load_experiment_config(load_config_from_entry(entry)) for entry in manifest["configs"]]


@pytest.mark.parametrize("domain,sweep_name", STAGES)
def test_smoke_sweep_is_short_and_publishes(domain, sweep_name, tmp_path):
    configs = _typed_configs(domain, sweep_name, tmp_path)
    assert configs
    for cfg in configs:
        loop = cfg.training.loop
        assert loop.iterations <= 100
        # A checkpoint is only saved after the warmup, and bundles need one.
        assert cfg.training.checkpoint.warmup < loop.iterations
        assert cfg.model_store.enabled
        assert cfg.model_store.collection == SMOKE_COLLECTION


def test_finetune_sweep_starts_from_a_parent(tmp_path):
    for cfg in _typed_configs("micropatterns", "smoke_micropatterns_ko_finetune", tmp_path):
        assert cfg.initialization.model_id is not None
        assert cfg.data.intervention.curriculum is not None
