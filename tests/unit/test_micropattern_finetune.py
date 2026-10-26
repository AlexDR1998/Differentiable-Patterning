"""Fine-tuning parents must match the run's data projection."""

import pytest
from omegaconf import OmegaConf

from Experiments.config import load_experiment_config
from Experiments.micropatterns import train


def _config(projection, excluded=("cell_fate_s2/SOX17",)):
    cfg = OmegaConf.load("Experiments/micropatterns/conf/base_config.yaml")
    cfg.data.dataset = "micropatterns_260726"
    cfg.data.micropattern.data_channels = 14 - len(excluded)
    cfg.data.micropattern.excluded_measurements = list(excluded)
    cfg.data.micropattern.projection = projection
    cfg.initialization.model_id = "parent"
    return load_experiment_config(cfg)


class _Bundle:
    def __init__(self, config):
        self.config = config

    def load_model(self, key):
        return "parent model"


@pytest.fixture
def registry(monkeypatch):
    bundles = {}

    class FakeRegistry:
        def __init__(self, root):
            pass

        def get(self, identifier):
            return bundles[identifier]

    monkeypatch.setattr("Experiments.model_registry.ModelRegistry", FakeRegistry)
    return bundles


@pytest.mark.parametrize("projection", ["max", "sum", "slice"])
def test_parent_on_the_same_projection_is_loaded(registry, projection):
    registry["parent"] = _Bundle(_config(projection))
    model, _ = train.load_initial_model(_config(projection), key=None, model_root="unused")
    assert model == "parent model"


def test_parent_on_another_projection_is_rejected(registry):
    registry["parent"] = _Bundle(_config("max"))
    with pytest.raises(ValueError, match="trained on max projections, but this run uses sum"):
        train.load_initial_model(_config("sum"), key=None, model_root="unused")


def test_run_name_marks_projections_other_than_max():
    names = {
        projection: train.build_run_name(_config(projection), "model", "optimiser")
        for projection in ("max", "sum", "slice")
    }
    assert "_pj" not in names["max"]
    assert "_pjsum_" in names["sum"] and "_pjslice_" in names["slice"]


def test_parent_with_other_excluded_channels_is_rejected(registry):
    registry["parent"] = _Bundle(_config("max", excluded=()))
    with pytest.raises(ValueError, match="same data.micropattern.excluded_measurements"):
        train.load_initial_model(_config("max"), key=None, model_root="unused")
