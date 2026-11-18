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


# ---------------------------------------------------------------------------
# check_parent_compatible and the fine-tune sweep writer
# ---------------------------------------------------------------------------

from pathlib import Path

from Experiments.config_workflow import generate_manifest, load_yaml
from Experiments.micropatterns.finetune import (
    FROM_PARENTS,
    check_finetune_sweep,
    check_parent_compatible,
    finetune_sweep,
)

BASE_CONFIG = "Experiments/micropatterns/conf/base_config.yaml"


def _typed(**overrides):
    cfg = OmegaConf.load(BASE_CONFIG)
    cfg.data.dataset = "micropatterns_260726"
    cfg.data.micropattern.data_channels = 13
    cfg.data.micropattern.train_replicates = [1, 2]
    cfg.data.micropattern.validation_replicates = [3, 4]
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return load_experiment_config(cfg)


class _NamedBundle:
    def __init__(self, model_id, config):
        self.id = model_id
        self.config = config


def test_matching_parent_has_no_problems():
    assert check_parent_compatible(_typed(), _typed()) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.downsample", 32),
        ("run.t", 77),
        ("data.micropattern.train_replicates", [3, 4]),
        ("data.micropattern.cleaning.stripe_block", 123),
        ("data.micropattern.histogram_percentiles", [1.0, 99.0]),
        ("model.channels", 7),
    ],
)
def test_each_parent_setting_is_checked(key, value):
    problems = check_parent_compatible(_typed(), _typed(**{key: value}))
    assert len(problems) == 1
    assert key.split(".")[0] in problems[0]


def test_cleaning_problem_names_only_the_differing_setting():
    (problem,) = check_parent_compatible(
        _typed(), _typed(**{"data.micropattern.cleaning.stripe_block": 123})
    )
    assert "stripe_block" in problem and "hot_pixel" not in problem


def _template(**grid):
    return {
        "experiment_name": "finetune_test",
        "entrypoint": "Experiments.micropatterns.train:run",
        "grid": {
            "initialization.model_id": [FROM_PARENTS],
            "data.knockout.curriculum": [["baseline", "ko_0h"], ["ko_0h"]],
            **grid,
        },
    }


def _parents():
    return [
        _NamedBundle("parent-a", _typed(**{"run.iterations": 3000, "run.repeat": 0})),
        _NamedBundle(
            "parent-b",
            _typed(**{
                "run.iterations": 9000,
                "run.repeat": 1,
                "data.micropattern.train_replicates": [3, 4],
                "data.micropattern.validation_replicates": [1, 2],
            }),
        ),
    ]


def test_written_sweep_pairs_each_child_with_its_parent(tmp_path):
    parents = _parents()
    sweep = finetune_sweep(_template(), parents, total_iterations=12000)

    assert check_finetune_sweep(sweep, BASE_CONFIG, parents) == 4
    manifest = generate_manifest(load_yaml(BASE_CONFIG), sweep, tmp_path)
    runs = {}
    for entry in manifest["configs"]:
        config = load_experiment_config(entry["config"])
        runs.setdefault(config.initialization.model_id, []).append(config)
    assert {config.run.iterations for config in runs["parent-a"]} == {9000}
    assert {config.run.iterations for config in runs["parent-b"]} == {3000}
    assert {config.run.repeat for config in runs["parent-b"]} == {1}
    assert {config.data.micropattern.train_replicates for config in runs["parent-b"]} == {(3, 4)}


def test_template_may_not_set_parent_settings():
    with pytest.raises(ValueError, match="come from the parent"):
        finetune_sweep(_template(**{"data.downsample": [8]}), _parents())


def test_template_needs_the_parent_placeholder():
    template = _template()
    template["grid"]["initialization.model_id"] = ["some-id"]
    with pytest.raises(ValueError, match=FROM_PARENTS):
        finetune_sweep(template, _parents())


def test_budget_must_leave_iterations_for_the_child():
    with pytest.raises(ValueError, match="uses up"):
        finetune_sweep(_template(), _parents(), total_iterations=9000)


def test_template_repeats_win_over_parent_repeats(tmp_path):
    parents = _parents()[:1]
    sweep = finetune_sweep(_template(**{"run.repeat": [5, 6]}), parents)
    manifest = generate_manifest(load_yaml(BASE_CONFIG), sweep, tmp_path)
    repeats = {load_experiment_config(e["config"]).run.repeat for e in manifest["configs"]}
    assert repeats == {5, 6}


def test_finetune_templates_are_not_generated_as_sweeps():
    # Folder-level tools only look at conf/experiments/*.yaml.
    templates = Path("Experiments/micropatterns/conf/experiments/finetune_templates")
    for path in templates.glob("*.yaml"):
        assert load_yaml(path)["grid"]["initialization.model_id"] == [FROM_PARENTS]
