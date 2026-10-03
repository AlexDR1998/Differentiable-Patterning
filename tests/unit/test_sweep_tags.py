"""Generated sweeps tag their W&B runs only with the settings that differ."""

from Experiments.config_workflow import distinguishing_keys, generate_manifest


BASE = {
    "seed": 0,
    "model": {"family": "NCA", "channels": 16},
    "run": {"iterations": 100},
    "logging": {"wandb": {"project": "p", "group": "g", "tags": None, "tag_keys": None}},
}


def _tag_keys(manifest):
    return [entry["config"]["logging"]["wandb"]["tag_keys"] for entry in manifest["configs"]]


def test_manifest_tags_only_the_varied_settings(tmp_path):
    sweep = {
        "experiment_name": "sweep",
        "grid": {
            "model.channels": [16, 32],
            "run.iterations": [500],
            "logging.wandb.project": ["p"],
        },
    }
    manifest = generate_manifest(BASE, sweep, tmp_path)

    # run.iterations is set but the same everywhere; seeds and names never count.
    assert _tag_keys(manifest) == [["model.channels"], ["model.channels"]]


def test_branch_only_settings_count_as_distinguishing(tmp_path):
    sweep = {
        "experiment_name": "sweep",
        "grid": {"model.family": ["NCA", "gNCA"]},
        "branches": [{"when": {"model.family": "gNCA"}, "grid": {"model.channels": [8]}}],
    }
    manifest = generate_manifest(BASE, sweep, tmp_path)

    assert _tag_keys(manifest)[0] == ["model.family", "model.channels"]


def test_sweep_can_choose_its_tag_keys(tmp_path):
    sweep = {
        "experiment_name": "sweep",
        "grid": {"model.channels": [16, 32], "logging.wandb.tag_keys": [["run.iterations"]]},
    }
    manifest = generate_manifest(BASE, sweep, tmp_path)

    assert _tag_keys(manifest) == [["run.iterations"], ["run.iterations"]]


def test_distinguishing_keys_compares_nested_values():
    assert distinguishing_keys(
        [
            {"data.timesteps": [0, 12], "x": {"a": 1}, "seed": 0},
            {"data.timesteps": [0, 24], "x": {"a": 1}, "seed": 1},
        ]
    ) == ["data.timesteps"]
