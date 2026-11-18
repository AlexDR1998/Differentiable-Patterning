from Experiments.micropatterns.final_losses import _rename_losses


def test_rename_losses_keeps_losses_and_drops_variation():
    metrics = {
        "validation/loss": 1.0,
        "validation/loss_component_raw/texture": 2.0,
        "validation/variation/spread": 3.0,
        "validation_rollout/loss": 4.0,
    }
    assert _rename_losses(metrics) == {
        "per_transition/loss": 1.0,
        "per_transition/loss_component_raw/texture": 2.0,
        "rollout/loss": 4.0,
    }


def test_job_parts_round_trip(tmp_path, monkeypatch):
    from Experiments.micropatterns import final_losses as module
    from Experiments import model_registry

    configs = module.job_configs(["a", "b", "c"], 7, tmp_path / "out", models_per_part=2)
    assert [c["model_ids"] for c in configs] == [["a", "b"], ["c"]]

    class Registry:
        def __init__(self, root):
            pass

        def get(self, model_id):
            return model_id

    def fake_final_losses(bundle, data_root, seed):
        return {"replicates": {"train": (1, 2), "validation": (3, 4)},
                "train": {"rollout/loss": 1.0}, "validation": {"rollout/loss": 2.0 + len(bundle)}}

    monkeypatch.setattr(model_registry, "ModelRegistry", Registry)
    monkeypatch.setattr(module, "final_losses", fake_final_losses)
    monkeypatch.delenv("MODEL_STORE_ROOT", raising=False)
    monkeypatch.delenv("DATA_PATH_BASE", raising=False)
    for index, config in enumerate(configs):
        module.run_job_part(config, index)
    loaded = module.load_final_losses(tmp_path / "out")
    assert sorted(loaded) == ["a", "b", "c"]
    assert loaded["c"]["seed"] == 7 and loaded["c"]["validation"]["rollout/loss"] == 3.0
    assert list(loaded["a"]["replicates"]["validation"]) == [3, 4]
