from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from Common.dataloader.cell_types import CellTypeRules, load_cell_type_rules
from Experiments.config import load_experiment_config
from Experiments.inverse_design.config import TextureConfig
from Experiments.inverse_design.optimise import (
    baseline_geometries,
    check_projection,
    reference_radius,
    rollout_steps,
    scaled_grid_shape,
    texture_bank,
)
from Experiments.inverse_design.records import load_geometry, publish_inverse_design_record
from NCA.inverse_design import (
    CellTypeShareObjective,
    FourierRadialGeometry,
    OptimisationConfig,
    coordinate_grid,
    optimise_geometry,
)

CELL_TYPES = Path("Experiments/micropatterns/conf/cell_types/260726_cell_types_sliced_nca.yaml")


def _data_config(projection):
    return SimpleNamespace(micropattern=SimpleNamespace(projection=projection))


def test_projection_of_rules_and_model_must_match():
    rules = load_cell_type_rules(CELL_TYPES)

    check_projection(rules, _data_config("slice"))
    with pytest.raises(ValueError, match="'slice'.*'max'"):
        check_projection(rules, _data_config("max"))


def test_rollout_steps_count_whole_steps():
    assert rollout_steps(48, 32) == 128
    with pytest.raises(ValueError):
        rollout_steps(6, 5)


def test_reference_radius_and_scaled_grid():
    y, x = np.mgrid[:101, :101]
    support = (np.hypot(y - 50, x - 50) < 20)[None]

    assert scaled_grid_shape((100, 100), 1.5) == (151, 151)
    assert np.isclose(reference_radius(support, (101, 101)), 20 * 2 / 100, rtol=0.02)


def test_baselines_keep_the_circle_area():
    grid = coordinate_grid((201, 201))
    areas = {
        name: float(jnp.sum(geometry.level_set(grid) < 0))
        for name, geometry in baseline_geometries(0.4, (0.15, 0.3)).items()
    }

    assert len(areas) == 5
    assert all(np.isclose(area, areas["circle"], rtol=0.03) for area in areas.values())


def test_texture_bank_has_one_texture_per_slot_in_state_channel_order():
    rng = np.random.default_rng(0)
    support = np.zeros((2, 31, 31), dtype=bool)
    support[:, 5:26, 5:26] = True
    reference = {
        "measurements": rng.uniform(size=(2, 3, 31, 31)).astype(np.float32),
        "available": np.ones((2, 3), dtype=bool),
        "support": support,
        "schema": SimpleNamespace(
            group_names=("a", "b"),
            group_measurement_indices=((0, 1), (2,)),
            primary_measurements=(2, 0),
        ),
    }

    bank = texture_bank(reference, TextureConfig(bank_size=2, patch_size=5, patch_overlap=2), (21, 21), 0)

    assert bank.shape == (2, 2, 21, 21)
    assert not np.array_equal(bank[0], bank[1])


class _IdentityNCA(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=3)

    def __call__(self, state, boundary_callback, key):
        del key
        return boundary_callback(state)


def test_record_round_trips_the_best_geometry(tmp_path):
    cfg = load_experiment_config(
        yaml.safe_load(Path("Experiments/inverse_design/conf/base_config.yaml").read_text())
    )
    rules = CellTypeRules(
        channels={"TBXT": "TBXT"},
        thresholds={"TBXT": 0.5},
        rules={"Notochord": {"TBXT": "high"}},
    )
    names = ("TBXT", "SOX17")
    objective = CellTypeShareObjective.from_rules(rules, names, "Notochord")
    biology = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 15)[None, None, None, :], (2, 2, 15, 15))
    config = OptimisationConfig(
        iterations=2, learning_rate=0.05, total_steps=1, evaluation_seeds=2, fixed_radius=True
    )
    geometry = FourierRadialGeometry(2, radius=0.5, key=jax.random.PRNGKey(0), initial_scale=0.1)
    result = optimise_geometry(_IdentityNCA(), geometry, biology, jax.random.PRNGKey(1), objective, config)
    bundle = SimpleNamespace(
        id="model", path=tmp_path, manifest=SimpleNamespace(artifact=SimpleNamespace(sha256="abc"))
    )

    path = publish_inverse_design_record(
        store_root=tmp_path,
        run_id="run",
        cfg=cfg,
        bundle=bundle,
        cell_types_path=CELL_TYPES,
        cell_type_rules=rules,
        state_channel_names=names,
        geometry_settings={"harmonics": 2, "radius_range": [0.1, 0.95], "coefficient_limit": 0.35},
        total_steps=1,
        boundary_mode="soft",
        baselines={"circle": (geometry, result.evaluations[0])},
        result=result,
        biology_bank=biology,
    )

    loaded = load_geometry(path)
    summary = yaml.safe_load((path / "summary.yaml").read_text())
    assert jnp.array_equal(loaded.raw_cos, result.best_geometry.raw_cos)
    assert np.isclose(summary["optimised"]["labels"]["Notochord"], result.best_evaluation.mean)
    assert set(np.load(path / "states.npz")) >= {"best_mask", "baseline_masks", "biology_bank"}


class _DecayNCA(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=6)

    def __call__(self, state, boundary_callback, key):
        return boundary_callback(0.95 * state + 0.01 * jax.random.normal(key, state.shape))


def test_run_wires_bundle_rules_textures_and_record(tmp_path, monkeypatch):
    import Experiments.inverse_design.optimise as entrypoint

    value = yaml.safe_load(Path("Experiments/inverse_design/conf/base_config.yaml").read_text())
    value["checkpoint"] = {"model_id": "fake", "store_root": str(tmp_path)}
    value["logging"]["backend"] = "none"
    value["inverse_design"].update(
        iterations=2, batch_size=2, log_every=1, evaluation_seeds=2, baseline_ellipses=[0.15]
    )
    value["inverse_design"]["rollout"].update(hours=12, scan_kind="lax")
    value["inverse_design"]["texture"].update(bank_size=2, patch_size=5, patch_overlap=2)
    cfg = load_experiment_config(value)

    data_config = SimpleNamespace(
        dataset="micropatterns_260726",
        micropattern=SimpleNamespace(projection="slice", noise_strength=0.005),
    )
    bundle = SimpleNamespace(
        id="fake",
        path=tmp_path,
        manifest=SimpleNamespace(artifact=SimpleNamespace(sha256="abc")),
        config=SimpleNamespace(
            data=data_config,
            trainer=SimpleNamespace(boundary_mode="soft"),
            run=SimpleNamespace(t=2),
        ),
        load_model=lambda key=None: _DecayNCA(),
    )
    y, x = np.mgrid[:25, :25]
    support = np.repeat((np.hypot(y - 12, x - 12) < 8)[None], 2, axis=0)
    reference = {
        "measurements": np.random.default_rng(0).uniform(size=(2, 4, 25, 25)).astype(np.float32),
        "available": np.ones((2, 4), dtype=bool),
        "support": support,
        "schema": SimpleNamespace(
            state_channels=("TBXT", "SOX17", "SOX2", "FOXA2"),
            group_names=("s1", "s2"),
            group_measurement_indices=((0, 1, 2), (3,)),
            primary_measurements=(0, 1, 2, 3),
        ),
    }
    monkeypatch.setattr(entrypoint, "open_registry_bundle", lambda model_id, store_root: bundle)
    monkeypatch.setattr(entrypoint, "load_reference", lambda data_config: reference)

    result = entrypoint.run(cfg)

    records = list((tmp_path / "inverse_design").glob("*/*/summary.yaml"))
    assert len(records) == 1
    summary = yaml.safe_load(records[0].read_text())
    assert set(summary["baselines"]) == {"circle", "ellipse_0.15_0deg", "ellipse_0.15_45deg"}
    assert summary["rollout"]["grid_shape"] == [39, 39]
    assert summary["rollout"]["total_steps"] == 2
    assert result.evaluation_iterations == (0, 1, 2)


def test_evaluation_panel_shows_mask_fates_cell_types_and_share_per_rollout():
    from Experiments.inverse_design.wandb_log import evaluation_panel
    from NCA.inverse_design import BinaryEvaluation

    rules = load_cell_type_rules(CELL_TYPES)
    names = ("SOX2", "TBXT", "SOX17", "FOXA2")
    evaluation = BinaryEvaluation(
        mask=np.ones((8, 9)), final_states=np.zeros((3, 5, 8, 9)), scores=np.zeros(3), border_fraction=0.0
    )

    panel = evaluation_panel(evaluation, np.zeros((3, 8, 9)), names, rules, rows=2)

    assert panel.shape == (16, 36, 3)


def test_cell_type_prevalences_sum_to_one_over_the_colony():
    from Experiments.micropatterns.rendering import cell_type_prevalences, render_cell_types

    rules = load_cell_type_rules(CELL_TYPES)
    names = ("SOX2", "TBXT", "SOX17", "FOXA2")
    states = np.random.default_rng(0).uniform(size=(3, 4, 10, 10))
    mask = np.zeros((10, 10))
    mask[2:8, 2:8] = 1.0

    prevalences = cell_type_prevalences(states, names, rules, mask)
    image = render_cell_types(states, names, rules, mask)

    assert np.allclose(sum(prevalences.values()), 1.0)
    assert image.shape == (3, 10, 10, 3)
    assert np.all(image[:, 0, 0] == 0.0)


def test_trajectory_video_ends_on_the_best_evaluation():
    from Experiments.inverse_design.wandb_log import trajectory_video
    from NCA.inverse_design import binary_trajectories, evaluate_binary

    rules = load_cell_type_rules(CELL_TYPES)
    names = ("SOX2", "TBXT", "SOX17", "FOXA2")
    geometry = FourierRadialGeometry(1, radius=0.5)
    biology = jax.random.uniform(jax.random.PRNGKey(0), (2, 4, 11, 11))
    config = OptimisationConfig(
        iterations=1, learning_rate=1.0, total_steps=3, evaluation_seeds=3, noise_strength=0.01
    )
    objective = CellTypeShareObjective.from_rules(rules, (*names, "mask", "hidden"), "Notochord")
    model = _DecayNCA()
    key = jax.random.PRNGKey(4)

    evaluation = evaluate_binary(model, geometry, biology, key, objective, config)
    mask, trajectories = binary_trajectories(model, geometry, biology, key, config, seeds=2)
    video = trajectory_video(mask, trajectories, (*names, "mask", "hidden"), rules)

    assert trajectories.shape == (2, 4, 6, 11, 11)
    assert np.allclose(mask, evaluation.mask)
    assert np.allclose(trajectories[:, -1], np.asarray(evaluation.final_states)[:2], atol=1.0e-5)
    assert video.shape == (4, 3, 22, 22)


def test_cell_type_legend_is_an_rgb_image():
    from Experiments.inverse_design.wandb_log import cell_type_legend

    legend = cell_type_legend(load_cell_type_rules(CELL_TYPES))

    assert legend.ndim == 3 and legend.shape[-1] == 3 and legend.dtype == np.uint8
