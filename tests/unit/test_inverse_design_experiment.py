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

    bank = texture_bank(reference, TextureConfig(bank_size=2, patch_size=5, patch_overlap=2), (21, 21), 0, 8)

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


def _fake_run_setup(tmp_path, monkeypatch, density_mode="none", texture_file=None):
    import Experiments.inverse_design.optimise as entrypoint

    value = yaml.safe_load(Path("Experiments/inverse_design/conf/base_config.yaml").read_text())
    value["checkpoint"] = {"model_id": "fake", "store_root": str(tmp_path)}
    value["logging"]["backend"] = "none"
    value["inverse_design"].update(
        iterations=2, batch_size=2, log_every=1, evaluation_seeds=2, baseline_ellipses=[0.15]
    )
    value["inverse_design"]["rollout"].update(hours=12, scan_kind="lax")
    if texture_file is None:
        value["inverse_design"]["texture"].update(file=None, bank_size=2, patch_size=5, patch_overlap=2)
    else:
        value["inverse_design"]["texture"].update(file=str(texture_file), bank_size=2)
    value["inverse_design"]["density"].update(mode=density_mode, baselines=[0.5])
    cfg = load_experiment_config(value)

    data_config = SimpleNamespace(
        dataset="micropatterns_260726",
        downsample=8,
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
    monkeypatch.setattr(entrypoint, "open_registry_bundle", lambda model_id, store_root: bundle)
    monkeypatch.setattr(entrypoint, "load_reference", lambda data_config: _fake_reference())
    return entrypoint, cfg


def _fake_reference():
    y, x = np.mgrid[:25, :25]
    support = np.repeat((np.hypot(y - 12, x - 12) < 8)[None], 2, axis=0)
    names = ("TBXT", "SOX17", "SOX2", "FOXA2")
    return {
        "measurements": np.random.default_rng(0).uniform(size=(2, 4, 25, 25)).astype(np.float32),
        "available": np.ones((2, 4), dtype=bool),
        "support": support,
        "replicates": (1, 2),
        "schema": SimpleNamespace(
            state_channels=names,
            measurement_channels=tuple(SimpleNamespace(marker=name) for name in names),
            group_names=("s1", "s2"),
            group_measurement_indices=((0, 1, 2), (3,)),
            primary_measurements=(0, 1, 2, 3),
        ),
    }


def test_run_wires_bundle_rules_textures_and_record(tmp_path, monkeypatch):
    entrypoint, cfg = _fake_run_setup(tmp_path, monkeypatch)

    result = entrypoint.run(cfg)

    records = list((tmp_path / "inverse_design").glob("*/*/summary.yaml"))
    assert len(records) == 1
    summary = yaml.safe_load(records[0].read_text())
    assert set(summary["baselines"]) == {"circle", "ellipse_0.15_0deg", "ellipse_0.15_45deg"}
    assert summary["rollout"]["grid_shape"] == [39, 39]
    assert summary["rollout"]["total_steps"] == 2
    assert summary["density"] is None
    assert result.evaluation_iterations == (0, 1, 2)
    assert result.best_density is None


def test_run_with_density_learns_it_and_records_it(tmp_path, monkeypatch):
    from Experiments.inverse_design.records import load_density

    entrypoint, cfg = _fake_run_setup(tmp_path, monkeypatch, density_mode="optimise")

    result = entrypoint.run(cfg)

    record = next((tmp_path / "inverse_design").glob("*/*/summary.yaml")).parent
    summary = yaml.safe_load((record / "summary.yaml").read_text())
    assert "circle_density_0.5x" in summary["baselines"]
    assert summary["baselines"]["circle_density_0.5x"]["density"] < summary["baselines"]["circle"]["density"]
    assert np.isclose(summary["density"]["best"], float(result.best_density.value))
    assert len(summary["density"]["training_colonies"]) == 4
    assert np.isclose(float(load_density(record).value), summary["density"]["best"])
    assert set(np.load(record / "states.npz")) >= {"gap_values", "channel_panels", "layer_coverage"}


def test_density_config_converts_and_rejects_unknown_keys():
    value = yaml.safe_load(Path("Experiments/inverse_design/conf/base_config.yaml").read_text())
    cfg = load_experiment_config(value)
    assert cfg.inverse_design.density.mode == "none"
    assert cfg.inverse_design.density.markers == ("LMBR", "SOX2")
    assert cfg.inverse_design.geometry.optimise

    del value["inverse_design"]["density"]
    assert load_experiment_config(value).inverse_design.density.mode == "none"

    value["inverse_design"]["density"] = {"mode": "optimise", "gain": 2.0}
    with pytest.raises(ValueError, match="gain"):
        load_experiment_config(value)
    value["inverse_design"]["density"] = {"mode": "sometimes"}
    with pytest.raises(ValueError, match="density mode"):
        load_experiment_config(value)


def test_segment_cells_splits_patches_and_keeps_only_whole_interior_cells():
    from Experiments.micropatterns.texture_synthesis import segment_cells

    y, x = np.mgrid[:30, :30]
    support = np.hypot(y - 14.5, x - 14.5) < 15
    signal = np.zeros((30, 30), dtype=np.float32)
    # Two touching bright blobs, one lone blob and one on the colony edge
    for centre_y, centre_x in ((10, 10), (10, 14), (20, 20), (1, 15)):
        signal += np.exp(-((y - centre_y) ** 2 + (x - centre_x) ** 2) / 3.0)
    coverage = signal > 0.2
    measurements = np.stack([signal, 2.0 * signal])

    split = segment_cells(measurements, support, coverage, signal, 2, 0.8, 2)
    whole = segment_cells(measurements, support, coverage, signal, 0, 0.8, 2)

    assert len(split) == 3 and len(whole) == 2
    for mask, values in split:
        assert values.shape == (2, *mask.shape)
        assert np.allclose(values[1][mask], 2.0 * values[0][mask])


def test_place_cells_never_overlaps_or_cuts_cells():
    from Experiments.micropatterns.texture_synthesis import place_cells

    square = (np.ones((3, 3), dtype=bool), np.full((2, 3, 3), 0.7, dtype=np.float32))
    plus = (np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), np.full((1, 3, 3), 0.4, dtype=np.float32))
    panels = [(np.asarray([0, 2]), [square]), (np.asarray([1]), [plus])]

    textures, ranks, coverage = place_cells(panels, 3, (40, 40), 2, 10, seed=0)

    assert textures.shape == (2, 3, 40, 40) and ranks.shape == (2, 2, 40, 40)
    assert np.all((coverage > 0.3) & (coverage < 1.0))
    for layer in range(2):
        # Each panel holds only its own cells, and every placed square is whole
        assert np.allclose(textures[layer, 0][ranks[layer, 0] >= 0], 0.7)
        assert np.allclose(textures[layer, 1][ranks[layer, 1] >= 0], 0.4)
        assert np.count_nonzero(ranks[layer, 0] >= 0) % 9 == 0
        # Both panels of a place share its rank
        assert set(np.unique(ranks[layer, 0])) == set(np.unique(ranks[layer, 1]))


def test_cell_library_collects_cells_and_gap_values_per_panel():
    from Experiments.micropatterns.reference import cell_library

    reference = _fake_reference()
    library = cell_library(reference, ("SOX2",), split_distance=1, min_area=1)

    assert [list(indices) for indices, _ in library["panels"]] == [[0, 1, 2], [3]]
    assert all(cells for _, cells in library["panels"])
    assert library["gap_values"].shape == (4,)
    assert library["measurement_gap_values"].shape == (4,)
    assert 0.0 < library["density"] < 1.0
    assert [row["from_markers"] for row in library["coverage"]] == [True, False, True, False]


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


def test_trajectory_with_density_ends_on_the_evaluation():
    from NCA.inverse_design import CellDensity, CellGaps, binary_trajectories, evaluate_binary, pack_cells

    names = ("SOX2", "TBXT", "SOX17", "FOXA2", "mask", "hidden")
    rules = load_cell_type_rules(CELL_TYPES)
    geometry = FourierRadialGeometry(1, radius=0.5)
    textures = jax.random.uniform(jax.random.PRNGKey(0), (2, 1, 4, 11, 11))
    ranks = jax.random.uniform(jax.random.PRNGKey(1), (2, 1, 1, 11, 11))
    biology = jnp.stack([pack_cells(texture, rank) for texture, rank in zip(textures, ranks)])
    config = OptimisationConfig(iterations=1, learning_rate=1.0, total_steps=2, evaluation_seeds=2)
    objective = CellTypeShareObjective.from_rules(rules, names, "Notochord")
    gaps = CellGaps(
        gap_values=jnp.zeros(4), channel_panels=jnp.zeros(4, dtype=int),
        layer_coverage=jnp.ones(1), panels=1,
    )
    density = CellDensity(0.6)
    key = jax.random.PRNGKey(4)

    evaluation = evaluate_binary(_DecayNCA(), geometry, biology, key, objective, config, gaps, density)
    _, trajectories = binary_trajectories(_DecayNCA(), geometry, biology, key, config, 2, gaps, density)

    assert np.isclose(evaluation.density, 0.6)
    assert np.allclose(trajectories[:, -1], np.asarray(evaluation.final_states), atol=1.0e-5)


def test_texture_settings_are_saved_per_projection_and_rescaled(tmp_path):
    from Experiments.micropatterns.texture_settings import (
        DEFAULT_TEXTURE_SETTINGS_FILE,
        TEXTURE_SETTING_KEYS,
        CellSettings,
        load_cell_settings,
        load_texture_settings,
        save_texture_settings,
        scaled_patch_geometry,
    )

    # The file keys are TextureConfig's, apart from the bank size and the file itself
    assert set(TEXTURE_SETTING_KEYS) == set(TextureConfig.__dataclass_fields__) - {"bank_size", "file"}
    for projection in ("max", "sum", "slice"):
        load_texture_settings(DEFAULT_TEXTURE_SETTINGS_FILE, projection)
        load_cell_settings(DEFAULT_TEXTURE_SETTINGS_FILE, projection)

    path = tmp_path / "textures.yaml"
    settings = {**load_texture_settings(DEFAULT_TEXTURE_SETTINGS_FILE, "max"), "patch_downsample": 4}
    cells = CellSettings(placement="layers", pieces="segments", markers=["SOX2"])
    save_texture_settings(path, "max", settings, CellSettings(), note={"downsample": 4})
    save_texture_settings(path, "slice", {**settings, "patch_size": 21}, cells, note={"downsample": 4})
    assert load_texture_settings(path, "max") == settings
    assert load_texture_settings(path, "slice")["patch_size"] == 21
    assert load_cell_settings(path, "slice") == cells
    with pytest.raises(ValueError, match="pieces: nuclei"):
        CellSettings(placement="tiled", pieces="segments")
    with pytest.raises(KeyError, match="sum"):
        load_texture_settings(path, "sum")
    # 11 px at downsample 4 is about 5 px at 8 (always odd and at least 3)
    assert scaled_patch_geometry(11, 7, 3, 4, 8) == (5, 4, 2)
    assert scaled_patch_geometry(11, 7, 3, None, 8) == (11, 7, 3)
    assert scaled_patch_geometry(5, 2, 1, 4, 32)[0] == 3


def test_texture_file_and_config_values_cannot_both_be_set():
    value = yaml.safe_load(Path("Experiments/inverse_design/conf/base_config.yaml").read_text())
    value["inverse_design"]["texture"]["patch_size"] = 9
    with pytest.raises(ValueError, match="patch_size"):
        load_experiment_config(value)
    value["inverse_design"]["texture"]["file"] = None
    assert load_experiment_config(value).inverse_design.texture.patch_size == 9


def test_run_takes_the_texture_settings_for_the_model_projection(tmp_path, monkeypatch):
    from Experiments.micropatterns.texture_settings import (
        DEFAULT_TEXTURE_SETTINGS_FILE,
        CellSettings,
        load_texture_settings,
        save_texture_settings,
    )

    path = tmp_path / "textures.yaml"
    settings = load_texture_settings(DEFAULT_TEXTURE_SETTINGS_FILE, "max")
    save_texture_settings(path, "max", settings, CellSettings())
    save_texture_settings(path, "slice", {**settings, "patch_size": 9, "patch_downsample": 4}, CellSettings())
    entrypoint, cfg = _fake_run_setup(tmp_path, monkeypatch, texture_file=path)

    entrypoint.run(cfg)

    summary = yaml.safe_load(next((tmp_path / "inverse_design").glob("*/*/summary.yaml")).read_text())
    assert summary["texture"]["projection"] == "slice"
    assert summary["texture"]["patch_size"] == 9
    # 9 px at downsample 4 is 5 px at the model's downsample 8
    assert summary["texture"]["used_patch_pixels"]["size"] == 5
