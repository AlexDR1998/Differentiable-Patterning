"""Colony size × cell density sweep: geometry, profile rollouts, fronts, results files and jobs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Experiments.micropatterns.size_density_sweep import (
    SizeDensitySettings,
    front_positions,
    front_speed,
    job_configs,
    kymograph_values,
    load_size_density_results,
    profile_positions,
    radial_bins,
    ring_label_shares,
    save_size_density_results,
    settings_from_dict,
    size_masks,
    sweep_grid_shape,
)
from Experiments.micropatterns.texture_settings import CellSettings
from Experiments.notebook_jobs import part_path, read_job_entry, write_job_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CELL_TYPES = "Experiments/micropatterns/conf/cell_types/260726_cell_types_sliced_nca.yaml"


def disc(size=30, radius=13):
    rows, cols = np.indices((size, size))
    return (rows - (size - 1) / 2) ** 2 + (cols - (size - 1) / 2) ** 2 <= radius**2


def test_grid_keeps_the_training_margin_and_is_odd():
    # Margin 14.5 - 13 = 1.5 px around a 20 px colony
    assert sweep_grid_shape((31, 29), 13, 20) == (43, 43)
    # Never smaller than the data grid
    assert sweep_grid_shape((31, 29), 13, 5) == (31, 31)
    assert sweep_grid_shape((30, 30), 13, 5) == (31, 31)


def test_radial_bins_and_masks():
    bins = radial_bins((21, 21), 2.0, 10.0)
    assert bins[10, 10] == 0
    assert bins[10, 15] == 1  # 5 px = 10 µm from the centre
    assert bins[10, 19] == 1  # 18 µm
    masks = size_masks((41, 41), [20.0, 40.0], 2.0)
    assert masks.shape == (2, 41, 41)
    # Radii of 5 and 10 px: areas close to pi r^2
    np.testing.assert_allclose(masks.sum(axis=(1, 2)), np.pi * np.array([5.0, 10.0]) ** 2, rtol=0.1)


class NoisyModel(eqx.Module):
    def __call__(self, state, boundary_callback, key=None):
        return boundary_callback(0.9 * state + 0.1 * jax.random.uniform(key, state.shape))


def test_profile_rollout_matches_sampled_rollout():
    from NCA.trainer.intervention import (
        rollout_batch_with_blocked_channel_profiles,
        rollout_model_with_blocked_channel_sampled,
    )

    model = NoisyModel()
    mask = disc(15, 6)
    groups = np.where(mask, radial_bins((15, 15), 1.0, 2.0), -1)
    n_groups = int(groups.max()) + 1
    states = jax.random.uniform(jax.random.PRNGKey(0), (2, 4, 15, 15))
    keys = jnp.stack([jax.random.PRNGKey(1), jax.random.PRNGKey(2)])
    boundary = jnp.asarray(mask[None], dtype=jnp.float32)
    fate, thresholds = jnp.asarray([0, 2]), jnp.asarray([0.5, 0.4])

    final, profiles, counts = rollout_batch_with_blocked_channel_profiles(
        model, states, boundary, "hard", keys, 5, 1, jnp.asarray(2), 3, fate, thresholds, jnp.asarray(groups), n_groups,
    )
    assert profiles.shape == (2, 6, n_groups, 3)
    assert counts.shape == (2, 6, n_groups, 4)
    for index in range(2):
        frames = np.asarray(rollout_model_with_blocked_channel_sampled(
            model, states[index], boundary, "hard", keys[index], 5, 1, jnp.asarray(2), jnp.arange(6),
        ))
        np.testing.assert_allclose(final[index], frames[-1], rtol=1e-5, atol=1e-6)
        for step, frame in enumerate(frames):
            for group in range(n_groups):
                inside = groups == group
                np.testing.assert_allclose(profiles[index, step, group], frame[:3, inside].mean(axis=1), rtol=1e-4)
                codes = (frame[0, inside] > 0.5) + 2 * (frame[2, inside] > 0.4)
                np.testing.assert_array_equal(counts[index, step, group], np.bincount(codes, minlength=4))


def test_fronts_follow_a_moving_edge():
    hours = np.arange(0.0, 25.0)
    positions = np.arange(0.0, 200.0, 10.0)[::-1]  # decreasing, as distance from the edge often is
    fronts_true = 20.0 + 3.0 * hours
    values = 1.0 / (1.0 + np.exp((positions[None] - fronts_true[:, None]) / 8.0))
    fronts = front_positions(values, positions, 0.5)

    np.testing.assert_allclose(fronts, fronts_true, atol=1.0)
    assert abs(front_speed(hours, fronts) - 3.0) < 0.05
    assert abs(front_speed(hours, fronts, window=(10, 20)) - 3.0) < 0.05
    # No crossing: no front
    assert np.isnan(front_positions(np.ones((1, 5)), np.arange(5.0), 0.5)[0])
    assert np.isnan(front_speed(hours[:1], fronts[:1]))


def test_ring_label_shares_sum_to_one():
    from Common.dataloader.cell_types import load_cell_type_rules

    rules = load_cell_type_rules(REPOSITORY_ROOT / CELL_TYPES)
    counts = np.random.default_rng(0).integers(0, 20, size=(3, 4, 2 ** len(rules.markers)))
    counts[1, 2] = 0  # an empty bin
    shares = ring_label_shares(counts, rules)

    assert shares.shape[:2] == (3, 4)
    np.testing.assert_allclose(np.nansum(shares, axis=-1)[counts.sum(axis=-1) > 0], 1.0, atol=1e-6)
    assert np.all(np.isnan(shares[1, 2]))


def test_job_parts_keep_seeds_and_grid():
    settings = SizeDensitySettings(
        data_root="data", store_root="models", diameters_um=(200.0, 500.0), replicates=(1, 2),
    )
    records = [{"model_id": "a", "projection": "max", "downsample": 8}, {"model_id": "b", "projection": "max", "downsample": 8}]

    by_size = job_configs(settings, records, "out", split="size")
    assert len(by_size) == 8
    assert [_c["job_offset"] for _c in by_size] == [0, 0, 1, 1, 2, 2, 3, 3]
    # Every part uses the grid of the largest colony
    assert {_c["settings"]["grid_diameter_um"] for _c in by_size} == {500.0}
    assert [_c["job_offset"] for _c in job_configs(settings, records, "out")] == [0, 2]
    again = settings_from_dict(by_size[0]["settings"])
    assert again.diameters_um == (200.0,)
    assert again.grid_diameter() == 500.0


def test_whole_sweep_runs_with_fake_models_and_data(monkeypatch, tmp_path):
    """The full loop (placement at every size, batched profile rollouts,
    counts, saving, a job part) on a tiny model, with the loaders replaced."""
    import Experiments.micropatterns.density_ko_sweep as density_sweep
    import Experiments.micropatterns.size_density_sweep as sweep
    import Experiments.model_registry as registry_module

    class TinyModel(eqx.Module):
        N_CHANNELS: int = eqx.field(static=True, default=6)

        def __call__(self, state, boundary_callback, key=None):
            return boundary_callback(0.9 * state + 0.05)

    bundle = SimpleNamespace(
        id="fake-model",
        load_model=lambda key, implementation: TinyModel(),
        config=SimpleNamespace(
            run=SimpleNamespace(t=4),
            data=SimpleNamespace(knockout=SimpleNamespace(curriculum=("baseline",)), downsample=8),
            trainer=SimpleNamespace(boundary_mode="hard"),
        ),
    )
    monkeypatch.setattr(registry_module, "ModelRegistry", lambda root: SimpleNamespace(get=lambda _id: bundle))
    names = ("SOX17", "SOX2", "TBXT", "FOXA2")
    support = disc(30, 13)
    rng = np.random.default_rng(0)
    cell = lambda: (np.ones((2, 2), dtype=bool), rng.random((2, 2, 2)).astype(np.float32))
    library = {
        "panels": [(np.array([0, 1]), [cell() for _ in range(5)]), (np.array([2, 3]), [cell() for _ in range(5)])],
        "measurement_gap_values": np.zeros(4, dtype=np.float32),
        "gap_values": np.zeros(4, dtype=np.float32),
        "pixels": [rng.random((40, 2)).astype(np.float32), rng.random((40, 2)).astype(np.float32)],
    }
    schema = SimpleNamespace(primary_measurements=(0, 1, 2, 3), group_names=("cell_fate_s1", "rna_expression"))
    monkeypatch.setattr(density_sweep, "load_training_histogram_bins", lambda bundle, root: None)
    # The true colony: a smaller disc on the 30-pixel data grid
    true_support = disc(30, 11)
    monkeypatch.setattr(density_sweep, "load_condition_data", lambda *args: {
        "boundary": (true_support if len(args) > 3 and args[3] == 2 else support)[None].astype(np.float32),
        "state_channel_names": names, "nodal_channel": 1,
        "initial_state": rng.random((len(names), 30, 30)).astype(np.float32),
    })
    monkeypatch.setattr(density_sweep, "stacked_colonies", lambda data, replicates: {
        "support": np.stack([support] * len(replicates)), "schema": schema, "source_files": None,
    })
    # 5 µm pixels: the training colony is 130 µm across
    monkeypatch.setattr(density_sweep, "bundle_alignment", lambda bundle: (None, 1.0, 5.0))
    monkeypatch.setattr(density_sweep, "reference_cell_counts", lambda *args, **kwargs: [])
    monkeypatch.setattr(density_sweep, "replicate_cell_counts", lambda rows, replicate, panels=None: {
        "cells_per_mm2": 3000.0, "cells_per_segment": 1.0,
    })
    monkeypatch.setattr(density_sweep, "replicate_cell_library", lambda *args, **kwargs: (library, 0.4))

    settings = SizeDensitySettings(
        data_root="data", store_root="models", cell_type_files={"slice": CELL_TYPES},
        cells={"slice": CellSettings(placement="tiled").to_dict()},
        diameters_um=(60.0, 100.0, 160.0), rings=3, density_start=2000, density_stop=5000, density_count=2,
        ko_hours=(6.0,), snapshot_hour=12, record_every_hours=3, profile_bin_um=10.0,
        duplicate_runs=2, max_batch=3,
    )
    record = {"model_id": "fake-model", "projection": "slice", "downsample": 8, "notes": "fake"}
    results, warnings = sweep.run_size_density_sweep(settings, [record])

    assert len(results) == 2
    first = results[0]
    sizes, densities, conditions, times = 3, 2, 2, 5
    n_bins = len(first["profile_radii_um"])
    assert first["grid_shape"] == [37, 37]
    assert first["conditions"] == ["baseline", "ko_6h"]
    np.testing.assert_allclose(first["profile_hours"], [0, 3, 6, 9, 12])
    assert first["pattern_counts"].shape[:4] == (sizes, densities, conditions, 3)
    assert first["channel_means"].shape == (sizes, densities, conditions, 3, 4)
    assert first["profiles"].shape == (sizes, densities, conditions, times, n_bins, 4)
    assert first["profile_counts"].shape[:5] == (sizes, densities, conditions, times, n_bins)
    assert first["final_states"].shape == (sizes, densities, conditions, 4, 37, 37)
    assert results[1]["final_states"] is None
    # Equal density at every size: cells grow with the colony area
    np.testing.assert_allclose(first["cells_per_mm2"], np.broadcast_to([2000.0, 5000.0], (3, 2)), rtol=0.15)
    assert np.all(np.diff(first["cells_per_colony"][:, 0]) > 0)
    # Bins beyond the smallest colony are empty
    assert np.isnan(first["profiles"][0, 0, 0, 0, -1, 0])
    assert np.all(np.isfinite(first["profiles"][-1, 0, 0, 0, :-1]))
    # Pixel counts per bin match the colony's pixels
    assert first["profile_counts"][1, 0, 0, 0].sum() == first["colony_masks"][1].sum()
    assert first["seed"] != results[1]["seed"]

    edge = profile_positions(first, 2, "edge_um")
    assert edge[0] == 80.0 - 5.0
    # The edge bin of every size sits inside its colony
    for size in range(sizes):
        filled = np.isfinite(first["profiles"][size, 0, 0, 0, :, 0])
        assert np.all(profile_positions(first, size, "edge_um")[filled] >= 0.0)
    values = kymograph_values(first, "SOX2")
    assert values.shape == (sizes, densities, conditions, times, n_bins)

    save_size_density_results(tmp_path / "results.npz", results)
    loaded = load_size_density_results(tmp_path / "results.npz")
    assert loaded[0]["conditions"] == ("baseline", "ko_6h")
    np.testing.assert_array_equal(loaded[1]["profile_counts"], results[1]["profile_counts"])

    # A remote job part, as a pod runs it: manifest entry -> part file, skipped when rerun
    manifest = write_job_manifest(
        tmp_path / "manifest.yaml", "size_density_fake", job_configs(settings, [record], str(tmp_path / "out"), split="size"),
    )
    index, config = read_job_entry(manifest, index=2)
    output = sweep.run_job_part(config, index)
    assert output == part_path(tmp_path / "out", 2)
    part = load_size_density_results(tmp_path / "out")
    # The size-160 part keeps the grid and seeds of the full run
    assert part[0]["grid_shape"] == [37, 37]
    assert [_r["seed"] for _r in part] == [_r["seed"] for _r in results]
    np.testing.assert_allclose(part[0]["profiles"][0], first["profiles"][2], atol=1e-3, equal_nan=True)
    modified = output.stat().st_mtime_ns
    sweep.run_job_part(config, index)
    assert output.stat().st_mtime_ns == modified

    # References: every size at the measured density, and the true colony of each replicate (2 is the smaller disc)
    settings = replace(settings, replicates=(1, 2), include_measured_density=True, include_true_data=True)
    results, warnings = sweep.run_size_density_sweep(settings, [record])
    assert [r["source"] for r in results] == (["sweep"] * 2 + ["measured_density"] * 2 + ["true_data"] * 2) * 2
    measured = [r for r in results if r["source"] == "measured_density"]
    true = [r for r in results if r["source"] == "true_data"]
    sweep_runs = [r for r in results if r["source"] == "sweep"]
    for result in measured:
        np.testing.assert_allclose(result["densities"], [3000.0])
        assert result["pattern_counts"].shape[:3] == (sizes, 1, conditions)
        np.testing.assert_allclose(result["cells_per_mm2"][:, 0], 3000.0, rtol=0.2)
    assert [r["seed"] for r in measured] == [r["seed"] for r in sweep_runs]
    assert [r["seed"] for r in true] == [r["seed"] for r in sweep_runs]
    for result in true:
        # One size: each replicate's own outline, centred on the sweep's grid
        disc_of = true_support if result["replicate"] == 2 else support
        assert result["colony_masks"].shape == (1, 37, 37)
        assert result["colony_masks"][0].sum() == disc_of.sum()
        np.testing.assert_allclose(result["diameters_um"], [2 * np.sqrt(disc_of.sum() / np.pi) * 5.0])
        assert result["profiles"].shape[:3] == (1, 1, conditions)
        assert result["cell_pattern_counts"] is None
        assert result["profile_counts"][0, 0, 0, 0].sum() == disc_of.sum()
    # Every replicate's true colony has the same radial bins, so their kymographs can be averaged
    assert len({len(r["profile_radii_um"]) for r in true}) == 1
    assert sweep.mean_kymograph(true, "SOX2").shape[:3] == (1, 1, conditions)
    assert [r["initial_states"] is not None for r in true] == [True, False, True, False]
    assert true[0]["initial_states"].shape == (1, 1, 4, 37, 37)
    assert settings.trajectories(1) == 2 * (2 * 3 * 2 + 2 * 3 + 2) * 2
    save_size_density_results(tmp_path / "references.npz", results)
    assert [r["source"] for r in load_size_density_results(tmp_path / "references.npz")] == [r["source"] for r in results]
    # Files from before the references load as sweep results
    assert all(r["source"] == "sweep" for r in load_size_density_results(tmp_path / "results.npz"))


def test_jobs_run_each_replicates_true_data_once_and_average_every_replicate():
    from Experiments.micropatterns.density_ko_sweep import DensityKoSettings
    from Experiments.micropatterns.density_ko_sweep import job_configs as density_job_configs

    record = {"model_id": "m", "projection": "slice", "downsample": 8}
    size_settings = SizeDensitySettings(
        data_root="d", store_root="s", replicates=(1, 2), diameters_um=(200.0, 500.0),
        include_true_data=True, include_measured_density=True,
    )
    parts = job_configs(size_settings, [record], "out", split="size")
    # Each replicate's true data runs in its first size part only
    assert [p["settings"]["include_true_data"] for p in parts] == [True, False, True, False]
    assert all(tuple(p["settings"]["measured_replicates"]) == (1, 2) for p in parts)
    density_settings = DensityKoSettings(data_root="d", store_root="s", replicates=(1, 2), include_true_data=True)
    parts = density_job_configs(density_settings, [record], "out", split="replicate")
    # Each part runs the true data of its own replicate
    assert [p["settings"]["include_true_data"] for p in parts] == [True, True]
    assert all(tuple(p["settings"]["measured_replicates"]) == (1, 2) for p in parts)


def test_mean_kymograph_pools_runs():
    from Common.dataloader.cell_types import label_names, load_cell_type_rules
    from Experiments.micropatterns.size_density_sweep import mean_kymograph

    rules = load_cell_type_rules(REPOSITORY_ROOT / CELL_TYPES)
    rng = np.random.default_rng(1)
    shape = (1, 1, 1, 2, 3)
    runs = [
        {
            "state_channel_names": ("SOX2",),
            "profiles": rng.random((*shape, 1)),
            "profile_counts": rng.integers(0, 9, size=(*shape, 2 ** len(rules.markers))),
        }
        for _ in range(2)
    ]
    np.testing.assert_allclose(
        mean_kymograph(runs, "SOX2"), (runs[0]["profiles"][..., 0] + runs[1]["profiles"][..., 0]) / 2
    )
    label = label_names(rules)[0]
    shares = mean_kymograph(runs, label, rules)
    assert shares.shape == shape
    assert np.all((shares >= 0) & (shares <= 1))
