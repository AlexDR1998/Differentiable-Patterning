"""Results files and remote job manifests of the density × KO sweep."""

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from Experiments.micropatterns.density_ko_sweep import (
    DensityKoSettings,
    actual_densities,
    capped_densities,
    density_converter,
    job_configs,
    load_density_ko_results,
    save_density_ko_results,
    settings_from_dict,
)
from Experiments.notebook_jobs import part_path, read_job_entry, write_job_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def fake_result(label, initial_states=True):
    return {
        "model_id": label,
        "label": label,
        "projection": "max",
        "measured_density": 0.4,
        "state_channel_names": ("SOX2", "NODAL"),
        "groups": {"all": "All selected models"},
        "density_unit": "per_mm2",
        "densities": np.array([1000.0, 2000.0]),
        "covered_shares": np.array([0.2, 0.4]),
        "cells_per_mm2": np.array([990.0, 2010.0]),
        "cells_per_colony": np.array([190.0, 390.0]),
        "requested_ko_hours": np.array([0.0, 24.0, 48.0]),
        "realised_ko_hours": np.array([0.0, 24.0, 48.0]),
        # [densities, KO times, rings, patterns]
        "pattern_counts": np.arange(2 * 3 * 4 * 8, dtype=float).reshape(2, 3, 4, 8),
        "initial_states": np.ones((2, 2, 5, 5)) if initial_states else None,
        "colony": np.eye(5, dtype=bool),
    }


def records():
    return [
        {"model_id": "model-a", "projection": "max", "downsample": 4, "notes": "a"},
        {"model_id": "model-b", "projection": "sum", "downsample": 4, "notes": None},
    ]


def test_results_file_round_trip(tmp_path):
    path = tmp_path / "results.npz"
    save_density_ko_results(path, [fake_result("a"), fake_result("b", initial_states=False)])

    loaded = load_density_ko_results(path)

    assert [result["label"] for result in loaded] == ["a", "b"]
    assert loaded[0]["state_channel_names"] == ("SOX2", "NODAL")
    np.testing.assert_array_equal(loaded[0]["pattern_counts"], fake_result("a")["pattern_counts"])
    assert loaded[1]["initial_states"] is None


def test_folder_of_parts_loads_in_part_order(tmp_path):
    for index, label in [(2, "c"), (0, "a"), (1, "b")]:
        save_density_ko_results(part_path(tmp_path, index), [fake_result(label)])

    loaded = load_density_ko_results(tmp_path)

    assert [result["label"] for result in loaded] == ["a", "b", "c"]
    assert loaded[0]["initial_states"] is not None


def test_job_parts_keep_the_seeds_of_a_single_run():
    settings = DensityKoSettings(data_root="data", store_root="models", replicates=(1, 2, 4))

    by_model = job_configs(settings, records(), "out", split="model")
    by_replicate = job_configs(settings, records(), "out", split="replicate")

    # A single run counts (model, replicate) jobs 0, 1, 2 for model a and 3, 4, 5 for b
    assert [config["job_offset"] for config in by_model] == [0, 3]
    assert [config["job_offset"] for config in by_replicate] == [0, 1, 2, 3, 4, 5]
    assert [config["settings"]["replicates"] for config in by_replicate[:3]] == [(1,), (2,), (4,)]
    assert [config["records"][0]["model_id"] for config in by_replicate] == ["model-a"] * 3 + ["model-b"] * 3


def test_manifest_round_trip_and_job_environment(tmp_path, monkeypatch):
    settings = DensityKoSettings(
        data_root="data", store_root="models", replicates=(1, 3),
        cell_type_files={"max": "Experiments/micropatterns/conf/cell_types/x.yaml"},
    )
    configs = job_configs(settings, records(), "/workspace/writeable/out/job", split="model")
    manifest = write_job_manifest(tmp_path / "manifest.yaml", "density_ko_job", configs)

    index, config = read_job_entry(manifest, index=1)
    assert index == 1
    assert settings_from_dict(config["settings"]) == settings
    assert config["records"][0]["model_id"] == "model-b"
    assert config["job_offset"] == 2

    # Indexed Kubernetes jobs: worker 1 of 2, first completion index -> part 1
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "0")
    monkeypatch.setenv("JOB_WORKER_INDEX", "1")
    monkeypatch.setenv("JOB_WORKER_COUNT", "2")
    assert read_job_entry(manifest)[0] == 1


def test_launchers_count_the_parts(tmp_path):
    settings = DensityKoSettings(data_root="data", store_root="models")
    manifest = write_job_manifest(
        tmp_path / "manifest.yaml", "density_ko_job",
        job_configs(settings, records(), "out", split="model"),
    )
    script = f'source "{REPOSITORY_ROOT / "launch/manifest.sh"}"; manifest_count "{manifest}"; manifest_experiment_name "{manifest}"'
    output = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split()

    assert output == ["2", "density_ko_job"]


def test_settings_ranges():
    settings = DensityKoSettings(
        data_root="data", store_root="models", density_count=3, ko_interval=12.0,
        replicates=(1, 2), duplicate_runs=2,
    )

    np.testing.assert_allclose(settings.ko_hours(), [0.0, 12.0, 24.0, 36.0, 48.0])
    assert len(settings.densities()) == 3
    assert settings.trajectories(2) == 2 * 2 * 2 * 3 * 5
    assert replace(settings, replicates=(5,)).trajectories(1) == 2 * 3 * 5


def test_parts_from_different_workers_get_different_files(tmp_path, monkeypatch):
    settings = DensityKoSettings(data_root="data", store_root="models", replicates=(1, 2))
    manifest = write_job_manifest(
        tmp_path / "manifest.yaml", "density_ko_job",
        job_configs(settings, records(), str(tmp_path / "out"), split="replicate"),
    )
    # 4 parts over 2 workers, as launch_batch_multi_job.sh shares them out
    monkeypatch.setenv("JOB_WORKER_COUNT", "2")
    indices = []
    for worker in range(2):
        for completion in range(2):
            monkeypatch.setenv("JOB_WORKER_INDEX", str(worker))
            monkeypatch.setenv("JOB_COMPLETION_INDEX", str(completion))
            index, config = read_job_entry(manifest)
            indices.append(index)
            save_density_ko_results(part_path(config["output_dir"], index), [fake_result(str(index))])

    assert sorted(indices) == [0, 1, 2, 3]
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "part_000.npz", "part_001.npz", "part_002.npz", "part_003.npz",
    ]
    assert [result["label"] for result in load_density_ko_results(tmp_path / "out")] == ["0", "1", "2", "3"]


def test_actual_densities_are_cut_at_what_the_layers_hold():
    result = {
        **fake_result("a"),
        "densities": np.array([1000.0, 3000.0, 5000.0]),
        "cells_per_mm2": np.array([1004.0, 2996.0, 3782.0]),
        "max_density": 3782.0,
    }

    np.testing.assert_array_equal(actual_densities(result), [1004.0, 2996.0, 3782.0])
    np.testing.assert_array_equal(capped_densities(result), [False, False, True])
    # Older results swept the covered share
    share = {**result, "density_unit": "share", "covered_shares": np.array([0.2, 0.4, 0.5])}
    np.testing.assert_array_equal(actual_densities(share), [0.2, 0.4, 0.5])


def test_density_converter_maps_between_units_both_ways():
    cells = np.array([100.0, 200.0, 400.0, 400.0])  # the last point was capped
    coverage = np.array([0.1, 0.3, 0.5, 0.5])
    forward, inverse = density_converter(cells, coverage)

    np.testing.assert_allclose(forward([100.0, 150.0, 300.0]), [0.1, 0.2, 0.4])
    np.testing.assert_allclose(inverse(forward([120.0, 350.0])), [120.0, 350.0])
    # Beyond the sweep the end segments continue
    np.testing.assert_allclose(forward([50.0, 500.0]), [0.0, 0.6])
    assert density_converter([1.0, 1.0], [0.2, 0.2]) is None
    assert density_converter([1.0, 2.0], [np.nan, np.nan]) is None


def test_density_conversions_skip_the_plotted_unit_and_missing_counts():
    import pandas as pd

    from Experiments.micropatterns.ko_evaluation import density_conversions

    summary = pd.DataFrame({
        "density": [100.0, 200.0, 400.0] * 2,
        "actual_density": [100.0, 200.0, 400.0] * 2,
        "cells_per_colony": [100.0, 200.0, 400.0] * 2,
        "cells_per_mm2": [500.0, 1000.0, 2000.0] * 2,
        "covered_share": [np.nan] * 6,  # not recorded
    })

    conversions = density_conversions(summary, "per_colony", ["per_colony", "per_mm2", "share"])

    assert list(conversions) == ["per_mm2"]
    forward, inverse = conversions["per_mm2"]
    np.testing.assert_allclose(forward([300.0]), [1500.0])
    np.testing.assert_allclose(inverse([1500.0]), [300.0])


def fake_library():
    """Two staining panels of small square cells, as ``cell_library`` returns them."""
    rng = np.random.default_rng(0)
    cell = lambda: (np.ones((2, 2), dtype=bool), rng.random((2, 2, 2)).astype(np.float32))
    return {
        "panels": [(np.array([0, 1]), [cell() for _ in range(5)]), (np.array([2, 3]), [cell() for _ in range(5)])],
        "measurement_gap_values": np.zeros(4, dtype=np.float32),
        "gap_values": np.full(3, -1.0, dtype=np.float32),
    }


class FakeSchema:
    primary_measurements = (0, 2, 3)


def disc(size=40, radius=18):
    rows, cols = np.indices((size, size))
    return (rows - (size - 1) / 2) ** 2 + (cols - (size - 1) / 2) ** 2 <= radius**2


def test_seeds_keep_their_distance_and_order():
    from Experiments.micropatterns.texture_synthesis import scatter_seeds

    seeds = scatter_seeds(disc(), 120, 2.0, seed=3)
    again = scatter_seeds(disc(), 120, 2.0, seed=3)

    assert seeds.shape == (120, 2)
    # More than random placement can fit: fewer points come back
    assert len(scatter_seeds(disc(), 400, 2.0, seed=3)) < 400
    np.testing.assert_array_equal(seeds, again)
    gaps = np.hypot(*(seeds[:, None] - seeds[None]).transpose(2, 0, 1))
    assert gaps[~np.eye(len(seeds), dtype=bool)].min() >= 2.0


def test_tiled_cells_never_overlap_and_are_counted_exactly():
    from Experiments.micropatterns.texture_synthesis import tile_cells

    mask = disc()
    seeds = np.array([[10.0, 10.0], [10.0, 30.0], [10.6, 10.6]])
    state, coverage, weights, cells = tile_cells(seeds, 1.5, np.eye(3, 2), np.full(2, -1.0), mask)

    assert cells == 3
    assert np.isclose(weights.sum(), 3.0)
    assert coverage.max() <= 1.0 + 1e-6
    # An isolated cell covers about pi r^2 model pixels
    assert abs(coverage[:20, 20:].sum() - np.pi * 1.5**2) < 1.0
    np.testing.assert_allclose(state[:, ~mask & (coverage == 0)], -1.0)


def test_tiled_states_hit_the_requested_cell_numbers():
    from Experiments.micropatterns.density_ko_sweep import tiled_cell_states

    mask = disc()
    tiled = tiled_cell_states(fake_library(), FakeSchema(), mask, "per_colony", [10, 100, 300], seed=1)

    np.testing.assert_array_equal(tiled["cells"], [10, 100, 300])
    np.testing.assert_allclose(tiled["cell_weights"].sum(axis=(1, 2)), [10, 100, 300], rtol=1e-5)
    assert tiled["initial_states"].shape == (3, 3, 40, 40)
    assert np.all(np.diff(tiled["shares"]) > 0)
    # Cells of 4 pixels: 300 of them cannot keep their area in about 1000 pixels
    assert tiled["squeeze"] < 1.0


def test_tiled_states_hit_the_requested_coverage():
    from Experiments.micropatterns.density_ko_sweep import tiled_cell_states

    tiled = tiled_cell_states(fake_library(), FakeSchema(), disc(), "share", [0.2, 0.6], seed=1)

    assert np.all(tiled["shares"] >= [0.2, 0.6])
    assert np.all(tiled["shares"] < [0.25, 0.65])
    assert tiled["cells"][0] < tiled["cells"][1]


def test_weighted_pattern_counts():
    from Common.dataloader.cell_type_shares import pattern_counts

    values = np.array([[0.0, 1.0, 1.0, 0.0]])
    counts = pattern_counts(values, [0.5], np.array([0, 0, 0, -1]), 1, weights=[0.5, 0.25, 0.25, 9.0])

    np.testing.assert_allclose(counts, [[0.5, 0.5]])


def test_ring_channel_means():
    from Experiments.micropatterns.density_ko_sweep import ring_channel_means

    colony = disc(20, 9)
    rows, cols = np.indices(colony.shape)
    state = np.stack([np.ones(colony.shape), np.hypot(rows - 9.5, cols - 9.5)])
    means = ring_channel_means(state, colony, 3)

    assert means.shape == (3, 2)
    np.testing.assert_allclose(means[:, 0], 1.0)
    assert np.all(np.diff(means[:, 1]) > 0)  # distance grows outwards


def test_whole_sweep_runs_with_fake_models_and_data(monkeypatch, tmp_path):
    """The full sweep loop (placement, batched rollouts, counts, channel means,
    saving) on a tiny stand-in model, with the data loaders replaced."""
    import equinox as eqx
    import pytest
    from types import SimpleNamespace

    import Experiments.micropatterns.density_ko_sweep as sweep
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
    }
    schema = SimpleNamespace(primary_measurements=(0, 1, 2, 3))
    monkeypatch.setattr(sweep, "load_training_histogram_bins", lambda bundle, root: None)
    monkeypatch.setattr(sweep, "load_condition_data", lambda *args: {
        "boundary": support[None].astype(np.float32), "state_channel_names": names, "nodal_channel": 1,
    })
    monkeypatch.setattr(sweep, "stacked_colonies", lambda data, replicates: {
        "support": np.stack([support] * len(replicates)), "schema": schema, "source_files": None,
    })
    monkeypatch.setattr(sweep, "bundle_alignment", lambda bundle: (None, 1.0, 5.0))
    monkeypatch.setattr(sweep, "reference_cell_counts", lambda *args, **kwargs: [])
    monkeypatch.setattr(sweep, "replicate_cell_counts", lambda rows, replicate: {
        "cells_per_mm2": 1000.0, "cells_per_segment": 1.0,
    })
    monkeypatch.setattr(sweep, "replicate_cell_library", lambda *args, **kwargs: (library, 0.4))
    cell_types = "Experiments/micropatterns/conf/cell_types/260726_cell_types_sliced_nca.yaml"

    for placement in ("tiled", "layers"):
        settings = DensityKoSettings(
            data_root="data", store_root="models", cell_type_files={"slice": cell_types}, rings=3,
            density_unit="per_colony", density_start=20, density_stop=120, density_count=3,
            ko_start=0, ko_stop=12, ko_interval=12, snapshot_hour=12, replicates=(1,),
            duplicate_runs=2, placement=placement, max_batch=4,
        )
        record = {"model_id": "fake-model", "projection": "slice", "downsample": 8, "notes": "fake"}
        results, warnings = sweep.run_density_ko_sweep(settings, [record])

        assert len(results) == 2
        for result in results:
            assert result["pattern_counts"].shape[:3] == (3, 2, 3)
            assert result["channel_means"].shape == (3, 2, 3, 4)
            assert np.all(np.isfinite(result["channel_means"]))
        if placement == "tiled":
            np.testing.assert_array_equal(results[0]["cells_per_colony"], [20, 70, 120])
            assert results[0]["cell_pattern_counts"].shape == results[0]["pattern_counts"].shape
            # Every 0h cell counted once over the colony
            np.testing.assert_allclose(results[0]["cell_pattern_counts"].sum(axis=(2, 3)), [[20] * 2, [70] * 2, [120] * 2])
        else:
            assert results[0]["cell_pattern_counts"] is None
        save_density_ko_results(tmp_path / f"{placement}.npz", results)
        loaded = load_density_ko_results(tmp_path / f"{placement}.npz")
        assert loaded[0]["placement"] == placement
        np.testing.assert_allclose(loaded[1]["channel_means"], results[1]["channel_means"])

    # A remote job part, as a pod runs it: manifest entry -> part file, skipped when rerun
    from Experiments.micropatterns.density_ko_sweep import run_job_part

    manifest = write_job_manifest(
        tmp_path / "manifest.yaml", "density_ko_fake", job_configs(settings, [record], str(tmp_path / "out")),
    )
    index, config = read_job_entry(manifest, index=0)
    output = run_job_part(config, index)
    assert output == part_path(tmp_path / "out", 0)
    assert len(load_density_ko_results(tmp_path / "out")) == 2
    modified = output.stat().st_mtime_ns
    run_job_part(config, index)
    assert output.stat().st_mtime_ns == modified
