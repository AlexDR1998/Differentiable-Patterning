"""Results files and remote job manifests of the density × KO sweep."""

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from Experiments.micropatterns.density_ko_sweep import (
    DensityKoSettings,
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
