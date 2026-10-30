import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from pathlib import Path

from Experiments.config import config_to_dict, impulse_experiment_config_from_mapping
from Experiments.impulse.analysis import (
    distance_over_time,
    graft,
    graft_switch_map,
    grow_attractors,
    load_impulse_results,
    load_saved_intervention,
    move_intervention,
    summary_row,
    scale_intervention,
    success_rate,
    switch_matrix,
)
from Experiments.impulse.config_helpers import build_intervention
from Experiments.impulse.records import impulse_size
from NCA.trainer.impulse import ExternalTargetPairSource, NCAImpulseOptimiser, TargetedObjective
from NCA.trainer.impulse.perturbation import perturbation


class RoundingModel(eqx.Module):
    """Every value moves halfway to the nearest integer, so integers are attractors."""

    N_CHANNELS: int = eqx.field(static=True, default=2)

    def __call__(self, state, boundary_callback=lambda x: x, key=None):
        return boundary_callback(state + 0.5 * (jnp.round(state) - state))


def _global_impulse(value, channels=2):
    intervention = perturbation(
        mode={"channel": "all", "spatial": "global"},
        CHANNELS=channels,
        OBS_CHANNELS=channels,
        x=jnp.zeros((1, channels, 4, 4)),
        WIDTH=1.0,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_at(lambda item: item.values, intervention, jnp.full_like(intervention.values, value))


def _patterns():
    # Patterns 0, 1, 2 start slightly off their integer attractors
    return jnp.stack([jnp.full((2, 4, 4), k + 0.2) for k in range(3)])


def test_switch_matrix_counts_where_each_pattern_ends():
    model = RoundingModel()
    key = jax.random.PRNGKey(1)
    attractors = grow_attractors(model, _patterns(), steps=20, samples=3, key=key)
    references = attractors[:, 0]

    baseline = switch_matrix(model, None, attractors, references, 20, 2, key)
    shifted = switch_matrix(model, _global_impulse(1.0), attractors, references, 20, 2, key)

    assert attractors.shape == (3, 3, 2, 4, 4)
    assert jnp.allclose(references[:, 0, 0, 0], jnp.arange(3.0), atol=1e-4)
    assert jnp.allclose(baseline, jnp.eye(3))
    # 0 -> 1, 1 -> 2, and 2 -> 3, which is nearest pattern 2
    assert jnp.allclose(shifted, jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 1]]))


def test_distance_over_time_and_success_rate_across_impulse_scales():
    model = RoundingModel()
    key = jax.random.PRNGKey(2)
    references = jnp.stack([jnp.full((2, 4, 4), float(k)) for k in range(3)])
    start = jnp.zeros((5, 2, 4, 4))
    impulse = _global_impulse(1.0)

    distances, trajectory = distance_over_time(model, impulse, start, references, 10, 2, key)
    rates = success_rate(
        model,
        [scale_intervention(impulse, s) for s in (0.2, 1.0, 2.0)],
        start,
        references,
        1,
        10,
        2,
        key,
    )

    assert distances.shape == (5, 10, 3)
    assert trajectory.shape == (5, 10, 2, 4, 4)
    assert jnp.allclose(distances[:, -1, 1], 0.0, atol=1e-3)
    assert rates.tolist() == [0.0, 1.0, 0.0]


def test_move_intervention_sets_the_local_centre():
    intervention = perturbation(
        mode={"channel": "all", "spatial": "local"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=jnp.zeros((1, 2, 4, 4)),
        WIDTH=0.1,
        key=jax.random.PRNGKey(0),
    )

    moved = move_intervention(intervention, (0.25, 0.75))

    assert jnp.allclose(moved.get_location(), jnp.asarray([0.25, 0.75]), atol=1e-5)


def test_load_impulse_results_reads_saved_run_summaries(tmp_path):
    value = yaml.safe_load(Path("Experiments/impulse/conf/base_config.yaml").read_text())
    value["impulse"]["pair_source"].update(source_index=2, target_index=0)
    cfg = impulse_experiment_config_from_mapping(value)
    eqx.tree_serialise_leaves(tmp_path / "run.eqx", _global_impulse(1.0))
    (tmp_path / "run.json").write_text(
        json.dumps({"model_id": "model-1", "config": config_to_dict(cfg)})
    )
    (tmp_path / "unrelated.json").write_text(json.dumps({"other": 1}))

    results = load_impulse_results(tmp_path)

    assert len(results) == 1
    assert results[0]["model_id"] == "model-1"
    assert (results[0]["source"], results[0]["target"]) == (2, 0)
    assert results[0]["config"] == cfg


def _write_old_format(directory, name, result, summary):
    """The loose .eqx/.npz/.json files that impulse runs wrote before impulse records."""

    eqx.tree_serialise_leaves(directory / f"{name}.eqx", result.best_intervention)
    np.savez_compressed(
        directory / f"{name}.npz",
        initial_states=np.asarray(result.initial_states),
        perturbed_trajectory=np.asarray(result.perturbed_trajectory),
    )
    (directory / f"{name}.json").write_text(json.dumps(summary))


@pytest.mark.parametrize("spatial, duration", [("local", 1), ("patch", 3)])
def test_old_format_interventions_reload_without_the_model(tmp_path, spatial, duration):
    value = yaml.safe_load(Path("Experiments/impulse/conf/base_config.yaml").read_text())
    value["impulse"]["intervention"].update(
        channels="all", spatial=spatial, patch_size=3, duration=duration
    )
    cfg = impulse_experiment_config_from_mapping(value)
    model = RoundingModel()
    # [pattern, time, C, H, W]
    trajectories = jnp.stack([jnp.full((1, 2, 6, 6), k + 0.2) for k in range(2)])
    intervention = build_intervention(
        cfg.impulse.intervention, 2, model, trajectories, jax.random.PRNGKey(1)
    )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=ExternalTargetPairSource(trajectories[:1, 0], trajectories[1:, 0]),
        intervention=intervention,
        objective=TargetedObjective(),
        optimiser=optax.adam(0.1),
        observed_channels=2,
        rollout_steps=4,
        loss_functions=[lambda x, y, **_: jnp.mean((x - y) ** 2, axis=(-3, -2, -1))],
    )
    result = optimiser.train(iterations=3, batch_size=2, key=jax.random.PRNGKey(2), evaluation_steps=4)
    # Old runs saved a schema 7 config with an output folder
    config = config_to_dict(cfg)
    config["schema_version"] = 7
    config["impulse"]["output"] = {"directory": "impulse_runs", "base_env": "IMPULSE_OUTPUT_PATH"}
    _write_old_format(tmp_path, "run", result, {"model_id": "model-1", "config": config})

    [saved] = load_impulse_results(tmp_path)
    loaded, initial_states = load_saved_intervention(saved)

    trained = result.best_intervention
    states = trajectories[:, 0]
    assert jnp.any(trained.values != 0)
    assert loaded.duration == duration
    assert jnp.array_equal(initial_states, result.initial_states)
    assert jnp.array_equal(loaded.deltas(states), trained.deltas(states))
    size = impulse_size(loaded, initial_states[:1])
    assert size["impulse_l2"] > 0 and size["footprint"] > 0
    row = summary_row(saved)
    assert (row["spatial"], row["duration"], row["transition"]) == (spatial, duration, "0 -> 1")
    assert row["budget"] == (9 * duration if spatial == "patch" else None)


def test_interventions_saved_before_patch_mode_still_load(tmp_path):
    # Saved leaves of a local intervention before PATCH_SIZE existed, in field order
    location, values = jnp.asarray([0.3, -0.2]), jnp.full((1, 2, 4, 4), 0.5)
    eqx.tree_serialise_leaves(tmp_path / "old.eqx", [location, values, 2, 2, 0.1])
    skeleton = perturbation(
        mode={"channel": "all", "spatial": "local"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=jnp.zeros((1, 2, 4, 4)),
        WIDTH=0.1,
        key=jax.random.PRNGKey(0),
    )

    loaded = eqx.tree_deserialise_leaves(tmp_path / "old.eqx", skeleton)

    assert jnp.array_equal(loaded.location, location)
    assert jnp.array_equal(loaded.values, values)


def test_graft_copies_the_target_into_a_window_around_the_centre():
    sources, targets = jnp.zeros((2, 2, 9, 9)), jnp.ones((2, 2, 9, 9))

    grafted = graft(sources, targets, 3, (0.5, 0.5))

    # Pixel i is at i / 8, so (0.5, 0.5) is pixel (4, 4)
    expected = jnp.zeros((9, 9)).at[3:6, 3:6].set(1.0)
    assert jnp.array_equal(grafted, jnp.broadcast_to(expected, (2, 2, 9, 9)))
    assert jnp.array_equal(graft(sources, targets, 0, (0.5, 0.5)), sources)
    assert jnp.array_equal(graft(sources, targets, 18, (0.0, 1.0)), targets)


def test_graft_switch_map_shows_that_a_small_graft_stays_a_stable_mixed_state():
    # Every pixel of the rounding model keeps its own integer, so a small
    # graft neither spreads nor dies out: only a graft of the whole lattice switches.
    sources, targets = jnp.zeros((2, 2, 8, 8)), jnp.ones((2, 2, 8, 8))
    references = jnp.stack([sources[0], targets[0]])

    rates = graft_switch_map(
        RoundingModel(), sources, targets, references, 1, [0, 3, 16], [(0.5, 0.5)], 4, 2, jax.random.PRNGKey(0)
    )

    assert rates.tolist() == [[0.0], [0.0], [1.0]]
    final = grow_attractors(RoundingModel(), graft(sources[:1], targets[:1], 3, (0.5, 0.5)), 4, 1, jax.random.PRNGKey(0))
    assert float(final.sum()) == 2 * 9
