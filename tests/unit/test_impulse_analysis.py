import json

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml
from pathlib import Path

from Experiments.config import config_to_dict, impulse_experiment_config_from_mapping
from Experiments.impulse.analysis import (
    distance_over_time,
    grow_attractors,
    load_impulse_results,
    move_intervention,
    scale_intervention,
    success_rate,
    switch_matrix,
)
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
