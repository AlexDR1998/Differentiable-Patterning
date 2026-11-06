import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Common.dataloader.cell_types import CellTypeRules, label_names, pixel_label_shares

from NCA.inverse_design.density import (
    CellDensity,
    CellGaps,
    fill_level,
    pack_cells,
)
from NCA.inverse_design.geometry import FourierRadialGeometry, coordinate_grid
from NCA.inverse_design.initial_state import (
    PatchSamplingConfig,
    build_initial_state,
    sample_circular_patches,
)
from NCA.inverse_design.objectives import CellTypeShareObjective, GeometryPenalties
from NCA.inverse_design.optimise import (
    OptimisationConfig,
    evaluate_binary,
    optimise_geometry,
)
from NCA.inverse_design.rollout import rollout_final, rollout_trajectory


def test_fourier_geometry_has_gradients_for_every_parameter():
    geometry = FourierRadialGeometry(
        3,
        radius=0.6,
        key=jax.random.PRNGKey(0),
        initial_scale=0.1,
        angle=0.2,
    )
    grid = coordinate_grid((21, 19))

    gradients = jax.grad(
        lambda candidate: jnp.sum(
            candidate.occupancy(grid, 0.06)
            * jnp.linspace(0.5, 1.5, 21)[:, None]
        )
    )(geometry)

    assert jnp.isfinite(gradients.raw_radius)
    assert jnp.isfinite(gradients.raw_angle)
    assert jnp.all(jnp.isfinite(gradients.raw_cos))
    assert jnp.all(jnp.isfinite(gradients.raw_sin))
    assert gradients.raw_cos.shape == (3,)


def test_patch_sampling_is_key_driven_and_stops_gradients():
    reference = jnp.arange(2 * 15 * 15, dtype=jnp.float32).reshape(2, 15, 15)
    reference_mask = jnp.ones((15, 15), dtype=bool)
    config = PatchSamplingConfig(5, ((0, 1),))

    sample_a = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(4), config
    )
    sample_b = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(4), config
    )
    sample_c = sample_circular_patches(
        reference, reference_mask, (17, 13), jax.random.PRNGKey(5), config
    )
    gradient = jax.grad(
        lambda values: jnp.sum(
            sample_circular_patches(
                values, reference_mask, (10, 10), jax.random.PRNGKey(4), config
            )
        )
    )(reference)

    assert jnp.array_equal(sample_a, sample_b)
    assert not jnp.array_equal(sample_a, sample_c)
    assert jnp.all(gradient == 0)


def test_initial_state_remains_differentiable_through_occupancy():
    biology = jnp.ones((2, 7, 7))
    occupancy = jnp.full((7, 7), 0.4)

    gradient = jax.grad(
        lambda mask: jnp.sum(build_initial_state(biology, mask, 4, boundary_mode="soft"))
    )(occupancy)

    # Two biological channels plus the recurrent boundary channel.
    assert jnp.allclose(gradient, 3.0)


# Notochord as in the cell type files, plus a two-clause Mesoderm rule with
# clause weights and an overlapping Endoderm rule (so "Several" occurs).
RULES = CellTypeRules(
    channels={"SOX17": "s1/SOX17", "TBXT": "s2/TBXT", "FOXA2": "s2/FOXA2"},
    thresholds={"SOX17": 0.2, "TBXT": 0.13, "FOXA2": 0.4},
    rules={
        "Notochord": {"SOX17": "low", "TBXT": "high", "FOXA2": "high"},
        "Endoderm": {"SOX17": "high", "TBXT": "any", "FOXA2": "any"},
        "Mesoderm": [
            {"SOX17": "low", "TBXT": "high", "FOXA2": "low"},
            {"SOX17": "any", "TBXT": "high", "FOXA2": "any"},
        ],
    },
    clause_weights={"Mesoderm": [0.6, 0.3]},
)
STATE_CHANNELS = ("TBXT", "SOX17", "LEF1", "FOXA2", "hidden")


def test_cell_type_share_matches_rule_file_labels():
    # Values away from every threshold, so the soft share has saturated
    levels = jnp.asarray([0.05, 0.3, 0.6, 0.9])
    state = levels[jax.random.randint(jax.random.PRNGKey(1), (5, 12, 12), 0, 4)]
    values ={marker: np.asarray(state[STATE_CHANNELS.index(marker)]) for marker in RULES.markers}
    expected = pixel_label_shares(values, RULES)

    for label in label_names(RULES):
        objective = CellTypeShareObjective.from_rules(
            RULES, STATE_CHANNELS, label, temperature=1.0e-4
        )
        assert np.allclose(objective.probability(state, hard=True), expected[label], atol=1.0e-6)
        assert np.allclose(objective.probability(state), expected[label], atol=1.0e-3)


def test_cell_type_share_has_marker_gradients():
    objective = CellTypeShareObjective.from_rules(RULES, STATE_CHANNELS, "Notochord")
    state = jnp.full((5, 4, 4), 0.15)
    gradient = jax.grad(lambda x: jnp.sum(objective.probability(x)))(state)

    assert jnp.all(gradient[STATE_CHANNELS.index("TBXT")] > 0)
    assert jnp.all(gradient[STATE_CHANNELS.index("SOX17")] < 0)
    assert jnp.all(gradient[STATE_CHANNELS.index("LEF1")] == 0)


class _IdentityNCA(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=3)

    def __call__(self, state, boundary_callback, key):
        del key
        return boundary_callback(state)


class _DecayNCA(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=3)

    def __call__(self, state, boundary_callback, key):
        noise = 0.01 * jax.random.normal(key, state.shape)
        return boundary_callback(0.9 * state + noise)


def test_checkpointed_rollout_matches_lax_rollout():
    state = jax.random.uniform(jax.random.PRNGKey(2), (3, 9, 9))
    mask = jnp.ones((1, 9, 9))

    def final_sum(initial, scan_kind):
        return jnp.sum(
            rollout_final(_DecayNCA(), initial, mask, "soft", jax.random.PRNGKey(3), 6, scan_kind=scan_kind)
        )

    lax_value, lax_gradient = jax.value_and_grad(final_sum)(state, "lax")
    checkpointed_value, checkpointed_gradient = jax.value_and_grad(final_sum)(state, "checkpointed")

    assert jnp.allclose(lax_value, checkpointed_value)
    assert jnp.allclose(lax_gradient, checkpointed_gradient)


def test_trajectory_ends_on_the_final_rollout_state():
    state = jax.random.uniform(jax.random.PRNGKey(2), (3, 9, 9))
    mask = jnp.ones((1, 9, 9))
    key = jax.random.PRNGKey(3)

    trajectory = rollout_trajectory(_DecayNCA(), state, mask, "soft", key, 4)
    final = rollout_final(_DecayNCA(), state, mask, "soft", key, 4)

    assert trajectory.shape == (5, 3, 9, 9)
    assert jnp.allclose(trajectory[0], state)
    assert jnp.allclose(trajectory[-1], final)


def _first_channel_objective(**options):
    # Target: channel 0 high (one marker, so two patterns: low, high)
    return CellTypeShareObjective((0,), (0.2,), (0.0, 1.0), temperature=0.05, **options)


def test_optimiser_keeps_radius_fixed_and_updates_shape():
    initial = FourierRadialGeometry(
        2, radius=0.45, key=jax.random.PRNGKey(0), initial_scale=0.1
    )
    # Channel 0 is high only on the right of the grid, so a lopsided shape scores more.
    biology = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 15)[None, None, None, :], (2, 1, 15, 15)
    )
    logged = []

    result = optimise_geometry(
        _IdentityNCA(),
        initial,
        biology,
        jax.random.PRNGKey(8),
        _first_channel_objective(),
        OptimisationConfig(
            iterations=3,
            learning_rate=5.0e-2,
            total_steps=1,
            batch_size=2,
            mask_softness=0.08,
            final_mask_softness=0.02,
            noise_strength=0.001,
            fixed_radius=True,
            log_every=2,
            evaluation_seeds=2,
        ),
        log_fn=lambda iteration, metrics, evaluation: logged.append((iteration, evaluation)),
    )

    assert result.loss_history.shape == (3,)
    assert jnp.array_equal(result.geometry.raw_radius, initial.raw_radius)
    assert not jnp.array_equal(result.geometry.raw_cos, initial.raw_cos)
    assert result.evaluation_iterations == (0, 2, 3)
    assert [iteration for iteration, evaluation in logged if evaluation is not None] == [0, 2, 3]
    assert result.best_iteration in result.evaluation_iterations
    assert result.best_evaluation.scores.shape == (2,)


def test_binary_evaluation_counts_mask_pixels():
    geometry = FourierRadialGeometry(1, radius=0.5)
    biology = jnp.ones((1, 1, 21, 21))
    config = OptimisationConfig(iterations=1, learning_rate=1.0, total_steps=2, evaluation_seeds=3)

    high = evaluate_binary(
        _IdentityNCA(), geometry, biology, jax.random.PRNGKey(0), _first_channel_objective(), config
    )
    low = evaluate_binary(
        _IdentityNCA(), geometry, 0.0 * biology, jax.random.PRNGKey(0), _first_channel_objective(), config
    )

    assert set(np.unique(np.asarray(high.mask))) == {0.0, 1.0}
    assert np.allclose(high.scores, 1.0) and np.allclose(low.scores, 0.0)
    assert high.border_fraction == 0.0


def _lattice_cells(size, layers=1, softness=1.0e-3, seed=0):
    # 2x2 cells on a 3-pixel lattice (4/9 of the grid covered per layer), one
    # random value and rank per cell; the gaps are 0
    rng = np.random.default_rng(seed)
    textures = np.zeros((layers, 1, size, size), dtype=np.float32)
    ranks = np.full((layers, 1, size, size), -1.0, dtype=np.float32)
    for layer in range(layers):
        # Each layer's lattice is shifted by one pixel, so the next fills gaps
        for top in range(layer, size - 1 - layer, 3):
            for left in range(layer, size - 1 - layer, 3):
                textures[layer, 0, top:top + 2, left:left + 2] = rng.uniform(0.5, 1.0)
                ranks[layer, 0, top:top + 2, left:left + 2] = rng.random()
    gaps = CellGaps(
        gap_values=jnp.zeros(1),
        channel_panels=jnp.zeros(1, dtype=int),
        layer_coverage=jnp.full(layers, float(np.mean(ranks[0] >= 0))),
        panels=1,
        softness=softness,
    )
    return pack_cells(jnp.asarray(textures), jnp.asarray(ranks)), gaps


def test_fill_level_fills_the_layers_in_turn():
    coverage = jnp.asarray([0.5, 0.4])

    assert np.isclose(float(fill_level(0.25, coverage)), 0.5)
    assert np.isclose(float(fill_level(0.5, coverage)), 1.0)
    assert np.isclose(float(fill_level(0.6, coverage)), 1.5)
    assert np.isclose(float(fill_level(0.95, coverage)), 2.0)


def test_density_keeps_whole_cells_and_has_a_gradient():
    packed, gaps = _lattice_cells(60)
    textures, _ = gaps.unpack(packed)
    key = jax.random.PRNGKey(1)

    for density in (0.15, 0.3):
        image = gaps.apply(packed, CellDensity(density), key)[0]
        assert abs(float(jnp.mean(image > 0)) - density) < 0.04
        # Every cell is either kept whole with its own values or left out
        cells = np.asarray(image).reshape(20, 3, 20, 3)[:, :2, :, :2]
        values = np.asarray(textures[0, 0]).reshape(20, 3, 20, 3)[:, :2, :, :2]
        kept = np.all(np.isclose(cells, values, atol=1.0e-3), axis=(1, 3))
        empty = np.all(np.abs(cells) < 1.0e-3, axis=(1, 3))
        assert np.mean(kept | empty) > 0.98
    sparse = gaps.apply(packed, CellDensity(0.15), key)
    dense = gaps.apply(packed, CellDensity(0.3), key)
    assert bool(jnp.all((sparse > 0) <= (dense > 0)))
    assert not jnp.array_equal(sparse, gaps.apply(packed, CellDensity(0.15), jax.random.PRNGKey(2)))

    soft_packed, soft_gaps = _lattice_cells(60, softness=0.05)
    gradient = jax.grad(
        lambda raw: jnp.mean(soft_gaps.apply(soft_packed, eqx.tree_at(lambda d: d.raw, CellDensity(0.3), raw), key))
    )(CellDensity(0.3).raw)
    assert float(gradient) > 0.0


def test_second_layer_fills_between_the_first():
    packed, gaps = _lattice_cells(60, layers=2)

    covered = gaps.coverage(packed, CellDensity(0.6), jax.random.PRNGKey(0))

    assert float(gaps.max_density) > 0.6
    # The shifted lattices are not quite independent, so only roughly 0.6
    assert abs(float(jnp.mean(covered)) - 0.6) < 0.08
    assert float(jnp.mean(covered)) > float(gaps.layer_coverage[0])


def test_optimiser_learns_the_density_and_can_keep_the_geometry():
    # Channel 0 is high wherever there are cells, so more cells score more.
    initial = FourierRadialGeometry(2, radius=0.45, key=jax.random.PRNGKey(0), initial_scale=0.1)
    packed, gaps = _lattice_cells(15, softness=0.1)
    biology = jnp.stack([packed, packed])
    config = OptimisationConfig(
        iterations=3, learning_rate=0.2, total_steps=1, batch_size=2,
        fixed_radius=True, optimise_geometry=False, evaluation_seeds=2,
    )
    logged = []

    result = optimise_geometry(
        _IdentityNCA(), initial, biology, jax.random.PRNGKey(8), _first_channel_objective(), config,
        log_fn=lambda iteration, metrics, evaluation: logged.append(metrics),
        gaps=gaps, density=CellDensity(0.3),
    )

    assert float(result.density.value) > 0.3
    assert jnp.array_equal(result.geometry.raw_cos, initial.raw_cos)
    assert result.best_density is not None and result.best_evaluation.density is not None
    assert "density" in logged[0]


def test_density_needs_both_parts_and_something_to_optimise():
    import pytest

    geometry = FourierRadialGeometry(1, radius=0.5)
    biology = jnp.ones((1, 1, 9, 9))
    config = OptimisationConfig(iterations=1, learning_rate=1.0, total_steps=1, optimise_geometry=False)
    with pytest.raises(ValueError, match="Nothing to optimise"):
        optimise_geometry(_IdentityNCA(), geometry, biology, jax.random.PRNGKey(0), _first_channel_objective(), config)
    with pytest.raises(ValueError, match="both"):
        evaluate_binary(
            _IdentityNCA(), geometry, biology, jax.random.PRNGKey(0), _first_channel_objective(),
            config, density=CellDensity(0.5),
        )
