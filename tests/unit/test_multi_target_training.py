import jax
import jax.numpy as jnp
import pytest

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from Common.trainer import loss_multi_target
from Common.trainer.loss_multi_target import multi_target_loss
from NCA.trainer.data_augmenter.micropattern import DataAugmenter

COMPONENT_NAMES = ("l2", "texture", "channel_mean", "radial", "correlation")


def test_multi_target_passes_texture_augmentation_flags(monkeypatch):
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    prediction = jnp.zeros((2, 1, schema.n_state_channels, 4, 4))
    target = jnp.zeros((2, 1, schema.n_measurement_channels, 4, 4))
    observed = []

    def texture_cost(
        *args, random_channel_shuffle=False, random_crop=False
    ):
        observed.append((random_channel_shuffle, random_crop))
        x, y = args[:2]
        return jnp.zeros((x.shape[0], y.shape[0], x.shape[1]))

    monkeypatch.setattr(loss_multi_target, "_texture_cost", texture_cost)
    loss_multi_target.multi_target_pairwise_costs(
        prediction,
        target,
        jnp.ones((4, 4), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(0),
        {
            "random_channel_shuffle": True,
            "random_crop": True,
            "multi_target_weights": {"texture": 1.0},
        },
    )

    assert observed == [(True, True)]


def test_multi_target_loss_is_invariant_to_groupwise_batch_order():
    schema = MICROPATTERN_260726_SCHEMA
    prediction = jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 10, 8, 8))
    target = jnp.take(prediction, jnp.asarray(schema.target_to_state), axis=2)
    for time in range(2):
        for group, channels in enumerate(schema.group_measurement_indices):
            order = jnp.roll(jnp.arange(3), time + group)
            target = target.at[:, time, channels].set(target[order, time][:, channels])

    loss, components = multi_target_loss(
        prediction,
        target,
        jnp.ones((8, 8), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(1),
        {"multi_target_weights": {"texture": 0.0}},
    )

    assert jnp.allclose(loss, 0.0, atol=1e-5)
    component_total = sum(components[name] for name in COMPONENT_NAMES)
    assert jnp.allclose(loss, component_total + components["assignment_regularisation"])
    for group in schema.group_names:
        group_total = sum(components[f"group/{group}/{name}"] for name in COMPONENT_NAMES)
        assert jnp.allclose(components[f"group/{group}/total"], group_total)


def test_multi_target_loss_accepts_a_selected_schema():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    prediction = jax.random.uniform(
        jax.random.PRNGKey(10), (3, 2, schema.n_state_channels, 8, 8)
    )
    target = jnp.take(prediction, jnp.asarray(schema.target_to_state), axis=2)

    loss, components = multi_target_loss(
        prediction,
        target,
        jnp.ones((8, 8), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(11),
        {"multi_target_weights": {"texture": 0.0}},
    )

    assert jnp.allclose(loss, 0.0, atol=1e-5)
    assert "group/cell_fate_s1/total" in components
    assert not any(name.startswith("group/rna_expression/") for name in components)


def test_multi_target_loss_ignores_missing_group_timepoints():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    prediction = jax.random.uniform(
        jax.random.PRNGKey(30), (2, 2, schema.n_state_channels, 4, 4)
    )
    target = jnp.take(prediction, jnp.asarray(schema.target_to_state), axis=2)
    target = target.at[1, 1].set(1000.0)
    measurement_mask = jnp.ones(target.shape[:3], dtype=bool)
    measurement_mask = measurement_mask.at[1, 1].set(False)

    loss, _ = multi_target_loss(
        prediction,
        target,
        jnp.ones((4, 4), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(31),
        {"multi_target_weights": {"texture": 0.0}},
        measurement_mask=measurement_mask,
    )

    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_multi_target_assignment_does_not_cross_intervention_conditions():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    prediction = jnp.stack(
        [
            jnp.zeros((1, schema.n_state_channels, 2, 2)),
            jnp.ones((1, schema.n_state_channels, 2, 2)),
        ]
    )
    target = jnp.take(prediction[::-1], jnp.asarray(schema.target_to_state), axis=2)
    arguments = {
        "multi_target_weights": {
            "l2": 1.0,
            "texture": 0.0,
            "channel_mean": 0.0,
            "radial": 0.0,
            "correlation": 0.0,
        }
    }

    unrestricted, _ = multi_target_loss(
        prediction,
        target,
        jnp.ones((2, 2), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(33),
        arguments,
    )
    condition_matched, _ = multi_target_loss(
        prediction,
        target,
        jnp.ones((2, 2), dtype=bool),
        schema,
        None,
        jax.random.PRNGKey(33),
        arguments,
        assignment_groups=(-1, 0),
    )

    assert jnp.allclose(unrestricted, 0.0)
    assert jnp.all(condition_matched > 0)


def test_snapshot_reinjection_skips_unmeasured_channels():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    data = jnp.ones((2, 3, schema.n_measurement_channels, 2, 2))
    measurement_mask = jnp.zeros((2, 2, schema.n_measurement_channels), dtype=bool)
    augmenter = DataAugmenter(
        data,
        schema=schema,
        measurement_mask=measurement_mask,
        intermediate_reinjection_probability=1.0,
    )
    augmenter.noise_strength = 0.0
    states = [
        jnp.zeros((2, schema.n_state_channels, 2, 2)) for _ in range(2)
    ]
    targets = [value[1:] for value in data]

    result, _ = augmenter.advance_pool(
        states, targets, 0, jax.random.PRNGKey(32)
    )

    assert jnp.all(jnp.stack(result)[:, 1:] == 0)


def test_snapshot_reinjection_donors_preserve_intervention_condition():
    augmenter = object.__new__(DataAugmenter)
    augmenter.intervention_times = (-1, -1, 0, 0, 24, 24)
    labels = jnp.asarray(augmenter.intervention_times)

    for seed in range(10):
        donors = augmenter._matched_donors(
            jax.random.PRNGKey(seed), 6, jnp.arange(6)
        )
        assert jnp.array_equal(labels[donors], labels)


def test_multi_target_l2_is_computed_within_channel_groups():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1"])
    prediction = jnp.zeros((1, 1, schema.n_state_channels, 2, 2))
    target = jnp.ones((1, 1, schema.n_measurement_channels, 2, 2))
    boundary = jnp.asarray([[True, False], [False, False]])

    loss, components = multi_target_loss(
        prediction,
        target,
        boundary,
        schema,
        None,
        jax.random.PRNGKey(12),
        {
            "multi_target_weights": {
                "l2": 1.0,
                "texture": 0.0,
                "channel_mean": 0.0,
                "radial": 0.0,
                "correlation": 0.0,
            }
        },
    )

    assert jnp.allclose(loss, 1.0)
    assert jnp.allclose(components["l2"], 1.0)
    assert jnp.allclose(components["group/cell_fate_s1/l2"], 1.0)


def test_soft_assignment_components_reconstruct_loss():
    schema = MICROPATTERN_260726_SCHEMA
    key = jax.random.PRNGKey(4)
    prediction = jax.random.uniform(key, (3, 2, 10, 8, 8))
    target = jax.random.uniform(jax.random.fold_in(key, 1), (3, 2, 14, 8, 8))

    loss, components = multi_target_loss(
        prediction,
        target,
        jnp.ones((8, 8), dtype=bool),
        schema,
        None,
        key,
        {
            "assignment": "softmin",
            "assignment_tau": 0.05,
            "multi_target_weights": {"texture": 0.0},
        },
    )

    component_total = sum(components[name] for name in COMPONENT_NAMES)
    assert jnp.allclose(loss, component_total + components["assignment_regularisation"])
    assert jnp.all(components["assignment_entropy"] >= 0)
    assert components["group/rna_expression/texture"].shape == loss.shape


def test_snapshot_augmenter_outputs_unique_state_and_measurement_targets():
    data = jax.random.uniform(jax.random.PRNGKey(2), (3, 5, 14, 8, 8))
    augmenter = DataAugmenter(data, hidden_channels=2)
    augmenter.noise_strength = 0.0

    x, y = augmenter.initialize_pool(jax.random.PRNGKey(3))

    assert len(x) == len(y) == 3
    assert x[0].shape == (4, 12, 8, 8)
    assert y[0].shape == (4, 16, 8, 8)
    assert augmenter.OBS_CHANNELS == 10
    assert DataAugmenter.schema is MICROPATTERN_260726_SCHEMA


def test_reinjection_preserves_group_specific_duplicate_measurements():
    schema = MICROPATTERN_260726_SCHEMA
    batch, time, channel = jnp.meshgrid(
        jnp.arange(3), jnp.arange(5), jnp.arange(14), indexing="ij"
    )
    data = (1000 * batch + 100 * time + channel)[..., None, None].astype(jnp.float32)
    data = jnp.broadcast_to(data, (3, 5, 14, 2, 2))
    augmenter = DataAugmenter(data)
    augmenter.noise_strength = 0.0
    zeros = [jnp.zeros((4, 10, 2, 2)) for _ in range(3)]
    targets = [value[1:] for value in data]
    observed = {"cell_fate_s2": False, "protein_response": False}

    for seed in range(20):
        result, _ = augmenter.advance_pool(zeros, targets, 0, jax.random.PRNGKey(seed))
        result = jnp.stack(result)
        for batch_index in range(3):
            for time_index in range(1, 4):
                values = result[batch_index, time_index, :, 0, 0]
                if values[4] != 0:
                    donor = jnp.rint((values[4] - 100 * time_index - 5) / 1000).astype(int)
                    assert values[2] == 1000 * donor + 100 * time_index + 4
                    assert values[1] == 1000 * donor + 100 * time_index + 6
                    assert values[0] == 1000 * donor + 100 * time_index + 7
                    observed["cell_fate_s2"] = True
                if values[8] != 0:
                    donor = jnp.rint((values[8] - 100 * time_index - 11) / 1000).astype(int)
                    assert values[0] == 1000 * donor + 100 * time_index + 12
                    assert values[9] == 1000 * donor + 100 * time_index + 13
                    observed["protein_response"] = True

    assert all(observed.values())


def test_sharded_reinjection_matches_global_two_and_four_batch_permutations():
    for batch_count in (2, 4):
        data = jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(20), batch_count),
            (batch_count, 5, 14, 2, 2),
        )
        key = jax.random.fold_in(jax.random.PRNGKey(21), batch_count)
        global_augmenter = DataAugmenter(data)
        global_augmenter.noise_strength = 0.0
        zeros = [jnp.zeros((4, 10, 2, 2)) for _ in range(batch_count)]
        targets = [value[1:] for value in data]
        expected, _ = global_augmenter.advance_pool(zeros, targets, 0, key)

        split = batch_count // 2
        actual = []
        for indices in (jnp.arange(split), jnp.arange(split, batch_count)):
            local = DataAugmenter(data)
            local.noise_strength = 0.0
            local._global_batch_indices = indices
            local._sharded_global_key = key
            result, _ = local.advance_pool(
                [zeros[int(index)] for index in indices],
                [targets[int(index)] for index in indices],
                0,
                jax.random.fold_in(key, int(indices[0]) + 1),
            )
            actual.extend(result)

        assert jnp.allclose(jnp.stack(actual), jnp.stack(expected))


def test_snapshot_reinjection_resets_initial_state_and_honours_decay_start():
    batch, time, channel = jnp.meshgrid(
        jnp.arange(3), jnp.arange(5), jnp.arange(14), indexing="ij"
    )
    data = (1 + 1000 * batch + 100 * time + channel)[..., None, None].astype(
        jnp.float32
    )
    data = jnp.broadcast_to(data, (3, 5, 14, 2, 2))
    augmenter = DataAugmenter(
        data,
        intermediate_reinjection_probability=0.0,
        intermediate_reinjection_probability_end=1.0,
        intermediate_reinjection_decay_start_fraction=0.5,
        intermediate_reinjection_total_iterations=100,
    )
    augmenter.noise_strength = 0.0
    stale = [jnp.zeros((4, 10, 2, 2)) for _ in range(3)]
    targets = [value[1:] for value in data]

    before_decay, _ = augmenter.advance_pool(
        stale, targets, 25, jax.random.PRNGKey(30)
    )
    before_decay = jnp.stack(before_decay)
    assert jnp.all(before_decay[:, 0] != 0.0)
    assert jnp.all(before_decay[:, 1:] == 0.0)

    after_decay, _ = augmenter.advance_pool(
        stale, targets, 100, jax.random.PRNGKey(30)
    )
    after_decay = jnp.stack(after_decay)
    assert jnp.all(jnp.any(after_decay[:, 1:] != 0.0, axis=(2, 3, 4)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intermediate_reinjection_probability": -0.1},
        {"intermediate_reinjection_probability_end": 1.1},
        {"intermediate_reinjection_decay_start_fraction": 1.0},
    ],
)
def test_snapshot_reinjection_rejects_invalid_schedule(kwargs):
    data = jnp.zeros((2, 5, 14, 2, 2))
    with pytest.raises(ValueError):
        DataAugmenter(data, **kwargs)
