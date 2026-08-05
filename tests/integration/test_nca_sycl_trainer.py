import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from Common.trainer.loss_multi_target import multi_target_loss
from NCA.trainer.backend.sycl.batching import apply_flat_batched_nca
from NCA.trainer.backend.sycl.trainer import (
    configure_custom_call_synchronization,
    configure_regulariser_reduction,
    configure_stage_synchronization,
)
from NCA.trainer.backend.sycl.execution import SyclTwoTileExecution
from NCA.trainer.backend.sycl.scan import scan_carry_only
from NCA.trainer.backend.sycl.shard_map import filter_shard_map


class _BatchableReferenceModel:
    """Small portable stand-in for the SYCL model's two call paths."""

    @staticmethod
    def _update(state, key):
        return state + jnp.asarray(key[0], dtype=state.dtype)

    def __call__(self, state, boundary_callback, key):
        return boundary_callback(self._update(state, key))

    def batched_call(self, states, keys):
        offsets = keys[:, 0].astype(states.dtype)
        return states + offsets[:, None, None, None]


class _RecordingBatchableReferenceModel(_BatchableReferenceModel):
    def __init__(self):
        self.batch_sizes = []

    def batched_call(self, states, keys):
        self.batch_sizes.append(states.shape[0])
        return super().batched_call(states, keys)


def _two_device_execution():
    """Build an execution policy over the first two test devices."""
    if len(jax.devices()) < 2:
        pytest.skip("requires two devices")
    devices = list(jax.devices()[:2])
    mesh = Mesh(np.asarray(devices), (SyclTwoTileExecution.AXIS_NAME,))
    execution = object.__new__(SyclTwoTileExecution)
    execution.devices = devices
    execution.mesh = mesh
    execution.replicated_sharding = NamedSharding(mesh, P())
    execution.tile_sharding = NamedSharding(
        mesh, P(SyclTwoTileExecution.AXIS_NAME)
    )
    return execution


def test_custom_call_synchronization_configuration(monkeypatch):
    name = "NCA_SYCL_SYNCHRONIZE_CUSTOM_CALLS"
    monkeypatch.delenv(name, raising=False)
    assert not configure_custom_call_synchronization(None)
    assert configure_custom_call_synchronization(True)
    assert os.environ[name] == "1"
    assert not configure_custom_call_synchronization(False)
    assert name not in os.environ


def test_stage_synchronization_configuration(monkeypatch):
    name = "NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION"
    monkeypatch.delenv(name, raising=False)
    assert not configure_stage_synchronization(None)
    assert configure_stage_synchronization(True)
    assert os.environ[name] == "1"
    assert not configure_stage_synchronization(False)
    assert name not in os.environ


def test_regulariser_reduction_configuration(monkeypatch):
    name = "NCA_SYCL_REGULARISER_REDUCTION"
    monkeypatch.delenv(name, raising=False)
    assert configure_regulariser_reduction(None) == "atomic"
    assert configure_regulariser_reduction("two_stage") == "two_stage"
    assert os.environ[name] == "two_stage"
    assert configure_regulariser_reduction("atomic") == "atomic"
    with pytest.raises(ValueError, match="must be 'atomic', 'two_stage'"):
        configure_regulariser_reduction("invalid")


def test_tile_axis_contract_is_explicit_and_preserves_inner_batch_axis():
    states = [
        jnp.zeros((1, 4, 32, 7, 9), dtype=jnp.float32),
    ]
    keys = jnp.zeros((1, 2), dtype=jnp.uint32)

    local_states = SyclTwoTileExecution._remove_local_tile_axis(
        states, "state"
    )
    local_key = SyclTwoTileExecution._remove_local_tile_axis(keys, "PRNG key")

    assert local_states[0].shape == (4, 32, 7, 9)
    assert local_key.shape == (2,)
    restored = SyclTwoTileExecution._add_local_tile_axis(
        local_states, "state"
    )
    assert restored[0].shape == (1, 4, 32, 7, 9)


def test_tile_axis_contract_rejects_unsharded_two_tile_input():
    states = [jnp.zeros((2, 4, 32, 7, 9), dtype=jnp.float32)]
    with pytest.raises(ValueError, match="one physical value"):
        SyclTwoTileExecution._remove_local_tile_axis(states, "state")


def test_outer_batches_must_split_evenly_between_tiles():
    with pytest.raises(ValueError, match="positive multiple of 2"):
        SyclTwoTileExecution._split_between_tiles([0, 1, 2], "batches")


def test_four_outer_batches_pack_as_two_slots_and_restore_original_order():
    execution = _two_device_execution()
    batches = [
        jnp.full((3, 5), batch, dtype=jnp.float32) for batch in range(4)
    ]

    packed = execution._pack_items(
        batches, "test batches", expected_ndim=2
    )
    restored = execution._unpack_slots(packed)

    assert len(packed) == 2
    assert all(slot.shape == (2, 3, 5) for slot in packed)
    assert len(restored) == 4
    for batch, expected in zip(restored, batches):
        assert np.array_equal(np.asarray(batch), np.asarray(expected))


def test_data_augmenter_state_is_partitioned_by_contiguous_batch_halves():
    execution = _two_device_execution()

    class Augmenter:
        pass

    augmenter = Augmenter()
    augmenter.data_true = [jnp.asarray([batch]) for batch in range(4)]
    augmenter.data_saved = [jnp.asarray([10 + batch]) for batch in range(4)]
    augmenter.channel_timestep_mask = jnp.arange(8).reshape(4, 2)
    augmenter.knockout_times = jnp.arange(4)

    left = execution._slice_data_augmenter(augmenter, (0, 1), 0)
    right = execution._slice_data_augmenter(augmenter, (2, 3), 1)

    assert [int(value[0]) for value in left.data_true] == [0, 1]
    assert [int(value[0]) for value in right.data_true] == [2, 3]
    assert np.array_equal(
        np.asarray(left.channel_timestep_mask), [[0, 1], [2, 3]]
    )
    assert np.array_equal(np.asarray(right.knockout_times), [2, 3])
    assert all(value.device == execution.devices[0] for value in left.data_saved)
    assert all(value.device == execution.devices[1] for value in right.data_saved)


def test_global_donor_augmenter_keeps_full_truth_pool_on_each_tile():
    execution = _two_device_execution()

    class Augmenter:
        supports_global_donor_pool = True

    augmenter = Augmenter()
    augmenter.data_true = [jnp.asarray([batch]) for batch in range(4)]
    augmenter.data_saved = [jnp.asarray([10 + batch]) for batch in range(4)]

    left = execution._slice_data_augmenter(augmenter, (0, 1), 0)
    right = execution._slice_data_augmenter(augmenter, (2, 3), 1)

    assert [int(value[0]) for value in left.data_saved] == [10, 11, 12, 13]
    assert [int(value[0]) for value in right.data_saved] == [10, 11, 12, 13]
    assert np.array_equal(np.asarray(left._global_batch_indices), [0, 1])
    assert np.array_equal(np.asarray(right._global_batch_indices), [2, 3])


def test_pool_injections_preserve_global_half_rate_across_tiles():
    assert SyclTwoTileExecution._allocate_injections([6, 6], 0) == [3, 3]
    assert sum(SyclTwoTileExecution._allocate_injections([3, 4], 0)) == 3
    assert SyclTwoTileExecution._allocate_injections([0, 0], 0) == [0, 0]


def test_filtered_shard_map_traces_scan_with_tile_local_shapes():
    devices = np.asarray(jax.devices()[:1])
    mesh = Mesh(devices, ("tile",))

    def step(states, targets, steps):
        assert states.shape[0] == 1
        local_states = states[0]
        local_targets = targets[0]

        def scan_step(state, _):
            if state.ndim != 4:
                raise AssertionError(f"scan received non-local shape {state.shape}")
            return state + 1.0, None

        final = scan_carry_only(
            scan_step,
            local_states,
            jnp.arange(steps),
            kind="checkpointed",
        )
        loss = jnp.mean((final[:, :3] - local_targets[:, :3]) ** 2)
        return loss, final[None]

    mapped = filter_shard_map(
        step,
        mesh=mesh,
        in_specs=(P("tile"), P("tile"), P()),
        out_specs=(P(), P("tile")),
        check_rep=False,
    )
    sharding = NamedSharding(mesh, P("tile"))
    states = jax.device_put(
        jnp.zeros((1, 4, 5, 3, 3), dtype=jnp.float32), sharding
    )
    loss, final = eqx.filter_jit(mapped)(states, states, 2)

    assert loss.shape == ()
    assert final.shape == (1, 4, 5, 3, 3)


def test_gradient_wraps_shard_map_for_data_parallel_loss():
    if len(jax.devices()) < 2:
        pytest.skip("requires two devices")
    devices = np.asarray(jax.devices()[:2])
    mesh = Mesh(devices, ("tile",))
    sharding = NamedSharding(mesh, P("tile"))
    states = jax.device_put(
        jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5),
        sharding,
    )

    def local_loss(weight, state_shard):
        assert state_shard.shape == (1, 3, 5)
        state = state_shard[0]

        def body(carry, _):
            return jnp.tanh(carry * weight), None

        final = scan_carry_only(
            body, state, jnp.arange(4), kind="checkpointed"
        )
        loss = jax.lax.pmean(jnp.mean(final**2), "tile")
        return loss, final[None]

    distributed_loss = filter_shard_map(
        local_loss,
        mesh=mesh,
        in_specs=(P(), P("tile")),
        out_specs=(P(), P("tile")),
        check_rep=False,
    )
    (actual_loss, actual_final), actual_gradient = jax.jit(
        jax.value_and_grad(distributed_loss, has_aux=True)
    )(jnp.asarray(0.9), states)

    def reference(weight):
        def one(state):
            def body(carry, _):
                return jnp.tanh(carry * weight), None

            final = scan_carry_only(
                body, state, jnp.arange(4), kind="checkpointed"
            )
            return jnp.mean(final**2), final

        losses, finals = jax.vmap(one)(states)
        return jnp.mean(losses), finals

    (expected_loss, expected_final), expected_gradient = jax.value_and_grad(
        reference, has_aux=True
    )(jnp.asarray(0.9))
    assert jnp.allclose(actual_loss, expected_loss, atol=1e-6)
    assert jnp.allclose(actual_final, expected_final, atol=1e-6)
    assert jnp.allclose(actual_gradient, expected_gradient, atol=1e-6)


def test_multi_target_cost_gather_matches_global_loss_and_gradient():
    execution = _two_device_execution()
    schema = MICROPATTERN_260726_SCHEMA
    args = {"multi_target_weights": {"texture": 0.0}}
    boundary = jnp.ones((4, 4), dtype=bool)
    key = jax.random.PRNGKey(11)

    def local_loss(local_prediction, local_target):
        losses, _ = execution.multi_target_loss(
            local_prediction,
            local_target,
            boundary,
            schema,
            None,
            key,
            args,
        )
        return jax.lax.pmean(jnp.mean(losses), execution.AXIS_NAME)

    distributed_loss = filter_shard_map(
        local_loss,
        mesh=execution.mesh,
        in_specs=(P(execution.AXIS_NAME), P(execution.AXIS_NAME)),
        out_specs=P(),
        check_rep=False,
    )

    for batch_count in (2, 4):
        prediction = jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(10), batch_count),
            (batch_count, 2, schema.n_state_channels, 4, 4),
        )
        target = jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(12), batch_count),
            (batch_count, 2, schema.n_measurement_channels, 4, 4),
        )
        expected = lambda value: jnp.mean(
            multi_target_loss(
                value, target, boundary, schema, None, key, args
            )[0]
        )
        expected_loss, expected_gradient = jax.value_and_grad(expected)(prediction)
        actual_loss, actual_gradient = jax.jit(jax.value_and_grad(distributed_loss))(
            jax.device_put(prediction, execution.tile_sharding),
            jax.device_put(target, execution.tile_sharding),
        )
        assert jnp.allclose(actual_loss, expected_loss, atol=1e-6)
        assert jnp.allclose(actual_gradient, expected_gradient, atol=1e-5)


def test_sharded_loss_evenly_processes_four_outer_batches():
    execution = _two_device_execution()
    batches = [
        jnp.arange(12, dtype=jnp.float32).reshape(3, 4) + 10.0 * batch
        for batch in range(4)
    ]
    targets = [0.25 * batch for batch in batches]
    packed_batches = execution._pack_items(batches, "states", expected_ndim=2)
    packed_targets = execution._pack_items(targets, "targets", expected_ndim=2)
    tile_keys = execution._make_tile_array(
        jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    )

    def local_loss(weight, static_scale, states, local_targets, steps, key):
        del steps, key
        losses = jnp.stack(
            [
                jnp.mean((weight * state - target) ** 2)
                for state, target in zip(states, local_targets)
            ]
        )
        mean_loss, regularisers = execution.synchronise_loss(
            static_scale * jnp.mean(losses), {}
        )
        return mean_loss, (states, states, losses, regularisers, {})

    distributed_loss = execution.transform_loss(local_loss)
    weight = jnp.asarray(0.7, dtype=jnp.float32)
    static_scale = jnp.asarray(1.0, dtype=jnp.float32)
    (actual_loss, auxiliary), actual_gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(distributed_loss, has_aux=True)
    )(
        weight,
        static_scale,
        packed_batches,
        packed_targets,
        1,
        tile_keys,
    )

    def reference(candidate):
        losses = jnp.stack(
            [
                jnp.mean((candidate * state - target) ** 2)
                for state, target in zip(batches, targets)
            ]
        )
        return jnp.mean(losses)

    expected_loss, expected_gradient = jax.value_and_grad(reference)(weight)
    assert jnp.allclose(actual_loss, expected_loss, atol=1e-6)
    assert jnp.allclose(actual_gradient, expected_gradient, atol=1e-6)
    assert auxiliary[2].shape == (2, 2)


def test_host_callback_extracts_physical_single_device_shards():
    execution = _two_device_execution()
    devices = execution.devices
    mesh = Mesh(np.asarray(devices), ("tile",))
    sharding = NamedSharding(mesh, P("tile"))
    global_states = jax.device_put(
        jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5),
        sharding,
    )
    local_states = execution._local_tile_trees(
        [global_states], "test state"
    )

    assert len(local_states) == 2
    assert local_states[0][0].shape == (3, 5)
    assert local_states[1][0].shape == (3, 5)
    assert local_states[0][0].device == devices[0]
    assert local_states[1][0].device == devices[1]
    expected = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    assert np.array_equal(np.asarray(local_states[0][0]), expected[0])
    assert np.array_equal(np.asarray(local_states[1][0]), expected[1])


def test_checkpointed_carry_only_scan_matches_lax_value_and_gradient():
    xs = jnp.linspace(0.1, 0.4, 4, dtype=jnp.float32)

    def objective(initial, kind):
        def body(carry, value):
            return jnp.tanh(carry + value), None

        final = scan_carry_only(body, initial, xs, kind=kind)
        return jnp.sum(final**2)

    initial = jnp.linspace(-0.3, 0.3, 7, dtype=jnp.float32)
    lax_value, lax_gradient = jax.value_and_grad(objective)(initial, "lax")
    checkpointed_value, checkpointed_gradient = jax.value_and_grad(objective)(
        initial, "checkpointed"
    )

    assert jnp.allclose(checkpointed_value, lax_value, atol=1e-6)
    assert jnp.allclose(checkpointed_gradient, lax_gradient, atol=1e-6)


def test_sycl_trainer_flat_batch_matches_reference_tree_vmap():
    model = _BatchableReferenceModel()
    v_model = jax.vmap(model, in_axes=(0, None, 0), out_axes=0)
    reference = lambda x, callbacks, keys: jtu.tree_map(
        v_model, x, callbacks, keys
    )

    states = [
        jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5),
        jnp.arange(3 * 3 * 4 * 5, dtype=jnp.float32).reshape(3, 3, 4, 5),
    ]
    keys = [
        jnp.asarray([[1, 0], [2, 0]], dtype=jnp.uint32),
        jnp.asarray([[3, 0], [4, 0], [5, 0]], dtype=jnp.uint32),
    ]
    callbacks = [lambda value: 2.0 * value, lambda value: value - 7.0]

    expected = reference(states, callbacks, keys)
    actual = apply_flat_batched_nca(
        model, states, callbacks, keys, reference
    )

    assert jtu.tree_structure(actual) == jtu.tree_structure(expected)
    assert all(
        jnp.array_equal(actual_leaf, expected_leaf)
        for actual_leaf, expected_leaf in zip(
            jtu.tree_leaves(actual), jtu.tree_leaves(expected)
        )
    )


def test_sycl_trainer_keeps_outer_batch_leaves_as_separate_calls():
    model = _RecordingBatchableReferenceModel()
    states = [
        jnp.zeros((2, 3, 4, 5), dtype=jnp.float32),
        jnp.zeros((3, 3, 4, 5), dtype=jnp.float32),
    ]
    keys = [
        jnp.zeros((2, 2), dtype=jnp.uint32),
        jnp.zeros((3, 2), dtype=jnp.uint32),
    ]
    callbacks = [lambda value: value, lambda value: value]

    apply_flat_batched_nca(
        model,
        states,
        callbacks,
        keys,
        lambda *_: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    assert model.batch_sizes == [2, 3]


def test_concatenated_shared_weight_gradient_equals_example_sum():
    key_left, key_cotangent = jax.random.split(jax.random.PRNGKey(4))
    examples, cells, inputs, outputs = 5, 7, 6, 4
    activations = jax.random.normal(
        key_left, (examples, cells, inputs), dtype=jnp.float32
    )
    cotangents = jax.random.normal(
        key_cotangent, (examples, cells, outputs), dtype=jnp.float32
    )

    per_example = jax.vmap(lambda dy, x: dy.T @ x)(cotangents, activations)
    flat_gradient = (
        cotangents.reshape(examples * cells, outputs).T
        @ activations.reshape(examples * cells, inputs)
    )

    assert jnp.allclose(flat_gradient, jnp.sum(per_example, axis=0), atol=1e-5)
