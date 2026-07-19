import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from NCA.trainer.sycl_batching import apply_flat_batched_nca
from NCA.trainer.sycl_execution import SyclTwoTileExecution
from NCA.trainer.sycl_scan import scan_carry_only
from NCA.trainer.sycl_shard_map import filter_shard_map


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
    try:
        SyclTwoTileExecution._remove_local_tile_axis(states, "state")
    except ValueError as error:
        assert "exactly one outer-B state leaf" in str(error)
    else:
        raise AssertionError("A global two-tile input was accepted as local data")


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
    compiled = mapped.lower(states, states, 2).compile()
    loss, final = compiled(states, states, 2)

    assert loss.shape == ()
    assert final.shape == (1, 4, 5, 3, 3)


def test_compiled_filtered_shard_map_reuses_the_direct_mapped_callable():
    devices = np.asarray(jax.devices()[:1])
    mesh = Mesh(devices, ("tile",))
    traces = []

    def step(states):
        traces.append(states.shape)
        assert states.shape[0] == 1
        return states + 1.0

    mapped = filter_shard_map(
        step,
        mesh=mesh,
        in_specs=(P("tile"),),
        out_specs=P("tile"),
        check_rep=False,
    )
    sharding = NamedSharding(mesh, P("tile"))
    states = jax.device_put(jnp.zeros((1, 2, 3)), sharding)
    compiled = mapped.lower(states).compile()

    first = compiled(states)
    second = compiled(states)
    assert jnp.array_equal(first, jnp.ones_like(states))
    assert jnp.array_equal(second, jnp.ones_like(states))
    assert traces == [(1, 2, 3)]


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
