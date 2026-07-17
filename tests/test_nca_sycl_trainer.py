import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from NCA.trainer.sycl_batching import (
    apply_flat_batched_nca,
    shape_probe_losses,
    shape_probe_rollout,
)


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


def test_two_tile_shape_probe_does_not_interpret_unmapped_keys():
    states = jnp.zeros((2, 4, 32, 5, 6), dtype=jnp.float32)
    final, trajectory = shape_probe_rollout(states, 8)

    assert final.shape == (2, 4, 32, 5, 6)
    assert trajectory.shape == (2, 8, 4, 32, 5, 6)
    assert shape_probe_losses(final).shape == (2, 4)


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
