import jax
import jax.numpy as jnp
import equinox as eqx

from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from Common.trainer.abstract_data_augmenter_array import (
    DataAugmenterAbstract as LegacyArrayAugmenter,
)
from Common.trainer.batch_backend import make_batch_backend
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter
from NCA.trainer.data_augmenter_nca_terminal import TerminalCarryDataAugmenter
from NCA.trainer.data_augmenter_nca_texture import (
    DataAugmenter as TextureDataAugmenter,
)
from NCA.trainer.data_augmenter_nca_texture_array import (
    DataAugmenter as ArrayTextureDataAugmenter,
)
from NCA.trainer.data_augmenter_9ch_colony import (
    DataAugmenter as Grouped12DataAugmenter,
)
from NCA.trainer.data_augmenter_4ch_colony import (
    DataAugmenter as FourChannelDataAugmenter,
)
from NCA.trainer.NCA_regulariser import (
    boundary_regulariser,
    intermediate_reg,
    latent_size_regulariser,
)
from NCA.model.NCA_model_fast import NCA as FastNCA


def _data():
    return jnp.arange(2 * 4 * 3 * 5 * 6, dtype=jnp.float32).reshape(
        2, 4, 3, 5, 6
    ) / 1000.0


def test_dense_and_tree_basic_augmenters_have_fixed_key_parity():
    data = _data()
    tree = DataAugmenter(data, hidden_channels=1)
    dense = DataAugmenter(data, hidden_channels=1, batch_mode="array")

    tree_x, tree_y = tree.data_load(jax.random.PRNGKey(7))
    dense_x, dense_y = dense.data_load(jax.random.PRNGKey(7))

    assert isinstance(tree_x, list)
    assert dense_x.shape == (2, 3, 4, 5, 6)
    assert jnp.array_equal(jnp.stack(tree_x), dense_x)
    assert jnp.array_equal(jnp.stack(tree_y), dense_y)


def test_dense_augmenter_samples_the_time_axis():
    dense = DataAugmenter(_data(), hidden_channels=0, batch_mode="array")
    x, y = dense.split_x_y(1)

    sampled_x, sampled_y = dense.random_N_select(
        x, y, 2, jax.random.PRNGKey(4)
    )

    assert sampled_x.shape == (2, 2, 3, 5, 6)
    assert sampled_y.shape == (2, 2, 3, 5, 6)


def test_dense_augmenter_pads_near_sized_trajectory_lists_and_auxiliaries():
    first = jnp.ones((3, 2, 5, 6))
    second = 2.0 * jnp.ones((3, 2, 6, 7))
    dense = DataAugmenter([first, second], batch_mode="array")

    assert dense.data_saved.shape == (2, 3, 2, 6, 7)
    assert dense.spatial_padding == [(0, 1, 0, 1), (0, 0, 0, 0)]

    masks = dense.pad_and_stack_spatial(
        [jnp.ones((1, 5, 6)), jnp.ones((1, 6, 7))]
    )
    assert masks.shape == (2, 1, 6, 7)
    assert jnp.all(masks[0, :, -1] == 0)


def test_dense_shift_and_unshift_round_trip():
    dense = DataAugmenter(_data(), hidden_channels=0, batch_mode="array")
    key = jax.random.PRNGKey(9)

    shifted = dense.shift(dense.data_saved, 3, key)
    restored = dense.unshift(shifted, 3, key)

    assert jnp.array_equal(restored, dense.data_saved)


def test_logging_view_tracks_initialisation_but_excludes_hidden_channels():
    dense = DataAugmenter(_data(), hidden_channels=2, batch_mode="array")
    initialized = dense.pad(dense.duplicate_batches(dense.data_saved, 2), 1)
    dense.save_data(initialized)

    observed = dense.return_observed_data()

    assert observed.shape == (4, 4, 3, 7, 8)
    assert jnp.array_equal(observed[:, :, :3], initialized[:, :, :3])


def test_legacy_array_subclass_without_super_keeps_array_helpers():
    class LegacySubclass(LegacyArrayAugmenter):
        def __init__(self):
            self.OBS_CHANNELS = 1

    values = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    duplicated = LegacySubclass().duplicate_batches(values, 2)

    assert duplicated.shape == (4, 3, 4)
    assert jnp.array_equal(duplicated, jnp.concatenate([values, values]))


def test_texture_augmenter_has_tree_and_array_parity():
    data = jnp.arange(2 * 2 * 1 * 8 * 8, dtype=jnp.float32).reshape(
        2, 2, 1, 8, 8
    )
    tree = TextureDataAugmenter(data)
    dense = ArrayTextureDataAugmenter(data)
    tree.NOISE_CUTOFF = dense.NOISE_CUTOFF = 1
    init_key = jax.random.PRNGKey(40)
    tree.data_init(key=init_key)
    dense.data_init(key=init_key)

    assert jnp.allclose(jnp.stack(tree.data_saved), dense.data_saved)

    tree_x, tree_y = tree.split_x_y()
    dense_x, dense_y = dense.split_x_y()
    callback_key = jax.random.PRNGKey(41)
    tree_x, tree_y = tree.data_callback(tree_x, tree_y, 0, callback_key)
    dense_x, dense_y = dense.data_callback(dense_x, dense_y, 0, callback_key)

    assert jnp.allclose(jnp.stack(tree_x), dense_x)
    assert jnp.allclose(jnp.stack(tree_y), dense_y)


def test_grouped_12_targets_expand_to_the_full_model_state():
    class Model:
        N_CHANNELS = 32

        @staticmethod
        def real_to_latent(value):
            return value

    data = jax.random.uniform(jax.random.PRNGKey(45), (4, 5, 12, 7, 8))
    tree = Grouped12DataAugmenter(data, nca_model=Model())
    dense = Grouped12DataAugmenter(data, nca_model=Model(), batch_mode="array")

    tree_x, tree_y = tree.split_x_y()
    dense_x, dense_y = dense.split_x_y()

    expected = jnp.take(
        data[:, :-1],
        jnp.asarray(Grouped12DataAugmenter.schema.primary_measurements),
        axis=2,
    )
    assert dense_x.shape == (4, 4, 32, 7, 8)
    assert dense_y.shape == (4, 4, 12, 7, 8)
    assert jnp.array_equal(dense_x[:, :, :9], expected)
    assert jnp.all(dense_x[:, :, 9:] == 0)
    assert jnp.array_equal(jnp.stack(tree_x), dense_x)
    assert jnp.array_equal(jnp.stack(tree_y), dense_y)


def test_four_channel_targets_expand_to_the_full_model_state():
    class Model:
        N_CHANNELS = 32

        @staticmethod
        def real_to_latent(value):
            return value

    data = jax.random.uniform(jax.random.PRNGKey(46), (2, 5, 4, 7, 8))
    tree = FourChannelDataAugmenter(data, nca_model=Model())
    dense = FourChannelDataAugmenter(data, nca_model=Model(), batch_mode="array")

    tree_x, _ = tree.split_x_y()
    dense_x, _ = dense.split_x_y()

    assert dense_x.shape == (2, 4, 32, 7, 8)
    assert jnp.array_equal(dense_x[:, :, :4], data[:, :-1])
    assert jnp.all(dense_x[:, :, 4:] == 0)
    assert jnp.array_equal(jnp.stack(tree_x), dense_x)


def test_dense_terminal_carry_preserves_each_terminal_state():
    class AlwaysCarry(TerminalCarryDataAugmenter):
        TERMINAL_CARRY_ENABLED = True
        TERMINAL_CARRY_INITIAL = 1.0
        TERMINAL_CARRY_FINAL = 1.0

    data = jnp.zeros((2, 3, 1, 2, 2))
    augmenter = AlwaysCarry(data, batch_mode="array")
    x = jnp.stack(
        [
            jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 7.0)]),
            jnp.stack([jnp.ones((1, 2, 2)), jnp.full((1, 2, 2), 9.0)]),
        ]
    )

    carried = augmenter.propagate_with_terminal_carry(
        x, jnp.zeros_like(x), 0, jax.random.PRNGKey(0)
    )

    assert jnp.all(carried[0, -1] == 7.0)
    assert jnp.all(carried[1, -1] == 9.0)


def test_dense_model_application_matches_tree_and_applies_boundaries():
    class AddKey:
        def __call__(self, state, boundary_callback=lambda value: value, key=None):
            update = jnp.asarray(key[0], state.dtype) / 100.0
            return boundary_callback(state + update)

    states = jnp.zeros((2, 3, 2, 4, 5), dtype=jnp.float32)
    keys = jax.random.randint(
        jax.random.PRNGKey(1), (2, 3, 2), 0, 100, dtype=jnp.uint32
    )
    mask = jnp.ones((1, 4, 5), dtype=jnp.float32)
    callbacks = [model_boundary(mask), no_boundary()]

    dense_backend = make_batch_backend(states, "array")
    tree_backend = make_batch_backend(list(states), "tree")
    dense = dense_backend.apply_model(AddKey(), states, callbacks, keys)
    tree = tree_backend.apply_model(AddKey(), list(states), callbacks, list(keys))

    assert jnp.allclose(dense, jnp.stack(tree))
    assert jnp.all(dense[0, :, -1] == 1.0)


def test_dense_hard_boundary_is_a_batched_epilogue():
    backend = make_batch_backend(jnp.ones((2, 1, 2, 3, 3)), "array")
    first = jnp.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]]], dtype=jnp.float32)
    second = 1.0 - first

    result = backend.apply_boundaries(
        jnp.ones((2, 1, 2, 3, 3)),
        [hard_boundary(first), hard_boundary(second)],
    )

    assert jnp.array_equal(result[0, 0, 0], first[0])
    assert jnp.array_equal(result[1, 0, 0], second[0])


def test_dense_regularisers_match_tree_regularisers():
    state = jnp.arange(2 * 3 * 4 * 3 * 3, dtype=jnp.float32).reshape(
        2, 3, 4, 3, 3
    ) / 100.0
    callbacks = [no_boundary(), no_boundary()]
    aux = {"OBS_CHANNELS": 2, "BOUNDARY_CALLBACK": callbacks}

    dense_intermediate = intermediate_reg(None, None, None, state, None, aux, None)
    tree_intermediate = intermediate_reg(
        None, None, None, list(state), None, aux, None
    )
    dense_latent = latent_size_regulariser(None, state, None, None, None, aux, None)
    tree_latent = latent_size_regulariser(
        None, list(state), None, None, None, aux, None
    )
    dense_boundary = boundary_regulariser(
        None, state, None, None, None, aux, None
    )
    tree_boundary = boundary_regulariser(
        None, list(state), None, None, None, aux, None
    )

    assert jnp.allclose(dense_intermediate, tree_intermediate)
    assert jnp.allclose(dense_latent, tree_latent)
    assert jnp.allclose(dense_boundary, tree_boundary)


def test_fast_nca_dense_forward_and_parameter_gradients_match_tree():
    model = FastNCA(
        4,
        KERNEL_STR=["ID", "LAP", "GRAD"],
        FIRE_RATE=0.5,
        key=jax.random.PRNGKey(20),
    )
    model = eqx.tree_at(
        lambda candidate: candidate.layers[-1].weight,
        model,
        0.02
        * jax.random.normal(
            jax.random.PRNGKey(23), model.layers[-1].weight.shape
        ),
    )
    states = jax.random.normal(jax.random.PRNGKey(21), (2, 3, 4, 7, 8))
    keys = jax.random.randint(
        jax.random.PRNGKey(22), (2, 3, 2), 0, 100, dtype=jnp.uint32
    )
    callbacks = [no_boundary(), no_boundary()]
    dense_backend = make_batch_backend(states, "array")
    tree_backend = make_batch_backend(list(states), "tree")

    dense = dense_backend.apply_model(model, states, callbacks, keys)
    tree = jnp.stack(
        tree_backend.apply_model(model, list(states), callbacks, list(keys))
    )

    def loss(candidate, backend, values, candidate_keys):
        result = backend.apply_model(
            candidate, values, callbacks, candidate_keys
        )
        result = result if backend.is_array else jnp.stack(result)
        return jnp.mean(result**2)

    dense_grads = eqx.filter_grad(loss)(model, dense_backend, states, keys)
    tree_grads = eqx.filter_grad(loss)(
        model, tree_backend, list(states), list(keys)
    )

    assert jnp.allclose(dense, tree, rtol=2e-5, atol=2e-6)
    dense_leaves = jax.tree_util.tree_leaves(
        eqx.filter(dense_grads, eqx.is_array)
    )
    tree_leaves = jax.tree_util.tree_leaves(
        eqx.filter(tree_grads, eqx.is_array)
    )
    assert len(dense_leaves) == len(tree_leaves)
    for dense_leaf, tree_leaf in zip(dense_leaves, tree_leaves):
        assert jnp.allclose(dense_leaf, tree_leaf, rtol=2e-4, atol=2e-6)


def test_fast_nca_dense_diff_perception_matches_tree():
    model = FastNCA(
        4,
        KERNEL_STR=["ID", "LAP", "DIFF"],
        FIRE_RATE=0.5,
        key=jax.random.PRNGKey(30),
    )
    model = eqx.tree_at(
        lambda candidate: candidate.layers[-1].weight,
        model,
        0.02
        * jax.random.normal(
            jax.random.PRNGKey(33), model.layers[-1].weight.shape
        ),
    )
    states = jax.random.normal(jax.random.PRNGKey(31), (2, 3, 4, 7, 8))
    keys = jax.random.randint(
        jax.random.PRNGKey(32), (2, 3, 2), 0, 100, dtype=jnp.uint32
    )
    callbacks = [no_boundary(), no_boundary()]

    dense = make_batch_backend(states, "array").apply_model(
        model, states, callbacks, keys
    )
    tree = make_batch_backend(list(states), "tree").apply_model(
        model, list(states), callbacks, list(keys)
    )

    assert jnp.allclose(dense, jnp.stack(tree), rtol=2e-5, atol=2e-6)
