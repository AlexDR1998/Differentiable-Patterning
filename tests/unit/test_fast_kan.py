import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from Common.model.fast_kan import (
    FastRBFKAN,
    FastLinearSplineKANLayer,
    FastRBFKANLayer,
    plot_fast_rbf_kan_edges,
    plot_fast_rbf_kan_edge_norms,
    plot_top_fast_rbf_kan_edges,
)
from NCA.model.NCA_fast_KAN_model import FastKaNCA, SpatialFastRBFKANLayer


def test_fast_rbf_kan_layer_vector_shape():
    key = jax.random.PRNGKey(0)
    layer = FastRBFKANLayer(5, 3, num_basis=4, key=key)
    x = jnp.ones((5,))

    y = layer(x)

    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(y))
    assert layer.edge_norms().shape == (5, 3)
    spline_inputs = layer.spline_inputs_from_inputs(jnp.ones((7, 5)))
    edge_values = layer.evaluate_edge_functions(jnp.linspace(-1, 1, 9))
    edge_contributions = layer.edge_contributions_from_inputs(jnp.ones((7, 5)))
    assert spline_inputs.shape == (7, 5)
    assert edge_values.shape == (5, 3, 9)
    assert edge_contributions.shape == (5, 3, 7)
    assert jnp.all(jnp.isfinite(spline_inputs))
    assert jnp.all(jnp.isfinite(edge_contributions))


def test_fast_rbf_kan_rejects_non_vector_input():
    key = jax.random.PRNGKey(10)
    layer = FastRBFKANLayer(5, 3, num_basis=4, key=key)

    with pytest.raises(ValueError, match="expects a vector input"):
        layer(jnp.ones((5, 2)))


def test_fast_rbf_kan_requires_at_least_two_widths():
    key = jax.random.PRNGKey(11)

    with pytest.raises(ValueError, match="at least"):
        FastRBFKAN([16], key=key)


def test_fast_rbf_kan_base_activation_presets():
    key = jax.random.PRNGKey(12)
    x = jnp.linspace(-1, 1, 5)

    identity_layer = FastRBFKANLayer(5, 3, base_activation="identity", key=key)
    silu_layer = FastRBFKANLayer(5, 3, base_activation="silu", key=key)
    relu_layer = FastRBFKANLayer(5, 3, base_activation="relu", key=key)
    none_layer = FastRBFKANLayer(5, 3, base_activation="none", key=key)

    assert identity_layer(x).shape == (3,)
    assert silu_layer(x).shape == (3,)
    assert relu_layer(x).shape == (3,)
    assert none_layer(x).shape == (3,)
    assert identity_layer.base_activation_name == "identity"
    assert silu_layer.base_activation_name == "silu"
    assert relu_layer.base_activation_name == "relu"
    assert none_layer.base_activation_name == "none"
    assert none_layer.base_weight is None


def test_fast_rbf_kan_layer_spatial_vmap_shape():
    key = jax.random.PRNGKey(1)
    layer = SpatialFastRBFKANLayer(FastRBFKANLayer(5, 3, num_basis=4, key=key))
    x = jnp.ones((5, 7, 11))

    y = layer(x)

    assert y.shape == (3, 7, 11)
    assert jnp.all(jnp.isfinite(y))


def test_fast_linear_spline_kan_layer_vector_shape_and_edges():
    key = jax.random.PRNGKey(13)
    layer = FastLinearSplineKANLayer(
        5,
        3,
        num_basis=4,
        grid_min=-1.0,
        grid_max=1.0,
        base_activation="none",
        use_layernorm=False,
        key=key,
    )

    y = layer(jnp.zeros((5,)))
    edge_values = layer.evaluate_edge_functions(jnp.linspace(-2, 2, 9))
    edge_contributions = layer.edge_contributions_from_inputs(jnp.ones((7, 5)))

    assert y.shape == (3,)
    assert edge_values.shape == (5, 3, 9)
    assert edge_contributions.shape == (5, 3, 7)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(edge_values))
    assert jnp.all(jnp.isfinite(edge_contributions))


def test_edge_contribution_variance_reflects_visited_inputs():
    values = jnp.array([[[0.0, 1.0, 2.0]]])
    layer = FastLinearSplineKANLayer(
        1,
        1,
        num_basis=3,
        grid_min=-1.0,
        grid_max=1.0,
        base_activation="none",
        use_layernorm=False,
        key=jax.random.PRNGKey(17),
    )
    layer = eqx.tree_at(lambda l: l.spline_weight, layer, values)

    constant_inputs = jnp.zeros((5, 1))
    changing_inputs = jnp.linspace(-1, 1, 5)[:, None]

    constant_edges = layer.edge_contributions_from_inputs(constant_inputs)
    changing_edges = layer.edge_contributions_from_inputs(changing_inputs)

    assert jnp.allclose(jnp.var(constant_edges, axis=-1), 0.0)
    assert jnp.all(jnp.var(changing_edges, axis=-1) > 0.0)


def test_fast_linear_spline_kan_layer_spatial_vmap_and_jit():
    key = jax.random.PRNGKey(14)
    layer = SpatialFastRBFKANLayer(
        FastLinearSplineKANLayer(5, 3, num_basis=4, key=key)
    )
    jit_layer = eqx.filter_jit(layer)
    x = jnp.ones((5, 7, 11))

    y = jit_layer(x)

    assert y.shape == (3, 7, 11)
    assert jnp.all(jnp.isfinite(y))


def test_fast_linear_spline_extrapolation_modes_differ():
    values = jnp.array([[[1.0, 2.0, 4.0]]])
    xs = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    outputs = {}

    for extrapolation in ["constant", "zero", "linear"]:
        layer = FastLinearSplineKANLayer(
            1,
            1,
            num_basis=3,
            grid_min=-1.0,
            grid_max=1.0,
            base_activation="none",
            use_layernorm=False,
            extrapolation=extrapolation,
            key=jax.random.PRNGKey(15),
        )
        layer = eqx.tree_at(lambda l: l.spline_weight, layer, values)
        outputs[extrapolation] = layer.evaluate_edge_functions(xs)[0, 0]

    assert jnp.allclose(outputs["constant"], jnp.array([1.0, 1.0, 2.0, 4.0, 4.0]))
    assert jnp.allclose(outputs["zero"], jnp.array([0.0, 1.0, 2.0, 4.0, 0.0]))
    assert jnp.allclose(outputs["linear"], jnp.array([0.0, 1.0, 2.0, 4.0, 6.0]))


def test_fast_kan_nca_shape_and_zero_initial_update():
    key = jax.random.PRNGKey(2)
    nca = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP", "GRAD"],
        KAN_AUX={"num_basis": 4, "final_zero_init": True},
        key=key,
    )
    x = jax.random.normal(key, shape=(4, 6, 8))

    y = nca(x, key=key)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)


def test_fast_kan_layer_and_nca_jit():
    key = jax.random.PRNGKey(3)
    layer = eqx.filter_jit(FastRBFKANLayer(5, 3, num_basis=4, key=key))
    nca = eqx.filter_jit(
        FastKaNCA(
            4,
            KERNEL_STR=["ID", "LAP"],
            KAN_AUX={"num_basis": 4, "final_zero_init": True},
            key=key,
        )
    )

    y_layer = layer(jnp.ones((5,)))
    y_nca = nca(jnp.ones((4, 6, 8)), key=key)

    assert y_layer.shape == (3,)
    assert y_nca.shape == (4, 6, 8)


def test_fast_kan_nca_layer_io_diagnostics_shape():
    key = jax.random.PRNGKey(18)
    nca = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4, "hidden_features": 6},
        key=key,
    )
    x = jax.random.normal(key, shape=(4, 6, 8))

    layer_ios = nca.get_kan_layer_inputs_outputs(x)

    assert len(layer_ios) == 2
    assert layer_ios[0][0].shape == (nca.N_FEATURES, 6, 8)
    assert layer_ios[0][1].shape == (6, 6, 8)
    assert layer_ios[1][0].shape == (6, 6, 8)
    assert layer_ios[1][1].shape == (4, 6, 8)


def test_fast_kan_edge_plot_smoke():
    key = jax.random.PRNGKey(4)
    kan = FastRBFKAN([4, 3], num_basis=4, key=key)

    fig = plot_fast_rbf_kan_edges(
        kan,
        input_indices=[0, 1],
        output_indices=[0],
        xs=jnp.linspace(-1, 1, 8),
    )

    assert fig is not None


def test_fast_kan_top_edges_and_norm_plot_smoke():
    key = jax.random.PRNGKey(5)
    kan = FastRBFKAN([4, 3], num_basis=4, key=key)

    top_edges = kan.get_top_edges(k=3)
    norms = [edge["norm"] for edge in top_edges]
    fig_edges = plot_top_fast_rbf_kan_edges(
        kan,
        k=3,
        xs=jnp.linspace(-1, 1, 8),
    )
    fig_norms = plot_fast_rbf_kan_edge_norms(kan)

    assert len(top_edges) == 3
    assert norms == sorted(norms, reverse=True)
    assert fig_edges is not None
    assert fig_norms is not None


def test_fast_kan_nca_top_edges_include_feature_names():
    key = jax.random.PRNGKey(6)
    nca = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={"num_basis": 4},
        key=key,
    )

    top_edges = nca.get_top_edges(k=2)

    assert len(top_edges) == 2
    assert "input_name" in top_edges[0]
    assert "output_name" in top_edges[0]
    assert nca.get_config()["KAN_AUX"]["base_activation"] == "identity"


def test_fast_kan_nca_linear_spline_basis_shape():
    key = jax.random.PRNGKey(16)
    nca = FastKaNCA(
        4,
        KERNEL_STR=["ID", "LAP"],
        KAN_AUX={
            "basis": "linear_spline",
            "num_basis": 4,
            "extrapolation": "linear",
            "final_zero_init": True,
        },
        key=key,
    )
    x = jax.random.normal(key, shape=(4, 6, 8))

    y = nca(x, key=key)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert nca.get_config()["KAN_AUX"]["basis"] == "linear_spline"
    assert nca.get_config()["KAN_AUX"]["extrapolation"] == "linear"
