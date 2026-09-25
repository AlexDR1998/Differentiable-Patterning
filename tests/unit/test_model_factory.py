import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_hierarchical import HNCA
from NCA.model.NCA_model import NCA
from NCA.model.config import HNCAModelConfig, KANConfig, KANModelConfig, ModelConfig
from NCA.model.factory import build_model, build_model_config_string


def _model_config(family, **overrides):
    values = dict(
        family=family, channels=4, kernel_str=("ID", "LAP"), fire_rate=1.0, padding="CIRCULAR"
    )
    values.update(overrides)
    return ModelConfig(**values)


def _kan_config(**kan_overrides):
    return KANModelConfig(
        family="FastKaNCA", channels=4, kernel_str=("ID", "LAP"), fire_rate=1.0,
        padding="CIRCULAR", kan=KANConfig(**kan_overrides),
    )


def test_build_model_constructs_nca():
    model, cfg_str = build_model(_model_config("NCA"), key=jax.random.PRNGKey(0))

    assert type(model) is NCA
    assert cfg_str == "NCA_c4"


@pytest.mark.parametrize(
    ("family", "gated", "noise"),
    [("NCA", False, 0.0), ("gNCA", True, 0.0), ("nNCA", False, 0.01), ("gnNCA", True, 0.01)],
)
def test_build_model_maps_variant_families_to_nca_flags(family, gated, noise):
    model, cfg_str = build_model(_model_config(family), key=jax.random.PRNGKey(0))

    assert type(model) is NCA
    assert model.GATED is gated
    assert model.PARAMETER_NOISE_LEVEL == noise
    assert model.get_config()["MODEL"] == family
    assert cfg_str.startswith(f"{family}_c4")


@pytest.mark.parametrize(
    ("family", "portable_family"),
    [("NCA_sycl", "NCA"), ("NCA_fast", "NCA"), ("gNCA_sycl", "gNCA")],
)
def test_build_model_maps_retired_families(family, portable_family):
    model, cfg_str = build_model(_model_config(family), key=jax.random.PRNGKey(0))

    assert type(model) is NCA
    assert model.GATED is (portable_family == "gNCA")
    assert cfg_str.startswith(f"{portable_family}_c4")


# Saved .eqx files store leaves in this order, so these layouts must not change
# or old bundles stop loading. 4 channels x ("ID", "LAP") gives 8 features; the
# four (1, 1, 3, 3) arrays are the fixed spatial kernels, and the ints/float
# are N_CHANNELS, N_FEATURES, FIRE_RATE and the kernel scale.
_KERNELS = [("array", (1, 1, 3, 3))] * 4
_SCALARS = [("int", 4), ("int", 8), ("float", 1.0), ("int", 1)]
SAVED_LAYOUTS = {
    "NCA": [("array", (8, 8, 1, 1)), ("array", (4, 8, 1, 1)), ("array", (4, 1, 1))]
    + _SCALARS + _KERNELS,
    "gNCA": [("array", (8, 8, 1, 1)), ("array", (8, 8, 1, 1)), ("array", (8, 1, 1))]
    + _SCALARS + _KERNELS,
}
SAVED_LAYOUTS["nNCA"] = SAVED_LAYOUTS["NCA"]
SAVED_LAYOUTS["gnNCA"] = SAVED_LAYOUTS["gNCA"]


def _saved_layout(model):
    layout = []
    for leaf in jax.tree_util.tree_leaves(model):
        if eqx.is_array(leaf):
            layout.append(("array", tuple(leaf.shape)))
        elif isinstance(leaf, (bool, int, float)):
            layout.append((type(leaf).__name__, leaf))
    return layout


@pytest.mark.parametrize("family", sorted(SAVED_LAYOUTS))
def test_saved_parameter_layout_is_unchanged(family):
    model, _ = build_model(_model_config(family), key=jax.random.PRNGKey(0))

    assert _saved_layout(model) == SAVED_LAYOUTS[family]


def test_new_model_starts_with_zero_update():
    for family in SAVED_LAYOUTS:
        model, _ = build_model(_model_config(family, parameter_noise_level=0.0), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 7))
        assert jnp.allclose(model(x, key=jax.random.PRNGKey(2)), x)


def test_parameter_noise_perturbs_the_update():
    # Give the last layer non-zero weights, so the update is not zero
    def build(family):
        model, _ = build_model(_model_config(family, parameter_noise_level=0.1), key=jax.random.PRNGKey(0))
        return eqx.tree_at(lambda m: m.layers[2].bias, model, jnp.ones((4, 1, 1)))

    clean, noisy = build("NCA"), build("nNCA")
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 7))

    clean_a = clean(x, key=jax.random.PRNGKey(2))
    clean_b = clean(x, key=jax.random.PRNGKey(3))
    noisy_a = noisy(x, key=jax.random.PRNGKey(2))
    noisy_b = noisy(x, key=jax.random.PRNGKey(3))

    assert jnp.array_equal(clean_a, clean_b)
    assert not jnp.allclose(noisy_a, clean_a)
    assert not jnp.allclose(noisy_a, noisy_b)
    assert jnp.array_equal(noisy_a, noisy(x, key=jax.random.PRNGKey(2)))


def test_save_and_load_accept_path_without_suffix(tmp_path):
    source, _ = build_model(_model_config("gNCA"), key=jax.random.PRNGKey(0))
    source = eqx.tree_at(lambda m: m.layers[2].bias, source, jnp.ones((8, 1, 1)))
    source.save(tmp_path / "model")
    blank, _ = build_model(_model_config("gNCA"), key=jax.random.PRNGKey(5))

    loaded = blank.load(tmp_path / "model")

    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(source, eqx.is_array)),
                        jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_array)))
    )
    assert len(source.get_weights()) == 3


def test_build_model_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        build_model(_model_config("NCA", activation="softplus"), key=jax.random.PRNGKey(0))


def test_build_model_constructs_fast_kan_nca_with_defaults():
    model, cfg_str = build_model(_kan_config(), key=jax.random.PRNGKey(1))
    x = jnp.ones((4, 6, 7))
    y = model(x, key=jax.random.PRNGKey(2))

    assert isinstance(model, FastKaNCA)
    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert model.get_config()["KAN_AUX"]["base_activation"] == "identity"
    assert cfg_str == "FastKaNCA_c4_kb8"


def test_build_model_constructs_fast_kan_nca_with_kan_overrides():
    model, _ = build_model(
        _kan_config(
            basis="linear_spline",
            hidden_features=6,
            num_basis=4,
            base_activation="none",
            extrapolation="linear",
            use_layernorm=False,
        ),
        key=jax.random.PRNGKey(3),
    )

    assert model.KAN_AUX["basis"] == "linear_spline"
    assert model.KAN_AUX["hidden_features"] == 6
    assert model.KAN_AUX["num_basis"] == 4
    assert model.KAN_AUX["base_activation"] == "none"
    assert model.KAN_AUX["extrapolation"] == "linear"
    assert model.KAN_AUX["use_layernorm"] is False
    assert model.get_config()["KAN_AUX"]["basis"] == "linear_spline"


def test_build_model_config_string_marks_linear_spline_kan():
    cfg_str = build_model_config_string(
        _kan_config(basis="linear_spline", num_basis=12, extrapolation="zero")
    )

    assert cfg_str == "FastKaNCA_c4_kb12_klin_kexzero"


def test_build_model_constructs_hnca():
    model_config = HNCAModelConfig(
        family="HNCA", channels=8, kernel_str=("ID", "LAP"), fire_rate=1.0,
        padding="CIRCULAR", scale=2, obs_channels=2, child_gated=True,
    )
    model, cfg_str = build_model(model_config, key=jax.random.PRNGKey(4))

    assert isinstance(model, HNCA)
    assert cfg_str == "HNCA_c8_s2_cg"
