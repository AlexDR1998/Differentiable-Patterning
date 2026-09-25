import jax
import jax.numpy as jnp
import pytest

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_gated_model import gNCA
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


def test_build_model_constructs_gated_nca():
    model, cfg_str = build_model(_model_config("gNCA"), key=jax.random.PRNGKey(0))

    assert type(model) is gNCA
    assert cfg_str == "gNCA_c4"


@pytest.mark.parametrize(
    ("family", "model_class"),
    [("NCA_sycl", NCA), ("NCA_fast", NCA), ("gNCA_sycl", gNCA)],
)
def test_build_model_maps_retired_families(family, model_class):
    model, cfg_str = build_model(_model_config(family), key=jax.random.PRNGKey(0))

    assert type(model) is model_class
    assert cfg_str.startswith(f"{model_class.__name__}_c4")


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
