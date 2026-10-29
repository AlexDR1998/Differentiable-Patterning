import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from NCA.model.config import ModelConfig
from NCA.model.factory import build_model
from NCA.trainer.NCA_regulariser import firing_energy_regulariser


def _model_config(family, **overrides):
    values = dict(
        family=family, channels=4, kernel_str=("ID", "LAP"), fire_rate=0.5,
        padding="CIRCULAR", parameter_noise_level=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _build(family="aNCA", **overrides):
    model, _ = build_model(_model_config(family, **overrides), key=jax.random.PRNGKey(0))
    return model


def _with_update(model, gated=False):
    """Give the last layer a non-zero bias on the update channels, so dx = 1."""
    bias = model.layers[2].bias
    bias = bias.at[:-1].set(1.0)
    if gated:
        # GLU: update = a * sigmoid(b); set b large so the update is about a = 1
        bias = bias.at[4:8].set(20.0)
    return eqx.tree_at(lambda m: m.layers[2].bias, model, bias)


@pytest.mark.parametrize(
    ("family", "estimator", "cfg_str", "gated"),
    [
        ("aNCA", "straight_through", "aNCA_c4_fst", False),
        ("aNCA", "expected", "aNCA_c4_fexp", False),
        ("gaNCA", "straight_through", "gaNCA_c4_fst", True),
    ],
)
def test_build_adaptive_families(family, estimator, cfg_str, gated):
    model, actual_cfg_str = build_model(
        _model_config(family, firing_estimator=estimator), key=jax.random.PRNGKey(0)
    )

    assert actual_cfg_str == cfg_str
    assert model.FIRING == estimator
    assert model.GATED is gated
    assert model.get_config()["MODEL"] == family
    # One extra output channel for the firing logit
    assert model.layers[2].bias.shape[0] == (8 if gated else 4) + 1


@pytest.mark.parametrize(
    ("family", "estimator"),
    [("aNCA", "straight_through"), ("aNCA", "expected"), ("gaNCA", "expected")],
)
def test_new_adaptive_model_makes_no_update_and_fires_at_fire_rate(family, estimator):
    model = _build(family, firing_estimator=estimator)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 7))

    y = model(x, key=jax.random.PRNGKey(2))

    assert jnp.allclose(y[:-1], x[:-1])
    assert jnp.allclose(y[-1], 0.5)
    assert jnp.allclose(model.firing_probability(y), 0.5)


def test_expected_firing_is_deterministic_and_scales_update():
    model = _with_update(_build(firing_estimator="expected"))
    x = jnp.zeros((4, 6, 7))

    a = model(x, key=jax.random.PRNGKey(2))
    b = model(x, key=jax.random.PRNGKey(3))

    assert jnp.array_equal(a, b)
    assert jnp.allclose(a[:-1], 0.5)


@pytest.mark.parametrize(("family", "gated"), [("aNCA", False), ("gaNCA", True)])
def test_straight_through_cells_fire_all_channels_or_none(family, gated):
    model = _with_update(_build(family), gated=gated)
    x = jnp.zeros((4, 16, 16))

    change = model(x, key=jax.random.PRNGKey(2))[:-1] - x[:-1]
    fired = jnp.isclose(change, 1.0, atol=1e-6)
    still = jnp.isclose(change, 0.0, atol=1e-6)

    assert jnp.all(fired | still)
    # Every channel of a cell agrees
    assert jnp.all(fired == fired[:1])
    # With p = 0.5, some cells fire and some do not
    assert 0 < jnp.mean(fired[0]) < 1


def test_straight_through_gradient_reaches_firing_logit():
    model = _with_update(_build())
    x = jnp.zeros((4, 6, 7))

    def loss(m):
        return jnp.sum(m(x, key=jax.random.PRNGKey(2))[:-1])

    gradient = eqx.filter_grad(loss)(model)

    assert jnp.abs(gradient.layers[2].bias[-1]).sum() > 0


def test_firing_energy_is_mean_of_activity_channel():
    model = _build()
    states = [
        jnp.zeros((2, 4, 3, 3)).at[:, -1].set(0.25),
        jnp.zeros((2, 4, 3, 3)).at[:, -1].set(0.75),
    ]

    energy = firing_energy_regulariser(
        None, states, {"firing_probability": model.firing_probability}, None
    )

    assert jnp.allclose(energy, jnp.asarray([0.25, 0.75]))


def test_fixed_firing_models_have_no_firing_probability():
    model = _build("NCA")

    with pytest.raises(ValueError, match="learned firing rate"):
        model.firing_probability(jnp.zeros((4, 6, 7)))


def test_config_rejects_unknown_firing_estimator():
    with pytest.raises(ValueError, match="firing_estimator"):
        _model_config("aNCA", firing_estimator="reinforce")
