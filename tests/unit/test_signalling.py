import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from Experiments.config import _model_config as parse_model_config
from NCA.model.NCA_signalling import (
    STRUCTURE_CHANNELS,
    SignalNCA,
    axon_step,
    broadcast,
    broadcast_scales,
    cable_step,
    conductance_from_channel,
    excitable_step,
    pointer_read,
    steady_cable,
)
from NCA.model.config import SignalModelConfig
from NCA.model.factory import build_model
from NCA.trainer.NCA_regulariser import wire_cost_regulariser


def _wire(length=8, size=12, row=5):
    """Conductance 1 on a horizontal wire of ``length`` cells starting at column 1, 0 elsewhere."""
    return jnp.zeros((size, size)).at[row, 1:1 + length].set(1.0)


def _pulse(size=12, row=5, col=1):
    return jnp.zeros((1, size, size)).at[0, row, col].set(1.0)


def _run(step, s, n):
    for _ in range(n):
        s = step(s)
    return s


@pytest.mark.parametrize("padding", ["CIRCULAR", "ZEROS", "REFLECT", "REPLICATE"])
def test_zero_conductance_blocks_all_signalling(padding):
    s = jax.random.uniform(jax.random.PRNGKey(0), (2, 8, 8))
    g = jnp.zeros((8, 8))
    v = jnp.ones((2, 8, 8)) * 0.5
    assert jnp.allclose(cable_step(s, g, padding, 0.25), s)
    assert jnp.allclose(axon_step(s, g, v, padding), s)
    assert jnp.allclose(steady_cable(s, g, padding, 0.01, 20), s)
    assert jnp.allclose(excitable_step(s, g, padding, 0.25), s)
    assert jnp.allclose(pointer_read(s, g, 3 * v, padding), s)
    assert jnp.allclose(broadcast(s, jnp.ones((8, 8)), g, jnp.zeros((2, 4)), (1, 2, 4), padding), s)


@pytest.mark.parametrize("padding", ["CIRCULAR", "ZEROS"])
def test_propagation_conserves_total_signal(padding):
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    s = jax.random.uniform(keys[0], (2, 8, 8))
    g = jax.random.uniform(keys[1], (8, 8))
    v = jax.random.uniform(keys[2], (2, 8, 8), minval=-0.5, maxval=0.5)
    assert jnp.allclose(_run(lambda x: cable_step(x, g, padding, 0.25), s, 10).sum(), s.sum(), rtol=1e-5)
    assert jnp.allclose(_run(lambda x: axon_step(x, g, v, padding), s, 10).sum(), s.sum(), rtol=1e-5)
    assert jnp.allclose(steady_cable(s, g, padding, 0.01, 50).sum(), s.sum(), rtol=1e-4)


def test_cable_signal_stays_inside_the_wire():
    g = _wire()
    s = _run(lambda x: cable_step(x, g, "ZEROS", 0.25), _pulse(), 50)
    assert jnp.allclose(s[0] * (g == 0), 0.0)
    assert s[0, 5, 8] > 0  # reached the far end


def test_axon_carries_a_pulse_one_cell_per_substep_to_the_terminal():
    g = _wire(length=8)
    v = jnp.zeros((2, 12, 12)).at[1].set(1.0)  # pointing right
    s = _run(lambda x: axon_step(x, g, v, "ZEROS"), _pulse(col=1), 4)
    assert jnp.allclose(s[0], _pulse(col=5)[0])
    # The signal piles up at the end of the wire and does not leave it
    s = _run(lambda x: axon_step(x, g, v, "ZEROS"), s, 20)
    assert jnp.allclose(s[0], _pulse(col=8)[0])


def test_axon_is_directed():
    g = _wire(length=8)
    v = jnp.zeros((2, 12, 12)).at[1].set(-1.0)  # pointing left
    s = _run(lambda x: axon_step(x, g, v, "ZEROS"), _pulse(col=6), 3)
    assert jnp.allclose(s[0], _pulse(col=3)[0])


def test_steady_cable_spreads_a_signal_along_the_whole_wire_at_once():
    g = _wire()
    r = steady_cable(_pulse(), g, "ZEROS", 1e-4, 50)
    on_wire = r[0, 5, 1:9]
    # Nearly the same value (the wire mean) everywhere on the wire, nothing off it
    assert jnp.allclose(on_wire, 1 / 8, rtol=0.05)
    assert jnp.allclose(r[0] * (g == 0), 0.0, atol=1e-6)


def test_excitable_pulse_travels_along_the_wire_and_returns_to_rest():
    size = 32
    g = jnp.zeros((size, size)).at[5, 1:29].set(1.0)
    s = jnp.zeros((2, size, size)).at[0, 5, 1:3].set(1.0)
    s = _run(lambda x: excitable_step(x, g, "ZEROS", 0.25), s, 60)
    front = int(jnp.argmax(s[0, 5]))
    assert front > 12 and s[0, 5, front] > 0.8  # still a full pulse, far from the start
    assert jnp.allclose(s[0] * (g == 0), 0.0)
    s = _run(lambda x: excitable_step(x, g, "ZEROS", 0.25), s, 300)
    assert jnp.abs(s[0]).max() < 0.05


def test_excitable_dynamics_stay_bounded_for_large_values():
    g = _wire()
    s = jnp.zeros((2, 12, 12)).at[0, 5, 2].set(50.0).at[1, 5, 4].set(-50.0)
    s = _run(lambda x: excitable_step(x, g, "ZEROS", 0.25), s, 200)
    assert jnp.all(jnp.isfinite(s)) and jnp.abs(s).max() < 50.0


def test_pointer_reads_the_signal_at_the_offset():
    s = _pulse(col=7)
    offset = jnp.zeros((2, 12, 12)).at[0].set(-2.0).at[1].set(-3.0)
    read = pointer_read(s, jnp.ones((12, 12)), offset, "ZEROS")
    # Cell (7, 10) reads cell (5, 7)
    assert jnp.allclose(read[0], jnp.zeros((12, 12)).at[7, 10].set(1.0), atol=1e-6)


@pytest.mark.parametrize("padding", ["CIRCULAR", "ZEROS"])
def test_broadcast_global_channel_gives_receivers_the_mean(padding):
    s = jax.random.uniform(jax.random.PRNGKey(2), (2, 8, 8))
    logits = jnp.full((2, 4), -1e4).at[:, -1].set(0.0)  # global mean only
    received = broadcast(s, jnp.ones((8, 8)), jnp.ones((8, 8)), logits, (1, 2, 4), padding)
    assert jnp.allclose(received, s.mean(axis=(1, 2), keepdims=True) * jnp.ones_like(s), atol=1e-5)


def test_broadcast_blur_conserves_signal_when_periodic():
    s = _pulse()
    logits = jnp.zeros((1, 4))
    received = broadcast(s, jnp.ones((12, 12)), jnp.ones((12, 12)), logits, (1, 2, 4), "CIRCULAR")
    assert jnp.allclose(received.sum(), 1.0, atol=1e-5)
    assert received[0, 5, 1] < 1.0 and received[0, 0, 7] > 0  # spread out, including far away


def test_broadcast_scales():
    assert broadcast_scales(16) == (1, 2, 4, 8, 16)
    assert broadcast_scales(1) == (1,)


def _model_config(**overrides):
    values = dict(
        family="sNCA", channels=10, kernel_str=("ID", "LAP"), fire_rate=1.0,
        padding="ZEROS", signal_channels=2, signal_substeps=6,
    )
    values.update(overrides)
    return SignalModelConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "cfg_str", "gated"),
    [
        ({}, "sNCA_c10_sigcable2x6", False),
        ({"family": "gsNCA", "signal_mode": "axon"}, "gsNCA_c10_sigaxon2x6", True),
        ({"signal_diffusion": 0.1}, "sNCA_c10_sigcable2x6_D0.1", False),
        ({"signal_mode": "steady", "signal_screening": 0.1}, "sNCA_c10_sigsteady2x6_scr0.1", False),
        ({"signal_mode": "excitable"}, "sNCA_c10_sigexcitable2x6", False),
        ({"signal_mode": "broadcast", "signal_range": 4}, "sNCA_c10_sigbroadcast2x6_r4", False),
        ({"signal_mode": "pointer"}, "sNCA_c10_sigpointer2x6", False),
    ],
)
def test_build_signal_families(overrides, cfg_str, gated):
    model, actual = build_model(_model_config(**overrides), key=jax.random.PRNGKey(0))
    assert isinstance(model, SignalNCA)
    assert actual == cfg_str
    assert model.nca.GATED is gated
    assert model.N_CHANNELS == 10
    assert model.get_config()["MODEL"] == overrides.get("family", "sNCA")


def test_experiment_config_routes_signal_families():
    node = {
        "family": "sNCA", "channels": 10, "kernel_str": ["ID", "LAP"], "fire_rate": 1.0,
        "padding": "ZEROS", "signal_mode": "axon", "signal_substeps": 3,
    }
    config = parse_model_config(node)
    assert isinstance(config, SignalModelConfig)
    assert config.signal_mode == "axon" and config.signal_substeps == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"signal_mode": "wifi"}, {"signal_channels": 9}, {"signal_substeps": 0},
        {"signal_diffusion": 0.3}, {"signal_mode": "excitable", "signal_channels": 3},
        {"signal_screening": 0.0}, {"signal_range": 0},
    ],
)
def test_invalid_signal_configs_fail(overrides):
    with pytest.raises(ValueError):
        _model_config(**overrides)


def test_model_step_moves_signals_along_the_grown_wire():
    model, _ = build_model(_model_config(signal_mode="axon"), key=jax.random.PRNGKey(0))
    size = 12
    x = jnp.zeros((10, size, size))
    x = x.at[model.signal_start].set(_pulse(size, col=1)[0])
    # Structure channels: strong conductance on the wire, direction pointing right
    x = x.at[model.structure_start].set(jnp.where(_wire() > 0, 20.0, -20.0))
    x = x.at[model.structure_start + 2].set(20.0)
    y = model(x, key=jax.random.PRNGKey(1))
    # The new model makes no NCA update, so only the signal moves (6 substeps)
    assert jnp.allclose(y[model.signal_start], _pulse(size, col=7)[0], atol=1e-3)
    assert jnp.allclose(y[: model.signal_start], x[: model.signal_start])


@pytest.mark.parametrize("mode", list(STRUCTURE_CHANNELS))
def test_gradient_reaches_the_update_network_through_propagation(mode):
    model, _ = build_model(_model_config(signal_mode=mode, signal_range=4), key=jax.random.PRNGKey(0))
    x = jax.random.uniform(jax.random.PRNGKey(1), (10, 8, 8))
    # Give the last layer weights, so the NCA writes the conductance channel
    w = model.nca.layers[2].weight
    model = eqx.tree_at(lambda m: m.nca.layers[2].weight, model, 0.1 * jnp.ones_like(w))

    def loss(diff, static):
        m = eqx.combine(diff, static)
        y = x
        for i in range(3):
            y = m(y, key=jax.random.PRNGKey(i))
        return jnp.sum(y[m.signal_start:m.structure_start, :4] ** 2)

    diff, static = model.partition()
    grads = jax.grad(loss)(diff, static)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.any(grads.nca.layers[2].weight != 0)
    if mode == "broadcast":
        assert jnp.any(grads.kernel_logits != 0)


@pytest.mark.parametrize("mode", ["cable", "broadcast"])
def test_save_and_load_round_trip(tmp_path, mode):
    model, _ = build_model(_model_config(signal_mode=mode), key=jax.random.PRNGKey(0))
    if mode == "broadcast":
        model = eqx.tree_at(lambda m: m.kernel_logits, model, model.kernel_logits + 1.5)
    model.save(tmp_path / "signal")
    fresh, _ = build_model(_model_config(signal_mode=mode), key=jax.random.PRNGKey(5))
    loaded = fresh.load(tmp_path / "signal")
    assert jnp.allclose(loaded.nca.layers[0].weight, model.nca.layers[0].weight)
    if mode == "broadcast":
        assert jnp.allclose(loaded.kernel_logits, 1.5)


def test_wire_cost_is_mean_conductance():
    model, _ = build_model(_model_config(), key=jax.random.PRNGKey(0))
    states = [jnp.zeros((2, 10, 3, 3)), jnp.zeros((2, 10, 3, 3)).at[:, -1].set(4.0)]
    cost = wire_cost_regulariser(None, states, {"conductance": model.conductance}, None)
    assert jnp.allclose(cost, jnp.asarray([conductance_from_channel(0.0), 0.5]))
