import jax
import jax.numpy as jnp
import pytest

from Experiments.config_helpers import build_model, build_model_config_string, build_tags
from Experiments.emoji.config_helpers import (
    build_data_augmenter,
    build_data_config_string,
    build_filename,
    load_data as load_emoji_data,
)
from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_model_fast import NCA as FastNCA
from NCA.model.NCA_sycl import NCA as SyclNCA
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch
from NCA.trainer.data_augmenter_9ch_colony import DataAugmenter as DataAugmenterGrouped


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _cfg(value):
    if isinstance(value, dict):
        return ConfigDict({key: _cfg(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_cfg(item) for item in value]
    return value


def _base_cfg(family):
    return _cfg(
        {
            "data": {"data_channels": 4},
            "model": {
                "family": family,
                "channels": 4,
                "kernel_str": ["ID", "LAP"],
                "fire_rate": 1.0,
                "padding": "CIRCULAR",
            },
        }
    )


def test_build_model_constructs_nca():
    cfg = _base_cfg("NCA")
    model, cfg_str = build_model(cfg, key=jax.random.PRNGKey(0))

    assert isinstance(model, NCA)
    assert cfg_str.startswith("NCA")


def test_build_model_constructs_fast_nca_and_marks_filename():
    cfg = _base_cfg("NCA_fast")
    model, cfg_str = build_model(cfg, key=jax.random.PRNGKey(0))

    assert isinstance(model, FastNCA)
    assert cfg_str.startswith("NCA_fast_c4")


def test_build_model_constructs_sycl_nca_and_marks_filename():
    cfg = _base_cfg("NCA_sycl")
    model, cfg_str = build_model(cfg, key=jax.random.PRNGKey(0))

    assert isinstance(model, SyclNCA)
    assert cfg_str.startswith("NCA_sycl_c4")


def test_build_model_constructs_fast_kan_nca_with_defaults():
    cfg = _base_cfg("FastKaNCA")
    model, cfg_str = build_model(cfg, key=jax.random.PRNGKey(1))
    x = jnp.ones((4, 6, 7))
    y = model(x, key=jax.random.PRNGKey(2))

    assert isinstance(model, FastKaNCA)
    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert model.get_config()["KAN_AUX"]["base_activation"] == "identity"
    assert "FastKaNCA" in cfg_str


def test_build_model_constructs_fast_kan_nca_with_kan_overrides():
    cfg = _base_cfg("FastKaNCA")
    cfg.model.kan = {
        "basis": "linear_spline",
        "hidden_features": 6,
        "num_basis": 4,
        "base_activation": "none",
        "extrapolation": "linear",
        "use_layernorm": False,
    }
    model, _ = build_model(cfg, key=jax.random.PRNGKey(3))

    assert model.KAN_AUX["basis"] == "linear_spline"
    assert model.KAN_AUX["hidden_features"] == 6
    assert model.KAN_AUX["num_basis"] == 4
    assert model.KAN_AUX["base_activation"] == "none"
    assert model.KAN_AUX["extrapolation"] == "linear"
    assert model.KAN_AUX["use_layernorm"] is False
    assert model.get_config()["KAN_AUX"]["basis"] == "linear_spline"
    assert model.get_config()["KAN_AUX"]["extrapolation"] == "linear"


def test_build_model_config_string_handles_missing_kan_section():
    cfg = _base_cfg("FastKaNCA")
    cfg_str = build_model_config_string(cfg)

    assert "kb8" in cfg_str
    assert "pad" not in cfg_str
    assert "kbaseidentity" not in cfg_str


def test_build_model_config_string_marks_linear_spline_kan():
    cfg = _base_cfg("FastKaNCA")
    cfg.model.kan = {
        "basis": "linear_spline",
        "num_basis": 12,
        "extrapolation": "zero",
    }

    cfg_str = build_model_config_string(cfg)

    assert "kb12" in cfg_str
    assert "klin" in cfg_str
    assert "kexzero" in cfg_str


def test_build_tags_truncates_long_values_for_wandb():
    cfg = _cfg(
        {
            "data": {
                "sequence": [
                    "alien_monster.png",
                    "microbe.png",
                    "very_long_emoji_filename_that_would_break_wandb_tags.png",
                ]
            }
        }
    )

    tags = build_tags(cfg)

    assert all(1 <= len(tag) <= 64 for tag in tags)
    assert "data.sequence:al_mi_ve" in tags


def test_build_tags_uses_emoji_sequence_alias():
    cfg = _cfg(
        {
            "data": {
                "sequence": [
                    "avocado.png",
                    "mushroom.png",
                    "lizard.png",
                    "lizard.png",
                ]
            }
        }
    )

    assert "data.sequence:av_mu_li" in build_tags(cfg)


def test_build_tags_omits_wandb_routing_tags():
    cfg = _cfg(
        {
            "data": {
                "sequence": ["avocado.png", "mushroom.png"],
            },
            "logging": {
                "wandb": {
                    "project": "KAN-NCA",
                    "group": "fast-kan",
                    "tags": None,
                }
            },
        }
    )

    tags = build_tags(cfg)

    assert "data.sequence:av_mu" in tags
    assert not any(tag.startswith("logging.wandb.project:") for tag in tags)
    assert not any(tag.startswith("logging.wandb.group:") for tag in tags)


def test_emoji_filename_uses_short_sequence_and_omits_runtime_noise():
    cfg = _cfg(
        {
            "data": {
                "data_channels": 4,
                "sequence": [
                    "avocado.png",
                    "mushroom.png",
                    "lizard.png",
                    "lizard.png",
                ],
                "batches": 2,
                "downsample": 1,
                "pad": [10, 10, 10, 10],
                "regenerate": True,
                "shift_amount": 10,
                "noise_strength": 0.005,
            },
            "model": {
                "family": "FastKaNCA",
                "channels": 12,
                "kernel_str": ["ID", "LAP", "GRAD"],
                "fire_rate": 0.5,
                "padding": "REPLICATE",
                "activation": "relu",
                "kan": {
                    "num_basis": 16,
                    "hidden_features": None,
                    "base_activation": "identity",
                    "use_layernorm": True,
                    "final_zero_init": True,
                },
            },
            "loss": {
                "primary": ["l2"],
                "layers": ["decoded"],
                "regulariser_coeffs": {
                    "intermediate_state": 0.0,
                    "boundary": 0.0,
                    "contiguous_growth": 0.0,
                    "update_sensitivity": 0.0,
                    "perturbation_conservation": 0.0,
                },
            },
            "run": {"filename_mode": "hydra", "t": 64, "iterations": 1000},
            "optimiser": {"learn_rate": 0.0003, "decay_rate": 0.99},
        }
    )
    data_cfg_str = build_data_config_string(cfg)
    data_augmenter, data_augmenter_cfg_str = build_data_augmenter(cfg)
    filename = build_filename(
        cfg,
        build_model_config_string(cfg),
        data_cfg_str,
        data_augmenter_cfg_str,
    )

    assert data_augmenter is not None
    assert "data_av_mu_li" in filename
    assert "avocado" not in filename
    assert "layersdecoded" not in filename
    assert "pad" not in filename
    assert "shift" not in filename
    assert "noise" not in filename
    assert filename.count("regenTrue") == 1
    assert len(filename) < 180


def _multi_attractor_cfg(pairs, target_repeats=2):
    return _cfg(
        {
            "data": {
                "task": "multi_attractor",
                "pairs": pairs,
                "target_repeats": target_repeats,
                "batches": 2,
                "downsample": 1,
                "crop_square": False,
                "regenerate": False,
            }
        }
    )


def test_multi_attractor_data_builds_independent_pair_trajectories(monkeypatch):
    values = {"crab.png": 1.0, "microbe.png": 2.0}

    def fake_load(sequence, **kwargs):
        value = values[sequence[0]]
        return jnp.full((1, 1, 4, 8, 8), value)

    monkeypatch.setattr("Experiments.emoji.config_helpers.load_emoji_sequence", fake_load)
    cfg = _multi_attractor_cfg(
        [
            {"initial": "crab.png", "target": "microbe.png"},
            {"initial": "microbe.png", "target": "crab.png"},
        ]
    )

    data, cfg_str = load_emoji_data(cfg, impath="/tmp/emojis/")

    assert data.shape == (2, 3, 4, 8, 8)
    assert jnp.all(data[0, 0] == 1.0)
    assert jnp.all(data[0, 1:] == 2.0)
    assert jnp.all(data[1, 0] == 2.0)
    assert jnp.all(data[1, 1:] == 1.0)
    assert "data_multi_cr2mi-mi2cr" in cfg_str


def test_multi_attractor_patch_initial_condition(monkeypatch):
    image = jnp.ones((1, 1, 4, 8, 8))
    monkeypatch.setattr(
        "Experiments.emoji.config_helpers.load_emoji_sequence",
        lambda sequence, **kwargs: image,
    )
    cfg = _multi_attractor_cfg(
        [
            {
                "initial": {"image": "crab.png", "mode": "patch", "size": 2},
                "target": "crab.png",
            }
        ],
        target_repeats=1,
    )

    data, _ = load_emoji_data(cfg, impath="/tmp/emojis/")

    assert data.shape == (1, 2, 4, 8, 8)
    assert data[0, 0].sum() == 4 * 2 * 2
    assert data[0, 1].sum() == 4 * 8 * 8


def test_multi_attractor_requires_pairs():
    cfg = _multi_attractor_cfg([])

    with pytest.raises(ValueError, match="data.pairs must contain at least one pair"):
        load_emoji_data(cfg, impath="/tmp/emojis/")


def _micropattern_cfg(data_channels=12, knockout_mode=None):
    return _cfg(
        {
            "data": {
                "data_channels": data_channels,
                "batches": 2,
                "timesteps": [0, 12, 24, 36, 48],
                "downsample": 1,
                "noise_strength": 0.005,
            },
            "knockout": {
                "mode": knockout_mode,
                "time": None if knockout_mode is None else 0,
                "channel": "Nodal",
            },
        }
    )


def _patch_micropattern_loader(monkeypatch):
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    loaded_data_12 = jnp.arange(2 * 5 * 12 * 2 * 3, dtype=jnp.float32).reshape(
        2, 5, 12, 2, 3
    )
    loaded_data_4 = jnp.arange(2 * 5 * 4 * 2 * 3, dtype=jnp.float32).reshape(
        2, 5, 4, 2, 3
    )
    loaded_mask = jnp.ones((2, 1, 2, 3), dtype=jnp.float32)
    loaded_channel_mask_12 = jnp.ones((2, 4, 9), dtype=jnp.float32)
    loaded_channel_mask_4 = jnp.ones((4, 4), dtype=jnp.float32)
    channel_names_12 = [
        "A-LMBR",
        "A-TBXT",
        "A-SOX17",
        "A-SOX2",
        "B-LMBR",
        "B-TBXT",
        "B-SOX17",
        "B-FOXA2",
        "C-Cer1",
        "C-Lefty2",
        "C-Nodal",
        "D-LEF1",
    ]
    channel_names_4 = ["LMBR", "TBXT", "SOX17", "SOX2"]
    calls = {"4ch": [], "12ch": []}

    def fake_loader_12(**kwargs):
        calls["12ch"].append(kwargs)
        return (
            loaded_data_12,
            {"source": "12ch-test"},
            channel_names_12,
            loaded_mask,
            loaded_channel_mask_12,
        )

    def fake_loader_4(**kwargs):
        calls["4ch"].append(kwargs)
        return (
            loaded_data_4,
            {"source": "4ch-test"},
            channel_names_4,
            loaded_mask,
            loaded_channel_mask_4,
        )

    monkeypatch.setattr(
        micropattern_helpers,
        "load_micropattern_circle_nodal_knockout_9ch_explicit_colony",
        fake_loader_12,
    )
    monkeypatch.setattr(
        micropattern_helpers,
        "load_micropattern_circle_4ch_individual",
        fake_loader_4,
    )
    return micropattern_helpers, loaded_data_12, loaded_data_4, channel_names_12, calls


def test_micropattern_load_data_preserves_12_channel_data(monkeypatch):
    (
        micropattern_helpers,
        loaded_data_12,
        _,
        channel_names_12,
        calls,
    ) = _patch_micropattern_loader(monkeypatch)

    data, _, names, _, channel_mask, cfg_str = micropattern_helpers.load_data(
        _micropattern_cfg(data_channels=12),
        impath="/tmp/micropatterns/",
    )

    assert data.shape == (2, 5, 12, 14, 15)
    assert jnp.array_equal(data[:, :, :, 6:-6, 6:-6], loaded_data_12)
    assert names == channel_names_12
    assert channel_mask.shape == (2, 4, 9)
    assert "_c12_" in cfg_str
    assert len(calls["12ch"]) == 1
    assert calls["12ch"][0]["impath"] == "/tmp/micropatterns/"
    assert calls["4ch"] == []


def test_micropattern_load_data_uses_4_channel_group_a_loader(monkeypatch):
    (
        micropattern_helpers,
        _,
        loaded_data_4,
        _,
        calls,
    ) = _patch_micropattern_loader(monkeypatch)

    data, _, names, _, channel_mask, cfg_str = micropattern_helpers.load_data(
        _micropattern_cfg(data_channels=4),
        impath="/tmp/micropatterns/",
    )

    assert data.shape == (2, 5, 4, 14, 15)
    assert jnp.array_equal(data[:, :, :, 6:-6, 6:-6], loaded_data_4)
    assert names == ["A-LMBR", "A-TBXT", "A-SOX17", "A-SOX2"]
    assert channel_mask.shape == (2, 4, 4)
    assert "_c4_" in cfg_str
    assert len(calls["4ch"]) == 1
    assert calls["4ch"][0]["impath"] == "/tmp/micropatterns/A/*"
    assert calls["12ch"] == []


def test_micropattern_build_data_augmenter_selects_channel_specific_class():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    augmenter_12, _ = micropattern_helpers.build_data_augmenter(
        _micropattern_cfg(data_channels=12)
    )
    augmenter_4, _ = micropattern_helpers.build_data_augmenter(
        _micropattern_cfg(data_channels=4)
    )

    assert issubclass(augmenter_12, DataAugmenterGrouped)
    assert issubclass(augmenter_12, DataAugmenter4Ch)
    assert augmenter_12 is not augmenter_4
    assert issubclass(augmenter_4, DataAugmenter4Ch)


def test_micropattern_knockout_role_patterns_repeat_by_batch():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    assert micropattern_helpers.build_knockout_times(None, None, 4) == [
        None,
        None,
        None,
        None,
    ]
    assert micropattern_helpers.build_knockout_times("only_one_ko", 24, 3) == [
        24,
        24,
        24,
    ]
    assert micropattern_helpers.build_knockout_times("one_ko_and_baseline", 24, 5) == [
        24,
        None,
        24,
        None,
        24,
    ]
    assert micropattern_helpers.build_knockout_times("both_ko_and_baseline", None, 7) == [
        0,
        24,
        None,
        0,
        24,
        None,
        0,
    ]
    assert micropattern_helpers.build_knockout_times("only_both_ko", None, 5) == [
        0,
        24,
        0,
        24,
        0,
    ]


def test_micropattern_masked_reinject_only_uses_measured_channels():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    key = jax.random.PRNGKey(7)
    x = [jnp.arange(3 * 9, dtype=jnp.float32).reshape(3, 9, 1, 1)]
    x_true = [1000.0 + x[0]]
    selected = int(jnp.argsort(jax.random.uniform(key, shape=(2,)))[0])
    mask = jnp.zeros((1, 2, 9), dtype=jnp.float32).at[0, selected, 0].set(1.0)

    out = micropattern_helpers.masked_reinject_callback_bit(
        x,
        x_true,
        9,
        key,
        mask,
        jnp.array([-1], dtype=jnp.int32),
    )[0]

    propagated = x[0].at[1:].set(x[0][:-1]).at[0].set(x_true[0][0])
    assert out[selected + 1, 0, 0, 0] == x_true[0][selected + 1, 0, 0, 0]
    assert out[selected + 1, 1, 0, 0] == propagated[selected + 1, 1, 0, 0]
    assert jnp.array_equal(out[:, 2:], propagated[:, 2:])


def test_micropattern_expands_9_channel_mask_for_12_channel_loss():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    mask = jnp.arange(2 * 3 * 9, dtype=jnp.float32).reshape(2, 3, 9)
    expanded = micropattern_helpers.expand_channel_timestep_mask_for_loss(
        _micropattern_cfg(data_channels=12),
        mask,
    )

    expected = jnp.concatenate(
        [mask[..., 0:4], mask[..., 0:3], mask[..., 4:8], mask[..., 8:9]],
        axis=-1,
    )
    assert expanded.shape == (2, 3, 12)
    assert jnp.array_equal(expanded, expected)


def test_micropattern_nodal_zeroing_wins_after_reinject_for_ko_batches():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    x = [
        jnp.ones((4, 9, 1, 1), dtype=jnp.float32),
        2.0 * jnp.ones((4, 9, 1, 1), dtype=jnp.float32),
        3.0 * jnp.ones((4, 9, 1, 1), dtype=jnp.float32),
    ]
    x_true = [100.0 * (i + 1) * jnp.ones((4, 9, 1, 1), dtype=jnp.float32) for i in range(3)]
    mask = jnp.ones((3, 3, 9), dtype=jnp.float32)

    out = micropattern_helpers.masked_reinject_callback_bit(
        x,
        x_true,
        9,
        jax.random.PRNGKey(0),
        mask,
        jnp.array([0, 24, -1], dtype=jnp.int32),
    )

    assert jnp.all(out[0][:, 7] == 0.0)
    assert jnp.all(out[1][:2, 7] != 0.0)
    assert jnp.all(out[1][2:, 7] == 0.0)
    assert jnp.all(out[2][:, 7] != 0.0)


def test_micropattern_rejects_unsupported_channel_count():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    with pytest.raises(ValueError, match="Expected 4 or 12"):
        micropattern_helpers.build_data_augmenter(_micropattern_cfg(data_channels=5))

    with pytest.raises(ValueError, match="Expected 4 or 12"):
        micropattern_helpers.load_data(
            _micropattern_cfg(data_channels=5),
            impath="/tmp/micropatterns/",
        )


def test_micropattern_rejects_4_channel_knockout_data():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    cfg = _micropattern_cfg(data_channels=4, knockout_mode="only_one_ko")

    with pytest.raises(ValueError, match="no-knockout group-A data"):
        micropattern_helpers.build_data_augmenter(cfg)

    with pytest.raises(ValueError, match="no-knockout group-A data"):
        micropattern_helpers.load_data(cfg, impath="/tmp/micropatterns/")


def test_data_augmenter_4ch_colony_keeps_observable_channels():
    data = jnp.arange(1 * 5 * 4 * 2 * 3, dtype=jnp.float32).reshape(1, 5, 4, 2, 3)
    augmenter = DataAugmenter4Ch(data_true=data, hidden_channels=2)

    x, y = augmenter.split_x_y(1)

    assert x[0].shape == (4, 6, 2, 3)
    assert y[0].shape == (4, 6, 2, 3)
    assert jnp.array_equal(x[0][:, :4], data[0, :-1])
    assert jnp.array_equal(y[0][:, :4], data[0, 1:])
