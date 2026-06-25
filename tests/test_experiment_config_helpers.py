import jax
import jax.numpy as jnp

from Experiments.config_helpers import build_model, build_model_config_string, build_tags
from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_model import NCA


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
    assert cfg_str.startswith("model_NCA")


def test_build_model_constructs_fast_kan_nca_with_defaults():
    cfg = _base_cfg("FastKaNCA")
    model, cfg_str = build_model(cfg, key=jax.random.PRNGKey(1))
    x = jnp.ones((4, 6, 7))
    y = model(x, key=jax.random.PRNGKey(2))

    assert isinstance(model, FastKaNCA)
    assert y.shape == x.shape
    assert jnp.allclose(y, x)
    assert model.get_config()["KAN_AUX"]["base_activation"] == "identity"
    assert "model_FastKaNCA" in cfg_str


def test_build_model_constructs_fast_kan_nca_with_kan_overrides():
    cfg = _base_cfg("FastKaNCA")
    cfg.model.kan = {
        "hidden_features": 6,
        "num_basis": 4,
        "base_activation": "none",
        "use_layernorm": False,
    }
    model, _ = build_model(cfg, key=jax.random.PRNGKey(3))

    assert model.KAN_AUX["hidden_features"] == 6
    assert model.KAN_AUX["num_basis"] == 4
    assert model.KAN_AUX["base_activation"] == "none"
    assert model.KAN_AUX["use_layernorm"] is False


def test_build_model_config_string_handles_missing_kan_section():
    cfg = _base_cfg("FastKaNCA")
    cfg_str = build_model_config_string(cfg)

    assert "kb8" in cfg_str
    assert "kbaseidentity" in cfg_str


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
