

import hashlib
import os
from pathlib import Path

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_gated_noise_model import gnNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_model_fast import NCA as NCAFast
from NCA.model.NCA_sycl import NCA as NCASycl
from NCA.model.NCA_noise_model import nNCA


MAX_WANDB_TAG_LENGTH = 64
EXCLUDED_WANDB_TAG_KEYS = {
    "logging.wandb.project",
    "logging.wandb.group",
    "logging.wandb.tags",
    "model_store.root",
    "model_store.collection",
    "model_store.model_factory",
}


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def data_channel_count(cfg):
    """Return the measurement channel count for the configured data domain."""

    if _cfg_get(cfg.data, "dataset") == "emojis":
        return cfg.data.emoji.data_channels
    return cfg.data.micropattern.data_channels


def _compact_value(value):
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    return str(value)


def _sequence_alias(sequence):
    aliases = []
    for filename in _as_list(sequence):
        basename = str(filename).rsplit("/", 1)[-1].split(".", 1)[0]
        alias = basename[:2].lower()
        if aliases and aliases[-1] == alias:
            continue
        aliases.append(alias)
    return "_".join(aliases)


def _safe_wandb_tag(tag, max_length=MAX_WANDB_TAG_LENGTH):
    if len(tag) <= max_length:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    return f"{tag[: max_length - 9]}~{digest}"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def compact_nonzero_config_string(values, aliases=None):
    aliases = aliases or {}
    parts = []
    for key, value in values.items():
        if value is None or value == 0:
            continue
        parts.append(f"{aliases.get(key, key)}{_compact_value(value)}")
    return "_".join(parts)


def loss_terms(loss_config):
    return list(_cfg_get(loss_config, "terms", ()))


def loss_names(loss_config):
    return [str(_cfg_get(term, "type")) for term in loss_terms(loss_config)]


def loss_weights(loss_config):
    return [float(_cfg_get(term, "weight", 1.0)) for term in loss_terms(loss_config)]


def build_loss_filename(loss_config, include_loss_args=False):
    terms = loss_terms(loss_config)
    names = loss_names(loss_config)
    loss_str = "_".join(names).lower()
    for term in terms:
        name = str(_cfg_get(term, "type"))
        if "vgg" in name:
            loss_str += f"_vgg{str(_cfg_get(term, 'metric', 'l2')).lower()}"
            if _cfg_get(term, "random_crop", False): loss_str += "_rc"
            if _cfg_get(term, "random_channel_shuffle", False): loss_str += "_chshuffle"
        channel_importance = _cfg_get(term, "channel_importance", None)
        if channel_importance is not None:
            non_default = [f"{i + 1}x{_compact_value(w)}" for i, w in enumerate(channel_importance) if float(w) != 1.0]
            if non_default: loss_str += "_ci" + "-".join(non_default)
        if name == "multi_target":
            multi_target_weights = _cfg_get(term, "multi_target_weights", None)
        else:
            multi_target_weights = None
        if multi_target_weights is not None:
            loss_str += (
                f"_mtw_tex{_cfg_get(multi_target_weights, 'texture', 1.0):g}"
                f"_cm{_cfg_get(multi_target_weights, 'channel_mean', 0.0):g}"
                f"_corr{_cfg_get(multi_target_weights, 'correlation', 0.0):g}"
                f"_rad{_cfg_get(multi_target_weights, 'radial', 0.0):g}"
                f"_rchsh{int(bool(_cfg_get(term, 'random_channel_shuffle', False)))}"
                f"_rcr{int(bool(_cfg_get(term, 'random_crop', False)))}"
            )
            l2_weight = _cfg_get(multi_target_weights, "l2", 0.0)
            if float(l2_weight) != 0.0:
                loss_str += f"_l2{l2_weight:g}"
        if include_loss_args:
            keys = ("S", "K", "D", "epsilon", "sharpen", "samples", "tau", "normalize", "amplitude_penalty")
            arg_str = compact_nonzero_config_string({key: _cfg_get(term, key, None) for key in keys})
            if arg_str: loss_str += f"_{arg_str}"

    weights = loss_weights(loss_config)
    if any(weight != 1.0 for weight in weights):
        loss_str += "_cw" + "-".join(_compact_value(weight) for weight in weights)

    reg_str = compact_nonzero_config_string(
        _cfg_get(loss_config, "regularisers", {}),
        aliases={
            "boundary": "bd",
            "contiguous_growth": "cg",
            "intermediate_state": "is",
            "hidden_state_size": "hs",
            "localised_hidden": "lh",
            "perturbation_conservation": "pc",
            "update_sensitivity": "us",
        },
    )
    if reg_str:
        loss_str += f"_{reg_str}"
    return loss_str


def build_loss_args(loss_config, overrides=None):
    terms = loss_terms(loss_config)
    loss_args = {
        "component_weights": loss_weights(loss_config),
    }
    ignored = {"type", "weight", "layer"}
    for term in terms:
        values = dict(term.items()) if hasattr(term, "items") else vars(term)
        for key, value in values.items():
            if key in ignored or value is None:
                continue
            runtime_key = "internal_loss_func" if key == "metric" else key
            if key == "metric" and "vgg" in str(_cfg_get(term, "type")):
                runtime_key = "metric"
            previous = loss_args.get(runtime_key, value)
            if previous != value:
                raise ValueError(f"Loss terms require conflicting shared runtime value for {runtime_key!r}")
            loss_args[runtime_key] = value
    loss_args.setdefault("channels", None)
    loss_args.setdefault("experiment_groups", None)
    if overrides is not None:
        loss_args.update(overrides)
    return loss_args


def build_pool_admission_config(trainer_config):
    pool = trainer_config.pool_admission
    return {
        "enabled": pool.enabled,
        "relative_threshold": pool.relative_threshold,
        "previous_relative_threshold": pool.previous_relative_threshold,
        "absolute_threshold": pool.absolute_threshold,
        "ema_decay": pool.ema_decay,
        "warmup": pool.warmup,
    }


def set_matmul_precision(runtime_config):
    precision = runtime_config.precision
    if precision is None:
        return
    import jax

    jax.config.update("jax_default_matmul_precision", precision)


def _build_kan_aux(model_config):
    kan_cfg = _cfg_get(model_config, "kan", None)
    hidden_features = _cfg_get(kan_cfg, "hidden_features", None)
    kan_aux = {
        "basis": _cfg_get(kan_cfg, "basis", "rbf"),
        "num_basis": _cfg_get(kan_cfg, "num_basis", 8),
        "grid_min": _cfg_get(kan_cfg, "grid_min", -2.0),
        "grid_max": _cfg_get(kan_cfg, "grid_max", 2.0),
        "rbf_width": _cfg_get(kan_cfg, "rbf_width", None),
        "trainable_width": _cfg_get(kan_cfg, "trainable_width", True),
        "extrapolation": _cfg_get(kan_cfg, "extrapolation", "constant"),
        "use_base_branch": _cfg_get(kan_cfg, "use_base_branch", True),
        "base_activation": _cfg_get(kan_cfg, "base_activation", "identity"),
        "use_layernorm": _cfg_get(kan_cfg, "use_layernorm", True),
        "spline_init_scale": _cfg_get(kan_cfg, "spline_init_scale", 0.1),
        "base_init_scale": _cfg_get(kan_cfg, "base_init_scale", 0.1),
        "final_zero_init": _cfg_get(kan_cfg, "final_zero_init", True),
    }
    if hidden_features is not None:
        kan_aux["hidden_features"] = hidden_features
    return kan_aux


def _build_activation(model_config):
    import jax

    activation_name = _cfg_get(model_config, "activation", "relu")
    if activation_name in {None, "relu"}:
        return jax.nn.relu
    if activation_name == "tanh":
        return jax.nn.tanh
    if activation_name == "swish":
        return jax.nn.swish
    if activation_name == "gelu":
        return jax.nn.gelu
    if activation_name == "linear":
        return lambda x: x
    raise ValueError(f"Unsupported activation {activation_name}")


def build_model_config_string(model_config):
    from types import SimpleNamespace

    cfg = SimpleNamespace(model=model_config)
    cfg_str = (
        f"{cfg.model.family}"
        f"_c{cfg.model.channels}"
        # f"_k{_compact_value(list(cfg.model.kernel_str))}"
        # f"_fr{cfg.model.fire_rate}"
    )
    activation = _cfg_get(cfg.model, "activation", None)
    if activation not in {None, "relu"}:
        cfg_str += f"_act{activation}"
    kernel_scale = _cfg_get(cfg.model, "kernel_scale", 1)
    if kernel_scale != 1:
        cfg_str += f"_ks{kernel_scale}"
    if cfg.model.family in {"nNCA", "gnNCA"}:
        cfg_str += f"_pn{_cfg_get(cfg.model, 'parameter_noise_level', 0.01)}"
    if cfg.model.family == "FastKaNCA":
        kan_cfg = _cfg_get(cfg.model, "kan", None)
        cfg_str += f"_kb{_cfg_get(kan_cfg, 'num_basis', 8)}"
        basis = _cfg_get(kan_cfg, "basis", "rbf")
        if basis == "linear_spline":
            cfg_str += "_klin"
        elif basis != "rbf":
            cfg_str += f"_k{basis}"
        extrapolation = _cfg_get(kan_cfg, "extrapolation", "constant")
        if extrapolation != "constant":
            cfg_str += f"_kex{extrapolation}"
        hidden_features = _cfg_get(kan_cfg, "hidden_features", None)
        if hidden_features is not None:
            cfg_str += f"_kh{hidden_features}"
        base_activation = _cfg_get(kan_cfg, "base_activation", "identity")
        if base_activation != "identity":
            cfg_str += f"_kbase{base_activation}"
        if not _cfg_get(kan_cfg, "use_layernorm", True):
            cfg_str += "_noln"
        if not _cfg_get(kan_cfg, "final_zero_init", True):
            cfg_str += "_nozero"
    return cfg_str


def build_model(model_config, key=None):
    """Construct a model solely from its reconstructable typed config."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(model=model_config)
    activation = _build_activation(model_config)
    kernel_scale = _cfg_get(cfg.model, "kernel_scale", 1)
    if cfg.model.family == "NCA":
        model = NCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            key=key,
        )
    elif cfg.model.family == "NCA_fast":
        model = NCAFast(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            key=key,
        )
    elif cfg.model.family == "NCA_sycl":
        model = NCASycl(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            key=key,
        )
    elif cfg.model.family == "gNCA":
        model = gNCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            key=key,
        )
    elif cfg.model.family == "nNCA":
        model = nNCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            PARAMETER_NOISE_LEVEL=_cfg_get(cfg.model, "parameter_noise_level", 0.01),
            key=key,
        )
    elif cfg.model.family == "gnNCA":
        model = gnNCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            PARAMETER_NOISE_LEVEL=_cfg_get(cfg.model, "parameter_noise_level", 0.01),
            key=key,
        )
    elif cfg.model.family == "FastKaNCA":
        model = FastKaNCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            KAN_AUX=_build_kan_aux(model_config),
            key=key,
        )
    else:
        raise ValueError(f"Unknown model family {cfg.model.family}")
    return model, build_model_config_string(model_config)


def resolve_checkpoint_path(checkpoint_config, env=None):
    """Resolve a configured checkpoint path and require an existing ``.eqx`` file.

    Relative paths are resolved against ``checkpoint.base_directory`` when it
    is set. Otherwise the environment variable named by
    ``checkpoint.base_env`` is used when available, followed by the current
    working directory.
    """

    configured_path = checkpoint_config.path
    if not configured_path:
        raise ValueError("checkpoint.path must be set")

    path = Path(str(configured_path)).expanduser()
    if path.suffix != ".eqx":
        path = path.with_suffix(".eqx")
    if not path.is_absolute():
        base_directory = checkpoint_config.base_directory
        environment = os.environ if env is None else env
        base_env = checkpoint_config.base_env
        if base_directory is None and base_env:
            base_directory = environment.get(str(base_env))
        if base_directory is not None:
            path = Path(str(base_directory)).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return path


def load_model_checkpoint(model_config, checkpoint_config, key=None, env=None):
    """Construct the configured model architecture and load checkpoint leaves.

    The model section must describe the same architecture used to create the
    checkpoint. Returns the loaded model, its compact configuration string,
    and the resolved checkpoint path.
    """

    model, model_cfg_str = build_model(model_config, key=key)
    checkpoint_path = resolve_checkpoint_path(checkpoint_config, env=env)
    model = model.load(checkpoint_path)
    return model, model_cfg_str, checkpoint_path


def build_tags(cfg, prefix=""):
    tags = []
    for key, value in cfg.items():
        if key == "seed":
            continue
        tag_key = f"{prefix}{key}"
        if tag_key in EXCLUDED_WANDB_TAG_KEYS:
            continue
        if value is None:
            continue
        if hasattr(value, "items"):
            tags.extend(build_tags(value, prefix=f"{tag_key}."))
        else:
            if tag_key == "data.emoji.sequence":
                value = _sequence_alias(value)
            else:
                value = _compact_value(value)
            tags.append(_safe_wandb_tag(f"{tag_key}:{value}"))
    return tags
