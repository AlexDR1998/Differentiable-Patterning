

import hashlib
import os
from pathlib import Path

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_gated_noise_model import gnNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_normalized_model import NormalizedNCA
from NCA.model.NCA_model_fast import NCA as NCAFast
from NCA.model.NCA_sycl import NCA as NCASycl
from NCA.model.NCA_noise_model import nNCA
from NCA.model.NCA_upsample_isotropic_model import uNCA as isouNCA
from NCA.model.NCA_upsample_model import uNCA


MAX_WANDB_TAG_LENGTH = 64
EXCLUDED_WANDB_TAG_KEYS = {
    "logging.wandb.project",
    "logging.wandb.group",
    "logging.wandb.tags",
}


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def compute_channel_statistics(data, channel_axis=2, epsilon=1e-6):
    """Compute fixed per-channel mean/std from a loaded trajectory array.

    The training data convention is ``[batch, time, channel, height, width]``
    (and this also supports extra leading dimensions). Statistics are computed
    over every axis except the channel axis.
    """
    import jax.numpy as jnp

    values = jnp.asarray(data)
    if values.ndim < 3:
        raise ValueError(f"Expected trajectory data with at least 3 dimensions, got {values.shape}")
    channel_axis %= values.ndim
    reduce_axes = tuple(axis for axis in range(values.ndim) if axis != channel_axis)
    mean = jnp.mean(values, axis=reduce_axes)
    std = jnp.maximum(jnp.std(values, axis=reduce_axes), epsilon)
    return mean, std


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


def uses_vgg_loss(loss_primary):
    return any("vgg" in loss_name for loss_name in _as_list(loss_primary))


def resolve_loss_layers(loss_primary, layers):
    loss_count = len(_as_list(loss_primary))
    layers = _as_list(layers)
    if loss_count == 0:
        return layers
    if not layers:
        return ["decoded"] * loss_count
    if len(layers) < loss_count:
        layers = layers + [layers[-1]] * (loss_count - len(layers))
    return layers[:loss_count]


def build_loss_filename(cfg, include_layers=False, include_loss_args=False):
    loss_str = "_".join(_as_list(cfg.loss.primary)).lower()
    layers = _cfg_get(cfg.loss, "layers", None)
    if include_layers and layers is not None:
        loss_str += f"_layers{'-'.join(resolve_loss_layers(cfg.loss.primary, layers)).lower()}"
    if uses_vgg_loss(cfg.loss.primary):
        vgg_internal = _cfg_get(
            cfg.loss,
            "vgg_internal",
            _cfg_get(_cfg_get(cfg.loss, "args", None), "internal_loss_func", "l2"),
        )
        loss_str += f"_vgg{str(vgg_internal).lower()}"
        if _cfg_get(cfg.loss, "random_crop", False):
            loss_str += "_rc"
        if _cfg_get(cfg.loss, "random_channel_shuffle", False):
            loss_str += "_chshuffle"

    channel_importance = _cfg_get(cfg.loss, "channel_importance", None)
    if channel_importance is not None:
        non_default = [
            f"{index + 1}x{_compact_value(weight)}"
            for index, weight in enumerate(channel_importance)
            if float(weight) != 1.0
        ]
        if non_default:
            loss_str += "_ci" + "-".join(non_default)

    component_weights = _cfg_get(cfg.loss, "component_weights", None)
    if component_weights is not None and any(
        float(weight) != 1.0 for weight in component_weights
    ):
        loss_str += "_cw" + "-".join(_compact_value(weight) for weight in component_weights)

    loss_args = _cfg_get(cfg.loss, "args", None)
    if "multi_target" in _as_list(cfg.loss.primary):
        multi_target_weights = _cfg_get(loss_args, "multi_target_weights", None)
        if multi_target_weights is not None:
            loss_str += (
                f"_mtw_tex{_cfg_get(multi_target_weights, 'texture', 1.0):g}"
                f"_cm{_cfg_get(multi_target_weights, 'channel_mean', 0.0):g}"
                f"_corr{_cfg_get(multi_target_weights, 'correlation', 0.0):g}"
                f"_rad{_cfg_get(multi_target_weights, 'radial', 0.0):g}"
            )
    if include_loss_args and loss_args is not None:
        uses_ott = any("ott" in loss_name for loss_name in _as_list(cfg.loss.primary))
        if uses_ott:
            sharpen = _cfg_get(loss_args, "sharpen", None)
            arg_values = {
                "S": _cfg_get(loss_args, "S", None),
                "K": _cfg_get(loss_args, "K", None),
                "D": _cfg_get(loss_args, "D", None),
                "epsilon": _cfg_get(loss_args, "epsilon", None),
                "sharpen": None if sharpen is None else str(sharpen).lower(),
                "internal_loss_func": _cfg_get(loss_args, "internal_loss_func", None),
            }
            aliases = {
                "S": "S",
                "K": "K",
                "D": "D",
                "epsilon": "eps",
                "sharpen": "sharp",
                "internal_loss_func": "metric",
            }
        else:
            arg_values = {
                "samples": _cfg_get(loss_args, "samples", None),
                "epsilon": _cfg_get(loss_args, "epsilon", None),
                "tau": _cfg_get(loss_args, "tau", None),
                "normalize": _cfg_get(loss_args, "normalize", None),
                "amplitude_penalty": _cfg_get(loss_args, "amplitude_penalty", None),
            }
            aliases = {
                "samples": "s",
                "epsilon": "eps",
                "tau": "tau",
                "normalize": "norm",
                "amplitude_penalty": "ap",
            }
        arg_str = compact_nonzero_config_string(arg_values, aliases=aliases)
        if arg_str:
            loss_str += f"_{arg_str}"

    reg_str = compact_nonzero_config_string(
        cfg.loss.regulariser_coeffs,
        aliases={
            "boundary": "bd",
            "contiguous_growth": "cg",
            "intermediate_state": "is",
            "latent_channel_match": "lcm",
            "latent_size": "ls",
            "perturbation_conservation": "pc",
            "update_sensitivity": "us",
        },
    )
    if reg_str:
        loss_str += f"_{reg_str}"
    return loss_str


def build_loss_args(cfg, overrides=None):
    loss_args_cfg = _cfg_get(cfg.loss, "args", None)
    layers = resolve_loss_layers(cfg.loss.primary, _cfg_get(cfg.loss, "layers", None))
    loss_args = {
        "channels": _cfg_get(loss_args_cfg, "channels", None),
        "experiment_groups": _cfg_get(loss_args_cfg, "experiment_groups", None),
        "S": _cfg_get(loss_args_cfg, "S", 1024),
        "K": _cfg_get(loss_args_cfg, "K", 5),
        "D": _cfg_get(loss_args_cfg, "D", 3),
        "sharpen": _cfg_get(loss_args_cfg, "sharpen", True),
        "epsilon": _cfg_get(loss_args_cfg, "epsilon", 0.1),
        "internal_loss_func": _cfg_get(loss_args_cfg, "internal_loss_func", "l2"),
        "samples": _cfg_get(loss_args_cfg, "samples", 128),
        "layers": layers,
        "random_crop": _cfg_get(cfg.loss, "random_crop", False),
        "random_channel_shuffle": _cfg_get(cfg.loss, "random_channel_shuffle", False),
        "channel_importance": _cfg_get(cfg.loss, "channel_importance", None),
        "component_weights": _cfg_get(cfg.loss, "component_weights", None),
    }
    vgg_internal = _cfg_get(cfg.loss, "vgg_internal", None)
    if vgg_internal is not None:
        loss_args["metric"] = vgg_internal
    for optional_key in (
        "normalize",
        "tau",
        "amplitude_penalty",
        "multi_target_weights",
        "radial_bins",
        "assignment",
        "assignment_tau",
        "texture_size",
    ):
        value = _cfg_get(loss_args_cfg, optional_key, None)
        if value is not None:
            loss_args[optional_key] = value
    if overrides is not None:
        loss_args.update(overrides)
    return loss_args


def build_pool_admission_config(cfg):
    return {
        "enabled": _cfg_get(cfg.trainer, "pool_admission_enabled", True),
        "relative_threshold": _cfg_get(cfg.trainer, "pool_admission_relative_threshold", 1.25),
        "previous_relative_threshold": _cfg_get(
            cfg.trainer, "pool_admission_previous_relative_threshold", 1.10
        ),
        "absolute_threshold": _cfg_get(cfg.trainer, "pool_admission_absolute_threshold", None),
        "ema_decay": _cfg_get(cfg.trainer, "pool_admission_ema_decay", 0.95),
        "warmup": _cfg_get(cfg.trainer, "pool_admission_warmup", None),
    }


def set_matmul_precision(cfg):
    precision = _cfg_get(_cfg_get(cfg, "system", None), "precision", None)
    if precision is None:
        return
    import jax

    jax.config.update("jax_default_matmul_precision", precision)


def _build_kan_aux(cfg):
    kan_cfg = _cfg_get(cfg.model, "kan", None)
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


def _build_activation(cfg):
    import jax

    activation_name = _cfg_get(cfg.model, "activation", "relu")
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


def build_model_config_string(cfg):
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
    if cfg.model.family == "NormalizedNCA":
        normalization = _cfg_get(cfg.model, "normalization", "none")
        if normalization != "none":
            cfg_str += f"_norm{normalization}"
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
    elif cfg.model.family in {"uNCA", "isouNCA"}:
        upsampler = _cfg_get(cfg.model, "upsampler", None)
        cfg_str += (
            f"_up{cfg.model.upscale_factor}"
            f"_ud{_cfg_get(upsampler, 'depth', 'none')}"
            f"_uw{_cfg_get(upsampler, 'width_factor', 'none')}"
        )
        if cfg.model.family == "uNCA":
            cfg_str += f"_fm{_cfg_get(upsampler, 'fourier_modes', 'none')}"
        else:
            cfg_str += f"_rad{_cfg_get(upsampler, 'radius', 'none')}"
    return cfg_str


def build_model(cfg, key=None):
    activation = _build_activation(cfg)
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
    elif cfg.model.family == "NormalizedNCA":
        model = NormalizedNCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            NORMALIZATION=_cfg_get(cfg.model, "normalization", "none"),
            NORMALIZATION_MEAN=_cfg_get(cfg.model, "normalization_mean", None),
            NORMALIZATION_STD=_cfg_get(cfg.model, "normalization_std", None),
            NORMALIZATION_EPS=_cfg_get(cfg.model, "normalization_eps", 1e-6),
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
            KAN_AUX=_build_kan_aux(cfg),
            key=key,
        )
    elif cfg.model.family == "uNCA":
        model = uNCA(
            N_CHANNELS=cfg.model.channels,
            O_CHANNELS=cfg.data.data_channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            UPSAMPLER_AUX={
                "depth": cfg.model.upsampler.depth,
                "width_factor": cfg.model.upsampler.width_factor,
                "fourier_modes": cfg.model.upsampler.fourier_modes,
                "upsample_factor": cfg.model.upscale_factor,
            },
            key=key,
        )
    elif cfg.model.family == "isouNCA":
        model = isouNCA(
            N_CHANNELS=cfg.model.channels,
            O_CHANNELS=cfg.data.data_channels,
            KERNEL_STR=cfg.model.kernel_str,
            ACTIVATION=activation,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            KERNEL_SCALE=kernel_scale,
            UPSAMPLER_AUX={
                "depth": cfg.model.upsampler.depth,
                "width_factor": cfg.model.upsampler.width_factor,
                "radius": cfg.model.upsampler.radius,
                "upsample_factor": cfg.model.upscale_factor,
            },
            key=key,
        )
    else:
        raise ValueError(f"Unknown model family {cfg.model.family}")
    return model, build_model_config_string(cfg)


def resolve_checkpoint_path(cfg, env=None):
    """Resolve a configured checkpoint path and require an existing ``.eqx`` file.

    Relative paths are resolved against ``checkpoint.base_directory`` when it
    is set. Otherwise the environment variable named by
    ``checkpoint.base_env`` is used when available, followed by the current
    working directory.
    """

    checkpoint_cfg = _cfg_get(cfg, "checkpoint", None)
    configured_path = _cfg_get(checkpoint_cfg, "path", None)
    if not configured_path:
        raise ValueError("checkpoint.path must be set")

    path = Path(str(configured_path)).expanduser()
    if path.suffix != ".eqx":
        path = path.with_suffix(".eqx")
    if not path.is_absolute():
        base_directory = _cfg_get(checkpoint_cfg, "base_directory", None)
        environment = os.environ if env is None else env
        base_env = _cfg_get(checkpoint_cfg, "base_env", "MODEL_SAVE_PATH")
        if base_directory is None and base_env:
            base_directory = environment.get(str(base_env))
        if base_directory is not None:
            path = Path(str(base_directory)).expanduser() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return path


def load_model_checkpoint(cfg, key=None, env=None):
    """Construct the configured model architecture and load checkpoint leaves.

    The model section must describe the same architecture used to create the
    checkpoint. Returns the loaded model, its compact configuration string,
    and the resolved checkpoint path.
    """

    model, model_cfg_str = build_model(cfg, key=key)
    checkpoint_path = resolve_checkpoint_path(cfg, env=env)
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
            if tag_key == "data.sequence":
                value = _sequence_alias(value)
            else:
                value = _compact_value(value)
            tags.append(_safe_wandb_tag(f"{tag_key}:{value}"))
    return tags
