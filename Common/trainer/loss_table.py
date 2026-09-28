"""The table of named losses that loss terms in the config can use.

``LOSSES`` maps each loss name (``loss.terms[i].type``) to a function that
takes the term's options (a plain dict, see ``NCA/trainer/objective.py``) and
returns the loss function used during training:

    loss(x, y, key, where, cache) -> float32 [N]

VGG losses also have a function that computes the target features once
before training, which are then passed back in as ``cache``.

To add a loss: write the function (in ``loss.py``, or ``loss_micropattern.py``
if it depends on the micropattern channel layout), add it here, and add its
config class to ``LOSS_TERM_CONFIGS`` in ``Common/trainer/config.py``.
``multi_target`` is not in this table; ``NCA/trainer/step.py`` handles it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

import Common.trainer.loss as loss
import Common.trainer.loss_micropattern as loss_micropattern
import Common.trainer.loss_ott as loss_ott
import Common.trainer.loss_vgg as loss_vgg
from Common.dataloader.micropattern_schemas import MICROPATTERN_GROUPED_12CH_SCHEMA
from Common.trainer.config import LOSS_TERM_CONFIGS


@dataclass(frozen=True)
class Loss:
    # options -> loss(x, y, key, where, cache)
    build: Callable
    # options -> precompute(targets, key, where), for losses that cache target features
    build_target_cache: Optional[Callable] = None


def _plain(function):
    """A loss with no options."""
    return lambda options: lambda x, y, key, where, cache: function(x, y, key, where)


def _with_aux(function, make_aux):
    """A loss whose options are passed as an ``aux`` dict built by ``make_aux``."""
    def build(options):
        aux = make_aux(options)
        return lambda x, y, key, where, cache: function(x, y, key, where, aux=aux, cache=cache)
    return build


def _without_cache(function, make_aux):
    """Like ``_with_aux``, for loss functions that take no ``cache`` argument."""
    def build(options):
        aux = make_aux(options)
        return lambda x, y, key, where, cache: function(x, y, key, where, aux=aux)
    return build


# --- Option dicts passed to the loss functions as ``aux`` ---

def _samples(options):
    return {"samples": options.get("samples")}


def _vgg_options(options):
    return {
        "vgg_metric": options.get("metric", "l2"),
        "samples": options.get("samples"),
        "vgg_params": options.get("vgg_params"),
        "random_crop": options.get("random_crop", False),
        "random_channel_shuffle": options.get("random_channel_shuffle", False),
        "channel_importance": options.get("channel_importance"),
    }


def _ott_options(options):
    return {
        "D": options.get("D"),
        "S": options.get("S"),
        "K": options.get("K"),
        "sharpen": options.get("sharpen", False),
        "epsilon": options.get("epsilon"),
        "internal_loss_func": options.get("internal_loss_func"),
    }


def _grouped_options(options):
    return {"channel_importance": options.get("channel_importance")}


def _summary_options(options):
    return {"radial_bins": options.get("radial_bins", 16), "epsilon": 1e-8}


def _channel_importance(options):
    importance = options.get("channel_importance")
    if importance is None:
        return jnp.ones(MICROPATTERN_GROUPED_12CH_SCHEMA.n_measurement_channels)
    return jnp.asarray(importance)


def _grouped_radial_options(options):
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA
    return {
        **_summary_options(options),
        "channel_weights": jnp.asarray(schema.measurement_weights) * _channel_importance(options),
    }


def _grouped_correlation_options(options):
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA
    importance = _channel_importance(options)
    pairs = jnp.asarray(schema.co_measurement_pairs)
    return {
        **_summary_options(options),
        "pairs": schema.co_measurement_pairs,
        "pair_weights": jnp.asarray(schema.correlation_pair_weights)
        * jnp.sqrt(importance[pairs[:, 0]] * importance[pairs[:, 1]]),
    }


def _vgg_target_cache(precompute):
    def build(options):
        aux = _vgg_options(options)
        return lambda targets, key, where: precompute(targets, key, where, aux=aux)
    return build


LOSSES = {
    # Pointwise, spectral and distribution losses
    "l1": Loss(_plain(loss.l1)),
    "l2": Loss(_plain(loss.l2)),
    "euclidean": Loss(_plain(loss.euclidean)),
    "cosine": Loss(_plain(loss.cosine)),
    "spectral": Loss(_plain(loss.spectral)),
    "spectral_no_phase": Loss(_plain(loss.spectral_no_phase)),
    "spectral_phase": Loss(_plain(loss.spectral_only_phase)),
    "bhattacharyya": Loss(_plain(loss.bhattacharyya_distance)),
    "kl_divergence": Loss(_plain(loss.kl_divergence)),
    "hellinger": Loss(_plain(loss.hellinger_distance)),
    "average_amplitude": Loss(_plain(loss.average_amplitude_distance)),
    # Sliced Wasserstein losses
    "sliced_wasserstein_spatial": Loss(_with_aux(loss.sliced_wasserstein_spatial, _samples)),
    "sliced_wasserstein_channel": Loss(_with_aux(loss.sliced_wasserstein_channel, _samples)),
    "sliced_wasserstein_full": Loss(_with_aux(loss.wasserstein_projected, _samples)),
    "sliced_wasserstein_rotational": Loss(_with_aux(loss.sliced_wasserstein_rotational, _samples)),
    "spectral_wasserstein_full": Loss(_with_aux(loss.spectral_wasserstein_projected, _samples)),
    # Radial profile and channel correlation summaries
    "radial_profile": Loss(_with_aux(loss.radial_profile_loss, _summary_options)),
    "channel_correlation": Loss(_with_aux(loss.channel_correlation_loss, _summary_options)),
    # VGG feature losses
    "vgg": Loss(
        _with_aux(loss_vgg.vgg_hyperspectral, _vgg_options),
        _vgg_target_cache(loss_vgg.precompute_vgg_hyperspectral_target),
    ),
    # Optimal transport texture losses
    "ott": Loss(_without_cache(loss_ott.ott_loss, _ott_options)),
    "ott_chstack": Loss(_without_cache(loss_ott.ott_channel_stack_loss, _ott_options)),
    # Grouped micropattern layout (loss_micropattern.py)
    "l2_grouped": Loss(_with_aux(loss_micropattern.l2_colony_grouped, _grouped_options)),
    "radial_profile_grouped": Loss(
        _with_aux(loss_micropattern.radial_profile_grouped_loss, _grouped_radial_options)
    ),
    "channel_correlation_grouped": Loss(
        _with_aux(loss_micropattern.channel_correlation_grouped_loss, _grouped_correlation_options)
    ),
    "vgg_grouped": Loss(
        _with_aux(loss_micropattern.vgg_hyperspectral_colony, _vgg_options),
        _vgg_target_cache(loss_micropattern.precompute_vgg_hyperspectral_colony_target),
    ),
    "vgg_grouped_and_l2": Loss(
        _with_aux(loss_micropattern.vgg_hyperspectral_colony_and_l2, _vgg_options),
        _vgg_target_cache(loss_micropattern.precompute_vgg_hyperspectral_colony_target),
    ),
    "ott_grouped": Loss(_without_cache(loss_micropattern.ott_grouped_loss, _ott_options)),
    "ott_grouped_and_l2": Loss(_without_cache(loss_micropattern.ott_grouped_and_l2_loss, _ott_options)),
}

# Losses that accept per-channel ``channel_importance`` weights
CHANNEL_IMPORTANCE_LOSSES = {
    "l2_grouped", "vgg_grouped", "vgg_grouped_and_l2",
    "radial_profile_grouped", "channel_correlation_grouped",
}

if set(LOSSES) != set(LOSS_TERM_CONFIGS) - {"multi_target"}:
    raise RuntimeError(
        "LOSSES and Common.trainer.config.LOSS_TERM_CONFIGS name different losses: "
        f"{sorted(set(LOSSES) ^ (set(LOSS_TERM_CONFIGS) - {'multi_target'}))}"
    )


def _names(loss_names):
    return [loss_names] if isinstance(loss_names, str) else list(loss_names)


def _check_channel_importance(names, options):
    importance = options.get("channel_importance")
    if importance is None:
        return
    if len(importance) != MICROPATTERN_GROUPED_12CH_SCHEMA.n_measurement_channels:
        raise ValueError(
            "loss term channel_importance must contain 12 target-channel weights for grouped micropattern losses"
        )
    if any(float(weight) < 0 for weight in importance):
        raise ValueError("loss term channel_importance cannot contain negative weights")
    if not any(float(weight) > 0 for weight in importance):
        raise ValueError("loss term channel_importance must contain at least one positive weight")
    unsupported = [name for name in names if name not in CHANNEL_IMPORTANCE_LOSSES]
    if unsupported:
        raise ValueError(
            "loss term channel_importance is only supported for grouped micropattern losses; "
            f"unsupported losses: {unsupported}"
        )


def build_loss_functions(loss_names, options):
    """Return the loss function for each name, in the same order.

    Parameters
    ----------
    loss_names : str or list of str
        keys of ``LOSSES``
    options : dict
        the loss terms' options, shared by all terms
    """
    names = _names(loss_names)
    unknown = [name for name in names if name not in LOSSES]
    if unknown:
        raise ValueError(f"Unknown losses: {unknown}")
    _check_channel_importance(names, options)
    return [LOSSES[name].build(options) for name in names]


def build_loss_initialiser(loss_names, options):
    """Return the target-feature precompute function for the first VGG loss, or None."""
    for name in _names(loss_names):
        if name in LOSSES and LOSSES[name].build_target_cache is not None:
            return LOSSES[name].build_target_cache(options)
    return None
