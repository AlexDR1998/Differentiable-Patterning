"""Losses for the grouped micropattern data layout.

The micropattern targets have 12 measurement channels in four experiment
groups (some markers were measured in more than one experiment), while the
NCA state has 9 unique biological channels. These losses map the state onto
the measurement layout (``duplicate_x_channels_9ch``) and keep channels from
the same experiment together. Everything that knows about this layout lives
here, so ``loss.py``, ``loss_vgg.py`` and ``loss_ott.py`` stay generic.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange, reduce, einsum

import Common.trainer.loss_ott as loss_ott
import Common.trainer.loss_vgg as loss_vgg
from Common.dataloader.micropattern_schemas import MICROPATTERN_GROUPED_12CH_SCHEMA
from Common.trainer.experiment_channel_grouping import (
    duplicate_x_channels_9ch,
    project_state_to_measurements,
    split_and_pad_by_experiment_groups_12ch,
)
from Common.trainer.loss import channel_correlation_loss, radial_profile_loss
from Common.trainer.loss_vgg import LOSS_DTYPE


# ---------------------------------------------------------------------
# Pointwise and summary losses
# ---------------------------------------------------------------------

def l2_colony_grouped(x,y,key,where,aux=None,cache=None):
    aux = {} if aux is None else aux
    x_full = duplicate_x_channels_9ch(x)
    _l2 = (x_full-y)**2
    if where is None:
        where_full = None
    elif where.shape[1] == y.shape[1]:
        where_full = where.astype(where.dtype)
    else:
        where_full = duplicate_x_channels_9ch(where).astype(where.dtype)
    base_weighting = jnp.array(
        [0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0],
        dtype=_l2.dtype,
    ) # Account for duplicate channels
    channel_importance = aux.get("channel_importance", None)
    if channel_importance is None:
        weighting = base_weighting
    else:
        channel_importance = jnp.asarray(channel_importance, dtype=_l2.dtype)
        if where is None:
            active = jnp.ones((x.shape[0], y.shape[1]), dtype=_l2.dtype)
        else:
            active = jnp.any(where_full, axis=(-1, -2)).astype(_l2.dtype)
        base_total = jnp.sum(active * base_weighting[None, :], axis=1, keepdims=True)
        weighted = base_weighting * channel_importance
        weighted_total = jnp.sum(active * weighted[None, :], axis=1, keepdims=True)
        scale = jnp.where(weighted_total > 0, base_total / weighted_total, 1.0)
        weighting = weighted[None, :] * scale
    _l2 = _l2 * weighting[..., None, None]
    _l2_loss = jnp.nan_to_num(jnp.mean(_l2,axis=[-1,-2,-3],where=where_full))
    return _l2_loss


def _grouped_summary_loss(loss_func, x, y, where, aux):
    """Map the grouped 9-channel state to the 12-channel target layout, then apply ``loss_func``."""
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA
    x = project_state_to_measurements(x[:, :schema.n_state_channels], schema)
    y = y[:, :schema.n_measurement_channels]
    if where is not None and where.shape[1] != schema.n_measurement_channels:
        where = project_state_to_measurements(where[:, :schema.n_state_channels], schema)
    return loss_func(x, y, where=where, aux=aux)


def radial_profile_grouped_loss(x,y,key=None,where=None,aux=None,cache=None):
    """Radial-profile loss on the grouped micropattern layout."""
    aux = {} if aux is None else dict(aux)
    aux.setdefault("channel_weights", MICROPATTERN_GROUPED_12CH_SCHEMA.measurement_weights)
    return _grouped_summary_loss(radial_profile_loss, x, y, where, aux)


def channel_correlation_grouped_loss(x,y,key=None,where=None,aux=None,cache=None):
    """Channel-correlation loss over co-measured channels of the grouped micropattern layout."""
    aux = {} if aux is None else dict(aux)
    aux.setdefault("pairs", MICROPATTERN_GROUPED_12CH_SCHEMA.co_measurement_pairs)
    aux.setdefault("pair_weights", MICROPATTERN_GROUPED_12CH_SCHEMA.correlation_pair_weights)
    return _grouped_summary_loss(channel_correlation_loss, x, y, where, aux)


# ---------------------------------------------------------------------
# VGG losses
# ---------------------------------------------------------------------

def _pad_grouped_12ch_values(values, pad_value=0):
    """Pad per-channel values in the four colony groups, as the grouped VGG image channels are padded."""
    groups = [values[0:4], values[4:8], values[8:11], values[11:12]]
    return jnp.concatenate(
        [
            jnp.pad(group, (0, (3 - group.shape[0] % 3) % 3), constant_values=pad_value)
            for group in groups
        ]
    )


def grouped_vgg_triplet_weights(
    channel_importance,
    where=None,
    random_channel_shuffle=False,
    key=None,
):
    """Map 12 per-channel weights onto the six grouped VGG triplets.

    A triplet gets the mean weight of its active channels, times the usual
    weighting for duplicated channels. The result is rescaled per sample so
    that uniform weights give the usual weighting exactly.
    """
    if random_channel_shuffle:
        base_weighting = jnp.array(
            [0.75, 0.75, 0.75, 0.75, 1.0, 1.0], dtype=LOSS_DTYPE
        )
    else:
        base_weighting = jnp.array(
            [0.5, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=LOSS_DTYPE
        )
    importance = _pad_grouped_12ch_values(
        jnp.asarray(channel_importance, dtype=LOSS_DTYPE)
    )
    validity = _pad_grouped_12ch_values(jnp.ones((12,), dtype=LOSS_DTYPE))

    if where is None:
        active = validity[None, :]
    else:
        active = jnp.any(where, axis=(-1, -2)).astype(LOSS_DTYPE)
        active = jax.vmap(_pad_grouped_12ch_values)(active) * validity[None, :]

    if random_channel_shuffle:
        if key is None:
            raise ValueError("key is required when random_channel_shuffle is enabled")
        permutation_key = jr.fold_in(key, 1)
        importance = loss_vgg.permute_grouped_channels(
            importance, permutation_key, (6, 6, 3, 3)
        )
        active = loss_vgg.permute_grouped_channels(
            active, permutation_key, (6, 6, 3, 3), axis=1
        )

    active = rearrange(active, "n (c vc) -> c n vc", vc=3)
    importance = rearrange(importance, "(c vc) -> c vc", vc=3)
    active_count = jnp.sum(active, axis=-1)
    triplet_importance = jnp.sum(
        active * importance[:, None, :], axis=-1
    ) / jnp.maximum(active_count, 1.0)

    base_total = jnp.sum(
        base_weighting[:, None] * (active_count > 0), axis=0, keepdims=True
    )
    weighted = base_weighting[:, None] * triplet_importance
    weighted_total = jnp.sum(weighted, axis=0, keepdims=True)
    scale = jnp.where(weighted_total > 0, base_total / weighted_total, 1.0)
    return weighted * scale


@eqx.filter_jit
def precompute_vgg_hyperspectral_colony_target(y, key, where=None, aux={"vgg_metric": "l2"}):
    """Target features for ``vgg_hyperspectral_colony`` (see loss_vgg.precompute_vgg_target)."""
    return loss_vgg.precompute_vgg_target(y, key, aux, split_and_pad_by_experiment_groups_12ch)


def vgg_hyperspectral_colony(x, y, key, where=None, aux={"vgg_metric": "l2"}, cache=None):
    """VGG loss on the grouped micropattern layout.

    Each experiment group is padded to a multiple of 3 channels, so VGG
    triplets never mix experiments.

    Parameters
    ----------
    x : float32 [N,9,WIDTH,HEIGHT]
        predictions
    y : float32 [N,12,WIDTH,HEIGHT]
        true data, with the duplicated measurement channels
    key : jax.random.PRNGKey
    where : boolean array [N,9 or 12,(),()]
        channels (and timesteps) to include
    aux : dict
        as for loss_vgg.vgg_hyperspectral, plus optional "channel_importance" (12 weights)
    cache : precomputed target features, or None

    Returns
    -------
    loss : float32 [N]
    """
    where_y = None
    if where is not None:
        x = x * where.astype(x.dtype)
        where_y = where if where.shape[1] == y.shape[1] else duplicate_x_channels_9ch(where)
        y = y * where_y.astype(y.dtype)

    x = duplicate_x_channels_9ch(x)
    x = split_and_pad_by_experiment_groups_12ch(x)
    y = split_and_pad_by_experiment_groups_12ch(y)

    use_target_cache = cache is not None and not aux.get("random_crop", False)
    random_channel_shuffle = aux.get("random_channel_shuffle", False) and not use_target_cache
    if random_channel_shuffle:
        x, y = loss_vgg.permute_matching_channel_groups(
            x,
            y,
            jr.fold_in(key, 1),
            group_sizes=(6, 6, 3, 3),
        )

    losses = loss_vgg.vgg_group_losses(x, y, key, aux, cache).astype(LOSS_DTYPE)

    channel_importance = aux.get("channel_importance", None)
    if channel_importance is not None:
        loss_weighting = grouped_vgg_triplet_weights(
            channel_importance,
            where=where_y,
            random_channel_shuffle=random_channel_shuffle,
            key=key,
        )
        losses = einsum(losses, loss_weighting, "c n i j k, c n -> c n i j k")
    else:
        # Account for the duplicated channels, which appear in two blocks
        if random_channel_shuffle:
            loss_weighting = jnp.array([0.75, 0.75, 0.75, 0.75, 1.0, 1.0], dtype=LOSS_DTYPE)
        else:
            loss_weighting = jnp.array([0.5, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=LOSS_DTYPE)
        losses = einsum(losses, loss_weighting, "c n i j k, c -> c n i j k")

    return reduce(losses, "c n () () () -> n", "mean")


def vgg_hyperspectral_colony_and_l2(x, y, key, where, aux={"vgg_metric": "l2"}, cache=None):
    return vgg_hyperspectral_colony(x, y, key, where, aux, cache) + l2_colony_grouped(x, y, key, where, aux)


# ---------------------------------------------------------------------
# Optimal transport losses
# ---------------------------------------------------------------------

def ott_grouped_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """OT loss computed separately on each experiment group of channels.

    Patches are taken at the same positions in every channel of a group.

    Parameters
    ----------
    x : float32 [N 9 H W]
        predictions
    y : float32 [N 12 H W]
        true data, with the duplicated measurement channels
    key : jax.random.PRNGKey
    where : boolean array [N 9 1 1]
        channels (and timesteps) to include
    aux : dict
        as for loss_ott.ott_loss

    Returns
    -------
    loss : float32 [N]
    """
    
    N = x.shape[0]
    C = x.shape[1]
    S = aux["S"]
    K = aux["K"]
    D = aux["D"]
    ott_kwargs = {
        "epsilon": aux["epsilon"],
        "internal_loss_func": aux["internal_loss_func"],
    }
    if aux["sharpen"]:
        x = loss_ott._sharpen(x,2)
        y = loss_ott._sharpen(y,2)
    x = duplicate_x_channels_9ch(x)
    if where is not None:
        where = duplicate_x_channels_9ch(where)
        x = x*where.astype(x.dtype)
        y = y*where.astype(y.dtype)
    
    def v_ot_loss(x,y,key):
        """OT loss for one group of one sample ``[C H W]``, averaged over scales."""
        keys = jr.split(key,2)
        v_ch_downsample_and_patch = jax.vmap(loss_ott._downsample_and_patch, in_axes=(0,None,None,None,None),out_axes=0) # vectorized over channels. Don't vectorize over keys - we want to select the same patches across channels in each group
        px = v_ch_downsample_and_patch(x,S,K,D,keys[0]) # C D S K*K
        py = v_ch_downsample_and_patch(y,S,K,D,keys[1]) # C D S K*K
        px = rearrange(px,"C D S Kk -> D S (C Kk)")
        py = rearrange(py,"C D S Kk -> D S (C Kk)")
        
        vscale_ot_loss = jax.vmap(loss_ott._ott_patch_loss,in_axes=(0,0,None),out_axes=(0))(px,py,ott_kwargs) # vectorized over scales which is then averaged over
        return jnp.mean(vscale_ot_loss)
    
    vv_ot_loss = jax.vmap(v_ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over N
    keys = jr.split(key,(N,4))
    
    # Each group of channels from the same experiment will have identically located patches selected.
    
    losses = jnp.stack([
        vv_ot_loss(x[:,0:4,:,:],y[:,0:4,:,:],keys[:,0]),
        vv_ot_loss(x[:,4:7,:,:],y[:,4:7,:,:],keys[:,1]),
        vv_ot_loss(x[:,7:11,:,:],y[:,7:11,:,:],keys[:,2]),
        vv_ot_loss(x[:,11:12,:,:],y[:,11:12,:,:],keys[:,3])
    ],axis=1)
    losses = jnp.mean(losses,axis=1) # N
    return losses


def ott_grouped_and_l2_loss(x, y, key, where=None, aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """Sum of ``ott_grouped_loss`` and the grouped l2 loss (without channel importance)."""
    return ott_grouped_loss(x, y, key, where, aux) + l2_colony_grouped(x, y, key, where)
