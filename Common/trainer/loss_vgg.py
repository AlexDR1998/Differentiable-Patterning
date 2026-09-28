"""VGG (LPIPS) feature losses for images with any number of channels.

The channels are split into blocks of 3, each block is passed through a
pretrained VGG network as if it were an RGB image, and the feature
distances are averaged. The micropattern version, which keeps channels from
the same experiment together, is in ``loss_micropattern.py``.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
from lpips_j.lpips import LPIPS
import jax.tree_util as jtu

import flax.linen as nn
from einops import rearrange,reduce,einsum
import jax.random as jr
from Common.trainer.experiment_channel_grouping import pad_to_multiple_of_3_channels

VGG_DTYPE = jnp.bfloat16
LOSS_DTYPE = jnp.float32
def to_vgg_dtype(x):
    return x.astype(VGG_DTYPE)


def to_loss_dtype(x):
    return x.astype(LOSS_DTYPE)


def cast_params_bf16(params):
    def f(z):
        if jnp.issubdtype(z.dtype, jnp.floating):
            return z.astype(VGG_DTYPE)
        return z
    return jax.tree.map(f, params)

def normalize_tensor(x, eps=1e-10):
    # Use `-1` because we are channel-last
    x = x.astype(LOSS_DTYPE)
    norm_factor = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
    return x / (norm_factor + eps)

def spatial_average(x, keepdims=True):
    # Mean over W, H
    x = x.astype(LOSS_DTYPE)
    x = jnp.mean(x, axis=[1, 2], keepdims=keepdims)
    return x

class LPIPS_WITH_FEATURES(LPIPS):
    def features(self, x):
        # Expects x in range [0,1]
        # x = ((x + 1.0) / 2.0).astype(VGG_DTYPE)
        # x = x * 2.0 - 1.0
        x = x.astype(VGG_DTYPE)
        return self.vgg(x) # expects x in range [0,1]


class LPIPS_L2(LPIPS_WITH_FEATURES):
    @nn.compact
    def __call__(self, x, t, key, aux): # pyright: ignore[reportIncompatibleMethodOverride]
        # x = self.vgg((x + 1) / 2)
        # t = self.vgg((t + 1) / 2)
        x = self.features(x)
        if aux.get("target_feats", None) is not None:
            t = aux["target_feats"]
        else:
            t = self.features(t)
            
        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])  # B fW fH fC
            
            # print(f"Feature map {f} shape: ",feats_x[i].shape,flush=True)

            diffs[i] = ((feats_x[i] - feats_t[i]) ** 2 ).astype(LOSS_DTYPE)      # B fW fH fC
            # print(f"Diffs {i} shape: ",diffs[i].shape,flush=True)

        # We should maybe vectorize this better
        # self.lins does B fW fH fC -> B fW fH 1
        # spatial_average does B fW fH 1 -> B 1 1 1
        res = [spatial_average(self.lins[i](diffs[i]), keepdims=True) for i in range(len(self.feature_names))] 
        # print("Res shapes: ",[r.shape for r in res],flush=True)
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val.astype(LOSS_DTYPE)


class LPIPS_OT_CH(LPIPS_WITH_FEATURES):
    @nn.compact
    def __call__(self, x, t, key, aux):# pyright: ignore[reportIncompatibleMethodOverride]
        # x = self.vgg((x + 1) / 2)
        x = self.features(x)
        if aux.get("target_feats", None) is not None:
            t = aux["target_feats"]
        else:
            t = self.features(t)
        # t = self.vgg((t + 1) / 2)
        # key = self.make_rng('projection')
        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])  # B fW fH fC
            W,H,C = feats_x[i].shape[1:]
            proj = jr.uniform(key,shape=(C,aux["samples"]),dtype=LOSS_DTYPE)
            proj = proj / jnp.linalg.norm(proj, axis=0, keepdims=True) # C samples

            x_proj = einsum(feats_x[i],proj,"b w h c , c s -> b w h s") # B fW fH samples
            t_proj = einsum(feats_t[i],proj,"b w h c , c s -> b w h s") # B fW fH samples

            x_proj = rearrange(x_proj,"b w h s -> b s (w h)")
            t_proj = rearrange(t_proj,"b w h s -> b s (w h)")

            x_proj = jnp.sort(x_proj, axis=-1)
            t_proj = jnp.sort(t_proj, axis=-1)

            # print(f"Projected and sorted feature map {f} shape: ",feats_x[i].shape,flush=True)

            _d = ((x_proj - t_proj) ** 2).astype(LOSS_DTYPE)       # B samples (fW fH)
            _d = reduce(_d,"b s wh -> b ","mean") # B

            diffs[i] = rearrange(_d,"b -> b 1 1 1")       # B 1 1 1

            # print(f"Diffs {i} shape: ",diffs[i].shape,flush=True)

        res = [diffs[i] for i in range(len(self.feature_names))]
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val.astype(LOSS_DTYPE)



class LPIPS_OT_SP(LPIPS_WITH_FEATURES):
    @nn.compact
    def __call__(self, x, t, key, aux):# pyright: ignore[reportIncompatibleMethodOverride]
        # x = self.vgg((x + 1) / 2)
        x = self.features(x)
        if aux.get("target_feats", None) is not None:
            t = aux["target_feats"]
        else:
            t = self.features(t)
        # t = self.vgg((t + 1) / 2)
        # key = self.make_rng('projection')
        feats_x, feats_t, diffs = {}, {}, {}
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])  # B fW fH fC
            W,H,C = feats_x[i].shape[1:]
            proj = jr.uniform(key,shape=(W,H,aux["samples"]),dtype=LOSS_DTYPE)
            proj = proj / jnp.linalg.norm(proj, axis=(0,1), keepdims=True) # C samples

            x_proj = einsum(feats_x[i],proj,"b w h c , w h s -> b s c") # B samples C
            t_proj = einsum(feats_t[i],proj,"b w h c , w h s -> b s c") # B samples C

            # x_proj = rearrange(x_proj,"b c s -> b s c")
            # t_proj = rearrange(t_proj,"b c s -> b s c")

            x_proj = jnp.sort(x_proj, axis=-1)
            t_proj = jnp.sort(t_proj, axis=-1)

            # print(f"Projected and sorted feature map {f} shape: ",feats_x[i].shape,flush=True)

            _d = ((x_proj - t_proj) ** 2).astype(LOSS_DTYPE)       # B samples C
            _d = reduce(_d,"b s c -> b ","mean") # B

            diffs[i] = rearrange(_d,"b -> b 1 1 1")       # B 1 1 1

            # print(f"Diffs {i} shape: ",diffs[i].shape,flush=True)

        res = [diffs[i] for i in range(len(self.feature_names))]
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val.astype(LOSS_DTYPE)


lpips_variants = {
    "otch": LPIPS_OT_CH(),
    "otsp": LPIPS_OT_SP(),
    "l2": LPIPS_L2(),
}


def to_vgg_groups(x):
    """[N, 3*G, W, H] -> [G, N, W, H, 3]: one RGB-like image per block of 3 channels."""
    x = rearrange(x, "n (c vc) x y -> c n x y vc", vc=3)
    return x.astype(VGG_DTYPE)


# ---------------------------------------------------------------------
# Precompute target features
# - to save compute time, sometimes we can just compute the true data
#   target features once at the start, effectively halving VGG calls.
# ---------------------------------------------------------------------

def precompute_vgg_target(y, key, aux, arrange_channels):
    """
    Initialise the VGG parameters and compute the target features once.

    Parameters:
        y : Pytree[Batches] of float32 [N,CHANNELS,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
        aux : dict with "vgg_metric" (and "samples" for the OT variants)
        arrange_channels : function [N,C,W,H] -> [N,3*G,W,H] that pads or
            groups the channels into blocks of 3
    Returns:
        {
            "vgg_params": params,
            "target_feats": Pytree[Batches] of [G, ...VGG feature pytree...],
        }
    """
    y = jtu.tree_map(lambda batch: to_vgg_groups(arrange_channels(batch)), y)
    lpips_model = lpips_variants[aux["vgg_metric"]]
    init_key, call_key = jr.split(key, 2)
    params = lpips_model.init(init_key, y[0][0], y[0][0], call_key, aux=aux)
    params = cast_params_bf16(params)

    def features(groups):
        return jax.vmap(
            lambda group: lpips_model.apply(params, group, method=lpips_model.features)
        )(groups)

    return {
        "vgg_params": params,
        "target_feats": jtu.tree_map(features, y),
    }


@eqx.filter_jit
def precompute_vgg_hyperspectral_target(y, key, where=None, aux={"vgg_metric": "l2"}):
    return precompute_vgg_target(y, key, aux, pad_to_multiple_of_3_channels)


def random_crop_to_vgg_input(x,key):
    def crop_image(im,key): # Takes [W H C] and returns [min(224,W,H) min(224,W,H) C]
        w,h,_ = im.shape
        keys = jr.split(key,2)
        crop_size = min(224, w, h)
        max_x = w - crop_size
        max_y = h - crop_size
        x_start = jr.randint(keys[0], (), 0, max_x + 1)
        y_start = jr.randint(keys[1], (), 0, max_y + 1)
        cropped = jax.lax.dynamic_slice(im, (x_start,y_start,0), (crop_size, crop_size, im.shape[2]))
        return cropped
    keys = jr.split(key,(x.shape[0],x.shape[1])) # one key per N and channel group
    crop_image_vmap = jax.vmap(jax.vmap(crop_image, in_axes=(0,0)), in_axes=(0,0))
    x = crop_image_vmap(x, keys)
    return x


def _permute_matching_channels(x, y, key):
    """
    Randomly permute matched channel order before making 3-channel VGG inputs.

    x, y: float array [N, C, H, W]
    key: jax.random.PRNGKey
    returns: x, y with shape [N, C, H, W]
    """
    perm = jr.permutation(key, x.shape[1])
    x = jnp.take(x, perm, axis=1)
    y = jnp.take(y, perm, axis=1)
    return x, y


def permute_matching_channel_groups(x, y, key, group_sizes):
    """
    Randomly permute matched channel order within fixed experiment groups.

    x, y: float array [N, C, H, W]
    key: jax.random.PRNGKey
    group_sizes: sequence of int summing to C
    returns: x, y with shape [N, C, H, W]
    """
    keys = jr.split(key, len(group_sizes))
    xs = []
    ys = []
    start = 0
    for group_size, group_key in zip(group_sizes, keys):
        end = start + group_size
        perm = jr.permutation(group_key, group_size)
        xs.append(jnp.take(x[:, start:end], perm, axis=1))
        ys.append(jnp.take(y[:, start:end], perm, axis=1))
        start = end
    return jnp.concatenate(xs, axis=1), jnp.concatenate(ys, axis=1)


def permute_grouped_channels(x, key, group_sizes, axis=0):
    """Apply the same deterministic within-group permutation used for grouped VGG inputs."""
    keys = jr.split(key, len(group_sizes))
    outputs = []
    start = 0
    for group_size, group_key in zip(group_sizes, keys):
        end = start + group_size
        perm = jr.permutation(group_key, group_size)
        outputs.append(jnp.take(x, start + perm, axis=axis))
        start = end
    return jnp.concatenate(outputs, axis=axis)

# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def vgg_group_losses(x, y, key, aux, cache=None):
    """
    VGG loss for each block of 3 channels.

    x, y : float32 [N, 3*G, W, H], channels already padded or grouped
    cache : precomputed target features, or None
    Returns float32 [G, N, 1, 1, 1]
    """
    x = to_vgg_groups(x)
    y = to_vgg_groups(y)

    lpips_model = lpips_variants[aux["vgg_metric"]]

    if aux.get("vgg_params", None) is None:
        init_key, call_key = jr.split(key, 2)
        params = lpips_model.init(init_key, x[0], y[0], call_key, aux=aux)
        params = cast_params_bf16(params)
    else:
        params = aux["vgg_params"]

    keys = jr.split(key, x.shape[0])
    if aux.get("random_crop", False):
        # For each N and channel group, select a random 224*224 sized crop,
        # as this is the input size that VGG was trained on. For larger resolutions, this 
        # should speed up training.
        x = random_crop_to_vgg_input(x, key)
        y = random_crop_to_vgg_input(y, key)
        cache = None # Can't use cached features if we are randomly cropping.

    if cache is None:
        return jax.vmap(
            lpips_model.apply,
            in_axes=(None, 0, 0, 0, None),
        )(params, x, y, keys, aux)

    def apply_one(xi, yi, ki, ti):
        aux_i = {**aux, "target_feats": ti}
        return lpips_model.apply(params, xi, yi, ki, aux=aux_i)

    return jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(x, y, keys, cache)


def vgg_hyperspectral(x, y, key, where=None, aux={"vgg_metric": "l2"}, cache=None):
    """
        Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions
        y : float32 [N,CHANNELS,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N,CHANNELS,(),()]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        aux : dict
            "vgg_metric", and optionally "vgg_params", "samples", "random_crop", "random_channel_shuffle"
        cache : precomputed target features (see precompute_vgg_hyperspectral_target), or None
        Returns
        -------
        loss : float32 [N]
            loss reduced over channel and spatial axes
    """
    if where is not None:
        x = x * where.astype(x.dtype)
        y = y * where.astype(y.dtype)

    x = pad_to_multiple_of_3_channels(x)
    y = pad_to_multiple_of_3_channels(y)

    use_target_cache = cache is not None and not aux.get("random_crop", False)
    if aux.get("random_channel_shuffle", False) and not use_target_cache:
        x, y = _permute_matching_channels(x, y, jr.fold_in(key, 0))

    losses = vgg_group_losses(x, y, key, aux, cache)
    return reduce(losses.astype(LOSS_DTYPE), "c n () () () -> n", "mean")
