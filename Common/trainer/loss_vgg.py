import jax.numpy as jnp
import jax
import equinox as eqx
from typing import Optional, Any
from lpips_j.lpips import LPIPS
import ott
import jax.tree_util as jtu

import flax.linen as nn
from jax.scipy.ndimage import map_coordinates
from einops import rearrange,reduce,einsum,repeat,reduce
import jax.random as jr
from pyparsing import Optional
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch,pad_to_multiple_of_3_channels

VGG_DTYPE = jnp.bfloat16
LOSS_DTYPE = jnp.float32
def to_vgg_dtype(x):
    return x.astype(VGG_DTYPE)


def to_loss_dtype(x):
    return x.astype(LOSS_DTYPE)


# def tree_stop_bf16(x):
#     return jax.tree.map(lambda z: jax.lax.stop_gradient(z.astype(VGG_DTYPE)), x)


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
    # print("Spatial average input shape: ",x.shape,flush=True)
    # Mean over W, H
    x = x.astype(LOSS_DTYPE)
    x = jnp.mean(x, axis=[1, 2], keepdims=keepdims)
    # print("Spatial average output shape: ",x.shape,flush=True)
    return x

class LPIPS_WITH_FEATURES(LPIPS):
    def features(self, x):
        # x = ((x + 1.0) / 2.0).astype(VGG_DTYPE)
        x = x * 2.0 - 1.0
        x = x.astype(VGG_DTYPE)
        return self.vgg(x)


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


def oti_loss(X,Y,aux):
    """
        Computes linear OT loss between two single channel images X and Y
        Parameters
            X: np.ndarray of shape [H W], source image
            Y: np.ndarray of shape [H W], target image
        Returns
            OT loss: float

    """
    metric = {
        "l2": [ott.geometry.costs.Euclidean()]*aux["dims"],
        "l2_squared": [ott.geometry.costs.SqEuclidean()]*aux["dims"],
        "l1": [ott.geometry.costs.PNormP(1)]*aux["dims"],
    }
    geom = ott.geometry.grid.Grid(grid_size=X.shape,epsilon=aux['epsilon'],cost_fns=metric[aux["internal_loss_func"]])
    if aux["normalize"]:
        X = X/ (X.sum()+1e-8)
        Y = Y/ (Y.sum()+1e-8)
    problem = ott.problems.linear.linear_problem.LinearProblem(geom,a=X.ravel(),b=Y.ravel(),tau_a=aux["tau"],tau_b=aux["tau"])
    
    solver = ott.solvers.linear.sinkhorn.Sinkhorn(min_iterations=64,max_iterations=64)
    out = solver(problem)
    
    return out.reg_ot_cost


class LPIPS_EMD_SP(LPIPS_WITH_FEATURES):
    @nn.compact
    def __call__(self, x, t, key, aux): # pyright: ignore[reportIncompatibleMethodOverride]
        # x = self.vgg((x + 1) / 2)
        x = self.features(x)
        if aux.get("target_feats", None) is not None:
            t = aux["target_feats"]
        else:
            t = self.features(t)
        # t = self.vgg((t + 1) / 2)
        # key = self.make_rng('projection')
        feats_x, feats_t, diffs = {}, {}, {}
        v_emd_loss = jax.vmap(oti_loss, in_axes=(0,0,None))
        vv_emd_loss = jax.vmap(v_emd_loss, in_axes=(0,0,None))
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])  # B fW fH fC - normalised to be probability distribution over fC
            feats_x_reshaped = rearrange(feats_x[i],"b fw fh fc -> b fc fw fh")
            feats_t_reshaped = rearrange(feats_t[i],"b fw fh fc -> b fc fw fh")
            _aux = {
                "epsilon":aux["epsilon"],
                "internal_loss_func":aux["internal_loss_func"],
                "tau":aux["tau"],
                "normalize":aux["normalize"],
                "dims":2
            }
            emd_loss = vv_emd_loss(feats_x_reshaped, feats_t_reshaped, _aux) # B fC
            emd_loss = reduce(emd_loss,"b fc -> b","mean") # B
            diffs[i] = rearrange(emd_loss,"b -> b 1 1 1")


        res = [diffs[i] for i in range(len(self.feature_names))]
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val.astype(LOSS_DTYPE)
    

class LPIPS_EMD_FULL(LPIPS_WITH_FEATURES): 
    @nn.compact
    def __call__(self, x, t, key, aux): # pyright: ignore[reportIncompatibleMethodOverride]
        # x = self.vgg((x + 1) / 2)
        x = self.features(x)
        if aux.get("target_feats", None) is not None:
            t = aux["target_feats"]
        else:
            t = self.features(t)
        # t = self.vgg((t + 1) / 2)
        # key = self.make_rng('projection')
        feats_x, feats_t, diffs = {}, {}, {}
        v_emd_loss = jax.vmap(oti_loss, in_axes=(0,0,None))
        # vv_emd_loss = jax.vmap(v_emd_loss, in_axes=(0,0,None))
        for i, f in enumerate(self.feature_names):
            feats_x[i], feats_t[i] = normalize_tensor(x[f]), normalize_tensor(t[f])  # B fW fH fC - normalised to be probability distribution over fC
            feats_x_reshaped = rearrange(feats_x[i],"b fw fh fc -> b fc fw fh")
            feats_t_reshaped = rearrange(feats_t[i],"b fw fh fc -> b fc fw fh")
            _aux = {
                "epsilon":aux["epsilon"],
                "internal_loss_func":aux["internal_loss_func"],
                "tau":aux["tau"],
                "normalize":aux["normalize"],
                "dims":3
            }
            emd_loss = v_emd_loss(feats_x_reshaped, feats_t_reshaped, _aux) # B

            # emd_loss = reduce(emd_loss,"b fc -> b","mean") # B
            diffs[i] = rearrange(emd_loss,"b -> b 1 1 1")


        res = [diffs[i] for i in range(len(self.feature_names))]
        
        val = res[0]
        for i in range(1, len(res)):
            val += res[i]
        return val.astype(LOSS_DTYPE)
    

lpips_ot_ch = LPIPS_OT_CH()
lpips_ot_sp = LPIPS_OT_SP()
lpips_emd_sp = LPIPS_EMD_SP()
lpips_emd_full = LPIPS_EMD_FULL()
lpips_l2 = LPIPS_L2()

lpips_variants = {
    "otch": lpips_ot_ch,
    "otsp": lpips_ot_sp,
    "l2": lpips_l2,
    "emdsp": lpips_emd_sp,
    "emdfull": lpips_emd_full,
}



# @eqx.filter_jit
# def vgg(x,y, key,where=None,aux={"vgg_metric":"l2"}):
#     """
#     NOTE THAT CHANNELS IS TRUNCATED TO 3
#     NOTE WHERE HAS NO EFFECT HERE

#     Parameters
#     ----------
#     x : float32 [N,CHANNELS,WIDTH,HEIGHT]
#         predictions
#     y : float32 [N,CHANNELS,WIDTH,HEIGHT]
#         true data
#     key : jax.random.PRNGKey
#         Jax random number key. 

#     Returns
#     -------
#     loss : float32 [N]

#     """
#     x = rearrange(x,"n c x y->n x y c")[...,:3]
#     y = rearrange(y,"n c x y->n x y c",)[...,:3]
#     lpips_model = lpips_variants[aux["vgg_metric"]]
#     # L-pips expects inputs in the range [-1,1], but we almost always use data in the range [0,1]
#     x = x*2-1
#     y = y*2-1
        
#     params = lpips_model.init(key, x, y, key, aux=aux)
#     loss = lpips_model.apply(params, x, y, key, aux=aux)

#     return loss
	



# ---------------------------------------------------------------------
# Precompute target features
# - to save compute time, sometimes we can just compute the true data
#   target features once at the start, effectively halving VGG calls.
# ---------------------------------------------------------------------

@eqx.filter_jit
def precompute_vgg_hyperspectral_target(y, key, where=None, aux={"vgg_metric": "l2"}):
    """
    Parameters:
        y : Pytree[Batches] of float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
            Jax random number key.
        where : Pytree[Batches] of boolean array [N,CHANNELS,(),()]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
    Returns:
        {
            "vgg_params": params,
            "target_feats": target_feats,
        }

    target_feats has shape/pytree structure:
        [num_channel_groups, ...VGG feature pytree...]
    """
    def pre_process_one_batch(y):
        # y = y * 2 - 1
        # if where is not None:
            # y = y * where.astype(y.dtype)
        y = pad_to_multiple_of_3_channels(y)
        y = rearrange(y, "n (c vc) x y -> c n x y vc", vc=3)
        y = y.astype(VGG_DTYPE)
        return y
    def apply_to_one_batch(y):
        def one_group(yg):
            feats = lpips_model.apply(params, yg, method=lpips_model.features)
            return feats

        target_feats = jax.vmap(one_group)(y)
        return target_feats
    
    y = jtu.tree_map(pre_process_one_batch, y)
    lpips_model = lpips_variants[aux["vgg_metric"]]
    init_key, call_key = jr.split(key, 2)
    params = lpips_model.init(init_key, y[0][0], y[0][0], call_key, aux=aux)
    params = cast_params_bf16(params)
    target_feats = jtu.tree_map(apply_to_one_batch, y)


    return {
        "vgg_params": params,
        "target_feats": target_feats,
    }


@eqx.filter_jit
def precompute_vgg_hyperspectral_colony_target(y, key, where=None, aux={"vgg_metric": "l2"}):
    """
    Parameters:
        y : Pytree[Batches] of float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
            Jax random number key.
        where : Pytree[Batches] of boolean array [N,CHANNELS,(),()]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
    Returns:
        {
            "vgg_params": params,
            "target_feats": target_feats,
        }
    """
    def pre_process_one_batch(y):
        # y = y * 2 - 1

        # if where is not None:
            # where_y = duplicate_x_channels_9ch(where)
            # y = y * where_y.astype(y.dtype)

        y = split_and_pad_by_experiment_groups_12ch(y)
        y = rearrange(y, "n (c vc) x y -> c n x y vc", vc=3)
        y = y.astype(VGG_DTYPE)

        return y
    def apply_to_one_batch(y):
        def one_group(yg):
            feats = lpips_model.apply(params, yg, method=lpips_model.features)
            return feats

        target_feats = jax.vmap(one_group)(y)
        return target_feats



    y = jtu.tree_map(pre_process_one_batch, y)

    lpips_model = lpips_variants[aux["vgg_metric"]]
    init_key, call_key = jr.split(key, 2)
    params = lpips_model.init(init_key, y[0][0], y[0][0], call_key, aux=aux)
    params = cast_params_bf16(params)

    target_feats = jtu.tree_map(apply_to_one_batch, y)

    return {
        "vgg_params": params,
        "target_feats": target_feats,
    }



def _random_crop_to_vgg_input(x,key):
    def crop_image(im,key): # Takes [W H C] and returns [224 224 C]
        w,h,_ = im.shape
        keys = jr.split(key,2)
        crop_size = 224
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


def _permute_matching_channel_groups(x, y, key, group_sizes):
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

# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def vgg_hyperspectral(x, y, key, where=None, aux={"vgg_metric": "l2"}, cache=None):
    """
        Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions
        y : float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N,CHANNELS,(),()]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        Returns
        -------
        loss : float32 [N]
            loss reduced over channel and spatial axes

        Optional aux entries:
            aux["vgg_params"]
            aux["target_feats"]

        If provided, skips VGG target forward pass.
    """

    # x = x * 2 - 1
    # y = y * 2 - 1

    if where is not None:
        x = x * where.astype(x.dtype)
        y = y * where.astype(y.dtype)

    x = pad_to_multiple_of_3_channels(x)
    y = pad_to_multiple_of_3_channels(y)

    use_target_cache = cache is not None and not aux.get("random_crop", False)
    if aux.get("random_channel_shuffle", False) and not use_target_cache:
        x, y = _permute_matching_channels(x, y, jr.fold_in(key, 0))

    x = rearrange(x, "n (c vc) x y -> c n x y vc", vc=3)
    y = rearrange(y, "n (c vc) x y -> c n x y vc", vc=3)

    x = x.astype(VGG_DTYPE)
    y = y.astype(VGG_DTYPE)

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
        x = _random_crop_to_vgg_input(x, key)
        y = _random_crop_to_vgg_input(y, key)
        cache = None # Can't use cached features if we are randomly cropping.
        

    if cache is None:
        losses = jax.vmap(
            lpips_model.apply,
            in_axes=(None, 0, 0, 0, None),
        )(params, x, y, keys, aux)
    else:
        # target_feats = aux["target_feats"]
        target_feats = cache

        def apply_one(xi, yi, ki, ti):
            aux_i = {**aux, "target_feats": ti}
            return lpips_model.apply(params, xi, yi, ki, aux=aux_i)

        losses = jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            x,
            y,
            keys,
            target_feats,
        )

    loss = reduce(losses.astype(LOSS_DTYPE), "c n () () () -> n", "mean")
    return loss



def vgg_hyperspectral_colony(x, y, key, where=None, aux={"vgg_metric": "l2"}, cache=None):
    """
    
        Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions
        y : float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
            true data
        key : jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N,CHANNELS,(),()]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        Returns
        -------
        loss : float32 [N]
            loss reduced over channel and spatial axes

        Optional aux entries:
            aux["vgg_params"]
            aux["target_feats"]

        If provided, skips VGG target forward pass.
    """

    # x = x * 2 - 1
    # y = y * 2 - 1

    if where is not None:
        x = x * where.astype(x.dtype)
        where_y = duplicate_x_channels_9ch(where)
        y = y * where_y.astype(y.dtype)

    x = duplicate_x_channels_9ch(x)
    x = split_and_pad_by_experiment_groups_12ch(x)
    y = split_and_pad_by_experiment_groups_12ch(y)

    use_target_cache = cache is not None and not aux.get("random_crop", False)
    random_channel_shuffle = aux.get("random_channel_shuffle", False) and not use_target_cache
    if random_channel_shuffle:
        x, y = _permute_matching_channel_groups(
            x,
            y,
            jr.fold_in(key, 1),
            group_sizes=(6, 6, 3, 3),
        )

    x = rearrange(x, "n (c vc) x y -> c n x y vc", vc=3)
    y = rearrange(y, "n (c vc) x y -> c n x y vc", vc=3)

    x = x.astype(VGG_DTYPE)
    y = y.astype(VGG_DTYPE)

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
        x = _random_crop_to_vgg_input(x, key)
        y = _random_crop_to_vgg_input(y, key)
        cache = None # Can't use cached features if we are randomly cropping.
        
    if cache is None:
        losses = jax.vmap(
            lpips_model.apply,
            in_axes=(None, 0, 0, 0, None),
        )(params, x, y, keys, aux)
    else:
        target_feats = cache

        def apply_one(xi, yi, ki, ti):
            aux_i = {**aux, "target_feats": ti}
            return lpips_model.apply(params, xi, yi, ki, aux=aux_i)

        losses = jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            x,
            y,
            keys,
            target_feats,
        )

    if random_channel_shuffle:
        loss_weighting = jnp.array(
            [0.75, 0.75, 0.75, 0.75, 1.0, 1.0],
            dtype=LOSS_DTYPE,
        )
    else:
        loss_weighting = jnp.array(
            [0.5, 1.0, 0.5, 1.0, 1.0, 1.0],
            dtype=LOSS_DTYPE,
        )

    losses = einsum(
        losses.astype(LOSS_DTYPE),
        loss_weighting,
        "c n i j k, c -> c n i j k",
    )

    loss = reduce(losses, "c n () () () -> n", "mean")
    return loss





# def vgg_hyperspectral_colony(x,y,key,where=None,aux={"vgg_metric":"l2"}):
#     """

#         Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
#         Parameters
#         ----------
#         x : float32 [N,CHANNELS,WIDTH,HEIGHT]
#             predictions
#         y : float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
#             true data
#         key : jax.random.PRNGKey
#             Jax random number key.
#         where : boolean array [N,CHANNELS,(),()]
#             Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
#         Returns
#         -------
#         loss : float32 [N]
#             loss reduced over channel and spatial axes
#     """

#     # Scale to [-1,1] for lpips
    
#     x = x*2-1
#     y = y*2-1

#     # x has 8 channels but y has 11. Some specified x channels need to be repeated to match the channels in y
#     # Apply where mask

#     if where is not None:
#         x = x*where.astype(x.dtype)
#         where_y = duplicate_x_channels_9ch(where)
#         y = y*where_y.astype(y.dtype)



#     x = duplicate_x_channels_9ch(x)
#     x = split_and_pad_by_experiment_groups_12ch(x)
#     y = split_and_pad_by_experiment_groups_12ch(y)		
#     x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
#     y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)

#     lpips_model = lpips_variants[aux["vgg_metric"]]
#     init_key, call_key = jr.split(key, 2)
#     params = lpips_model.init(init_key, x[0], y[0], call_key, aux=aux)
    
#     keys = jr.split(key, x.shape[0])
    
#     losses = jax.vmap(lpips_model.apply, in_axes=(None,0,0,0,None))(params, x, y, keys, aux) # C N () () ()
#     # print("VGG losses shape: ",losses.shape,flush=True)
#     # Weight different loss channels - some are duplicate channels from specifying colonies, others are dummy channels introduced by vgg groupings
#     loss_weighting = jnp.array([0.5,1.0,0.5,1.0,1.0,1.0]) # Should there be an extra 1.0 here?
#     # loss_weighting = jnp.array([0.5,1.0,0.5,1.0,1.0]) # Should there be an extra 1.0 here?
#     print("Loss weighting shape: ",loss_weighting.shape,flush=True)
#     print("Losses shape before weighting: ",losses.shape,flush=True)

#     losses = einsum(losses,loss_weighting,"c n i j k , c -> c n i j k")
#     loss = reduce(losses,"c n () () () -> n","mean")
#     return loss




# def vgg_hyperspectral(x,y,key,where=None,aux={"vgg_metric":"l2","cache_target_features":True}):
#     """

#         Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
#         Parameters
#         ----------
#         x : float32 [N,CHANNELS,WIDTH,HEIGHT]
#             predictions
#         y : float32 [N,CHANNELS_DUPLICATED,WIDTH,HEIGHT]
#             true data
#         key : jax.random.PRNGKey
#             Jax random number key.
#         where : boolean array [N,CHANNELS,(),()]
#             Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
#         Returns
#         -------
#         loss : float32 [N]
#             loss reduced over channel and spatial axes
#     """

#     # Scale to [-1,1] for lpips
#     x = x*2-1
#     if where is not None:
#         x = x*where.astype(x.dtype)
#     x = pad_to_multiple_of_3_channels(x)
#     x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
    
    
#     if aux.get("cache_target_features", False):
#         y = y*2-1
#         if where is not None:
#             y = y*where.astype(y.dtype)
#         y = pad_to_multiple_of_3_channels(y)		
#         y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)

#     # Apply where mask

#     lpips_model = lpips_variants[aux["vgg_metric"]]

#     init_key, call_key = jr.split(key, 2)
#     params = lpips_model.init(init_key, x[0], y[0], call_key, aux=aux)
#     keys = jr.split(key, x.shape[0])
#     losses = jax.vmap(lpips_model.apply, in_axes=(None,0,0,0,None))(params, x, y, keys, aux) # C N () () ()
#     loss = reduce(losses,"c n () () () -> n","mean")
#     return loss
