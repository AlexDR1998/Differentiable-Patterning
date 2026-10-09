"""Optimal transport texture losses.

Random patches are sampled from each image at several scales, and the
entropy-regularised optimal transport cost between the patch sets of the
prediction and the target is the loss. The micropattern version, which keeps
channels from the same experiment together, is in ``loss_micropattern.py``.
"""

import jax.numpy as np
import jax
import numpy as onp
import equinox as eqx
import ott
from einops import rearrange,reduce,repeat
import jax.random as jr


def _make_gaussian_kernel(sigma, nstds):
    """Normalised 2D Gaussian kernel as a numpy array (call outside jit)."""
    import math
    sigma_x = float(sigma) * 0.5
    extent = float(nstds) * sigma_x
    kmax = math.ceil(max(1.0, extent))
    
    coords = onp.linspace(-kmax, kmax, 2 * kmax + 1, dtype=onp.float32)
    y, x = onp.meshgrid(coords, coords, indexing='ij')
    
    gb = onp.exp(-0.5 * (x**2 / (sigma_x**2) + y**2 / (sigma_x**2)))
    
    return (gb / onp.sum(gb)).astype(onp.float32)

# Pre-compute commonly used kernels as constants
_GAUSSIAN_CACHE = {
    (1, 1): _make_gaussian_kernel(1, 1),
    (2, 2): _make_gaussian_kernel(2, 2),
    (3, 3): _make_gaussian_kernel(3, 3),
    (5, 5): _make_gaussian_kernel(5, 5),
}

def _gaussian(sigma, nstds):
    """Normalised 2D Gaussian kernel, taken from ``_GAUSSIAN_CACHE`` when possible so it works under jit.

    Parameters
    ----------
    sigma : float
        width scale (the standard deviation is ``sigma / 2``)
    nstds : float
        kernel half-width in standard deviations
    """
    key = (sigma, nstds)
    if key in _GAUSSIAN_CACHE:
        return _GAUSSIAN_CACHE[key]
    else:
        # Fallback: compute it (will fail if called inside JIT with non-static args)
        return _make_gaussian_kernel(sigma, nstds)

def _sharpen(X,k):
    """Unsharp-mask images ``[N C H W]``: ``X + 2 (X - blur)``, with a Gaussian blur of scale ``k``."""
    C = X.shape[1]
    kernel = _gaussian(k,k)
    # Kernel shape for depthwise conv: [kh, kw, in_features_per_group=1, out_features=C]
    kernel = repeat(kernel,"kh kw -> kh kw () C", C=C)
    # Input shape: [N, H, W, C]
    X_reshaped = rearrange(X,"N C H W -> N H W C")
    
    # Layouts of input, kernel and output
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    blur = jax.lax.conv_general_dilated(
        X_reshaped,
        kernel,
        window_strides=(1,1),
        padding='SAME',
        dimension_numbers=dimension_numbers,
        feature_group_count=C,  # Each channel processed independently
    )
    # Rearrange back to [N C H W]
    
    blur = rearrange(blur, "N H W C -> N C H W")
    return X + 2*(X - blur)


def _sample_random_patches(X,S,K,key):
    """Sample ``S`` random KxK patches from an image ``[H W]``; returns ``[S K*K]``."""
    H,W = X.shape
    keys = jr.split(key,2)
    ys = jr.randint(keys[0],(S,),0,H)
    xs = jr.randint(keys[1],(S,),0,W)
    X_pad = np.pad(X,((0,K),(0,K)),'edge')
    def select_patch(X,ix,iy,K):
        return jax.lax.dynamic_slice(X, (iy, ix), (K, K))
    vpatch = jax.vmap(select_patch,(None,0,0,None))
    patches = vpatch(X_pad,xs,ys,K)
    patches = rearrange(patches,"S x y -> S (x y)")
    return patches
    
def _downsample_and_patch(X,S,K,D,key):
    """Sample ``S`` random KxK patches from an image ``[H W]`` and from ``D`` successive 2x downsamplings of it.

    Returns ``[D+1, S, K*K]``.
    """
    patches = [_sample_random_patches(X,S,K,key=key)]
    Xd = X
    keys = jr.split(key,D)
    for d in range(D):
        Xd = np.pad(Xd,((0,Xd.shape[0]%2),(0,Xd.shape[1]%2)),'reflect')
        Xd = reduce(Xd,"(h 2) (w 2) -> h w", 'mean')
        pd = _sample_random_patches(Xd,S,K,key=keys[d])
        patches.append(pd)
    patches = np.stack(patches,axis=0)
    return patches


def _ott_patch_loss(PX,PY,aux):
    """Entropy-regularised OT cost between two patch sets ``[S K*K]``; ``aux`` holds "epsilon" and "internal_loss_func"."""
    
    metric = {
        "l2": ott.geometry.costs.Euclidean(),
        "l2_squared": ott.geometry.costs.SqEuclidean(),
        "l1": ott.geometry.costs.PNormP(1),
        "cos": ott.geometry.costs.Cosine(),
        "arccos": ott.geometry.costs.Arccos(n=2),
    }
    geom = ott.geometry.pointcloud.PointCloud(PX, PY, epsilon=aux["epsilon"], cost_fn=metric[aux["internal_loss_func"]])
    ot = ott.solvers.linear.solve(geom,min_iterations=64,max_iterations=64)
    # print(
    #     " Sinkhorn has converged: ",
    #     ot.converged,
    #     "\n",
    #     "Error upon last iteration: ",
    #     ot.errors[(ot.errors > -1)][-1],
    #     "\n",
    #     "Sinkhorn required ",
    #     np.sum(ot.errors > -1),
    #     " iterations to converge. \n",
    #     "Entropy regularized OT cost: ",
    #     ot.ent_reg_cost,
    #     "\n",
    #     "OT cost (without entropy): ",
    #     np.sum(ot.matrix * ot.geom.cost_matrix),
    #     "\n",
    #     # "Time taken (s): ",
    #     # t2 - t1,
    # )
    # ot_cost = ot.matrix 
    # return np.sum(ot.matrix * ot.geom.cost_matrix)
    return ot.ent_reg_cost


@eqx.filter_jit
def ott_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """OT loss between the patches of each channel of x and y, at several scales.

    Parameters
    ----------
    x : float32 [N C H W]
        predictions
    y : float32 [N C H W]
        true data
    key : jax.random.PRNGKey
    where : boolean array [N C 1 1]
        channels (and timesteps) to include
    aux : dict
        S : number of patches per scale
        K : patch size (KxK)
        D : number of 2x downsampling steps
        sharpen : sharpen images before sampling patches
        epsilon : entropic regularisation
        internal_loss_func : patch cost, one of "l2", "l2_squared", "l1", "cos", "arccos"

    Returns
    -------
    loss : float32 [N]
    """
    N = x.shape[0]
    C = x.shape[1]
    S = aux["S"]
    K = aux["K"]
    D = aux["D"]
    # ep = aux["epsilon"]
    ott_kwargs = {
        "epsilon": aux["epsilon"],
        "internal_loss_func": aux["internal_loss_func"],
    }
    keys = jr.split(key, (N,C))
    if aux["sharpen"]:
        x = _sharpen(x,2)
        y = _sharpen(y,2)

    def ot_loss(x,y,key):
        """OT loss for one channel ``[H W]``, averaged over scales."""
        ks = jr.split(key,2)
        px = _downsample_and_patch(x,S,K,D,key=ks[0])
        py = _downsample_and_patch(y,S,K,D,key=ks[1])
        vscale_ot_loss = jax.vmap(_ott_patch_loss,in_axes=(0,0,None),out_axes=(0))(px,py,ott_kwargs) # vectorized over scales
        return np.mean(vscale_ot_loss)
    
    v_ot_loss = jax.vmap(ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over channels
    vv_ot_loss = jax.vmap(v_ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over N
    losses = vv_ot_loss(x,y,keys) # N C
    where = where[:,:,0,0]
    return np.nan_to_num(np.mean(losses,axis=1,where=where)) # N
        


@eqx.filter_jit
def ott_channel_stack_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """OT loss on patches taken at the same positions in every channel and stacked into one vector.

    ``where`` is ignored.

    Parameters
    ----------
    x : float32 [N C H W]
        predictions
    y : float32 [N C H W]
        true data
    key : jax.random.PRNGKey
    where : None
        not supported
    aux : dict
        S : number of patches per scale
        K : patch size (KxK)
        D : number of 2x downsampling steps
        sharpen : sharpen images before sampling patches
        epsilon : entropic regularisation
        internal_loss_func : patch cost, one of "l2", "l2_squared", "l1", "cos", "arccos"

    Returns
    -------
    loss : float32 [N]
    """
    N = x.shape[0]
    C = x.shape[1]
    S = aux["S"]
    K = aux["K"]
    D = aux["D"]
    # ep = aux["epsilon"]
    ott_kwargs = {
        "epsilon": aux["epsilon"],
        "internal_loss_func": aux["internal_loss_func"],
    }
    if aux["sharpen"]:
        x = _sharpen(x,2)
        y = _sharpen(y,2)

    def v_ot_loss(x,y,key):
        """OT loss for one sample ``[C H W]``, averaged over scales."""
        keys = jr.split(key,2)
        v_ch_downsample_and_patch = jax.vmap(_downsample_and_patch, in_axes=(0,None,None,None,None),out_axes=0) # vectorized over channels
        px = v_ch_downsample_and_patch(x,S,K,D,keys[0]) # C D S K*K
        py = v_ch_downsample_and_patch(y,S,K,D,keys[1]) # C D S K*K
        px = rearrange(px,"C D S Kk -> D S (C Kk)")
        py = rearrange(py,"C D S Kk -> D S (C Kk)")
        vscale_ot_loss = jax.vmap(_ott_patch_loss,in_axes=(0,0,None),out_axes=(0))(px,py,ott_kwargs) # vectorized over scales
        return np.mean(vscale_ot_loss)
    vv_ot_loss = jax.vmap(v_ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over N
    keys = jr.split(key,N)
    losses = vv_ot_loss(x,y,keys) # N
    return losses
