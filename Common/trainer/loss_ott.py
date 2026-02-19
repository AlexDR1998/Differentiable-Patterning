import jax.numpy as np
import jax
#from ott.geometry import pointcloud
#from ott.tools import sinkhorn_divergence
#from ott.problems.linear import linear_problem
#from ott.solvers.linear import sinkhorn
import equinox as eqx
import ott
from einops import rearrange,reduce,einsum,repeat
import jax.random as jr
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch,pad_to_multiple_of_3_channels


def _make_gaussian_kernel(sigma, nstds):
    """
    Helper to create a Gaussian kernel with concrete Python values.
    Must be called outside of JIT context.
    """
    import math
    sigma_x = float(sigma) * 0.5
    extent = float(nstds) * sigma_x
    kmax = math.ceil(max(1.0, extent))
    
    coords = np.linspace(-kmax, kmax, 2*kmax + 1)
    y, x = np.meshgrid(coords, coords, indexing='ij')
    
    gb = np.exp(-0.5 * (x**2 / (sigma_x**2) + y**2 / (sigma_x**2)))
    
    return gb / np.sum(gb)

# Pre-compute commonly used kernels as constants
_GAUSSIAN_CACHE = {
    (1, 1): _make_gaussian_kernel(1, 1),
    (2, 2): _make_gaussian_kernel(2, 2),
    (3, 3): _make_gaussian_kernel(3, 3),
    (5, 5): _make_gaussian_kernel(5, 5),
}

def _gaussian(sigma, nstds):
    """
    Creates a normalized 2D Gaussian kernel. JIT-compatible via pre-computed cache.
    
    Parameters:
        sigma: float, standard deviation scaling factor
        nstds: float, number of standard deviations for kernel extent
    
    Returns:
        Normalized 2D Gaussian kernel as jax array
    """
    key = (sigma, nstds)
    if key in _GAUSSIAN_CACHE:
        return _GAUSSIAN_CACHE[key]
    else:
        # Fallback: compute it (will fail if called inside JIT with non-static args)
        return _make_gaussian_kernel(sigma, nstds)

def _sharpen(X,k):
    """
        Sharpens the images with a gaussian kernel of size 2*k+1
        Parameters:
            X: np.ndarray of shape [N C H W] where N=Batches, C=channels, H=height, W=width
            k: int, size parameter for the gaussian kernel
        Returns:
            np.ndarray of shape [N C H W], sharpened images
    """
    C = X.shape[1]
    kernel = _gaussian(k,k)
    # Kernel shape for depthwise conv: [kh, kw, in_features_per_group=1, out_features=C]
    kernel = repeat(kernel,"kh kw -> kh kw () C", C=C)
    # Input shape: [batch=1, H, W, in_features=C]
    X_reshaped = rearrange(X,"N C H W -> N H W C")
    
    # Use explicit dimension_numbers to ensure correct interpretation
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    blur = jax.lax.conv_general_dilated(
        X_reshaped,
        kernel,
        window_strides=(1,1),
        padding='SAME',
        dimension_numbers=dimension_numbers,
        feature_group_count=C,  # Each channel processed independently
    )
    # Rearrange back to [C H W]
    
    blur = rearrange(blur, "N H W C -> N C H W")
    return X + 2*(X - blur)


def _sample_random_patches(X,S,K,key):
    """
        Samples S random patches of size KxK from image X of shape [H W]
		Parameters:
			X: float32 [H W]
			S: int, number of patches to sample
			K: int, size of patches (KxK)
			key: jax.random.PRNGKey
		Returns:
			patches: float32 [S K*K], sampled patches
    """
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
    """
        Downsamples image X by factor of 2 D times and samples S random patches of size KxK from it.
        Parameters:
            X: float32 [H W]
            S: int, number of patches to sample
            K: int, size of patches (KxK)
            D: int, number of downsampling layers
            key: jax.random.PRNGKey
        Returns:
            patches: float32 [D S K*K], sampled patches
    """
    patches = [_sample_random_patches(X,S,K,key=key)]
    Xd = X
    keys = jr.split(key,D-1)
    for d in range(D-1):
        Xd = np.pad(Xd,((0,Xd.shape[0]%2),(0,Xd.shape[1]%2)),'reflect')
        Xd = reduce(Xd,"(h 2) (w 2) -> h w", 'mean')
        pd = _sample_random_patches(Xd,S,K,key=keys[d])
        patches.append(pd)
    patches = np.stack(patches,axis=0)
    return patches


def _ott_patch_loss(PX,PY,aux):
    """
        Computes linear OT loss between 2 point clouds of size S in dimensionality K*K. 
        Parameters:
            PX: float32 [S K*K]
            PY: float32 [S K*K]
        Returns:
            ot_cost: float32, OT cost between patches sampled from X and Y
    """
    
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
    """
        Computes OT loss between images x and y by sampling random patches at multiple scales.

        Parameters
        ----------
        x : float32 [N C H W]
            predictions
        y : float32 [N C H W]
            true data
        key: jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N C]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        aux : dict
            Additional parameters for the loss function. Includes D, S, K
                S : int - number of patches to sample
                K : int - size of patches (KxK)
                D : int - number of downsampling steps
                Sharpen: bool - whether to sharpen images before computing loss
        Returns
        -------
        loss : float32 [N]
            loss 

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
        """
            OT loss for a single channel/timestep
            Parameters:
                x: float32 [H W]
                y: float32 [H W]
                k: jax.random.PRNGKey
            Returns:
                loss: float32
        """
        ks = jr.split(key,2)
        px = _downsample_and_patch(x,S,K,D,key=ks[0])
        py = _downsample_and_patch(y,S,K,D,key=ks[1])
        vscale_ot_loss = jax.vmap(_ott_patch_loss,in_axes=(0,0,None),out_axes=(0))(px,py,ott_kwargs) # vectorized over scales
        return np.mean(vscale_ot_loss)
    
    v_ot_loss = jax.vmap(ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over channels
    vv_ot_loss = jax.vmap(v_ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over N
    losses = vv_ot_loss(x,y,keys) # N C
    where = where[:,:,0,0]
    print(f"OTT losses shape: {losses.shape}")
    print(f"where shape: {where.shape}")
    return np.nan_to_num(np.mean(losses,axis=1,where=where)) # N
        


@eqx.filter_jit
def ott_channel_stack_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """
        Computes OT loss between images x and y by sampling random patches at multiple scales, 
        sampling the same patches across all channels and flattening those "stack" patches to the OT pointcloud space
        
        Where mask is currently not supported for this loss.

        Parameters
        ----------
        x : float32 [N C H W]
            predictions
        y : float32 [N C H W]
            true data
        key: jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N C]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        aux : dict
            Additional parameters for the loss function. Includes D, S, K
                S : int - number of patches to sample
                K : int - size of patches (KxK)
                D : int - number of downsampling steps
                Sharpen: bool - whether to sharpen images before computing loss
        Returns
        -------
        loss : float32 [N]
            loss 

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
        """
            OT loss for a single timestep
            Parameters:
                x: float32 [C H W]
                y: float32 [C H W]
                k: jax.random.PRNGKey
            Returns:
                loss: float32
        """
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








def ott_grouped_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """
        Computes OT loss between images x and y by grouping channels based on experiment and ott_loss on each group.
        Parameters
        ----------
        x : float32 [N C=8 H W]
            predictions
        y : float32 [N C=11 H W]
            true data - with some duplicate channels from different experiment groups
        key: jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N C 1 1]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        aux : dict
            Additional parameters for the loss function. Includes D, S, K, Sharpen
                S : int - number of patches to sample
                K : int - size of patches (KxK)
                D : int - number of downsampling steps
                Sharpen: bool - whether to sharpen images before computing loss
        Returns
        -------
        loss : float32 [N]
            loss reduced over channel and spatial axes
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
    x = duplicate_x_channels_9ch(x)
    # where = rearrange(where,"n c -> n c 1 1")
    if where is not None:
        where = duplicate_x_channels_9ch(where)
        x = x*where.astype(x.dtype)
        y = y*where.astype(y.dtype)
    
    def v_ot_loss(x,y,key):
        """
            OT loss for a single timestep
            Parameters:
                x: float32 [C H W]
                y: float32 [C H W]
                k: jax.random.PRNGKey
            Returns:
                loss: float32
        """
        keys = jr.split(key,2)
        v_ch_downsample_and_patch = jax.vmap(_downsample_and_patch, in_axes=(0,None,None,None,None),out_axes=0) # vectorized over channels. Don't vectorize over keys - we want to select the same patches across channels in each group
        px = v_ch_downsample_and_patch(x,S,K,D,keys[0]) # C D S K*K
        py = v_ch_downsample_and_patch(y,S,K,D,keys[1]) # C D S K*K
        px = rearrange(px,"C D S Kk -> D S (C Kk)")
        py = rearrange(py,"C D S Kk -> D S (C Kk)")
        
        vscale_ot_loss = jax.vmap(_ott_patch_loss,in_axes=(0,0,None),out_axes=(0))(px,py,ott_kwargs) # vectorized over scales which is then averaged over
        return np.mean(vscale_ot_loss)
    
    vv_ot_loss = jax.vmap(v_ot_loss,in_axes=(0,0,0),out_axes=0) # Vectorized over N
    keys = jr.split(key,(N,4))
    
    # Each group of channels from the same experiment will have identically located patches selected.
    
    # losses = np.stack([
    #     vv_ot_loss(x[:,0:4,:,:],y[:,0:4,:,:],keys[:,0]),
    #     vv_ot_loss(x[:,0:3,:,:],y[:,4:7,:,:],keys[:,1]),
    #     vv_ot_loss(x[:,4:8,:,:],y[:,7:11,:,:],keys[:,2]),
    #     vv_ot_loss(x[:,8:9,:,:],y[:,11:12,:,:],keys[:,3])
    # ],axis=1)
    losses = np.stack([
        vv_ot_loss(x[:,0:4,:,:],y[:,0:4,:,:],keys[:,0]),
        vv_ot_loss(x[:,4:7,:,:],y[:,4:7,:,:],keys[:,1]),
        vv_ot_loss(x[:,7:11,:,:],y[:,7:11,:,:],keys[:,2]),
        vv_ot_loss(x[:,11:12,:,:],y[:,11:12,:,:],keys[:,3])
    ],axis=1)
    losses = np.mean(losses,axis=1) # N
    # losses = vv_ot_loss(x,y,keys) # N
    return losses



def ott_grouped_and_l2_loss(x,y,key,where=None,aux={"D":3,"S":1024,"K":5,"sharpen":True,"epsilon":0.1,"internal_loss_func":"l2"}):
    """
        Computes OT loss between images x and y by grouping channels based on experiment and ott_loss on each group.
        Parameters
        ----------
        x : float32 [N C=8 H W]
            predictions
        y : float32 [N C=11 H W]
            true data - with some duplicate channels from different experiment groups
        key: jax.random.PRNGKey
            Jax random number key.
        where : boolean array [N C]
            Mask to apply to x and y before calculating loss, to select which timesteps and channels we care about.
        aux : dict
            Additional parameters for the loss function. Includes D, S, K
                S : int - number of patches to sample
                K : int - size of patches (KxK)
                D : int - number of downsampling steps
                Sharpen: bool - whether to sharpen images before computing loss
        Returns
        -------
        loss : float32 [N]
            loss reduced over channel and spatial axes
    """
    loss_ott = ott_grouped_loss(x,y,key,where,aux)
    x_full = duplicate_x_channels_9ch(x)
    _l2 = (x_full-y)**2
    weighting = np.array([0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0]) # Account for duplicate channels
    _l2 = einsum(_l2,weighting,"n c x y , c -> n c x y")
    where_full = duplicate_x_channels_9ch(where).astype(where.dtype)
    l2_loss = np.nan_to_num(np.mean(_l2,axis=[-1,-2,-3],where=where_full))
    return loss_ott + l2_loss



def emd_loss(x,y,key,where,aux={"epsilon":0.01,"internal_loss_func":"l2","normalize":True,"tau":1.0}):

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
            "l2": [ott.geometry.costs.Euclidean()]*2,
            "l2_squared": [ott.geometry.costs.SqEuclidean()]*2,
            "l1": [ott.geometry.costs.PNormP(1)]*2,
        }
        geom = ott.geometry.grid.Grid(grid_size=X.shape,epsilon=aux['epsilon'],cost_fns=metric[aux["internal_loss_func"]])
        if aux["normalize"]:
            X = X/ (X.sum()+1e-8)
            Y = Y/ (Y.sum()+1e-8)
        problem = ott.problems.linear.linear_problem.LinearProblem(geom,a=X.ravel(),b=Y.ravel(),tau_a=aux["tau"],tau_b=aux["tau"])
        
        solver = ott.solvers.linear.sinkhorn.Sinkhorn(min_iterations=64,max_iterations=64)
        out = solver(problem)
        
        return out.reg_ot_cost
    v_oti_loss = jax.vmap(oti_loss, in_axes=(0,0,None),out_axes=0)
    vv_oti_loss = jax.vmap(v_oti_loss, in_axes=(0,0,None),out_axes=0)
    losses = vv_oti_loss(x,y,aux) # Shape N C
    losses_avg_intensity = (reduce(x,"n c x y -> n c", 'mean') - reduce(y,"n c x y -> n c", 'mean'))**2
    if aux["amplitude_penalty"]:
        losses = losses + losses_avg_intensity
    where = where[:,:,0,0] # Shape N C
    return np.nan_to_num(np.mean(losses,axis=1,where=where)) # N