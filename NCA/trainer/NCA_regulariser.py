import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
from Common.utils import key_pytree_gen
from einops import repeat, reduce, rearrange, einsum


def _is_dict_leaf(value):
    return isinstance(value, dict)


def _state(example):
    return example["latent"] if isinstance(example, dict) else example

@eqx.filter_jit
def intermediate_reg(x,x_new,vv_nca,aux,key):
    """
    Intermediate state regulariser - tracks how much of x is outwith [0,1]
    

    Parameters
    ----------
    x : float32 array [B,N,CHANNELS,_,_]
        NCA state
    x_new : float32 array [B,N,CHANNELS,_,_]
        Updated NCA state
    vv_nca : Callable
        NCA update function - doubly vectorised to work on [B,N,CHANNELS,_,_]
    aux : Any
        Auxiliary information
    key : jax.random.PRNGKey
        Jax random number key
    Returns
    -------
    reg : float
        float tracking how much of x is outwith range [0,1]

    """
    def _reg(x_new,full=True):
        # if not full:
            # x = x[:,:self.OBS_CHANNELS]
        x_new = _state(x_new)
        return jnp.mean(jnp.abs(x_new)+jnp.abs(x_new-1)-1)
    return jnp.array(jtu.tree_map(_reg,x_new,is_leaf=_is_dict_leaf))
        # v_intermediate_reg = lambda x:jnp.array(jax.tree_util.tree_map(self.intermediate_reg,x))  # noqa: E731
		
def boundary_regulariser(x,x_new,vv_nca,aux,key):
    """
    Penalise the model for any nonzero components outside the boundary mask
    Parameters
    ----------
    x : float32 PyTree [BATCH] Array [,N,CHANNELS,_,_]
        NCA state
    Returns
    -------
    reg : float32 PyTree [BATCH]
    
    """
    x_new = jtu.tree_map(_state, x_new, is_leaf=_is_dict_leaf)
    x_in_bound = jax.tree_util.tree_map(lambda f,x:f(x),aux["BOUNDARY_CALLBACK"],x_new)
    x_out_bound = jax.tree_util.tree_map(lambda x,y: x-y,x_new,x_in_bound)
    return jnp.array(jax.tree_util.tree_map(lambda x: jnp.mean(jnp.abs(x)),x_out_bound))
@eqx.filter_jit
def contiguous_growth_regulariser(x,x_new,vv_nca,aux,key):
    """
    Contiguous state regulariser. For the observable channels, penalises any growth of those channels that occurs more than
    N cells out from the current block of high cells. Intended to stop regions of cells growing seemingly out of nowhere.

    NOTE: VMAP THIS OVER BATCHES

    Parameters
    ----------
    x : float32 array [N,CHANNELS,_,_]
        NCA state
    x_previous : float32 array [N,CHANNELS,_,_]
        Previous NCA state
    Returns
    -------
    reg : float
        float tracking how much of growth of x in observable channels occurs outwith the bounding region of high observable cells in x_previous 

    """
    def _reg(x_new,x):
        x_new = _state(x_new)
        x = _state(x)
        x_new = x_new[:,:aux["OBS_CHANNELS"]]
        x = x[:,:aux["OBS_CHANNELS"]]
        dx = jax.nn.relu(x_new - x) # How much obs growth
        # kernel = jnp.array([[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
        kernel = jnp.ones((3,3),dtype=jnp.float32)
        kernel = repeat(kernel,"w h -> O I w h",O=1,I=aux["OBS_CHANNELS"])
        dilation = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
        )
        dilation = 1 - jax.nn.sigmoid((dilation-5.0)*10.0)
        dilation = repeat(dilation,"N () w h -> N C w h",C=aux["OBS_CHANNELS"])
        err = jnp.mean(dilation*dx)
        return err
    return jnp.array(jtu.tree_map(_reg,x_new,x,is_leaf=_is_dict_leaf))

def update_sensitivity_regulariser(x,x_new,vv_nca,aux,key):
    """
    Measures NCA update step sensitivity to small changes in inputs. Computes a second update step with a small amount of noise added to the input.
    Minimized by NCA model that is insensitive to small changes in input.

    Parameters
    ----------
        x: PyTree [Batch] of Arrays [N C H W]
        x_new: PyTree [Batch] of Arrays [N C H W]
        vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
        key: Jax PRNGkey
    Returns:
        Sensitivity: List [Batch] of floats
    """

    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(x)])
    x_noise = jtu.tree_map(
        lambda x,key: {**x, "latent": x["latent"] + noise_amount * jr.normal(key, shape=x["latent"].shape)} if isinstance(x, dict) else x+noise_amount*jr.normal(key,shape=x.shape),
        x,
        key_array_noise,
        is_leaf=_is_dict_leaf,
    ) # x with gaussian noise added
    key_array_nca = key_pytree_gen(key,(len(x),_state(x[0]).shape[0]))
    x_new_noise = vv_nca(x_noise,aux["BOUNDARY_CALLBACK"],key_array_nca)
    diffs = jtu.tree_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(_state(x_new)-_state(x_new_noise))),
        x,
        x_noise,
        x_new,
        x_new_noise,
        is_leaf=_is_dict_leaf,
    )

    return jnp.array(diffs)

def perturbation_conservation_regulariser(x,x_new,vv_nca,aux,key):
    """
    Measures NCA update step sensitivity to small changes in inputs. Computes a second update step with a small amount of noise added to the input.
    Minimized by NCA model that is linearly proportional to small changes in input. I.e. if input is changed by dx, output should change by ~dx


    Parameters
    ----------
        x: PyTree [Batch] of Arrays [N C H W]
        x_new: PyTree [Batch] of Arrays [N C H W]
        vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
        key: Jax PRNGkey
    Returns:
        Loss: List [Batch] of floats
    """
    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(x)])
    x_noise = jtu.tree_map(
        lambda x,key: {**x, "latent": x["latent"] + noise_amount * jr.normal(key, shape=x["latent"].shape)} if isinstance(x, dict) else x+noise_amount*jr.normal(key,shape=x.shape),
        x,
        key_array_noise,
        is_leaf=_is_dict_leaf,
    ) # x with gaussian noise added
    key_array_nca = key_pytree_gen(key,(len(x),_state(x[0]).shape[0]))
    x_new_noise = vv_nca(x_noise,aux["BOUNDARY_CALLBACK"],key_array_nca)

    diffs = jtu.tree_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(jnp.abs(_state(x_new)-_state(x_new_noise))-jnp.abs(_state(x)-_state(x_noise)))),
        x,
        x_noise,
        x_new,
        x_new_noise,
        is_leaf=_is_dict_leaf,
    )
    return jnp.array(diffs)