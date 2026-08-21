import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
from Common.model.boundary import hard_boundary, model_boundary, no_boundary
from einops import repeat, reduce, rearrange, einsum


def _batch_map(function, *values):
    return jnp.asarray(jtu.tree_map(function, *values))


# def _is_dict_leaf(value):
#     return isinstance(value, dict)


# def _state(example):
#     return example["latent"] if isinstance(example, dict) else example


@eqx.filter_jit
def intermediate_reg(state, next_state, context, key):
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
    def _reg(x_new_proc,full=True):
        # if not full:
            # x = x[:,:self.OBS_CHANNELS]
        # x_new = _state(x_new)
        return jnp.mean(jnp.abs(x_new_proc)+jnp.abs(x_new_proc-1)-1)
    return _batch_map(_reg, next_state)
        # v_intermediate_reg = lambda x:jnp.array(jax.tree_util.tree_map(self.intermediate_reg,x))  # noqa: E731


def hidden_state_size_regulariser(state, next_state, context, key):
    """
    Regulariser to encourage the model to keep the size of the latent representation small, by penalising the mean value of the latent channels.

    Parameters
    ----------
    x: PyTree [Batch] of Arrays [N C H W]
    x_new: PyTree [Batch] of Arrays [N C H W]
    x_proc: PyTree [Batch] of Arrays [N L h w]
    x_new_proc: PyTree [Batch] of Arrays [N L h w]
    vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
    key: Jax PRNGkey
    Returns
    -------
    reg : float32 Array [BATCH]
        float tracking how much latent space is being used, by mean value of latent channels

    """
    def _reg(x_new):
        return jnp.mean(jnp.abs(x_new[:,context["observed_channels"]:]))
    return _batch_map(_reg, next_state)


def boundary_regulariser(state, next_state, context, key):
    """Penalise state channels that are nonzero outside the spatial mask.

    ``x_new`` is a PyTree of outer-B leaves shaped ``[N,C,H,W]``. For a
    ``model_boundary``, its final fixed mask channel(s) are excluded from the
    penalty; every other channel is weighted by ``1 - spatial_mask``. A
    ``hard_boundary`` has no dedicated mask channel, so all channels are
    included. The returned array has shape ``[B]``.
    """
    del state, key

    select_state = context.get("boundary_state_selector", lambda value: value)

    def _reg(callback, state):
        if isinstance(callback, no_boundary):
            return jnp.zeros((), dtype=state.dtype)
        state = select_state(state)
        if isinstance(callback, model_boundary):
            mask = jnp.asarray(callback.MASK, dtype=state.dtype)
            boundary_channels = mask.shape[0]
            if boundary_channels >= state.shape[-3]:
                raise ValueError(
                    "Boundary regularisation requires at least one non-mask "
                    "state channel"
                )
            values = state[:, :-boundary_channels]
            spatial_mask = jnp.max(mask, axis=0)
        elif isinstance(callback, hard_boundary):
            values = state
            spatial_mask = jnp.asarray(callback.MASK, dtype=state.dtype)
        else:
            raise TypeError(
                "Unsupported boundary callback for regularisation: "
                f"{type(callback).__name__}"
            )
        outside_weight = 1.0 - spatial_mask
        return jnp.mean(jnp.abs(values) * outside_weight)

    callbacks = context["boundary_callbacks"]
    return jnp.asarray(jtu.tree_map(_reg, callbacks, next_state))
@eqx.filter_jit
def contiguous_growth_regulariser(state, next_state, context, key):
    """
    Contiguous state regulariser. For the observable channels, penalises any growth of those channels that occurs more than
    N cells out from the current block of high cells. Intended to stop regions of cells growing seemingly out of nowhere.

    NOTE: VMAP THIS OVER BATCHES

    Parameters
    ----------
        x: PyTree [Batch] of Arrays [N C H W]
        x_new: PyTree [Batch] of Arrays [N C H W]
        x_proc: PyTree [Batch] of Arrays [N L h w]
        x_new_proc: PyTree [Batch] of Arrays [N L h w]
        vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
        key: Jax PRNGkey
    Returns
    -------
        Growth : Array [Batch] float
            float array tracking how much of growth of x_proc_new in observable channels occurs outwith the bounding region of high observable cells in x_proc

    """
    def _reg(x, x_new):
        x_proc = x[:,:context["observed_channels"]]
        x_new_proc = x_new[:,:context["observed_channels"]]
        dx = jax.nn.relu(x_new_proc - x_proc) # How much obs growth
        # kernel = jnp.array([[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
        kernel = jnp.ones((3,3),dtype=jnp.float32)
        kernel = repeat(kernel,"w h -> O I w h",O=1,I=context["observed_channels"])
        dilation = jax.lax.conv_general_dilated(
            lhs=x_proc,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
        )
        dilation = 1 - jax.nn.sigmoid((dilation-5.0)*10.0)
        dilation = repeat(dilation,"N () w h -> N C w h",C=context["observed_channels"])
        err = jnp.mean(dilation*dx)
        return err
    return _batch_map(_reg, state, next_state)


def localised_hidden_regulariser(state, next_state, context, key):
    """
        Encourages NCA to only use the hidden channels in regions where the observable channels are active. Penalises hidden channel activity in regions where observable channels are low.

    Parameters
    ----------
        x: PyTree [Batch] of Arrays [N C H W]
        x_new: PyTree [Batch] of Arrays [N C H W]
        x_proc: PyTree [Batch] of Arrays [N L h w]
        x_new_proc: PyTree [Batch] of Arrays [N L h w]
        vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
        key: Jax PRNGkey
    Returns:
        Sensitivity: Array [Batch] of floats
    """

    def _reg(x_new_proc):
        x_new_proc_obs = x_new_proc[:,:context["observed_channels"]]
        x_new_proc_hidden = x_new_proc[:,context["observed_channels"]:]
        err = jnp.mean(jax.nn.relu(0.5-jnp.max(x_new_proc_obs,axis=1,keepdims=True))*jnp.abs(x_new_proc_hidden))
        # err = jnp.mean(err,axis=(0,1)) # mean over N and C_hidden
        return err
    return _batch_map(_reg, next_state)



def update_sensitivity_regulariser(state, next_state, context, key):
    """
    Measures NCA update step sensitivity to small changes in inputs. Computes a second update step with a small amount of noise added to the input.
    Minimized by NCA model that is insensitive to small changes in input.

    Parameters
    ----------
        x: PyTree [Batch] of Arrays [N C H W]
        x_new: PyTree [Batch] of Arrays [N C H W]
        x_proc: PyTree [Batch] of Arrays [N L h w]
        x_new_proc: PyTree [Batch] of Arrays [N L h w]
        vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
        key: Jax PRNGkey
    Returns:
        Sensitivity: Array [Batch] of floats
    """

    from Common.utils import key_pytree_gen

    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(state)])
    x_noise = jtu.tree_map(lambda x,key: x+noise_amount*jr.normal(key,shape=x.shape),state,key_array_noise)
    key_array_nca = key_pytree_gen(key,(len(state),state[0].shape[0]))
    x_new_noise = context["model"](x_noise,context["boundary_callbacks"],key_array_nca)
    diffs = _batch_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(x_new-x_new_noise)),
        state,
        x_noise,
        next_state,
        x_new_noise,
    )

    return jnp.asarray(diffs)

def perturbation_conservation_regulariser(state, next_state, context, key):
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
        Loss: Array[Batch] of floats
    """
    from Common.utils import key_pytree_gen

    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(state)])
    x_noise = jtu.tree_map(lambda x,key: x+noise_amount*jr.normal(key,shape=x.shape),state,key_array_noise)
    key_array_nca = key_pytree_gen(key,(len(state),state[0].shape[0]))
    x_new_noise = context["model"](x_noise,context["boundary_callbacks"],key_array_nca)

    diffs = _batch_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(jnp.abs(x_new-x_new_noise)-jnp.abs(x-x_noise))),
        state,
        x_noise,
        next_state,
        x_new_noise,

    )
    return jnp.asarray(diffs)
