"""Regularisers evaluated on every NCA update step during training.

Each has the signature ``(state, next_state, context, key)``. ``state`` and
``next_state`` are lists (one entry per batch) of arrays shaped ``[N, C, H, W]``,
``context`` is a dict of runtime values (e.g. ``observed_channels``,
``boundary_callbacks``, ``model``), and the result has one value per batch.
"""
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


@eqx.filter_jit
def intermediate_reg(state, next_state, context, key):
    """Penalise state values outside [0, 1]."""
    def _reg(x_new_proc,full=True):
        return jnp.mean(jnp.abs(x_new_proc)+jnp.abs(x_new_proc-1)-1)
    return _batch_map(_reg, next_state)


def hidden_state_size_regulariser(state, next_state, context, key):
    """Penalise the mean absolute value of the hidden channels."""
    def _reg(x_new):
        return jnp.mean(jnp.abs(x_new[:,context["observed_channels"]:]))
    return _batch_map(_reg, next_state)


def boundary_regulariser(state, next_state, context, key):
    """Penalise state channels that are nonzero outside the spatial mask.

    Each channel is weighted by ``1 - spatial_mask``. For a ``model_boundary``
    the trailing mask channel(s) are left out; a ``hard_boundary`` has no mask
    channel, so all channels are included.
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
    """Penalise growth of observed channels away from existing high cells.

    Growth is penalised where the 3x3 neighbourhood sum of the observed
    channels is below about 5, to stop patches of cells appearing out of nowhere.
    """
    def _reg(x, x_new):
        x_proc = x[:,:context["observed_channels"]]
        x_new_proc = x_new[:,:context["observed_channels"]]
        dx = jax.nn.relu(x_new_proc - x_proc) # How much obs growth
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
    """Penalise hidden channel activity where all observed channels are low (< 0.5)."""

    def _reg(x_new_proc):
        x_new_proc_obs = x_new_proc[:,:context["observed_channels"]]
        x_new_proc_hidden = x_new_proc[:,context["observed_channels"]:]
        err = jnp.mean(jax.nn.relu(0.5-jnp.max(x_new_proc_obs,axis=1,keepdims=True))*jnp.abs(x_new_proc_hidden))
        return err
    return _batch_map(_reg, next_state)



def update_sensitivity_regulariser(state, next_state, context, key):
    """Penalise how much the update changes when noise (std 0.1) is added to the input."""

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
    """Penalise updates that do not carry small input perturbations through unchanged.

    Noise (std 0.1) is added to the input; if the input changes by dx, the
    output should change by about dx.
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
