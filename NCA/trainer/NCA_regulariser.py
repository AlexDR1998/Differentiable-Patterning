import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
from Common.utils import key_pytree_gen
from einops import repeat, reduce, rearrange, einsum


# Target channel layout:
#   A: LMBR, TBXT, SOX17, SOX2
#   B: LMBR, TBXT, SOX17, FOXA2
#   C: CER1, LEFTY2, NODAL
#   D: LEF1
# Group D contains no channel pair and therefore contributes no correlation.
CO_MEASUREMENT_GROUPS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10),
)

# Maps the duplicated 12-channel target layout back to the nine unique model
# channels. This makes repeated biological pairs explicit rather than relying
# on hard-coded pair weights.
TARGET_TO_UNIQUE_CHANNEL = (0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7, 8)


def _build_grouped_correlation_pairs():
    target_pairs = []
    unique_pairs = []
    for group in CO_MEASUREMENT_GROUPS:
        for left_offset, left_channel in enumerate(group):
            for right_channel in group[left_offset + 1:]:
                target_pairs.append((left_channel, right_channel))
                unique_pairs.append(tuple(sorted((
                    TARGET_TO_UNIQUE_CHANNEL[left_channel],
                    TARGET_TO_UNIQUE_CHANNEL[right_channel],
                ))))

    pair_counts = {
        pair: unique_pairs.count(pair)
        for pair in set(unique_pairs)
    }
    pair_weights = [1.0 / pair_counts[pair] for pair in unique_pairs]
    return target_pairs, pair_weights

# LMBR, TBXT, and SOX17 occur in both groups A and B. Correlations between
# those duplicated markers receive half weight in each group, so together they
# count once. Pairs involving SOX2, FOXA2, or group-C markers occur only once.
_CORRELATION_PAIRS, _CORRELATION_WEIGHTS = _build_grouped_correlation_pairs()
CORRELATION_PAIR_I = jnp.array(
    [pair[0] for pair in _CORRELATION_PAIRS], dtype=jnp.int32
)
CORRELATION_PAIR_J = jnp.array(
    [pair[1] for pair in _CORRELATION_PAIRS], dtype=jnp.int32
)
CORRELATION_PAIR_WEIGHTS = jnp.array(_CORRELATION_WEIGHTS, dtype=jnp.float32)

# Repeated measurements of LMBR, TBXT, and SOX17 should together carry the
# same weight as one uniquely measured channel.
RADIAL_CHANNEL_WEIGHTS = jnp.array(
    [
        1.0 / TARGET_TO_UNIQUE_CHANNEL.count(unique_channel)
        for unique_channel in TARGET_TO_UNIQUE_CHANNEL
    ],
    dtype=jnp.float32,
)


def _duplicate_grouped_prediction_channels(prediction):
    """Map nine unique model channels to the 12-channel target layout."""
    return jnp.concatenate(
        [
            prediction[:, 0:4],
            prediction[:, 0:3],
            prediction[:, 4:8],
            prediction[:, 8:9],
        ],
        axis=1,
    )


def _masked_channel_correlations(values, spatial_mask, epsilon):
    """Return Pearson channel correlations for every trajectory timestep."""
    time_count, channel_count = values.shape[:2]
    pixels = values.reshape(time_count, channel_count, -1)
    mask = spatial_mask.reshape(-1).astype(values.dtype)
    pixel_count = jnp.maximum(jnp.sum(mask), 1.0)
    means = jnp.sum(pixels * mask[None, None], axis=-1) / pixel_count
    centred = (pixels - means[..., None]) * mask[None, None]
    covariance = jnp.einsum("tcp,tdp->tcd", centred, centred)
    squared_norms = jnp.sum(centred**2, axis=-1)
    denominator = jnp.sqrt(
        squared_norms[:, :, None] * squared_norms[:, None, :] + epsilon
    )
    return covariance / denominator, squared_norms


def channel_correlation_regulariser(
    predictions,
    targets,
    channel_time_masks,
    boundary_masks=None,
    epsilon=1e-8,
):
    """Match within-experiment channel correlations to the measured targets.

    Parameters use pytrees/lists of batches. Predictions contain the nine
    unique biological channels in their first nine positions, while targets
    use the 12-channel co-measurement layout described above. Only measured
    channel pairs with non-constant target data contribute.
    """
    if boundary_masks is None:
        boundary_masks = [None] * len(predictions)

    batch_losses = []
    for prediction, target, channel_time_mask, boundary_mask in zip(
        predictions, targets, channel_time_masks, boundary_masks
    ):
        prediction = _duplicate_grouped_prediction_channels(prediction[:, :9])
        target = target[:, :12]
        if boundary_mask is None:
            spatial_mask = jnp.ones(target.shape[-2:], dtype=target.dtype)
        else:
            spatial_mask = jnp.squeeze(boundary_mask, axis=0)

        prediction_corr, _ = _masked_channel_correlations(
            prediction, spatial_mask, epsilon
        )
        target_corr, target_squared_norms = _masked_channel_correlations(
            target, spatial_mask, epsilon
        )

        pair_prediction = prediction_corr[
            :, CORRELATION_PAIR_I, CORRELATION_PAIR_J
        ]
        pair_target = target_corr[:, CORRELATION_PAIR_I, CORRELATION_PAIR_J]
        measured_channels = jnp.any(channel_time_mask, axis=(-1, -2))
        measured_pairs = (
            measured_channels[:, CORRELATION_PAIR_I]
            & measured_channels[:, CORRELATION_PAIR_J]
        )
        variable_targets = (
            target_squared_norms[:, CORRELATION_PAIR_I] > epsilon
        ) & (
            target_squared_norms[:, CORRELATION_PAIR_J] > epsilon
        )
        valid_pairs = measured_pairs & variable_targets
        weights = CORRELATION_PAIR_WEIGHTS[None] * valid_pairs
        squared_error = (pair_prediction - pair_target) ** 2
        batch_losses.append(
            jnp.sum(weights * squared_error) / jnp.maximum(jnp.sum(weights), 1.0)
        )

    return jnp.stack(batch_losses)


def _radial_profiles(values, spatial_mask, radial_bins, epsilon):
    """Return annular mean intensities and a mask of non-empty annuli."""
    width, height = values.shape[-2:]
    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(width, dtype=values.dtype),
        jnp.arange(height, dtype=values.dtype),
        indexing="ij",
    )
    spatial_mask = spatial_mask.astype(bool)
    mask = spatial_mask.astype(values.dtype)
    pixel_count = jnp.sum(mask)
    centre_x = jnp.sum(grid_x * mask) / jnp.maximum(pixel_count, 1.0)
    centre_y = jnp.sum(grid_y * mask) / jnp.maximum(pixel_count, 1.0)
    radius = jnp.sqrt((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2)
    max_radius = jnp.max(jnp.where(spatial_mask, radius, 0.0))
    normalized_radius = radius / jnp.maximum(max_radius, epsilon)
    bin_indices = jnp.minimum(
        jnp.floor(normalized_radius * radial_bins).astype(jnp.int32),
        radial_bins - 1,
    )
    annuli = (
        bin_indices[None] == jnp.arange(radial_bins)[:, None, None]
    ) & spatial_mask[None]
    annuli = annuli.astype(values.dtype)
    annulus_counts = jnp.sum(annuli, axis=(-1, -2))
    profile_sums = jnp.einsum("tchw,rhw->tcr", values, annuli)
    profiles = profile_sums / jnp.maximum(annulus_counts[None, None], 1.0)
    return profiles, annulus_counts > 0


def radial_profile_regulariser(
    predictions,
    targets,
    channel_time_masks,
    boundary_masks=None,
    radial_bins=16,
    epsilon=1e-8,
):
    """Match per-channel radial mean-intensity profiles to measured targets.

    Radius is measured from the centroid of the boundary mask and normalized
    by its largest in-mask radius. Predictions use the nine unique biological
    channels, while targets use the duplicated 12-channel co-measurement
    layout. Duplicate target observations are half-weighted so each biological
    channel has equal total importance.
    """
    if radial_bins <= 0:
        raise ValueError("radial_bins must be positive")
    if boundary_masks is None:
        boundary_masks = [None] * len(predictions)

    batch_losses = []
    for prediction, target, channel_time_mask, boundary_mask in zip(
        predictions, targets, channel_time_masks, boundary_masks
    ):
        prediction = _duplicate_grouped_prediction_channels(prediction[:, :9])
        target = target[:, :12]
        if boundary_mask is None:
            spatial_mask = jnp.ones(target.shape[-2:], dtype=bool)
        else:
            spatial_mask = jnp.squeeze(boundary_mask, axis=0).astype(bool)

        prediction_profiles, nonempty_bins = _radial_profiles(
            prediction, spatial_mask, radial_bins, epsilon
        )
        target_profiles, _ = _radial_profiles(
            target, spatial_mask, radial_bins, epsilon
        )
        measured_channels = jnp.any(channel_time_mask, axis=(-1, -2))
        weights = (
            measured_channels[:, :, None]
            * RADIAL_CHANNEL_WEIGHTS[None, :, None]
            * nonempty_bins[None, None, :]
        )
        squared_error = (prediction_profiles - target_profiles) ** 2
        batch_losses.append(
            jnp.sum(weights * squared_error) / jnp.maximum(jnp.sum(weights), 1.0)
        )

    return jnp.stack(batch_losses)


# def _is_dict_leaf(value):
#     return isinstance(value, dict)


# def _state(example):
#     return example["latent"] if isinstance(example, dict) else example


@eqx.filter_jit
def intermediate_reg(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
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
    return jnp.array(jtu.tree_map(_reg,x_new_proc))
        # v_intermediate_reg = lambda x:jnp.array(jax.tree_util.tree_map(self.intermediate_reg,x))  # noqa: E731
		

def latent_size_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
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
        return jnp.mean(jnp.abs(x_new[:,aux["OBS_CHANNELS"]:]))
    return jnp.array(jtu.tree_map(_reg,x_new))


def boundary_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
    """
    Penalise the model for any nonzero components outside the boundary mask
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
        float tracking how much of x_new is outwith the boundary mask
    """
    # x_new = jtu.tree_map(_state, x_new, is_leaf=_is_dict_leaf)
    x_in_bound = jax.tree_util.tree_map(lambda f,x:f(x),aux["BOUNDARY_CALLBACK"],x_new)
    x_out_bound = jax.tree_util.tree_map(lambda x,y: x-y,x_new,x_in_bound)
    return jnp.array(jax.tree_util.tree_map(lambda x: jnp.mean(jnp.abs(x)),x_out_bound))
@eqx.filter_jit
def contiguous_growth_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
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
    def _reg(x_new,x,x_proc,x_new_proc):
        
        
        x_new_proc = x_new_proc[:,:aux["OBS_CHANNELS"]]
        x_proc = x_proc[:,:aux["OBS_CHANNELS"]]
        dx = jax.nn.relu(x_new_proc - x_proc) # How much obs growth
        # kernel = jnp.array([[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
        kernel = jnp.ones((3,3),dtype=jnp.float32)
        kernel = repeat(kernel,"w h -> O I w h",O=1,I=aux["OBS_CHANNELS"])
        dilation = jax.lax.conv_general_dilated(
            lhs=x_proc,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",
        )
        dilation = 1 - jax.nn.sigmoid((dilation-5.0)*10.0)
        dilation = repeat(dilation,"N () w h -> N C w h",C=aux["OBS_CHANNELS"])
        err = jnp.mean(dilation*dx)
        return err
    return jnp.array(jtu.tree_map(_reg,x_new,x,x_proc,x_new_proc))

def update_sensitivity_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
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

    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(x)])
    
    x_noise = jtu.tree_map(lambda x,key: x+noise_amount*jr.normal(key,shape=x.shape),x,key_array_noise)
    key_array_nca = key_pytree_gen(key,(len(x),x[0].shape[0]))
    x_new_noise = vv_nca(x_noise,aux["BOUNDARY_CALLBACK"],key_array_nca)
    diffs = jtu.tree_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(x_new-x_new_noise)),
        x,
        x_noise,
        x_new,
        x_new_noise,
    )

    return jnp.array(diffs)

def perturbation_conservation_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
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
    noise_amount = 0.1
    key_array_noise = key_pytree_gen(key,[len(x)])

    x_noise = jtu.tree_map(lambda x,key: x+noise_amount*jr.normal(key,shape=x.shape),x,key_array_noise)
    key_array_nca = key_pytree_gen(key,(len(x),x[0].shape[0]))
    x_new_noise = vv_nca(x_noise,aux["BOUNDARY_CALLBACK"],key_array_nca)

    diffs = jtu.tree_map(
        lambda x,x_noise,x_new,x_new_noise: jnp.mean(jnp.abs(jnp.abs(x_new-x_new_noise)-jnp.abs(x-x_noise))),
        x,
        x_noise,
        x_new,
        x_new_noise,
        
    )
    return jnp.array(diffs)


def latent_channel_match_regulariser(x,x_new,x_proc,x_new_proc,vv_nca,aux,key):
    """
    Regulariser to encourage the model to match the first N latent channels to be downsampled versions of the output channnels.

    Parameters
    ----------
    x: PyTree [Batch] of Arrays [N C H W]
    x_new: PyTree [Batch] of Arrays [N C H W]
    x_proc: PyTree [Batch] of Arrays [N L h w]
    x_new_proc: PyTree [Batch] of Arrays [N L h w]
    """
    OBS_CHANNELS = aux["OBS_CHANNELS"]
    real_to_latent = aux["REAL_TO_LATENT"]
    x_latent = jtu.tree_map(real_to_latent,x_new_proc)
    
    losses = jtu.tree_map(
        lambda x,x_latent:jnp.mean(jnp.abs(x[:,:OBS_CHANNELS]-x_latent[:,:OBS_CHANNELS])),
        x_new,
        x_latent,
    )
    return jnp.array(losses)
