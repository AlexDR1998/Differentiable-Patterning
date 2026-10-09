"""Generic image losses.

Every loss takes predictions ``x`` and targets ``y`` of shape
``[N, CHANNELS, WIDTH, HEIGHT]`` and returns one value per sample, ``[N]``.
The table that maps loss names in the config to these functions is in
``loss_table.py``.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
from jax.scipy.ndimage import map_coordinates
from einops import rearrange,reduce,einsum,repeat
import jax.random as jr


@jax.jit
def cosine(x,y,key=None,where=None,aux=None,cache=None):
	"""Negative cosine similarity, normalised by the norms of the whole ``x`` and ``y`` arrays."""
	return -jnp.nan_to_num(jnp.mean((x*y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)),axis=[-1,-2,-3],where=where))

@jax.jit
def l2(x,y,key=None,where=None,aux=None,cache=None):
	"""Mean squared error."""
	
	return jnp.nan_to_num(jnp.mean((x-y)**2,axis=[-1,-2,-3],where=where))

@jax.jit
def l1(x,y,key=None,where=None,aux=None,cache=None):
	"""Mean absolute error."""
	return jnp.nan_to_num(jnp.mean(jnp.abs(x-y),axis=[-1,-2,-3],where=where))

@jax.jit
def euclidean(x,y,key=None,where=None,aux=None,cache=None):
	"""Root mean squared error."""
	return jnp.nan_to_num(jnp.sqrt(jnp.mean(((x-y)**2),axis=[-1,-2,-3],where=where)))

@eqx.filter_jit
def sliced_wasserstein_spatial(x,y,key=None,where=None,aux=None,cache=None):
	"""Sliced Wasserstein distance with ``aux["samples"]`` (default 64) random spatial projections."""
	
	WIDTH = x.shape[2]
	HEIGHT = x.shape[3]
	
	if aux["samples"] is None:
		SAMPLES = 64
	else:
		SAMPLES = aux["samples"]
	
	proj_directions = jr.uniform(key,(WIDTH,HEIGHT,SAMPLES))
	proj_directions = proj_directions / jnp.linalg.norm(proj_directions,axis=(0,1),keepdims=True)

	x_proj = einsum(x,proj_directions,"n channels width height , width height samples -> samples n channels")
	y_proj = einsum(y,proj_directions,"n channels width height , width height samples -> samples n channels")

	x_sorted = jnp.sort(x_proj,axis=-1)
	y_sorted = jnp.sort(y_proj,axis=-1)

	return jnp.nan_to_num(jnp.mean((x_sorted - y_sorted)**2,axis=[0,2]))


@eqx.filter_jit
def sliced_wasserstein_channel(x,y,key=None,where=None,aux=None,cache=None):
	"""Sliced Wasserstein distance between pixel distributions, with ``aux["samples"]`` (default 64) random projections across channels."""
	
	CHANNELS = x.shape[1]
	if aux["samples"] is None:
		SAMPLES = 64
	else:
		SAMPLES = aux["samples"]
	
	proj_directions = jr.uniform(key,(CHANNELS,SAMPLES))
	proj_directions = proj_directions / jnp.linalg.norm(proj_directions,axis=(0),keepdims=True)

	x_proj = einsum(x,proj_directions,"n channels width height , channels samples -> n samples width height")
	y_proj = einsum(y,proj_directions,"n channels width height , channels samples -> n samples width height")

	x_proj = rearrange(x_proj,"n s w h -> s n (w h)")
	y_proj = rearrange(y_proj,"n s w h -> s n (w h)")

	x_sorted = jnp.sort(x_proj,axis=-1)
	y_sorted = jnp.sort(y_proj,axis=-1)

	return jnp.nan_to_num(jnp.mean((x_sorted - y_sorted)**2,axis=[0,2]))


@eqx.filter_jit
def sliced_wasserstein_rotational(x,y,key=None,where=None,aux=None,cache=None):

	"""Sliced Wasserstein distance between 1D profiles of the images rotated by ``aux["samples"]`` (default 64) random angles."""
	
	WIDTH = x.shape[2]
	HEIGHT = x.shape[3]
	
	if aux["samples"] is None:
		SAMPLES = 64
	else:
		SAMPLES = aux["samples"]
	
	angles = jr.uniform(key,(SAMPLES,),minval=0.0,maxval=360.0)
	v_rotate_project = jax.vmap(_rotate_and_project, in_axes=(0,None),out_axes=0) # rotates array of shape [C, WIDTH, HEIGHT] by a given angle and projects to [C,WIDTH]
	vv_rotate_project = jax.vmap(v_rotate_project, in_axes=(0,None),out_axes=0) # rotates array of shape [N, C, WIDTH,HEIGHT] by a given angle and projects to [N, C, W]
	vvv_rotate_project = jax.vmap(vv_rotate_project, in_axes=(None,0),out_axes=0) # rotates array of shape [N, C, WIDTH,HEIGHT] by an array of angles [SAMPLES] and projects to [SAMPLES, N, C, W]

	x_proj = vvv_rotate_project(x,angles) # shape [SAMPLES, N, C, W]
	y_proj = vvv_rotate_project(y,angles) # shape [SAMPLES, N, C, W]
	# x_proj = jnp.mean(x_rotated,axis=-1) # shape [SAMPLES, N, C, W]
	# y_proj = jnp.mean(y_rotated,axis=-1) # shape [SAMPLES, N, C, W]
	x_proj = rearrange(x_proj,"s n c w -> (s c) n w")
	y_proj = rearrange(y_proj,"s n c w -> (s c) n w")
	x_sorted = jnp.sort(x_proj,axis=-1)
	y_sorted = jnp.sort(y_proj,axis=-1)

	return jnp.nan_to_num(jnp.mean((x_sorted - y_sorted)**2,axis=[0,2]))

def _get_rotation_grid(shape, angle_deg):
    
	ny, nx = shape
	y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
	# Center coordinates for rotation.
	y_center = (ny - 1) / 2.
	x_center = (nx - 1) / 2.
	y = y - y_center
	x = x - x_center

	# Convert angle to radians.
	theta = jnp.deg2rad(angle_deg)
	cos_theta = jnp.cos(theta)
	sin_theta = jnp.sin(theta)

	# Compute inverse rotation (to sample from the input image).
	x_rot = cos_theta * x + sin_theta * y
	y_rot = -sin_theta * x + cos_theta * y

	# Shift back.
	x_rot = x_rot + x_center
	y_rot = y_rot + y_center

	return y_rot, x_rot

def _rotate_and_project(arr, angle_deg):
	# arr shape: [W, H]
	# return shape: [W]
	coords = _get_rotation_grid(arr.shape, angle_deg)
	coords = jnp.stack(coords, axis=0)
	rotated = map_coordinates(arr, coords, order=1, mode='constant', cval=0.0)
	rotated = jnp.mean(rotated,axis=-1)
	return rotated

@eqx.filter_jit
def wasserstein_projected(x,y,key=None,where=None,aux=None,cache=None):
	"""Mean squared difference of random projections of the whole image (not sorted)."""
	
	CHANNELS = x.shape[1]
	WIDTH = x.shape[2]
	HEIGHT = x.shape[3]
	
	if aux["samples"] is None:
		SAMPLES = 64
	else:
		SAMPLES = aux["samples"]
	
	proj_directions = jr.uniform(key,(CHANNELS,WIDTH,HEIGHT,SAMPLES))
	proj_directions = proj_directions / jnp.linalg.norm(proj_directions,axis=(1,2),keepdims=True)

	x_proj = einsum(x,proj_directions,"n channels width height , channels width height samples -> n samples")
	y_proj = einsum(y,proj_directions,"n channels width height , channels width height samples -> n samples")

	# x_sorted = jnp.sort(x_proj,axis=1)
	# y_sorted = jnp.sort(y_proj,axis=1)
	x_sorted = x_proj
	y_sorted = y_proj

	return jnp.nan_to_num(jnp.mean((x_sorted - y_sorted)**2,axis=-1))

@eqx.filter_jit
def spectral_wasserstein_projected(x,y,key=None,where=None,aux=None,cache=None):
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	CHANNELS = fx.shape[1]
	WIDTH = fx.shape[2]
	HEIGHT = fx.shape[3]
	
	if aux["samples"] is None:
		SAMPLES = 64
	else:
		SAMPLES = aux["samples"]
	
	proj_directions = jr.uniform(key,(CHANNELS,WIDTH,HEIGHT,SAMPLES))
	proj_directions = proj_directions / jnp.linalg.norm(proj_directions,axis=(1,2),keepdims=True)

	x_proj = einsum(fx,proj_directions,"n channels width height , channels width height samples -> n samples")
	y_proj = einsum(fy,proj_directions,"n channels width height , channels width height samples -> n samples")

	# x_sorted = jnp.sort(x_proj,axis=1)
	# y_sorted = jnp.sort(y_proj,axis=1)
	x_sorted = x_proj
	y_sorted = y_proj

	return jnp.nan_to_num(jnp.abs(jnp.mean((x_sorted - y_sorted)**2,axis=-1)))


@jax.jit
def bhattacharyya_distance(x,y,key=None,where=None,aux=None,cache=None):
	"""Bhattacharyya distance between channel images, each normalised to unit L2 norm."""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.linalg.norm(x,axis=(-1,-2),keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.linalg.norm(y,axis=(-1,-2),keepdims=True)+eps)
	bc = jnp.sum(jnp.sqrt(x_norm*y_norm),axis=[-1,-2],keepdims=True,where=where)
	bc =-jnp.log(bc+eps)
	return jnp.nan_to_num(jnp.mean(bc,axis=[-1,-2,-3],where=where))

	# return -jnp.nan_to_num(jnp.log(bc + eps))

@jax.jit
def hellinger_distance(x,y,key=None,where=None,aux=None,cache=None):
	"""Hellinger distance between channel images, each normalised to unit L2 norm."""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.linalg.norm(x,axis=(-1,-2),keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.linalg.norm(y,axis=(-1,-2),keepdims=True)+eps)
	sqrt_diff = jnp.sqrt(x_norm) - jnp.sqrt(y_norm)
	H_bc = jnp.sqrt(jnp.sum(sqrt_diff**2,axis=[-1,-2],keepdims=True)) / jnp.sqrt(2) # Shape [N,CHANNELS,1,1]
	return jnp.nan_to_num(jnp.mean(H_bc,axis=[-1,-2,-3],where=where))

@jax.jit
def kl_divergence(x,y,key=None,where=None,aux=None,cache=None):
	"""KL divergence KL(x || y) between channel images, each normalised to sum to 1."""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.sum(x,axis=[-1,-2],keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.sum(y,axis=[-1,-2],keepdims=True)+eps)
	kl = jnp.sum(x_norm * jnp.log((x_norm + eps)/(y_norm + eps)),axis=[-1,-2],where=where,keepdims=True) # Shape [N C 1 1]
	
	return jnp.nan_to_num(jnp.mean(kl,axis=[-1,-2,-3],where=where))

@jax.jit
def average_amplitude_distance(x,y,key=None,where=None,aux=None,cache=None):
	"""Squared difference of the mean intensity of each channel.

	Ignores all spatial structure; useful alongside losses that rescale x and y.
	"""
	x_amp = jnp.mean(x,axis=[-1,-2],keepdims=True)
	y_amp = jnp.mean(y,axis=[-1,-2],keepdims=True)
	return jnp.nan_to_num(jnp.mean((x_amp - y_amp)**2,axis=[-1,-2,-3],where=where))


def masked_channel_correlations(values, spatial_mask=None, epsilon=1e-8):
	"""Return masked Pearson correlations and channel squared norms."""
	if spatial_mask is None:
		spatial_mask = jnp.ones(values.shape[-2:], dtype=values.dtype)
	spatial_mask = jnp.asarray(spatial_mask).reshape(values.shape[-2:]).astype(values.dtype)
	pixels = values.reshape(*values.shape[:-2], -1)
	mask = spatial_mask.reshape(-1)
	means = jnp.sum(pixels * mask, axis=-1) / jnp.maximum(mask.sum(), 1.0)
	centred = (pixels - means[..., None]) * mask
	covariance = jnp.einsum("...cp,...dp->...cd", centred, centred)
	squared_norms = jnp.sum(centred**2, axis=-1)
	denominator = jnp.sqrt(
		squared_norms[..., :, None] * squared_norms[..., None, :] + epsilon
	)
	return covariance / denominator, squared_norms


def channel_correlation_loss(x,y,key=None,where=None,aux=None,cache=None):
	"""L2 loss between masked within-image channel correlations."""
	aux = {} if aux is None else aux
	correlation_x, _ = masked_channel_correlations(
		x, aux.get("spatial_mask"), aux.get("epsilon", 1e-8)
	)
	correlation_y, target_norms = masked_channel_correlations(
		y, aux.get("spatial_mask"), aux.get("epsilon", 1e-8)
	)
	pairs = aux.get("pairs", tuple(zip(*jnp.triu_indices(x.shape[-3], 1))))
	pair_i = jnp.asarray([pair[0] for pair in pairs])
	pair_j = jnp.asarray([pair[1] for pair in pairs])
	weights = jnp.asarray(aux.get("pair_weights", jnp.ones(len(pairs))), dtype=x.dtype)
	active = jnp.ones(x.shape[:-3] + (x.shape[-3],), dtype=bool) if where is None else jnp.any(where, axis=(-1, -2))
	valid = active[..., pair_i] & active[..., pair_j]
	valid &= (target_norms[..., pair_i] > aux.get("epsilon", 1e-8)) & (target_norms[..., pair_j] > aux.get("epsilon", 1e-8))
	weights = weights * valid
	error = (correlation_x[..., pair_i, pair_j] - correlation_y[..., pair_i, pair_j]) ** 2
	return jnp.sum(weights * error, axis=-1) / jnp.maximum(jnp.sum(weights, axis=-1), 1.0)


def radial_profiles(values, spatial_mask=None, radial_bins=32, epsilon=1e-8):
	"""Return masked annular channel means and non-empty radial bins."""
	if radial_bins <= 0:
		raise ValueError("radial_bins must be positive")
	if spatial_mask is None:
		spatial_mask = jnp.ones(values.shape[-2:], dtype=bool)
	spatial_mask = jnp.asarray(spatial_mask).reshape(values.shape[-2:]).astype(bool)
	width, height = values.shape[-2:]
	grid_x, grid_y = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="ij")
	mask = spatial_mask.astype(values.dtype)
	centre_x = jnp.sum(grid_x * mask) / jnp.maximum(mask.sum(), 1.0)
	centre_y = jnp.sum(grid_y * mask) / jnp.maximum(mask.sum(), 1.0)
	radius = jnp.sqrt((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2)
	radius /= jnp.maximum(jnp.max(jnp.where(spatial_mask, radius, 0.0)), epsilon)
	indices = jnp.minimum((radius * radial_bins).astype(jnp.int32), radial_bins - 1)
	annuli = (indices[None] == jnp.arange(radial_bins)[:, None, None]) & spatial_mask
	counts = jnp.sum(annuli, axis=(-1, -2))
	profiles = jnp.einsum("...chw,rhw->...cr", values, annuli.astype(values.dtype))
	return profiles / jnp.maximum(counts, 1.0), counts > 0


def radial_profile_loss(x,y,key=None,where=None,aux=None,cache=None):
	"""L2 loss between masked per-channel radial intensity profiles."""
	aux = {} if aux is None else aux
	profiles_x, nonempty = radial_profiles(
		x, aux.get("spatial_mask"), aux.get("radial_bins", 16), aux.get("epsilon", 1e-8)
	)
	profiles_y, _ = radial_profiles(
		y, aux.get("spatial_mask"), aux.get("radial_bins", 16), aux.get("epsilon", 1e-8)
	)
	active = jnp.ones(x.shape[:-2], dtype=bool) if where is None else jnp.any(where, axis=(-1, -2))
	weights = jnp.asarray(aux.get("channel_weights", jnp.ones(x.shape[-3])), dtype=x.dtype)
	weights = active[..., :, None] * weights[..., None] * nonempty
	error = (profiles_x - profiles_y) ** 2
	return jnp.sum(weights * error, axis=(-1, -2)) / jnp.maximum(jnp.sum(weights, axis=(-1, -2)), 1.0)


@jax.jit
def spectral_no_phase(x,y,key=None,where=None,aux=None,cache=None):
	"""L2 distance between Fourier amplitudes (phase discarded)."""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	fx = jnp.abs(fx)
	fy = jnp.abs(fy)
	return l2(fx,fy,key,where=where)
        

@jax.jit
def spectral_only_phase(x,y,key=None,where=None,aux=None,cache=None):
	"""L2 distance between Fourier phases (spectra scaled to unit modulus)."""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	fx_phase = fx / (jnp.abs(fx)+1e-8)
	fy_phase = fy / (jnp.abs(fy)+1e-8)
	return jnp.nan_to_num(jnp.abs(l2(fx_phase,fy_phase,key,where=where)))


@jax.jit
def spectral(x,y,key=None,where=None,aux=None,cache=None):
	"""L2 distance between Fourier transforms."""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	return jnp.nan_to_num(jnp.abs(l2(fx,fy,key,where=where)))
