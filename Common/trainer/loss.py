from collections.abc import Sequence

import jax.numpy as jnp
import jax
#from ott.geometry import pointcloud
#from ott.tools import sinkhorn_divergence
#from ott.problems.linear import linear_problem
#from ott.solvers.linear import sinkhorn
#from eqxvision.models import alexnet
#from eqxvision.utils import CLASSIFICATION_URLS
import equinox as eqx
# from lpips_j.lpips import LPIPS
from jax.scipy.ndimage import map_coordinates
from einops import rearrange,reduce,einsum,repeat
import jax.random as jr
# from optax import l2_loss
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch,pad_to_multiple_of_3_channels
import Common.trainer.loss_ott as loss_ott
import Common.trainer.loss_vgg as loss_vgg
# import Common.trainer.loss_clip as loss_clip
#import eqxvision as eqv

#loaded_alexnet = alexnet(torch_weights=CLASSIFICATION_URLS['alexnet'])
#loaded_vgg11 = eqv.models.vgg11(torch_weights=CLASSIFICATION_URLS["vgg11"])


@jax.jit
def cosine(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
	"""
	return -jnp.nan_to_num(jnp.mean((x*y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)),axis=[-1,-2,-3],where=where))

@jax.jit
def l2(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
		"""
	
	return jnp.nan_to_num(jnp.mean((x-y)**2,axis=[-1,-2,-3],where=where))

@jax.jit
def l1(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
		"""
	return jnp.nan_to_num(jnp.mean(jnp.abs(x-y),axis=[-1,-2,-3],where=where))

@jax.jit
def euclidean(x,y,key=None,where=None,aux=None,cache=None):
	"""
		General format of loss functions here:

		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes

	"""
	return jnp.nan_to_num(jnp.sqrt(jnp.mean(((x-y)**2),axis=[-1,-2,-3],where=where)))

@eqx.filter_jit
def sliced_wasserstein_spatial(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Sliced Wasserstein distance in spatial domain

		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	
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
	"""
		Sliced Wasserstein distance across channels

		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	
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

	"""
		Sliced Wasserstein distance in spatial domain, using random rotations

		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	
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
	"""
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	
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
	# return loss_ott.spectral_wasserstein_projected(x,y,key,where,aux)
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
	"""
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
	"""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.linalg.norm(x,axis=(-1,-2),keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.linalg.norm(y,axis=(-1,-2),keepdims=True)+eps)
	bc = jnp.sum(jnp.sqrt(x_norm*y_norm),axis=[-1,-2],keepdims=True,where=where)
	bc =-jnp.log(bc+eps)
	print("loss shape before mean reduction:",bc.shape)
	return jnp.nan_to_num(jnp.mean(bc,axis=[-1,-2,-3],where=where))

	# return -jnp.nan_to_num(jnp.log(bc + eps))

@jax.jit
def hellinger_distance(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.linalg.norm(x,axis=(-1,-2),keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.linalg.norm(y,axis=(-1,-2),keepdims=True)+eps)
	sqrt_diff = jnp.sqrt(x_norm) - jnp.sqrt(y_norm)
	H_bc = jnp.sqrt(jnp.sum(sqrt_diff**2,axis=[-1,-2],keepdims=True)) / jnp.sqrt(2) # Shape [N,CHANNELS,1,1]
	print("loss shape before mean reduction:",H_bc.shape)
	return jnp.nan_to_num(jnp.mean(H_bc,axis=[-1,-2,-3],where=where))

@jax.jit
def kl_divergence(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	eps = 1e-6
	x_norm = (x+eps) / (jnp.sum(x,axis=[-1,-2],keepdims=True)+eps)
	y_norm = (y+eps) / (jnp.sum(y,axis=[-1,-2],keepdims=True)+eps)
	kl = jnp.sum(x_norm * jnp.log((x_norm + eps)/(y_norm + eps)),axis=[-1,-2],where=where,keepdims=True) # Shape [N C 1 1]
	
	return jnp.nan_to_num(jnp.mean(kl,axis=[-1,-2,-3],where=where))

@jax.jit
def average_amplitude_distance(x,y,key=None,where=None,aux=None,cache=None):
	"""
		Distance between average intensities of each channel and timestep. Removes all spatial information, 
		can be a useful auxiliary loss when combined with losses that re-normalise X and Y
	
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [N]
			loss reduced over channel and spatial axes
	"""
	x_amp = jnp.mean(x,axis=[-1,-2],keepdims=True)
	y_amp = jnp.mean(y,axis=[-1,-2],keepdims=True)
	return jnp.nan_to_num(jnp.mean((x_amp - y_amp)**2,axis=[-1,-2,-3],where=where))


@jax.jit
def random_sampled_euclidean(x,y,key,where=None,aux=16,cache=None):
	SAMPLES = aux
	x_r = jnp.einsum("ncxy->cxyn",x)
	y_r = jnp.einsum("ncxy->cxyn",y)
	x_sub = jax.random.choice(key,x_r.reshape((-1,x_r.shape[-1])),(SAMPLES,),False)
	y_sub = jax.random.choice(key,y_r.reshape((-1,y_r.shape[-1])),(SAMPLES,),False)
	return jnp.nan_to_num(jnp.sqrt(jnp.mean((x_sub-y_sub)**2,axis=0)))


@jax.jit
def spectral_no_phase(x,y,key=None,where=None,aux=None,cache=None):
	""" 
		l2 norm in fourier space (discarding phase information)

		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
	"""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	fx = jnp.abs(fx)
	fy = jnp.abs(fy)
	return l2(fx,fy,key,where=where)
        

@jax.jit
def spectral_only_phase(x,y,key=None,where=None,aux=None,cache=None):
	""" 
		l2 norm in fourier space, keeping only phase information.

		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
	"""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	fx_phase = fx / (jnp.abs(fx)+1e-8)
	fy_phase = fy / (jnp.abs(fy)+1e-8)
	return jnp.nan_to_num(jnp.abs(l2(fx_phase,fy_phase,key,where=where)))


@jax.jit
def spectral(x,y,key=None,where=None,aux=None,cache=None):
	""" 
		l2 norm in fourier space, keeping phase information.
		Weighted to emphasise importance of certain frequencies

		Parameters
		----------
		x : float32 [...,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [...,CHANNELS,WIDTH,HEIGHT]
			true data

		Returns
		-------
		loss : float32 array [...]
			loss reduced over channel and spatial axes
	"""
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	return jnp.nan_to_num(jnp.abs(l2(fx,fy,key,where=where)))



def vgg_hyperspectral_colony_and_l2(x,y,key,where,aux={"vgg_metric":"l2"},cache=None):
	vgg_loss = loss_vgg.vgg_hyperspectral_colony(x,y,key,where,aux,cache)
	_l2_loss = l2_colony_grouped(x,y,key,where,aux)
	return vgg_loss + _l2_loss


def l2_colony_grouped(x,y,key,where,aux=None,cache=None):
	
	x_full = duplicate_x_channels_9ch(x)
	_l2 = (x_full-y)**2
	weighting = jnp.array([0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0]) # Account for duplicate channels
	_l2 = einsum(_l2,weighting,"n c x y , c -> n c x y")
	where_full = duplicate_x_channels_9ch(where).astype(where.dtype)
	_l2_loss = jnp.nan_to_num(jnp.mean(_l2,axis=[-1,-2,-3],where=where_full))
	return _l2_loss


def build_loss_initialiser(loss_strings,loss_args):
	"""
		For VGG based losses, we want to pre-compute the target features once, and also initialise the model parameters once,
		and cache them for later computation. This makes things faster and more efficient.
	"""
	
	_vgg_aux = {
		"vgg_metric":loss_args["metric"] if "metric" in loss_args else "l2",
		"internal_loss_func":loss_args["internal_loss_func"] if "internal_loss_func" in loss_args else None,
		"epsilon":loss_args["epsilon"] if "epsilon" in loss_args else None,
		"tau":loss_args["tau"] if "tau" in loss_args else None,
		"normalize":loss_args["normalize"] if "normalize" in loss_args else None,
		"samples":loss_args["samples"] if "samples" in loss_args else None,
	}
	LOSS_FUNC_INITS = {
		"vgg":lambda y,key,where:loss_vgg.precompute_vgg_hyperspectral_target(y,key,where,aux=_vgg_aux),
		"vgg_grouped":lambda y,key,where:loss_vgg.precompute_vgg_hyperspectral_colony_target(y,key,where,aux=_vgg_aux),
		"vgg_grouped_and_l2":lambda y,key,where:loss_vgg.precompute_vgg_hyperspectral_colony_target(y,key,where,aux=_vgg_aux),
	}
	if isinstance(loss_strings,str):
		loss_strings = [loss_strings]
	
	if "vgg_grouped_and_l2" in loss_strings:
		return LOSS_FUNC_INITS["vgg_grouped_and_l2"]
	elif "vgg_grouped" in loss_strings:
		return LOSS_FUNC_INITS["vgg_grouped"]
	elif "vgg" in loss_strings:
		return LOSS_FUNC_INITS["vgg"]
	else:
		return None


def build_loss_functions(loss_strings,loss_args):
	"""
		Builds a list of loss functions based on the specified loss strings.
		If loss_string is a single string, returns a list with one loss function.
		If loss_string is a list of strings, returns a list of loss functions in the same order.



		Parameters
		----------
		loss_strings : str or list of str
			Loss function name(s) to build. Must be keys in the LOSS_FUNCS dictionary.
		loss_args : dict
			Dictionary of additional arguments for certain loss functions.
		Returns
		-------
		loss_funcs : list of functions
			List of loss functions corresponding to the input loss_strings.
	"""


	_ott_aux = {
		"D":loss_args["D"] if "D" in loss_args else None,
		"S":loss_args["S"] if "S" in loss_args else None,
		"K":loss_args["K"] if "K" in loss_args else None,
		"sharpen":loss_args["sharpen"] if "sharpen" in loss_args else False,
		"epsilon":loss_args["epsilon"] if "epsilon" in loss_args else None,
		"internal_loss_func":loss_args["internal_loss_func"] if "internal_loss_func" in loss_args else None,
	}
	_emd_aux = {
		"epsilon":loss_args["epsilon"] if "epsilon" in loss_args else None,
		"internal_loss_func":loss_args["internal_loss_func"] if "internal_loss_func" in loss_args else None,
		"normalize":loss_args["normalize"] if "normalize" in loss_args else None,
		"tau":loss_args["tau"] if "tau" in loss_args else None,
		"amplitude_penalty":loss_args["amplitude_penalty"] if "amplitude_penalty" in loss_args else False
	}

	_vgg_aux = {
		"vgg_metric":loss_args["metric"] if "metric" in loss_args else "l2",
		"internal_loss_func":loss_args["internal_loss_func"] if "internal_loss_func" in loss_args else None,
		"epsilon":loss_args["epsilon"] if "epsilon" in loss_args else None,
		"tau":loss_args["tau"] if "tau" in loss_args else None,
		"normalize":loss_args["normalize"] if "normalize" in loss_args else None,
		"samples":loss_args["samples"] if "samples" in loss_args else None,
		"vgg_params":loss_args["vgg_params"] if "vgg_params" in loss_args else None,
		"random_crop":loss_args["random_crop"] if "random_crop" in loss_args else False,
		"random_channel_shuffle":loss_args["random_channel_shuffle"] if "random_channel_shuffle" in loss_args else False,
		# "target_feats":loss_args["target_feats"] if "target_feats" in loss_args else None,
	}
	# _vision_extractor = None
	# for lstr in loss_strings:
	# 	if "clip" in lstr:
	# 		_vision_extractor = loss_clip.build_clip_vision_extractor() # Only actually load this if using clip loss, as it is quite big
	# 		break

	# _clip_aux = {
	# 	"clip_metric":loss_args["metric"] if "metric" in loss_args else "l2",
	# 	"normalize":loss_args["normalize"] if "normalize" in loss_args else None,
	# 	"vision_extractor":_vision_extractor
	# }

	LOSS_FUNCS = {
		"l2":l2,
		"l2_grouped":l2_colony_grouped,
		"l1":l1,
		"vgg":lambda x,y,key,where,cache:loss_vgg.vgg_hyperspectral(x,y,key,where,aux=_vgg_aux,cache=cache),
		"vgg_grouped":lambda x,y,key,where,cache:loss_vgg.vgg_hyperspectral_colony(x,y,key,where,aux=_vgg_aux,cache=cache),
		# "vgg_3ch":lambda x,y,key,where:loss_vgg.vgg(x,y,key,where,aux=_vgg_aux),
		"vgg_grouped_and_l2":lambda x,y,key,where,cache:vgg_hyperspectral_colony_and_l2(x,y,key,where,aux=_vgg_aux,cache=cache),
		# "clip_3ch":lambda x,y,key,where:loss_clip.clip_loss_3ch(x,y,key,where,aux=_clip_aux),
		# "clip_grouped":lambda x,y,key,where:loss_clip.clip_loss_colony(x,y,key,where,aux=_clip_aux),
		# "clip":lambda x,y,key,where:loss_clip.clip_loss_hyperspectral(x,y,key,where,aux=_clip_aux),
		# "clip_grouped_and_l2":lambda x,y,key,where:loss_clip.clip_loss_colony_and_l2(x,y,key,where,aux=_clip_aux),
		"euclidean":euclidean,
		"cosine":cosine,
		"spectral":spectral,
		"spectral_no_phase":spectral_no_phase,
		"spectral_phase":spectral_only_phase,
		"sliced_wasserstein_spatial":lambda x,y,key,where,cache:sliced_wasserstein_spatial(x,y,key,where,aux={"samples":loss_args["samples"]},cache=cache),
		"sliced_wasserstein_channel":lambda x,y,key,where,cache:sliced_wasserstein_channel(x,y,key,where,aux={"samples":loss_args["samples"]},cache=cache),
		"sliced_wasserstein_full":lambda x,y,key,where,cache:wasserstein_projected(x,y,key,where,aux={"samples":loss_args["samples"]},cache=cache),
		"sliced_wasserstein_rotational":lambda x,y,key,where,cache:sliced_wasserstein_rotational(x,y,key,where,aux={"samples":loss_args["samples"]},cache=cache),
		"spectral_wasserstein_full":lambda x,y,key,where,cache:spectral_wasserstein_projected(x,y,key,where,aux={"samples":loss_args["samples"]},cache=cache),
		"bhattacharyya":bhattacharyya_distance,
		"kl_divergence":kl_divergence,
		"hellinger":hellinger_distance,
		"average_amplitude":average_amplitude_distance,
		"ott":lambda x,y,key,where,cache:loss_ott.ott_loss(x,y,key,where,aux=_ott_aux),
		"ott_chstack":lambda x,y,key,where,cache:loss_ott.ott_channel_stack_loss(x,y,key,where,aux=_ott_aux),
		"ott_grouped":lambda x,y,key,where,cache:loss_ott.ott_grouped_loss(x,y,key,where,aux=_ott_aux),
		"ott_grouped_and_l2":lambda x,y,key,where,cache:loss_ott.ott_grouped_and_l2_loss(x,y,key,where,aux=_ott_aux),
		"emd_loss":lambda x,y,key,where,cache:loss_ott.emd_loss(x,y,key,where,aux=_emd_aux),
		
	}
	if isinstance(loss_strings,str):
		loss_funcs = [LOSS_FUNCS[loss_strings]]
	elif isinstance(loss_strings,Sequence):
		# loss_strings = list(loss_strings)
		loss_funcs = [LOSS_FUNCS[f] for f in loss_strings]
	else:
		raise ValueError("loss_strings must be a string or sequence of strings. Got {}".format(type(loss_strings)))
		
	return loss_funcs
