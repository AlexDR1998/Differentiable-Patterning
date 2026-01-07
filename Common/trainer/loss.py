import jax.numpy as jnp
import jax
#from ott.geometry import pointcloud
#from ott.tools import sinkhorn_divergence
#from ott.problems.linear import linear_problem
#from ott.solvers.linear import sinkhorn
#from eqxvision.models import alexnet
#from eqxvision.utils import CLASSIFICATION_URLS
import equinox as eqx
from lpips_j.lpips import LPIPS
from einops import rearrange,reduce,einsum,repeat
import jax.random as jr
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch,pad_to_multiple_of_3_channels
#import eqxvision as eqv

#loaded_alexnet = alexnet(torch_weights=CLASSIFICATION_URLS['alexnet'])
#loaded_vgg11 = eqv.models.vgg11(torch_weights=CLASSIFICATION_URLS["vgg11"])
lpips = LPIPS()

@jax.jit
def cosine(x,y,key=None,where=None,aux=None):
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
def l2(x,y,key=None,where=None,aux=None):
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
def l1(x,y,key=None,where=None,aux=None):
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
def euclidean(x,y,key=None,where=None,aux=None):
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

	
	


@jax.jit
def random_sampled_euclidean(x,y,key,where=None,aux=16):
	SAMPLES = aux
	x_r = jnp.einsum("ncxy->cxyn",x)
	y_r = jnp.einsum("ncxy->cxyn",y)
	x_sub = jax.random.choice(key,x_r.reshape((-1,x_r.shape[-1])),(SAMPLES,),False)
	y_sub = jax.random.choice(key,y_r.reshape((-1,y_r.shape[-1])),(SAMPLES,),False)
	return jnp.nan_to_num(jnp.sqrt(jnp.mean((x_sub-y_sub)**2,axis=0)))


@jax.jit
def spectral(x,y,key=None,where=None,aux=None):
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
def spectral_weighted(x,y,key=None,where=None,aux=None):
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
@eqx.filter_jit
def vgg(x,y, key,where=None,aux=None):
	"""
	NOTE THAT CHANNELS IS TRUNCATED TO 3
	NOTE WHERE HAS NO EFFECT HERE

	Parameters
	----------
	x : float32 [N,CHANNELS,WIDTH,HEIGHT]
		predictions
	y : float32 [N,CHANNELS,WIDTH,HEIGHT]
		true data
	key : jax.random.PRNGKey
		Jax random number key. 

	Returns
	-------
	loss : float32 [N]

	"""
	x = rearrange(x,"n c x y->n x y c")[...,:3]
	y = rearrange(y,"n c x y->n x y c",)[...,:3]
	
	
	# L-pips expects inputs in the range [-1,1], but we almost always use data in the range [0,1]
	x = x*2-1
	y = y*2-1
	params = lpips.init(key, x, y)
	loss = lpips.apply(params, x, y)
	return loss
	

def vgg_hyperspectral_colony(x,y,key,where=None,aux=None):
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
	"""
	
	# Scale to [-1,1] for lpips
	x = x*2-1
	y = y*2-1

	# x has 8 channels but y has 11. Some specified x channels need to be repeated to match the channels in y
	# Apply where mask
	
	if where is not None:
		x = x*where.astype(x.dtype)
		where_y = duplicate_x_channels_9ch(where)
		y = y*where_y.astype(y.dtype)



	x = duplicate_x_channels_9ch(x)
	x = split_and_pad_by_experiment_groups_12ch(x)
	y = split_and_pad_by_experiment_groups_12ch(y)		
	x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
	y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)

	params = lpips.init(key, x[0], y[0])
	losses = jax.vmap(lpips.apply, in_axes=(None,0,0))(params, x, y) # C N () () ()
	# print("VGG losses shape: ",losses.shape,flush=True)
	# Weight different loss channels - some are duplicate channels from specifying colonies, others are dummy channels introduced by vgg groupings
	loss_weighting = jnp.array([0.5,1.0,0.5,1.0,1.0,1.0]) # Should there be an extra 1.0 here?
	losses = einsum(losses,loss_weighting,"c n i j k , c -> c n i j k")
	loss = reduce(losses,"c n () () () -> n","mean")
	return loss


def vgg_hyperspectral_colony_and_l2(x,y,key,where=None,aux=None):
	vgg_loss = vgg_hyperspectral_colony(x,y,key,where,aux)
	x_full = duplicate_x_channels_9ch(x)
	_l2 = (x_full-y)**2
	weighting = jnp.array([0.5,0.5,0.5,1.0,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0]) # Account for duplicate channels
	_l2 = einsum(_l2,weighting,"n c x y , c -> n c x y")
	where_full = duplicate_x_channels_9ch(where).astype(where.dtype)
	l2_loss = jnp.nan_to_num(jnp.mean(_l2,axis=[-1,-2,-3],where=where_full))
	return vgg_loss + l2_loss

def vgg_hyperspectral(x,y,key,where=None,aux=None):
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
	"""
	
	# Scale to [-1,1] for lpips
	x = x*2-1
	y = y*2-1
	# Apply where mask
	
	if where is not None:
		x = x*where.astype(x.dtype)
		y = y*where.astype(y.dtype)

	x = pad_to_multiple_of_3_channels(x)
	y = pad_to_multiple_of_3_channels(y)		
	x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
	y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)
	params = lpips.init(key, x[0], y[0])
	losses = jax.vmap(lpips.apply, in_axes=(None,0,0))(params, x, y) # C N () () ()
	loss = reduce(losses,"c n () () () -> n","mean")
	return loss
# @eqx.filter_jit
# def vgg_fast(x,y,params):
# 	x = rearrange(x,"n c x y->n x y c")[...,:3]
# 	y = rearrange(y,"n c x y->n x y c",)[...,:3]
# 	loss = lpips.apply(params, x, y)
# 	return loss


# def vgg_init_params(x,y, key):
# 	x = rearrange(x,"n c x y->n x y c")[...,:3]
# 	y = rearrange(y,"n c x y->n x y c",)[...,:3]
# 	return lpips.init(key, x, y)