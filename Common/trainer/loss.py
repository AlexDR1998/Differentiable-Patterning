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
from einops import rearrange,reduce
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

# @jax.jit
# def sinkhorn_divergence_loss(x,y):
# 	"""
# 		Sinkhorn loss - OT distance between 2 point clouds in 2D space

# 		Parameters
# 		----------
# 		x : float32 [N_x,2]
# 			predictions
# 		y : float32 [N_y,2]
# 			true data

# 		Returns
# 		-------
# 		loss : float32 
# 			loss 

# 	"""


# 	geom = pointcloud.PointCloud(x,y)
# 	ot = sinkhorn_divergence.sinkhorn_divergence(
# 		geom,
# 		x=geom.x,
# 		y=geom.y,
# 		static_b=True,
# 	)
# 	return ot.divergence
# 	# ot = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
# 	# return ot.reg_ot_cost
	
	


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
	

def _split_and_pad_by_experiment_groups(x): 
	"""
		For VGG hyperspectral loss, sometimes we need to define which channels are aggregated together, as we compare corresponding blocks of 3 channels.
		TODO: This needs to be pure and able to be jitted. We probably need to pass this function in with the data
	"""
	# Ensure channels and experiment groups are sorted by experiment group 
	# channel_order_inds = jnp.argsort(experiment_groups)
	# experiment_groups = experiment_groups[channel_order_inds]
	# x = x[:,channel_order_inds]
	# # print("Experiment groups after sorting: ",experiment_groups,flush=True)
	# # Find indices to split at
	# experiment_groups = jnp.array([0,0,0,0,1,2,2,2])  # Hardcoded for jitting
	# diff = jnp.diff(experiment_groups)	
	# indices_to_split_at = jnp.where(diff != 0)[0] + 1
	indices_to_split_at = jnp.array([4,5])
	# Split and pad each block of channels
	# x_split = jnp.split(ary=x, indices_or_sections=indices_to_split_at, axis=1)
	x_split = [x[:,0:4],x[:,4:5],x[:,5:8]]
	x_split = [jnp.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0))) for x in x_split]

	# Recombine
	x = jnp.concatenate(x_split,axis=1)
	return x

def vgg_hyperspectral(x,y,key,where=None,aux=None):
	"""

		Takes x and y with > 3 channels and computes VGG loss on each 3-channel subset, averaging the result.
		Parameters
		----------
		x : float32 [N,CHANNELS,WIDTH,HEIGHT]
			predictions
		y : float32 [N,CHANNELS,WIDTH,HEIGHT]
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
	# experiment_groups = aux
	# Scale to [-1,1] for lpips
	x = x*2-1
	y = y*2-1

	# Apply where mask
	if where is not None:
		x = x*where.astype(x.dtype)
		y = y*where.astype(y.dtype)

	# if experiment_groups is None:
	# 	# First pad with zeros to make number of channels a multiple of 3
	# 	x = jnp.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0)))
	# 	y = jnp.pad(y,((0,0),(0,(3-y.shape[1]%3)%3),(0,0),(0,0)))
	# 	x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
	# 	y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)

	# else:
		# Split and pad by experiment groups
	x = _split_and_pad_by_experiment_groups(x)
	y = _split_and_pad_by_experiment_groups(y)		
	x = rearrange(x,"n (c vc) x y -> c n x y vc",vc=3)
	y = rearrange(y,"n (c vc) x y -> c n x y vc",vc=3)

	params = lpips.init(key, x[0], y[0])
	loss = reduce(jax.vmap(lpips.apply, in_axes=(None,0,0))(params, x, y),"c n () () () -> n","mean")
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