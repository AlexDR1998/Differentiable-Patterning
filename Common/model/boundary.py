import jax.numpy as jnp
import jax
import equinox as eqx
from einops import einsum,rearrange
from jaxtyping import Array, Float, Int, Key, Scalar

class model_boundary(object):
	"""
		Overwrite the last channels of an NCA state with a fixed mask (boundary condition)
	"""
	
	
	def __init__(self,mask = None):
		"""
		Parameters
		----------
		mask : float32 [MASK_CHANNELS,WIDTH,HEIGHT]
			array encoding structure or boundary conditions for NCA intermediate states
		"""
		assert len(mask.shape) == 3, "Mask should be of shape [MASK_CHANNELS,WIDTH,HEIGHT]"
		self.MASK = mask
		
	@eqx.filter_jit	
	def __call__(self,x):

		m_channels = self.MASK.shape[0]
		x_masked = x.at[-m_channels:].set(self.MASK)
		return x_masked
	


class no_boundary(object):
	"""
		No boundary condition: returns the state unchanged
	"""
	def __init__(self):
		return None
	
	def __call__(self,x):
		return x

class hard_boundary(object):
	def __init__(self,mask = None):
		"""
		Parameters
		----------
		mask : float32 [1,WIDTH,HEIGHT]
			multiplies every channel of the NCA state
		"""
		self.MASK = rearrange(mask,"() H W -> H W")
	
	@eqx.filter_jit
	def __call__(self,x):
		#if self.MASK is None:
		#	return x
		
		x_masked = einsum(x,self.MASK,'... C H W, H W -> ... C H W')
		return x_masked


BOUNDARY_MODES = ("soft", "uniform", "hard")


def build_boundary_callback(mode, mask):
	"""
		Boundary callback applied to the NCA state after each step.

		soft:    the mask is written into the last state channel(s)
		uniform: as soft, but that channel is 1 everywhere, so the model is
		         given no geometry (ablation of the adhesion mask)
		hard:    every channel is multiplied by the mask
	"""
	if mode == "soft":
		return model_boundary(mask)
	if mode == "uniform":
		return model_boundary(jnp.ones_like(mask))
	if mode == "hard":
		return hard_boundary(mask)
	raise ValueError(f"boundary mode must be one of {BOUNDARY_MODES}, got {mode!r}")
