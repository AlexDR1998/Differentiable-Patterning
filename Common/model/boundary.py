import jax.numpy as jnp
import jax
import equinox as eqx
from einops import einsum,rearrange
from jaxtyping import Array, Float, Int, Key, Scalar

class model_boundary(object):
	"""
		Callable object that forces intermediate NCA states to be fixed to boundary condition at specified channels
	"""
	
	
	def __init__(self,mask = None):
		"""
		Parameters
		----------
		mask : float32 [MASK_CHANNELS,WIDTH,HEIGHT]
			array encoding structure or boundary conditions for NCA intermediate states
		Returns
		-------
		None.

		"""

		self.MASK = mask
		
	@eqx.filter_jit	
	def __call__(self,x):

		m_channels = self.MASK.shape[0]
		x_masked = x.at[-m_channels:].set(self.MASK)
		return x_masked
	


class no_boundary(object):
	"""
		Callable object that does not enforce any boundary conditions on intermediate NCA states
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
			array encoding structure or boundary conditions for NCA intermediate states
		Returns
		-------
		None.

		"""
		self.MASK = rearrange(mask,"() H W -> H W")
	
	@eqx.filter_jit
	def __call__(self,x):
		#if self.MASK is None:
		#	return x
		
		x_masked = einsum(x,self.MASK,'... C H W, H W -> ... C H W')
		return x_masked
	



class trainable_boundary(eqx.Module):
	""" 
		Callable object that forces intermediate NCA states to be fixed to boundary condition at specified channels.
		Mask is trainable.
	"""
	mask: Array
	limit_function: callable
	m_channels: Int
	def __init__(self,mask = None,limit_function = jax.nn.sigmoid):
		"""
		Parameters
		----------
		mask : float32 [MASK_CHANNELS,WIDTH,HEIGHT]
			array encoding structure or boundary conditions for NCA intermediate states
		limit_function : callable, optional
			function that constrains mask values. The default is jax.nn.sigmoid.		
		Returns
		-------
		None.

		"""
		self.mask = mask
		self.limit_function = limit_function# Force mask to be constrained between 0 and 1
		self.m_channels = mask.shape[0]

	def __call__(self,x):
		x_masked = x.at[-self.m_channels:].set(self.limit_function(self.mask))
		return x_masked	

	def coverage(self):
		return jnp.sum(self.limit_function(self.mask))
	
	def get_mask(self):
		return self.limit_function(self.mask)