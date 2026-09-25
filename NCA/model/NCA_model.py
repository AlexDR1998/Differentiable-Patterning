import jax
import jax.numpy as jnp
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from Common.model.spatial_operators import Ops # Spatial stuff like gradients or laplacians
from einops import rearrange


def zero_conv(layer):
	"""Return a copy of a Conv2d layer with its weight (and bias) set to zero.

	Used on the last layer of an update network, so that a freshly built
	model starts by making no change to the state.
	"""
	layer = eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
	if layer.bias is not None:
		layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
	return layer


def gated_linear_unit(x):
	return jax.nn.glu(x, axis=0)


class NCA(AbstractModel):
	layers: list
	KERNEL_STR: list
	N_CHANNELS: int
	N_FEATURES: int
	FIRE_RATE: float
	op: Ops
	perception: callable # type: ignore
	# Flags are static so they are not written to saved .eqx files. This keeps
	# the saved layout identical to the old gNCA/nNCA/gnNCA classes.
	GATED: bool = eqx.field(static=True)
	PARAMETER_NOISE_LEVEL: float = eqx.field(static=True)

	def __init__(self,
			     N_CHANNELS,
				 KERNEL_STR=["ID","LAP"],
					 ACTIVATION=jax.nn.relu,
					 PADDING="CIRCULAR",
					 FIRE_RATE=1.0,
					 KERNEL_SCALE = 1,
					 key=None,
					 GATED=False,
					 PARAMETER_NOISE_LEVEL=0.0):
		"""
		

		Parameters
		----------
		N_CHANNELS : int
			Number of channels for NCA.
		KERNEL_STR : [STR], optional
			List of strings corresponding to convolution kernels. Can include "ID","DIFF","GRAD","LAP","AV", corresponding to
			identity, gradient norm, gradient, laplacian and average respectively. The default is ["ID","LAP"].
		ACTIVATION : callable, optional
			Activation function of the hidden layer. The default is relu.
		PADDING : str, optional
			Boundary padding used by the spatial kernels. The default is "CIRCULAR".
		FIRE_RATE : float, optional
			Probability that each pixel updates at each timestep. Defaults to 1, i.e. deterministic update
		KERNEL_SCALE : int, optional
			Radius of the spatial kernels. The default is 1.
		key : jax.random.PRNGKey, optional
			Jax random number key. Defaults to a key based on the current time.
		GATED : bool, optional
			If True, the last layer outputs 2*N_CHANNELS values and a gated linear unit
			combines them into the update (previously the gNCA model). The default is False.
		PARAMETER_NOISE_LEVEL : float, optional
			Standard deviation of Gaussian noise added to the network weights at every
			step (previously the nNCA model). 0 turns it off. The default is 0.

		Returns
		-------
		None.

		"""
		
		
		if key is None:
			key = jax.random.PRNGKey(int(time.time()))
		key1,key2 = jax.random.split(key,2)
		self.N_CHANNELS = N_CHANNELS
		self.FIRE_RATE = FIRE_RATE
		self.KERNEL_STR = KERNEL_STR
		self.GATED = GATED
		self.PARAMETER_NOISE_LEVEL = PARAMETER_NOISE_LEVEL
		N_WIDTH = 1
		self.op = Ops(PADDING=PADDING,dx=1,KERNEL_SCALE=KERNEL_SCALE,SMOOTHING=1)

		

		_kernel_length = len(KERNEL_STR)
		if "GRAD" in KERNEL_STR:
			_kernel_length+=1
		self.N_FEATURES = N_CHANNELS*_kernel_length*N_WIDTH
		
		def spatial_layer(X: Float[Array,"{self.N_CHANNELS} x y"])-> Float[Array, "H x y"]:
			output = []
			if "ID" in KERNEL_STR:
				output.append(X)
			if "DIFF" in KERNEL_STR:
				gradnorm = self.op.GradNorm(X)
				output.append(gradnorm)
			if "GRAD" in KERNEL_STR:
				grad = self.op.Grad(X)
				output.append(grad[0])
				output.append(grad[1])
			if "AV" in KERNEL_STR:
				output.append(self.op.Average(X))
			if "LAP" in KERNEL_STR:
				output.append(self.op.Lap(X))
			output = rearrange(output,"b C x y -> (b C) x y")
			return output
		self.perception = lambda x:spatial_layer(x)
		
		# The last layer starts at zero, so a new model makes no update
		out_channels = 2*self.N_CHANNELS if GATED else self.N_CHANNELS
		self.layers = [
			eqx.nn.Conv2d(in_channels=self.N_FEATURES,
						  out_channels=self.N_FEATURES,
						  kernel_size=1,
						  use_bias=False,
						  key=key1),
			ACTIVATION,
			zero_conv(eqx.nn.Conv2d(in_channels=self.N_FEATURES,
						  out_channels=out_channels,
						  kernel_size=1,
						  use_bias=True,
						  key=key2)),
			]
		if GATED:
			self.layers.append(gated_linear_unit)

	def get_config(self):
		"""
		Returns the model configuration as a dictionary.

		Returns
		-------
		dict
			dictionary of model hyperparameters

		"""
		name = "NCA"
		if self.PARAMETER_NOISE_LEVEL > 0:
			name = "nNCA"
		if self.GATED:
			name = "g" + name
		config = {
			"MODEL":name,
			"N_CHANNELS":self.N_CHANNELS,
			"KERNEL_STR":self.KERNEL_STR,
			"ACTIVATION":self.layers[1].__name__,
			"PADDING":self.op.PADDING,
			"FIRE_RATE":self.FIRE_RATE,
		}
		if self.PARAMETER_NOISE_LEVEL > 0:
			config["PARAMETER_NOISE_LEVEL"] = self.PARAMETER_NOISE_LEVEL
		return config

	def _noisy_layers(self, key):
		"""Return self.layers with Gaussian noise added to every weight and bias."""
		weights, rest = eqx.partition(self.layers, eqx.is_inexact_array)
		leaves, treedef = jax.tree_util.tree_flatten(weights)
		keys = jax.random.split(key, len(leaves))
		leaves = [w + self.PARAMETER_NOISE_LEVEL*jax.random.normal(k, w.shape, w.dtype) for w, k in zip(leaves, keys)]
		return eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), rest)
		
	def __call__(self,
				  	 x: Float[Array,"{self.N_CHANNELS} x y"],
					 boundary_callback=lambda x:x,
					 key=None)->Float[Array, "{self.N_CHANNEL} x y"]:
		"""
		

		Parameters
		----------
		x : float32 [N_CHANNELS,_,_]
			input NCA lattice state.
		boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
			function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
		key : jax.random.PRNGKey, optional
			Jax random number key, used for the fire mask and parameter noise. Defaults to a key based on the current time.

		Returns
		-------
		x : float32 [N_CHANNELS,_,_]
			output NCA lattice state.

		"""
		
		if key is None:
			key = jax.random.PRNGKey(int(time.time()))
		layers = self.layers
		if self.PARAMETER_NOISE_LEVEL > 0:
			# A separate key, so the noise is independent of the fire mask
			layers = self._noisy_layers(jax.random.split(key)[1])
		dx = self.perception(x)
		for layer in layers:
			dx = layer(dx)
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		x_new = x + sigma*dx
		return boundary_callback(x_new)

	def partition(self):
		"""
		Behaves like eqx.partition, but moves the hard coded kernels (a jax array) from the "trainable" pytree to the "static" pytree

		Returns
		-------
		diff : PyTree
			PyTree of same structure as NCA, with all non trainable parameters set to None
		static : PyTree
			PyTree of same structure as NCA, with all trainable parameters set to None

		"""
		
		total_diff,total_static = eqx.partition(self,eqx.is_inexact_array)
		ops_diff,ops_static = self.op.partition()
		where_ops = lambda m:m.op
		total_diff = eqx.tree_at(where_ops,total_diff,ops_diff)
		total_static = eqx.tree_at(where_ops,total_static,ops_static)
		return total_diff, total_static
		
	def run(self,
		    iters: Int[Scalar, ""],
			x: Float[Array, "{self.N_CHANNELS} x y"],
			callback=lambda x:x,
			key=None)->Float[Array,"{iters} {self.N_CHANNELS} x y"]:
		
		if key is None:
			key = jax.random.PRNGKey(int(time.time()))
		trajectory = []
		trajectory.append(x)

		for i in range(iters):
			key = jax.random.fold_in(key,i)
			x = self(x,callback,key=key)
			trajectory.append(x)
		return jnp.stack(trajectory)
		
