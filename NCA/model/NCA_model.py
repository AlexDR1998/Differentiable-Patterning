import jax
import jax.numpy as jnp
import equinox as eqx
import time
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from Common.model.spatial_operators import Ops # Spatial stuff like gradients or laplacians
from einops import rearrange

class NCA(AbstractModel):
	layers: list
	KERNEL_STR: list
	N_CHANNELS: int
	N_FEATURES: int
	FIRE_RATE: float
	op: Ops
	perception: callable
	def __init__(self,
			     N_CHANNELS,
				 KERNEL_STR=["ID","LAP"],
				 ACTIVATION=jax.nn.relu,
				 PADDING="CIRCULAR",
				 FIRE_RATE=1.0,
				 KERNEL_SCALE = 1,
				 key=jax.random.PRNGKey(int(time.time()))):
		"""
		

		Parameters
		----------
		N_CHANNELS : int
			Number of channels for NCA.
		KERNEL_STR : [STR], optional
			List of strings corresponding to convolution kernels. Can include "ID","DIFF","LAP","AV", corresponding to
			identity, derivatives, laplacian and average respectively. The default is ["ID","LAP"].
		ACTIVATION_STR : str, optional
			Decide which activation function to use. The default is "relu".
		PERIODIC : Boolean, optional
			Decide whether to have periodic or fixed boundaries. The default is True.
		FIRE_RATE : float, optional
			Probability that each pixel updates at each timestep. Defuaults to 1, i.e. deterministic update
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		None.

		"""
		
		
		key1,key2 = jax.random.split(key,2)
		self.N_CHANNELS = N_CHANNELS
		self.FIRE_RATE = FIRE_RATE
		self.KERNEL_STR = KERNEL_STR
		N_WIDTH = 1
		self.op = Ops(PADDING=PADDING,dx=1,KERNEL_SCALE=KERNEL_SCALE,SMOOTHING=1.0)

		

		_kernel_length = len(KERNEL_STR)
		if "GRAD" in KERNEL_STR:
			_kernel_length+=1
		self.N_FEATURES = N_CHANNELS*_kernel_length*N_WIDTH
		
		#@eqx.filter_jit
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
		
		self.layers = [
			eqx.nn.Conv2d(in_channels=self.N_FEATURES,
						  out_channels=self.N_FEATURES,
						  kernel_size=1,
						  use_bias=False,
						  key=key1),
			ACTIVATION,
			eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
						  out_channels=self.N_CHANNELS,
						  kernel_size=1,
						  use_bias=True,
						  key=key2)
			]
		
		
		# Initialise final layer to zero
		w_zeros = jnp.zeros((self.N_CHANNELS,self.N_FEATURES,1,1))
		b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
		w_where = lambda l: l.weight
		b_where = lambda l: l.bias
		self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
		self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)

	def get_config(self):
		"""
		Returns the model configuration as a dictionary.

		Returns
		-------
		dict
			dictionary of model hyperparameters

		"""
		
		return {
			"MODEL":"NCA",
			"N_CHANNELS":self.N_CHANNELS,
			"KERNEL_STR":self.KERNEL_STR,
			"ACTIVATION":self.layers[1].__name__,
			"PADDING":self.op.PADDING,
			"FIRE_RATE":self.FIRE_RATE,
		}
		
	def __call__(self,
			  	 x: Float[Array,"{self.N_CHANNELS} x y"],
				 boundary_callback=lambda x:x,
				 key: Key=jax.random.PRNGKey(int(time.time())))->Float[Array, "{self.N_CHANNEL} x y"]:
		"""
		

		Parameters
		----------
		x : float32 [N_CHANNELS,_,_]
			input NCA lattice state.
		boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
			function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		x : float32 [N_CHANNELS,_,_]
			output NCA lattice state.

		"""
		
		dx = self.perception(x)
		for layer in self.layers:
			dx = layer(dx)
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		x_new = x + sigma*dx
		return boundary_callback(x_new)

	def call_with_activations(self,
					   x: Float[Array,"{self.N_CHANNELS} x y"],
					   boundary_callback=lambda x:x,
					   key: Key=jax.random.PRNGKey(int(time.time()))):
		"""
		Parameters
		----------
		x : float32 [N_CHANNELS,_,_]
			input NCA lattice state.
		boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
			function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		x : float32 [N_CHANNELS,_,_]
			output NCA lattice state.
		activations : list of float32 [N_FEATURES,_,_]
			list of activations from each layer.

		
		"""
		dx = self.perception(x)
		activations = [dx]
		for layer in self.layers:
			dx = layer(dx)
			activations.append(dx)
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		x_new = x + sigma*dx
		return boundary_callback(x_new),activations



	def call_with_SAE(self,
				   x: Float[Array,"{self.N_CHANNELS} x y"],
				   SAE,
				   latent_edit={"mode":"none",
								"positions":None,
								"values":1.0},
				   boundary_callback=lambda x:x,
				   key: Key=jax.random.PRNGKey(int(time.time()))):
		layer_name_dict = {"perception":0,"linear_hidden":1,"activation":2,"linear_output":3,"gate_func":4}
		#vSAE = jax.vmap(SAE,in_axes=0,out_axes=0)
		vENC = jax.vmap(SAE.encode,in_axes=0,out_axes=0)
		vDEC = jax.vmap(SAE.decode,in_axes=0,out_axes=0)
		vGET = jax.vmap(SAE.get_top_k_feature_positions,in_axes=0,out_axes=0)
		if latent_edit["mode"]=="mult_top_k":
			vEDIT = jax.vmap(lambda l:SAE.mult_k_top_features(l,latent_edit["positions"],latent_edit["values"]),in_axes=0,out_axes=0)

		if latent_edit["mode"]=="set_absolute":
			vEDIT = jax.vmap(lambda l:SAE.set_features_at_positions(l,latent_edit["positions"],latent_edit["values"]),in_axes=0,out_axes=0)

		#dx = self.perception(x)
		dx = x
		layers = [self.perception,*self.layers]
		for i,layer in enumerate(layers):
			
			dx = layer(dx)
			if i == layer_name_dict[SAE.TARGET_LAYER]:
				dx = rearrange(dx, "F X Y -> (X Y) F")
				latents = vENC(dx)
				top_latent_values,top_latent_positions = vGET(latents)
				if latent_edit["mode"] in ["mult_top_k","set_absolute"]:
					latents = vEDIT(latents)
				dx = vDEC(latents)
				dx = rearrange(dx, "(X Y) F -> F X Y",Y=x.shape[-1])
				latents = rearrange(latents,"(X Y) L-> L X Y",Y=x.shape[-1])
				top_latent_positions = rearrange(top_latent_positions,"(X Y) K-> K X Y",Y=x.shape[-1])
				top_latent_values = rearrange(top_latent_values,"(X Y) K-> K X Y",Y=x.shape[-1])
		sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
		x_new = x + sigma*dx
		return boundary_callback(x_new),latents,top_latent_values,top_latent_positions
	
	def set_weights(self,weights):
		w0,w1,b1 = weights
		w_where = lambda l: l.weight
		b_where = lambda l: l.bias

		self.layers[0] = eqx.tree_at(w_where,self.layers[0],w0)
		self.layers[2] = eqx.tree_at(w_where,self.layers[2],w1)
		self.layers[2] = eqx.tree_at(b_where,self.layers[2],b1)

	def get_weights(self):
		"""Returns list of arrays of weights, for plotting purposes, or for manually adjusting weights with
		code that doesn't `just work' on PyTrees

		Returns:
			weights : list of arrays of trainable parameters 
		"""
		
		
		diff_self,_ = self.partition()
		ws,tree_def = jax.tree_util.tree_flatten(diff_self)
		return list(map(jnp.squeeze,ws))
		#return ws,tree_def
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
			key: Key =jax.random.PRNGKey(int(time.time())))->Float[Array,"{iters} {self.N_CHANNELS} x y"]:
		
		trajectory = []
		trajectory.append(x)
		for i in range(iters):
			key = jax.random.fold_in(key,i)
			x = self(x,callback,key=key)
			trajectory.append(x)
		return jnp.array(trajectory)
		
