import jax.numpy as jnp
import jax
import time
import jax.random as jr
import equinox as eqx
from jax.experimental import mesh_utils
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract as DataAugmenterAbstractTree
from jaxtyping import Array, Float, PyTree, Scalar, Int, Key
from Common.utils import key_array_gen
from einops import repeat

class DataAugmenterAbstract(DataAugmenterAbstractTree):
	
	def __init__(self,data_true,hidden_channels=0):
		"""
		Class for handling data augmentation for NCA training. 
		data_init is called before training,
		data_callback is called during training
		
		Also handles JAX array sharding, so all methods of NCA_trainer work
		on multi-gpu setups. Currently splits data onto different GPUs by batches

		Parameters
		----------
		data_true : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			true un-augmented data
		hidden_channels : int optional
			number of hidden channels to zero-pad to data. Defaults to zero
		"""
		self.OBS_CHANNELS = data_true.shape[2]
		
		data_true = jnp.pad(data_true,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0))) # Pad zeros onto hidden channels
		
		self.data_true = data_true
		self.data_saved = data_true

	def data_init(self,SHARDING = None):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		"""
		data = self.return_saved_data()
		if SHARDING is not None:
			data = self.duplicate_batches(data, SHARDING)
			shard = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((SHARDING,1,1,1,1)))
			data = jax.device_put(data,shard)
			jax.debug.visualize_array_sharding(data[:,0,0,0])

		
		self.save_data(data)
		return None
	
		
	def split_x_y(self,N_steps=1):
		"""
		Splits data into x (initial conditions) and y (final states). 
		Offset by N_steps in N, so x[:,N]->y[:,N+N_steps] is learned

		Parameters
		----------
		N_steps : int, optional
			How many steps along data trajectory to learn update rule for. The default is 1.

		Returns
		-------
		x : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
		return self.data_saved[:,:-N_steps],self.data_saved[:,N_steps:]
	
	
	
	@eqx.filter_jit
	def pad(self,data: Float[Array, "B N C W H"],am):
		"""
		
		Pads spatial dimensions with zeros

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			width to pad with zeros in spatial dimension

		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH+2*am,HEIGHT+2*am]
			data padded with zeros

		"""
		return jnp.pad(data,((0,0),(0,0),(0,0),(am,am),(am,am)))
	
	#@eqx.filter_jit
	def shift(self,data: Float[Array, "B N C W H"],am,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Randomly shifts each trajectory. 

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
			
		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

			
		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(data.shape[1],2))
		
		for b in range(data.shape[1]):
			data = data.at[b].set(jnp.roll(data[b],shifts[b],axis=(-1,-2)))
		return data
	
	def unshift(self,data: Float[Array, "B N C W H"],am,key):
		"""
		Randomly shifts each trajectory. If useing same key as shift(), it undoes that shift

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey
			Jax random number key.
			
		Returns
		-------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

			
		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(data.shape[1],2))
		
		for b in range(data.shape[1]):
			data = data.at[b].set(jnp.roll(data[b],-shifts[b],axis=(-1,-2)))
		return data
	
	
	
	
	def noise(self,
		   	  data: Float[Array, "B N C W H"],
			  am,
			  mode="full",
			  key=jax.random.PRNGKey(int(time.time()))):
		"""
		Adds uniform noise to the data
		
		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : float in (0,1)
			amount of noise, with 0 being none and 1 being pure noise
		full : boolean optional
			apply noise to observable channels, or all channels?. Defaults to True (all channels)
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		noisy : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			noisy data

		"""
		noisy = am*jax.random.uniform(key,shape=data.shape) + (1-am)*data

		if mode=="observable": # Overwrite correct data onto hidden channels
			noisy = noisy.at[:,:,self.OBS_CHANNELS:,:,:].set(data[:,:,self.OBS_CHANNELS:,:,:])
		elif mode=="hidden": # Overwrite correct data onto observable channels
			noisy = noisy.at[:,:,:self.OBS_CHANNELS,:,:].set(data[:,:,:self.OBS_CHANNELS,:,:])
		
		return noisy
		
	@eqx.filter_jit
	def duplicate_batches(self,data,B):
		"""
		Repeats data along batches axis by B

		Parameters
		----------
		data : float32[BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		B : int
			number of repetitions

		Returns
		-------
		data : float32[N,B*BATCHES,CHANNELS,WIDTH,HEIGHT]
			data augmented along batch axis

		"""
		
		return jnp.repeat(data,B,axis=0)
	
	def zero_random_circle(self, data: Float[Array, "B N C W H"], key: Key=jr.PRNGKey(int(time.time()))):
		
		B = data.shape[0]
		N = data.shape[1]
		def _zero_random_circle(image, key):
			# Get image dimensions
			height = image.shape[-1]
			width = image.shape[-2]

			# Generate random numbers for circle parameters
			key, subkey1, subkey2, subkey3 = jr.split(key, 4)
			center_x = jr.randint(subkey1, (), 0, width)
			center_y = jr.randint(subkey2, (), 0, height)
			max_radius = jnp.min(jnp.array([center_x, width - center_x, center_y, height - center_y]))
			radius = jr.randint(subkey3, (), 1, (max_radius + 1)/2)

			Y, X = jnp.ogrid[:height, :width]
			
			# Create the mask for the circle
			mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
			mask = repeat(mask,"H W -> () H W")
			image = jnp.where(mask, 0, image)

			return image


		keys = key_array_gen(key, (B,N))

		# Vectorize the zero_random_circle function to apply it to each image in the batch
		v_zeromap = jax.vmap(_zero_random_circle, in_axes=(0, 0))
		vv_zeromap = jax.vmap(v_zeromap, in_axes=(0, 0))

		# Apply the function to the batch of images
		modified_images = vv_zeromap(data, keys)

		return modified_images

# Example usage:
	
	def save_data(self,data):
		self.data_saved = data

	def return_saved_data(self):		
		return self.data_saved
	
	def return_true_data(self):
		return self.data_true
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		