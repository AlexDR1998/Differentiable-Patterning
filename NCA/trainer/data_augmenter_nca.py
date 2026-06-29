import jax.numpy as jnp
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils
from Common.utils import key_pytree_gen
# from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as DataAugmenterBasic
from NCA.trainer.data_augmenter_nca_basic import jittable_callback_bit
import itertools

class DataAugmenter(DataAugmenterBasic):
	

		
	def data_init(self,SHARDING = None):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		"""
		data = self.return_saved_data()
		if SHARDING is not None:
			# For Pytree version we have to shard over time axis?
			data = self.duplicate_batches(data, SHARDING)
			data = self.pad(data,10)
			shard = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((SHARDING,1,1,1,1)))
			data = jax.device_put(data,shard)
			jax.debug.visualize_array_sharding(data[:,0,0,0])
		else:	
			data = self.duplicate_batches(data, 4)
			data = self.pad(data, 10)

		
		self.save_data(data)
		return None
	
		
	#@eqx.filter_jit
	def data_callback(self,x,y,i,key):
		"""
		Called after every training iteration to perform data augmentation and processing		


		Parameters
		----------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states
		i : int
			Current training iteration - useful for scheduling mid-training data augmentation

		Returns
		-------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
		am=2
		
		
		if hasattr(self,"PREVIOUS_KEY"):
			x = self.unshift(x, am, self.PREVIOUS_KEY)
			y = self.unshift(y, am, self.PREVIOUS_KEY)

		x_true,_ =self.split_x_y(1)
				
		x = jittable_callback_bit(
			x,
			x_true,
			self.OBS_CHANNELS,
			jax.random.fold_in(key, 0),
		)

		x = self.shift(x,am,key=key)
		y = self.shift(y,am,key=key)
		#print(x[0].shape)
		#print(len(x))
		# if i < 10000:
		x = self.zero_random_circle(x,key=key)
		x = self.noise(x,0.005,key=key)

		#y = self.noise(y,0.01,key=jax.random.fold_in(key,2*i))
		self.PREVIOUS_KEY = key
		return x,y
		


		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
