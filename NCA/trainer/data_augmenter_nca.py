import jax.numpy as jnp
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils
from Common.utils import key_pytree_gen
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import itertools

class DataAugmenter(DataAugmenterAbstract):
	
	def __init__(self,data_true,hidden_channels=0):
		"""
		Class for handling data augmentation for NCA training. 
		data_init is called before training,
		data_callback is called during training
		
		Also handles JAX array sharding, so all methods of NCA_trainer work
		on multi-gpu setups. Currently splits data onto different GPUs by batches


		Modified version of DataAugmenter where each batch can have different spatial resolution/size
		Treat data as Pytree of trajectories, where each leaf is a different batch f32[N,CHANNEL,WIDTH,HEIGHT]
		Parameters
		----------
		data_true : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			true un-augmented data
		hidden_channels : int optional
			number of hidden channels to zero-pad to data. Defaults to zero
		"""
		self.OBS_CHANNELS = data_true[0].shape[1]
		data_tree = []
		try:
			for i in range(data_true.shape[0]): # if data is provided as big array, convert to list of arrays. If data is list of arrays, this will leave it unchanged
				data_tree.append(data_true[i])
		except:
			data_tree = data_true
		data_true = jax.tree_util.tree_map(lambda x: jnp.pad(x,((0,0),(0,hidden_channels),(0,0),(0,0))),data_tree) # Pad zeros onto hidden channels


		self.data_true = data_true
		self.data_saved = data_true
		
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
		am=10
		
		
		if hasattr(self,"PREVIOUS_KEY"):
			x = self.unshift(x, am, self.PREVIOUS_KEY)
			y = self.unshift(y, am, self.PREVIOUS_KEY)

		x_true,_ =self.split_x_y(1)
				
		propagate_xn = lambda x:x.at[1:].set(x[:-1])
		reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
		
		x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
		x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
		
				
		for b in range(len(x)//2):
			x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
		
			
		
		# if hasattr(self, "PREVIOUS_KEY"):
		# 	key = jax.random.fold_in(self.PREVIOUS_KEY,i)
		# else:
		# 	key=jax.random.PRNGKey(int(time.time()))
		x = self.shift(x,am,key=key)
		y = self.shift(y,am,key=key)
		#print(x[0].shape)
		#print(len(x))
		if i < 1000:
			x = self.zero_random_circle(x,key=key)
		x = self.noise(x,0.005,key=key)

		#y = self.noise(y,0.01,key=jax.random.fold_in(key,2*i))
		self.PREVIOUS_KEY = key
		return x,y
		

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		