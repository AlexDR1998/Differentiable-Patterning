import jax.numpy as jnp
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils
from Common.utils import key_pytree_gen
import itertools
class DataAugmenterAbstract(object):
	
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
		
		OVERWRITE IN SUBCLASS
		"""
		data = self.return_saved_data()
		self.save_data(data)
		return None
	
	def data_load(self):	
		x0,y0 = self.split_x_y(1)
		x0,y0 = self.data_callback(x0,y0,0)
		return x0,y0
	
	def data_callback(self,x,y,i):
		"""
		Called after every training iteration to perform data augmentation and processing		

		OVERWRITE IN SUBCLASS
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

		return x,y
		
	@eqx.filter_jit
	def random_N_select(self,x,y,n,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Randomly sample n pairs of states from x and y

		Parameters
		----------
		x : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states
		n : int < N-N_steps
			How many batches to sample.

		Returns
		-------
		x_sampled : float32[BATCHES,n,CHANNELS,WIDTH,HEIGHT]
			sampled initial conditions
		y_sampled : float32[BATCHES,n,CHANNELS,WIDTH,HEIGHT]
			sampled final states.

		"""
		#print(x)
		ns = jax.random.choice(key,jnp.arange(x[0].shape[0]),shape=(n,),replace=False)
		x_sampled = jax.tree_util.tree_map(lambda data:data[ns],x)
		y_sampled = jax.tree_util.tree_map(lambda data:data[ns],y)
		return x_sampled,y_sampled

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
		x = jax.tree_util.tree_map(lambda data:data[:-N_steps],self.data_saved)
		y = jax.tree_util.tree_map(lambda data:data[N_steps:],self.data_saved)
		return x,y
	
	@eqx.filter_jit
	def pad(self,data,am):
		"""
		
		Pads spatial dimensions with zeros

		Parameters
		----------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			width to pad with zeros in spatial dimension

		Returns
		-------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH+2*am,HEIGHT+2*am]
			data padded with zeros

		"""
		return jax.tree_util.tree_map(lambda x,am:jnp.pad(x,((0,0),(0,0),(am,am),(am,am))),data,[am]*len(data))

	
	@eqx.filter_jit
	def shift(self,data,am,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Randomly shifts each trajectory. 

		Parameters
		----------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
			
		Returns
		-------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(len(data),2))
		for b in range(len(data)):
			data[b] = jnp.roll(data[b],shifts[b],axis=(-1,-2))
		return data

	@eqx.filter_jit
	def unshift(self,data,am,key):
		"""
		Randomly shifts each trajectory. If useing same key as shift(), it undoes that shift

		Parameters
		----------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			data to augment.
		am : int
			possible width to shift by in spatial dimension
		key : jax.random.PRNGKey
			Jax random number key.
			
		Returns
		-------
		data : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
			data randomly shifted in spatial dimensions

		"""

		shifts = jax.random.randint(key,minval=-am,maxval=am,shape=(len(data),2))
		for b in range(len(data)):
			data[b] = jnp.roll(data[b],-shifts[b],axis=(-1,-2))
		return data

	
	def noise(self,data,am,full=True,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Adds uniform noise to the data
		
		Parameters
		----------
		data : PyTree BATCHES [float32[N,CHANNELS,WIDTH,HEIGHT]]
			data to augment.
		am : float in (0,1)
			amount of noise, with 0 being none and 1 being pure noise
		full : boolean optional
			apply noise to observable channels, or all channels?. Defaults to True (all channels)
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		noisy : PyTree BATCHES [float32[N,CHANNELS,WIDTH,HEIGHT]]
			noisy data

		"""
		key_array = key_pytree_gen(key, [len(data)])
		#print(data[0].shape)
		#noisy = am*jax.random.uniform(key,shape=data.shape) + (1-am)*data
		noisy = jax.tree_util.tree_map(lambda x,key:am*jax.random.uniform(key,shape=x.shape) + (1-am)*x,data,key_array)
		
		if not full:
			noisy = jax.tree_util.tree_map(lambda x,y:x.at[:,self.OBS_CHANNELS:].set(y[:,self.OBS_CHANNELS:]),noisy,data)
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
		data : float32[B*BATCHES,N,CHANNELS,WIDTH,HEIGHT]
			data augmented along batch axis

		"""

		list_repeated = list(itertools.repeat(data,B))
		array_repeated = jax.tree_util.tree_map(lambda x:jnp.array(x),list_repeated)

		return jax.tree_util.tree_flatten(array_repeated)[0]
	
	def save_data(self,data):
		self.data_saved = data

	def return_saved_data(self):		
		return self.data_saved
	
	def return_true_data(self):
		return self.data_true
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		