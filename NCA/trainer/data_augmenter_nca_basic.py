import equinox as eqx
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from NCA.trainer.data_augmenter import reinject_observations

class DataAugmenter(DataAugmenterAbstract):
	
	# def __init__(self,data_true,hidden_channels=0):
	# 	"""
	# 	Class for handling data augmentation for NCA training. 
	# 	data_init is called before training,
	# 	advance_pool is called during training
		
	# 	Also handles JAX array sharding, so all methods of NCA_trainer work
	# 	on multi-gpu setups. Currently splits data onto different GPUs by batches


	# 	Modified version of DataAugmenter where each batch can have different spatial resolution/size
	# 	Treat data as Pytree of trajectories, where each leaf is a different batch f32[N,CHANNEL,WIDTH,HEIGHT]
	# 	Parameters
	# 	----------
	# 	data_true : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
	# 		true un-augmented data
	# 	hidden_channels : int optional
	# 		number of hidden channels to zero-pad to data. Defaults to zero
	# 	"""
	# 	self.OBS_CHANNELS = data_true[0].shape[1]
	# 	data_tree = []
	# 	try:
	# 		for i in range(data_true.shape[0]): # if data is provided as big array, convert to list of arrays. If data is list of arrays, this will leave it unchanged
	# 			data_tree.append(data_true[i])
	# 	except:
	# 		data_tree = data_true
	# 	data_true = jax.tree_util.tree_map(lambda x: jnp.pad(x,((0,0),(0,hidden_channels),(0,0),(0,0))),data_tree) # Pad zeros onto hidden channels


	# 	self.data_true = data_true
	# 	self.data_saved = data_true
		
	def data_init(self,SHARDING = None):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		"""
		data = self.return_saved_data()	
		self.save_data(data)
		return None
	
		
	#@eqx.filter_jit
	def advance_pool(self,x,y,i,key):
		"""
		Called after every training iteration to perform data augmentation and processing		


		Parameters
		----------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Latent initial conditions used for previous training steps
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			True final states. Will be passed through unchanged here
		i : int
			Current training iteration - useful for scheduling mid-training data augmentation

		Returns
		-------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Latent initial conditions used for next training steps
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			True final states. Passed through unchanged here

		"""

		
		x_true,_ =self.split_x_y(1)
				
		x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS,key)

		#print(x[0].shape)
		#print(len(x))
		# if i < 10000:
		x = self.noise(x,0.005,key=key)

		#y = self.noise(y,0.01,key=jax.random.fold_in(key,2*i))
		self.PREVIOUS_KEY = key
		return x,y
		

@eqx.filter_jit
def jittable_callback_bit(x,x_true,OBS_CHANNELS,key):
	return reinject_observations(x, x_true, OBS_CHANNELS, key, fraction=0.5)
	
	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
