import jax
import time
import jax.numpy as np
import jax.random as jr
from jaxtyping import Float, Int, PyTree, Scalar, Array
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from einops import repeat

class DataAugmenter(DataAugmenterAbstract):
	"""
		Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
	"""
	def __init__(self,Ts,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.CALLBACK_PARAMS = {"INCREMENT_PROB":0.1,
						  		"RESET_PROB":0.001,
								"OVERWRITE_OBS_CHANNELS":False,
								"NOISE_FRACTION":0.01}
		B = len(self.data_saved)
		Ts = repeat(Ts,"T -> B T",B=B)
		self.Ts = list(Ts)

	def data_init(self,SHARDING=None):
		return None
	
	
	
	def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Splits data into x (initial conditions) and y (following sub trajectory of length L).
		So that x[:,i]->y[:,i+1:i+1+L] is learned

		Parameters
		----------
		L : Int
			Length of subdirectory

		Returns
		-------
		x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
			Following sub trajectories

		"""
		
		
		pos = list(jax.random.randint(key,shape=(len(self.data_saved),),minval=0,maxval=self.data_true[0].shape[0]-L-1))
		x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
		y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+L],self.data_saved,pos)
		
		return x,y
		
		
	def data_callback(self,
				   	  x: PyTree[Float[Array, "1 C W H"]], 
					  y: PyTree[Float[Array, "{L} C W H"]],
					  ts: Float[Array, "Batches {L}"],
					  i: Int[Scalar, ""],
					  L: Int[Scalar, ""],
					  key):
		
		""" Preserves intermediate hidden channels after each training iteration

		Returns:
			x : PyTree [BATCHES] f32[1,CHANNELS,WIDTH,HEIGHT]
				Initial conditions
			y : PyTree [BATCHES] f32[L,CHANNELS,WIDTH,HEIGHT]
				True trajectories that follow the initial conditions
		"""
		keys = jr.split(key,4)
		data = self.return_saved_data()
		ts = self.Ts
		B = len(data)
		N = data[0].shape[0]
		C = self.OBS_CHANNELS


		# y input contains important hidden channel information that we want to preserve during training
		#if hasattr(self,"SUBTRAJECTORY_LOCATION"):
		pos = self.SUBTRAJECTORY_LOCATION
		
		# Update position counters
		# with 
		#    probability p=0.1, increment each counter by 1, 
		#	 probability 1-p keep each counter the same,
		#    probability q=0.01, reset each counter to zero
		# If a counter reaches the end of the data, reset it to zero
		input_y_length = y[0].shape[0]
		
		reset_counters = jr.bernoulli(keys[0],p=self.CALLBACK_PARAMS["RESET_PROB"],shape=(B,)) # With very small probability at each step, reset to 0
		increment_counters = jr.bernoulli(keys[1],p=self.CALLBACK_PARAMS["INCREMENT_PROB"],shape=(B,)).astype(int) # With small probability at each step, advance to next step
		_pos_incremented = np.clip(np.array(self.SUBTRAJECTORY_LOCATION)+increment_counters,min=0,max=N-L-1)
		_pos_at_end = _pos_incremented == N-L-1
		reset_counters = np.logical_or(reset_counters,_pos_at_end)
		#_pos_reset = jax.random.randint(key,shape=(B,),minval=0,maxval=N-L-1)
		_pos_reset = np.zeros(shape=(B,)).astype(int)
		pos = np.where(reset_counters,_pos_reset,_pos_incremented)
		pos = list(pos)
		
		propagate_hidden_channels = lambda data,y,p: data.at[p+1:p+1+input_y_length,C:].set(y[:,C:])
		data = jax.tree_util.tree_map(propagate_hidden_channels,data,y,pos)
		
		# propagate_hidden_channels = lambda data,y,p,inc: data.at[p+1:p+1+L,C:].set(y[:,C:])*inc + data*(1-inc) 
		# data = jax.tree_util.tree_map(propagate_hidden_channels,data,y,pos,list(increment_counters))
		
		if self.CALLBACK_PARAMS["OVERWRITE_OBS_CHANNELS"]:
			for b in range(len(data)//2):
				data[b*2] = data[b*2].at[pos[b*2]+1:pos[b*2]+1+input_y_length,:C].set(y[b*2][:,:C])*increment_counters[b*2] + data[b*2]*increment_counters[b*2]

		self.save_data(data)


		# Sample new x,y and ts
		output_y_length = jr.randint(keys[2],shape=(1,),minval=1,maxval=L)[0]
		x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
		# TODO make sure noise never makes x negative
		x = self.noise(x,am=self.CALLBACK_PARAMS["NOISE_FRACTION"],mode="full",key=keys[3])
		y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+output_y_length],self.data_saved,pos)
		ts = jax.tree_util.tree_map(lambda data,p:data[p:p+1+output_y_length],ts,pos)
		#ts = self.Ts[:,pos:pos+L]
		
		self.SUBTRAJECTORY_LOCATION = pos
		return x,y,ts

		
	def data_load(self,L,key):
		data = self.return_saved_data()
		#pos = list(jax.random.randint(key,shape=(len(data),),minval=0,maxval=data[0].shape[0]-L-1))
		pos = list(np.zeros(len(data)).astype(int))
		x0 = jax.tree_util.tree_map(lambda d,p:d[p],data,pos)
		y0 = jax.tree_util.tree_map(lambda d,p:d[p+1:p+1+L],data,pos)
		ts = jax.tree_util.tree_map(lambda data,p:data[p:p+1+L],self.Ts,pos)
		#ts = self.Ts[:,pos:pos+L]
		self.SUBTRAJECTORY_LOCATION = pos
		return x0,y0,ts

		
	def initial_condition_loss(self, model, x0, args):
		
		t = args["t"]
		loss_func = args["loss_func"]
		_,y_true = self.split_x_y()
		y_true = jax.tree_util.tree_map(lambda y:y[:t],y_true)
		
		
		#print(f"Inner loop batch number: {len(x0)}")
		#print(f"Inner loop X shape: {x0[0].shape}")
		
		v_pde = lambda x:model(np.linspace(0,t,t+1),x)[1][1:] # Don't need to vmap over N
		vv_pde= lambda x: jax.tree_util.tree_map(v_pde,x) # different data batches can have different sizes
		y_pred = vv_pde(x0)
		v_loss_func = lambda x,y: np.array(jax.tree_util.tree_map(loss_func,x,y))
		loss = v_loss_func(y_true,y_pred)
		return np.mean(loss)
		#return 0
		