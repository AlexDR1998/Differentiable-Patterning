import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from functools import partial
from NCA.trainer.tensorboard_log import NCA_Train_log, kaNCA_Train_log, mNCA_Train_log, aNCA_Train_log
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_multihead_attention import aNCA
from NCA.trainer.data_augmenter_nca import DataAugmenter
from einops import repeat, reduce, rearrange, einsum
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary, hard_boundary, no_boundary
from tqdm import tqdm
from jaxtyping import Float,Array,Key
import time

class NCA_Trainer(object):
	"""
	General class for training NCA model to data trajectories
	"""
	
	def __init__(self,
			     NCA_model,
				 data,
				 model_filename=None,
				 DATA_AUGMENTER = DataAugmenter,
				 BOUNDARY_MASK = None, 
				 BOUNDARY_MODE = "soft", # "soft" or "hard"
				 SHARDING = None, 
				 GRAD_LOSS = True,
				 OBS_CHANNELS = None,
				 DATA_CHANNELS = None,
				 LOSS_TIME_CHANNEL_MASK = None,
				 MODEL_DIRECTORY="models/",
				 LOG_DIRECTORY="logs/"):
		"""
		

		Parameters
		----------
		
		NCA_model : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
			the NCA object to train
			
		data : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
			set of trajectories to train NCA on
		
		model_filename : str, optional
			name of directories to save tensorboard log and model parameters to.
			log at :	'logs/gradient_tape/model_filename/train'
			model at : 	'models/model_filename'
			if None, sets model_filename to current time
		
		DATA_AUGMENTER : object, optional
			DataAugmenter object. Has data_init and data_callback methods that can be re-written as needed. The default is DataAugmenter.
		BOUNDARY_MASK : float32 [N_BOUNDARY_CHANNELS,WIDTH,HEIGHT], optional
			Set of channels to keep fixed, encoding boundary conditions. The default is None.
		SHARDING : int, optional
			How many parallel GPUs to shard data across?. The default is None.
		
		directory : str
			Name of directory where all models get stored, defaults to 'models/'

		Returns
		-------
		None.

		"""
		self.NCA_model = NCA_model
		
		# Set up variables 
		self.CHANNELS = self.NCA_model.N_CHANNELS
		if OBS_CHANNELS is None:
			self.OBS_CHANNELS = data[0].shape[1]
		else:
			self.OBS_CHANNELS = OBS_CHANNELS
		# For some loss functions, the NCA observable channels don't necessarily match the data channels. Handle this here.
		if DATA_CHANNELS is None:
			self.DATA_CHANNELS = self.OBS_CHANNELS
		else:
			self.DATA_CHANNELS = DATA_CHANNELS
		
		
		self.SHARDING = SHARDING
		self.GRAD_LOSS = GRAD_LOSS
		self.LOSS_TIME_CHANNEL_MASK = LOSS_TIME_CHANNEL_MASK
		
		# Set up partial mask of channels / timesteps
		if self.LOSS_TIME_CHANNEL_MASK is None:
			self.LOSS_TIME_CHANNEL_MASK = jnp.ones((data.shape[1]-1,self.OBS_CHANNELS),dtype=jnp.float32)

		_model_kernel_length = len(self.NCA_model.KERNEL_STR)
		if "GRAD" in self.NCA_model.KERNEL_STR:
			_model_kernel_length+=1
		if GRAD_LOSS:
			self.LOSS_TIME_CHANNEL_MASK = repeat(self.LOSS_TIME_CHANNEL_MASK,"n c -> n (gc c) () ()",gc=_model_kernel_length)
			print("Timestep / Channel mask: ")
			print(self.LOSS_TIME_CHANNEL_MASK[:,:,0,0])
		else:
			self.LOSS_TIME_CHANNEL_MASK = rearrange(self.LOSS_TIME_CHANNEL_MASK,"n c -> n c () ()")
			print("Timestep / Channel mask: ")
			print(self.LOSS_TIME_CHANNEL_MASK[:,:,0,0])


		# Set up data and data augmenter class
		self._data_raw = data
		self.DATA_AUGMENTER = DATA_AUGMENTER(data,self.CHANNELS-self.DATA_CHANNELS)
		self.DATA_AUGMENTER.data_init(self.SHARDING)
		self.data = self.DATA_AUGMENTER.return_saved_data()
		self.BATCHES = len(self.data)
		print("Batches = "+str(self.BATCHES))
		# Set up boundary augmenter class
		# length of BOUNDARY_MASK PyTree should be same as number of batches
		
		self.BOUNDARY_CALLBACK = []
		for b in range(self.BATCHES):
			if BOUNDARY_MASK is not None:
				if BOUNDARY_MODE=="soft":
					self.BOUNDARY_CALLBACK.append(model_boundary(BOUNDARY_MASK[b]))
				elif BOUNDARY_MODE=="hard":
					self.BOUNDARY_CALLBACK.append(hard_boundary(BOUNDARY_MASK[b]))
			else:
				self.BOUNDARY_CALLBACK.append(no_boundary())
		
		self._LOG_DIRECTORY = LOG_DIRECTORY
		self._MODEL_DIRECTORY = MODEL_DIRECTORY
		self.model_filename = model_filename
		#print(jax.tree_util.tree_structure(self.BOUNDARY_CALLBACK))
		
	def setup_logging(self,BACKEND,wandb_args):
		# Set logging behvaiour based on provided filename
		print(f"Raw data shape: {jnp.array(self._data_raw).shape}")
		if self.model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			if BACKEND=="tensorboard":
				self.IS_LOGGING = True
				self.LOG_DIR = self._LOG_DIRECTORY+self.model_filename+"/train"
				if isinstance(self.NCA_model ,kaNCA):
					self.LOGGER = kaNCA_Train_log(self.LOG_DIR,self._data_raw)
				elif isinstance(self.NCA_model , mNCA):
					self.LOGGER = mNCA_Train_log(self.LOG_DIR,self._data_raw)
				elif isinstance(self.NCA_model , aNCA):
					self.LOGGER = aNCA_Train_log(self.LOG_DIR,self._data_raw)
				else:
					self.LOGGER = NCA_Train_log(self.LOG_DIR, self._data_raw)
				print("Logging training to: "+self.LOG_DIR)
			elif BACKEND=="wandb":
				self.IS_LOGGING = True
				self.LOG_DIR = self._LOG_DIRECTORY+self.model_filename+"/train"
				config = {"MODEL":self.NCA_model.get_config(),
			  			 "TRAINING":self.TRAIN_CONFIG}
				wandb_args["config"] = config
				
				if isinstance(self.NCA_model ,kaNCA):
					self.LOGGER = kaNCA_Train_log(data=self._data_raw,wandb_config=wandb_args)
				elif isinstance(self.NCA_model , mNCA):
					self.LOGGER = mNCA_Train_log(data=self._data_raw,wandb_config=wandb_args)
				elif isinstance(self.NCA_model , aNCA):
					self.LOGGER = aNCA_Train_log(data=self._data_raw,wandb_config=wandb_args)
				else:
					self.LOGGER = NCA_Train_log(data=self._data_raw,wandb_config=wandb_args)
				print("Logging training to: "+self.LOG_DIR)
		self.MODEL_PATH = self._MODEL_DIRECTORY+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
		
	@eqx.filter_jit	
	def loss_func(self,
			   	  x:Float[Array, "N CHANNELS x y"],  # noqa: F722
				  y:Float[Array, "N CHANNELS x y"],  # noqa: F722
				  key: Key)->Float[Array, "N"]:
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		key : jr.PRNGKey
			Jax random number key. Only useful for loss functions that are stochastic (i.e. subsampled).
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		x_obs = x[:,:self.OBS_CHANNELS]
		y_obs = y[:,:self.DATA_CHANNELS]
		if self.GRAD_LOSS:
			v_perception = jax.vmap(self.NCA_model.perception,in_axes=0,out_axes=0)
			x_obs = v_perception(x_obs)
			y_obs = v_perception(y_obs)
			x_obs = x_obs.at[:,self.OBS_CHANNELS:].set(0.1*x_obs[:,self.OBS_CHANNELS:])
			y_obs = y_obs.at[:,self.DATA_CHANNELS:].set(0.1*y_obs[:,self.DATA_CHANNELS:])
		# return self._loss_func(x_obs,y_obs,key,self.LOSS_TIME_CHANNEL_MASK)
		# if self.LOSS_FUNC_CHANNELS is not None:
		losses = []
		for idx, f in enumerate(self._loss_func):
			key = jr.fold_in(key,idx)
			# Get mask for channels that should be included in this loss function
			# Include channels where LOSS_FUNC_CHANNELS == idx or == -1
			channel_mask = (self.LOSS_FUNC_CHANNELS == idx) | (self.LOSS_FUNC_CHANNELS == -1)
			channel_mask = repeat(channel_mask,"c -> (gc c) () ()",gc=self.LOSS_TIME_CHANNEL_MASK.shape[1]//self.OBS_CHANNELS).astype(jnp.float32)
			# Select only the relevant channels
			
			loss_mask = einsum(self.LOSS_TIME_CHANNEL_MASK,channel_mask,"n c w h, c w h-> n c w h")
			# loss_aux = self.LOSS_FUNC_AUX[idx]
			# losses.append(f(x_obs, y_obs, key, loss_mask, loss_aux))
			losses.append(f(x_obs, y_obs, key, loss_mask))

		losses = jnp.array(losses)
	
	
	
		return reduce(losses,"loss_funcs N -> N","mean")
	@eqx.filter_jit
	def intermediate_reg(self,x,x_new,vv_nca,key):
		"""
		Intermediate state regulariser - tracks how much of x is outwith [0,1]
		
		NOTE: IS NOW TREE-MAPPED OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		full : boolean
			Flag for whether to only regularise observable channel (true) or all channels (false)
		Returns
		-------
		reg : float
			float tracking how much of x is outwith range [0,1]

		"""
		def _reg(x_new,full=True):
			# if not full:
				# x = x[:,:self.OBS_CHANNELS]
			return jnp.mean(jnp.abs(x_new)+jnp.abs(x_new-1)-1)
		return jnp.array(jtu.tree_map(_reg,x_new))
			# v_intermediate_reg = lambda x:jnp.array(jax.tree_util.tree_map(self.intermediate_reg,x))  # noqa: E731
		

	def boundary_regulariser(self,x,x_new,vv_nca,key):
		"""
		Penalise the model for any nonzero components outside the boundary mask
		Parameters
		----------
		x : float32 PyTree [BATCH] Array [,N,CHANNELS,_,_]
			NCA state
		Returns
		-------
		reg : float32 PyTree [BATCH]
		
		"""
		x_in_bound = jax.tree_util.tree_map(lambda f,x:f(x),self.BOUNDARY_CALLBACK,x_new)
		x_out_bound = jax.tree_util.tree_map(lambda x,y: x-y,x_new,x_in_bound)
		return jnp.array(jax.tree_util.tree_map(lambda x: jnp.mean(jnp.abs(x)),x_out_bound))
	@eqx.filter_jit
	def contiguous_growth_regulariser(self,x,x_new,vv_nca,key):
		"""
		Contiguous state regulariser. For the observable channels, penalises any growth of those channels that occurs more than
		N cells out from the current block of high cells. Intended to stop regions of cells growing seemingly out of nowhere.

		NOTE: VMAP THIS OVER BATCHES

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		x_previous : float32 array [N,CHANNELS,_,_]
			Previous NCA state
		Returns
		-------
		reg : float
			float tracking how much of growth of x in observable channels occurs outwith the bounding region of high observable cells in x_previous 

		"""
		def _reg(x_new,x):
			x_new = x_new[:,:self.OBS_CHANNELS]
			x = x[:,:self.OBS_CHANNELS]
			dx = jax.nn.relu(x_new - x) # How much obs growth
			# kernel = jnp.array([[1,1,1],[1,1,1],[1,1,1]],dtype=jnp.float32)
			kernel = jnp.ones((3,3),dtype=jnp.float32)
			kernel = repeat(kernel,"w h -> O I w h",O=1,I=self.OBS_CHANNELS)
			dilation = jax.lax.conv_general_dilated(
				lhs=x,
				rhs=kernel,
				window_strides=(1, 1),
				padding="SAME",
			)
			dilation = 1 - jax.nn.sigmoid((dilation-5.0)*10.0)
			dilation = repeat(dilation,"N () w h -> N C w h",C=self.OBS_CHANNELS)
			err = jnp.mean(dilation*dx)
			return err
		return jnp.array(jtu.tree_map(_reg,x_new,x))

	def update_sensitivity_regulariser(self,x,x_new,vv_nca,key):
		"""
		Measures NCA update step sensitivity to small changes in inputs. Computes a second update step with a small amount of noise added to the input.
		Minimized by NCA model that is insensitive to small changes in input.
	
		Parameters
		----------
			x: PyTree [Batch] of Arrays [N C H W]
			x_new: PyTree [Batch] of Arrays [N C H W]
			vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
			key: Jax PRNGkey
		Returns:
			Sensitivity: List [Batch] of floats
		"""

		noise_amount = 0.1
		key_array_noise = key_pytree_gen(key,[len(x)])
		x_noise = jtu.tree_map(lambda x,key:x+noise_amount*jr.normal(key,shape=x.shape),x,key_array_noise) # x with gaussian noise added
		key_array_nca = key_pytree_gen(key,(len(x),x[0].shape[0]))
		x_new_noise = vv_nca(x_noise,self.BOUNDARY_CALLBACK,key_array_nca)
		diffs = jtu.tree_map(lambda x,x_noise,x_new,x_new_noise:jnp.mean(jnp.abs(x_new-x_new_noise)/(jnp.abs(x-x_noise)+1e-8)),x,x_noise,x_new,x_new_noise)

		return jnp.array(diffs)
	
	def perturbation_conservation_regulariser(self,x,x_new,vv_nca,key):
		"""
		Measures NCA update step sensitivity to small changes in inputs. Computes a second update step with a small amount of noise added to the input.
		Minimized by NCA model that is linearly proportional to small changes in input. I.e. if input is changed by dx, output should change by ~dx


		Parameters
		----------
			x: PyTree [Batch] of Arrays [N C H W]
			x_new: PyTree [Batch] of Arrays [N C H W]
			vv_nca: Callable PyTree [Batch] of Arrays [N C H W], Callable, KeyArray -> PyTree [Batch] of Arrays [N C H W]
			key: Jax PRNGkey
		Returns:
			Loss: List [Batch] of floats
		"""
		noise_amount = 0.1
		key_array_noise = key_pytree_gen(key,[len(x)])
		x_noise = jtu.tree_map(lambda x,key:x+noise_amount*jr.normal(key,shape=x.shape),x,key_array_noise) # x with gaussian noise added
		key_array_nca = key_pytree_gen(key,(len(x),x[0].shape[0]))
		x_new_noise = vv_nca(x_noise,self.BOUNDARY_CALLBACK,key_array_nca)

		diffs = jtu.tree_map(lambda x,x_noise,x_new,x_new_noise:jnp.mean(jnp.abs(x_new-x_new_noise)-(jnp.abs(x-x_noise))),x,x_noise,x_new,x_new_noise)
		return jnp.array(diffs)

	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  REGULARISER_COEFFS = {
				  "intermediate_state":1.0,
				  "boundary": 0.0,
				  "contiguous_growth":1.0,
				  "update_sensitivity":0.0,
				  "perturbation_conservation":0.0
				},
			  WARMUP=64,
			  LOG_EVERY=40,
			  CLEAR_CACHE_EVERY=100,
			  WRITE_IMAGES=True,
			  LOSS_FUNC_STR = ["euclidean"],
			  LOSS_ARGS = {
				"channels":None,
				"experiment_groups":None
			  },			  
			  LOOP_AUTODIFF = "checkpointed",
			  SPARSE_PRUNING = False,
			  TARGET_SPARSITY = 0.5,
			  wandb_args={"project":"NCA",
				 		  "group":"group_1",
				 		  "tags":["training"]},
			  key=jr.PRNGKey(int(time.time()))):
		"""
		Perform t steps of NCA on x, compare output to y, compute loss and gradients of loss wrt model parameters, and update parameters.

		Parameters
		----------
		t : int
			number of NCA timesteps between x[N] and x[N+1]
		iters : int
			number of training iterations
		optimiser : optax.GradientTransformation
			the optax optimiser to use when applying gradient updates to model parameters.
			if None, constructs adamw with exponential learning rate schedule
		REGULARISERS : dict optional
			Strengths of various intermediate state regularisers. Defaults to 1.0
		WARMUP : int optional
			Number of iterations to wait for until starting model checkpointing
		LOG_EVERY : int optional
			Save output of model every LOG_EVERY steps
		WRITE_IMAGES : boolean
			Save images during logging
		LOSS_FUNC_STR : string
			Which loss function to use
		LOOP_AUTODIFF : string 
			How to save gradients through loop over timesteps. "checkpointed" or "lax"
		SPARSE_PRUNING : boolean
			Whether to prune model weights to a target sparsity
		TARGET_SPARSITY : float
			Target sparsity for model pruning - [0,1]
		key : jr.PRNGKey, optional
			Jax random number key. The default is jr.PRNGKey(int(time.time())).
		Returns
		-------

		None
		"""

		self.TRAIN_CONFIG = {
			"t":t,
			"iters":iters,
			"optimiser":optimiser,
			"REGULARISERS":REGULARISER_COEFFS,
			"WARMUP":WARMUP,
			"LOG_EVERY":LOG_EVERY,
			"CLEAR_CACHE_EVERY":CLEAR_CACHE_EVERY,
			"WRITE_IMAGES":WRITE_IMAGES,
			"LOSS_FUNC_STR":LOSS_FUNC_STR,
			"LOOP_AUTODIFF":LOOP_AUTODIFF,
			"SPARSE_PRUNING":SPARSE_PRUNING,
			"TARGET_SPARSITY":TARGET_SPARSITY
		}
		
		self.setup_logging("wandb",wandb_args=wandb_args)


		LOSS_FUNCS = {
			"l2":loss.l2,
			"l1":loss.l1,
			"vgg":loss.vgg_hyperspectral,#lambda x,y,key,where:loss.vgg_hyperspectral(x,y,key,where,experiment_groups=LOSS_ARGS["experiment_groups"]),
			"vgg_grouped":loss.vgg_hyperspectral_colony,
			"vgg_grouped_and_l2":loss.vgg_hyperspectral_colony_and_l2,
			"vgg_3ch":loss.vgg,
			"euclidean":loss.euclidean,
			"spectral":loss.spectral,
			"spectral_full":loss.spectral_weighted,
			# "rand_euclidean":lambda x,y,key:loss.random_sampled_euclidean(x,y,key=key)
		}

		if isinstance(LOSS_FUNC_STR,str):
			self._loss_func = [LOSS_FUNCS[LOSS_FUNC_STR]]
		elif isinstance(LOSS_FUNC_STR,list):
			self._loss_func = [LOSS_FUNCS[f] for f in LOSS_FUNC_STR]
			
		
		LOSS_FUNC_CHANNELS = LOSS_ARGS["channels"]
		if LOSS_FUNC_CHANNELS is not None:
			assert len(LOSS_FUNC_CHANNELS)==self.OBS_CHANNELS, "LOSS_FUNC_CHANNELS should be same length as number of observable channels"
		elif LOSS_FUNC_CHANNELS is None:
			LOSS_FUNC_CHANNELS = jnp.ones((self.OBS_CHANNELS,),dtype=jnp.int32)*-1
		self.LOSS_FUNC_CHANNELS = LOSS_FUNC_CHANNELS
		
		REG_FUNCS = {
			"intermediate_state":self.intermediate_reg,
			"boundary":self.boundary_regulariser,
			"contiguous_growth":self.contiguous_growth_regulariser,
			"update_sensitivity":self.update_sensitivity_regulariser,
			"perturbation_conservation":self.perturbation_conservation_regulariser
		}
		

		# Filter REG_FUNCS to the same set (optional but keeps things consistent)
		REGULARISER_COEFFS = {name:REGULARISER_COEFFS[name] for name in REGULARISER_COEFFS.keys() if REGULARISER_COEFFS[name]!=0.0}
		REG_FUNCS = {name: REG_FUNCS[name] for name in REGULARISER_COEFFS.keys()}
		#@partial(eqx.filter_jit,donate="all-except-first")
		@eqx.filter_jit
		def make_step(nca,x,y,t,opt_state,key):
			"""
			

			Parameters
			----------
			nca : object callable - (float32 [N_CHANNELS,_,_],PRNGKey) -> (float32 [N_CHANNELS,_,_])
				the NCA object to train
			x : float32 array [BATCHES,N,CHANNELS,_,_]
				NCA state
			y : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				true data
			t : int
				number of NCA timesteps between x[N] and x[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jr.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
				the NCA object with updated parameters
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [BATCHES,N,CHANNELS,_,_], float32 array [BATCHES,N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated NCA state after t iterations

			"""

			def apply_intermediate_regs(reg_logs,x,x_new,vv_nca,key):
				for name in REGULARISER_COEFFS.keys():
					reg_logs[name]+=REG_FUNCS[name](x,x_new,vv_nca,key)
				return reg_logs
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff,nca_static,x,y,t,key):
				# Gradient and values of loss function computed here
				_nca = eqx.combine(nca_diff,nca_static)
				v_nca = jax.vmap(_nca,in_axes=(0,None,0),out_axes=0,axis_name="N") # boundary is independant of time N
				vv_nca = lambda x,callback,key_array:jax.tree_util.tree_map(v_nca,x,callback,key_array)  # noqa: E731
				reg_logs_internal = {name: jnp.zeros(len(x)) for name in REGULARISER_COEFFS.keys()}
				_loss_func = lambda x,y,key:self.loss_func(x,y,key)  # noqa: E731
				v_loss_func = lambda x,y,key_array:jnp.array(jax.tree_util.tree_map(_loss_func,x,y,key_array))  # noqa: E731
				
				# Structuring this as function and lax.scan speeds up jit compile a lot
				def nca_step(carry,j): # function of type a,b -> a
					key,x,reg_logs_internal = carry
					# Apply NCA update step
					key = jr.fold_in(key,j)
					key_array = key_pytree_gen(key,(len(x),x[0].shape[0]))
					x_new = vv_nca(x,self.BOUNDARY_CALLBACK,key_array)
					reg_logs_internal = apply_intermediate_regs(reg_logs_internal,x,x_new,vv_nca,key)

					return (key,x_new,reg_logs_internal),None
				(
					key,
					x,
					reg_logs_internal
				),_ = eqx.internal.scan(
					nca_step,
					(
						key,
						x,
						reg_logs_internal
					),
					xs=jnp.arange(t),
					kind=LOOP_AUTODIFF
				)

				loss_key = key_pytree_gen(key, (len(x),))
				losses = v_loss_func(x, y, loss_key)
				reg_loss_internal = {name: REGULARISER_COEFFS[name]*jnp.mean(reg_logs_internal[name])/t for name in REGULARISER_COEFFS.keys()}
				mean_loss = jnp.mean(losses) + jnp.sum(jnp.array(list(reg_loss_internal.values())))
				return mean_loss,(x,losses,reg_loss_internal)

			nca_diff,nca_static = nca.partition()
			loss_x,grads = compute_loss(nca_diff,nca_static,x,y,t,key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			(mean_loss,(x,losses,reg_loss)) = loss_x
			return nca,x,y,t,opt_state,key,mean_loss,losses,reg_loss

		nca = self.NCA_model
		nca_diff,nca_static = nca.partition()
		

		#--- OPTIMISER ---
		# Set up optimiser
		if optimiser is None:
			schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
			self.OPTIMISER = optax.nadam(schedule)
			
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(nca_diff)
		
		# # Split data into x and y
		x,y = self.DATA_AUGMENTER.data_load(key)
		print(f"Initial x shape: {jnp.array(x).shape}, y shape: {jnp.array(y).shape}",flush=True)
		
		
		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		loss_diff = 0
		#prev_loss = 0
		mean_loss = 0
		loss_diff_thresh = 1e-2
		error = 0
		error_at = 0
		SPARSITY = jnp.concat((jnp.zeros(WARMUP),jnp.linspace(0,TARGET_SPARSITY,iters-WARMUP)))

		pbar = tqdm(range(iters))
		#--- Do training run ---
		for i in pbar:
			#prev_loss = mean_loss
			if i%CLEAR_CACHE_EVERY==0:
				#print(f"Clearing cache at step {i}")
				jax.clear_caches()
			key = jr.fold_in(key,i)

			#nca,opt_state,(mean_loss,(x,losses)) = make_step(nca, x, y, t, opt_state,key)
			nca,x_new,y_new,t,opt_state,key,mean_loss,losses,reg_loss = make_step(nca, x, y, t, opt_state,key)
			loss_diff = mean_loss - best_loss


			reg_loss["loss"] = mean_loss
			reg_loss["best_loss"] = best_loss
			pbar.set_postfix(reg_loss)

			if SPARSE_PRUNING:
				
				if i>WARMUP:

					ws,_ = nca.get_weights()
					sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=SPARSITY[i])
					pruner = jaxpruner.MagnitudePruning(
						sparsity_distribution_fn=sparsity_distribution,
						skip_gradients=True)
					ws = pruner.instant_sparsify(ws)[0]
					nca.set_weights(ws)

			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, reg_loss, x_new, i, nca,write_images=WRITE_IMAGES,LOG_EVERY=LOG_EVERY)
			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			
			# Do data augmentation update
			if error==0:
				#if i%UPDATE_DATA_EVERY==0 or i<WARMUP:
				if loss_diff<loss_diff_thresh or i<WARMUP:
					x,y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, key)
				
				
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.NCA_model = nca
						self.NCA_model.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						#tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
		
		if error==0:
			print("Training completed successfully")
		elif error==1:
			print("|-|-|-|-|-|-  Loss reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==2:
			print("|-|-|-|-|-|-  X reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==3:
			print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(error_at)+", optimisation probably diverging  -|-|-|-|-|-|")
		if error!=0 and model_saved==False:
			print("|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|")
		elif self.IS_LOGGING and model_saved:
			x,y = self.DATA_AUGMENTER.split_x_y(1)
			x,y = self.DATA_AUGMENTER.data_callback(x,y,0,key)
			#try:
			self.LOGGER.tb_training_end_log(
				self.NCA_model,
				x,
				t=t,
				NUMBER_OF_IMAGES=x[0].shape[0],
				boundary_callback=self.BOUNDARY_CALLBACK,
				SAVE_TRAJECTORY=True)
			# except Exception as e:
			# 	print("Error logging training end")
			# 	print(e)
			# 	pass
def _build_vgg_aux(experiment_groups):
	if experiment_groups is None:
		return None
	else:
		diff = jnp.diff(experiment_groups)	
		indices_to_split_at = jnp.where(diff != 0)[0] + 1
		return indices_to_split_at.astype(jnp.int32)