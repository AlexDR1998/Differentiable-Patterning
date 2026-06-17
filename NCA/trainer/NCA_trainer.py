from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import equinox as eqx
import datetime
# import Common.trainer.loss as loss
# import Common.trainer.loss_ott as loss_ott
from Common.trainer.loss import build_loss_functions,build_loss_initialiser
from NCA.trainer.tensorboard_log import NCA_Train_log, kaNCA_Train_log, mNCA_Train_log, aNCA_Train_log, NCA_knockout_Train_log
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_multihead_attention import aNCA
from NCA.trainer.data_augmenter_nca import DataAugmenter
import NCA.trainer.NCA_regulariser as regularisers
from einops import repeat, reduce, rearrange, einsum
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary, hard_boundary, no_boundary
from tqdm import tqdm
from jaxtyping import Float,Array,Key
import time


INTERNAL_LOOP_DTYPE = jnp.bfloat16 # dtype to use for values inside the loop over timesteps. Should be low precision to save memory, but not so low that it causes instability. Can experiment with bfloat16 or float16.
LOSS_DTYPE = jnp.float32 # dtype to use for loss values. Higher precision as it accumulates over many timesteps and batches.
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
				 LOSS_TIME_CHANNEL_MASK = None, # If none, is overwritten to ones mask that does nothing
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
			if None, sets model_filename to current time
		DATA_AUGMENTER : object, optional
			DataAugmenter object. Has data_init and data_callback methods that can be re-written as needed. The default is DataAugmenter.
		BOUNDARY_MASK : float32 [N_BOUNDARY_CHANNELS,WIDTH,HEIGHT], optional
			Set of channels to keep fixed, encoding boundary conditions. The default is None.
		BOUNDARY_MODE : string, optional
			Whether to apply boundary conditions as a soft regulariser ("soft") or a hard constraint ("hard"). The default is "soft".
		SHARDING : int, optional
			How many parallel GPUs to shard data across?. The default is None.
		GRAD_LOSS : boolean, optional
			Whether to compute loss on spatial gradients of x and y as well as on x and y themself. The default is True.
		OBS_CHANNELS : int, optional
			Number of channels in x and y to include in loss function. If None, set to all channels in data. The default is None.
		DATA_CHANNELS : int, optional
			Number of channels in y to include in loss function. If None, set to OBS_CHANNELS. The default is None.
			Can be different to OBS_CHANNELS, i.e. if training to duplicate data from multiple experiments
		LOSS_TIME_CHANNEL_MASK: float32 array [BATCHES, N, OBS_CHANNELS], optional
			Mask for which channels and timesteps to include in loss function. 1 for include, 0 for exclude. If None, set to ones mask that includes everything. The default is None.
		MODEL_DIRECTORY : str, optional
			Name of directory where model parameters get stored, defaults to 'models/'
		LOG_DIRECOTRY : str, optional
			Name of directory where tensorboard logs get stored, defaults to 'logs/'

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
		# Set up data and data augmenter class
		self._data_raw = data
		self.DATA_AUGMENTER = DATA_AUGMENTER(
			data_true=data,
			hidden_channels=self.CHANNELS-self.DATA_CHANNELS,
			nca_model=self.NCA_model
			)
		self.DATA_AUGMENTER.data_init(self.SHARDING)
		self.data = self.DATA_AUGMENTER.return_saved_data()
		self.BATCHES = len(self.data)
		print("Batches = "+str(self.BATCHES))
		
		# Set up partial mask of channels / timesteps
		if self.LOSS_TIME_CHANNEL_MASK is None:
			self.LOSS_TIME_CHANNEL_MASK = jnp.ones((self.BATCHES,data.shape[1]-1,self.OBS_CHANNELS),dtype=jnp.float32)

		_model_kernel_length = len(self.NCA_model.KERNEL_STR)
		if "GRAD" in self.NCA_model.KERNEL_STR:
			_model_kernel_length+=1
		if GRAD_LOSS:
			self.LOSS_TIME_CHANNEL_MASK = repeat(self.LOSS_TIME_CHANNEL_MASK,"b n c -> b n (gc c) () ()",gc=_model_kernel_length)
			print("Timestep / Channel mask: ")
			print(self.LOSS_TIME_CHANNEL_MASK[:,:,:,0,0])
		else:
			self.LOSS_TIME_CHANNEL_MASK = rearrange(self.LOSS_TIME_CHANNEL_MASK,"b n c -> b n c () ()")
			print("Timestep / Channel mask: ")
			print(self.LOSS_TIME_CHANNEL_MASK[:,:,:,0,0])

		self.LOSS_TIME_CHANNEL_MASK = list(self.LOSS_TIME_CHANNEL_MASK)
		# Set up boundary augmenter class
		# length of BOUNDARY_MASK PyTree should be same as number of batches
		

		# For NCA with latent state, boundary mask should be in the latent space
		if BOUNDARY_MASK is not None:
			BOUNDARY_MASK = self.NCA_model.real_to_latent(BOUNDARY_MASK)

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
		
	def setup_logging(self,BACKEND,wandb_args,KNOCKOUT_ARGS):
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
				# elif isinstance(self.NCA_model, uNCA):
					# self.LOGGER = uNCA_Train_log(self.LOG_DIR, self._data_raw)
				else:
					self.LOGGER = NCA_Train_log(self.LOG_DIR, self._data_raw)
				print("Logging training to: "+self.LOG_DIR)
			elif BACKEND=="wandb":
				self.IS_LOGGING = True
				self.LOG_DIR = self._LOG_DIRECTORY+self.model_filename+"/train"
				config = {"MODEL":self.NCA_model.get_config(),
			  			 "TRAINING":self.TRAIN_CONFIG}
				wandb_args["config"] = config
				
				if KNOCKOUT_ARGS["time"] is not None: # Nodal KO has differet logging behaviour
					self.LOGGER = NCA_knockout_Train_log(
						data=self._data_raw,
						wandb_config=wandb_args,
						knockout_time=KNOCKOUT_ARGS["time"],
						knockout_channel=KNOCKOUT_ARGS["channel"])
				else:
					self.LOGGER = NCA_Train_log(data=self._data_raw,wandb_config=wandb_args)
				print("Logging training to: "+self.LOG_DIR)
		self.MODEL_PATH = self._MODEL_DIRECTORY+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)

	@eqx.filter_jit	
	def loss_func(
		self,
		x_proc:Float[Array, "N CHANNELS x y"],  # noqa: F722
		y_proc:Float[Array, "N CHANNELS x y"],  # noqa: F722
		x_latent:Float[Array, "N L h w"],  # noqa: F722
		y_latent:Float[Array, "N L h w"],  # noqa: F722
		channel_time_mask:Float[Array, "N OBS_CHANNELS"],  # noqa: F722
		loss_cache:Dict[str, Float[Array, "N ..."]],  # noqa: F722
		key: Key)->Float[Array, " N"]:
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		channel_time_mask : float32 array [N,OBS_CHANNELS]
			Masks for which channels and timesteps to include in the loss. 1 for include, 0 for exclude. 
		loss_cache : float32 array [N,...] | None
			Optional precomputed cache for some loss functions, such as VGG. Can save computation by avoiding recomputing some latent features of y inputs.
		key : jr.PRNGKey
			Jax random number key. Only useful for loss functions that are stochastic (i.e. subsampled).
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		# if x.shape[-2:] != y.shape[-2:]:
			# x = jax.image.resize(x, y.shape, method="linear")
		
		x_proc_obs = x_proc[:,:self.OBS_CHANNELS]
		y_proc_obs = y_proc[:,:self.DATA_CHANNELS]
		x_lat_obs = x_latent[:,:self.OBS_CHANNELS]
		y_lat_obs = y_latent[:,:self.DATA_CHANNELS]
		if self.GRAD_LOSS:
			x_proc_obs = self.grad_loss_helper(x_proc_obs)
			y_proc_obs = self.grad_loss_helper(y_proc_obs)
			x_lat_obs = self.grad_loss_helper(x_lat_obs)
			y_lat_obs = self.grad_loss_helper(y_lat_obs)
		
		losses = []
		for idx, f in enumerate(self._loss_func):
			key = jr.fold_in(key,idx)
			# Get mask for channels that should be included in this loss function
			# Include channels where LOSS_FUNC_CHANNELS == idx or == -1
			channel_loss_mask = (self.LOSS_FUNC_CHANNELS == idx) | (self.LOSS_FUNC_CHANNELS == -1)
			channel_loss_mask = repeat(channel_loss_mask,"c -> (gc c) () ()",gc=channel_time_mask.shape[1]//self.OBS_CHANNELS).astype(jnp.float32)
			# Select only the relevant channels
			loss_mask = einsum(channel_time_mask,channel_loss_mask,"n c w h, c w h-> n c w h").astype(jnp.bool_)
			
			# Select whether each loss function applies to latents or decoded outputs, or both.
			if self.LOSS_FUNC_LAYERS[idx]=="decoded":
				losses.append(f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None)))
			elif self.LOSS_FUNC_LAYERS[idx]=="latent":
				losses.append(f(x_lat_obs, y_lat_obs, key, loss_mask, loss_cache.get("latent",None)))
			elif self.LOSS_FUNC_LAYERS[idx]=="both":
				losses.append(f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None)))
				losses.append(f(x_lat_obs, y_lat_obs, key, loss_mask, loss_cache.get("latent",None)))
			else:
				print(f"Warning: LOSS_FUNC_LAYERS[{idx}] is {self.LOSS_FUNC_LAYERS[idx]}, but should be either 'decoded' or 'latent'. Defaulting to 'decoded'.")
				losses.append(f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None)))
						
		losses = jnp.array(losses)
		return reduce(losses,"loss_funcs N -> N","mean")
	
	def grad_loss_helper(self,x):
		v_perception = jax.vmap(self.NCA_model.perception,in_axes=0,out_axes=0)
		base_channels = x.shape[1]
		x = v_perception(x)
		x = x.at[:,base_channels:].set(0.1*x[:,base_channels:])
		return x
	
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
				"experiment_groups":None,
				"S":1024,
				"K":5,
				"D":3,
				"sharpen":True,
				"epsilon":0.1,
				"internal_loss_func":"l2",
				"samples":128,
				"layers":["decoded"] # List of length LOSS_FUNC_STR, specifying whether each loss function applies to the "latent" or "decoded" outputs. Redundant for baseline NCA.
			  },
			  KNOCKOUT_ARGS = {
				  "time":None,
				  "channel":None
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

			log_x = jtu.tree_map(self.NCA_model.latent_to_real, x_new)
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
		NCA_MODEL_AUX_ARGS : dict, optional
			Additional arguments for the NCA model
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
		
		self.setup_logging("wandb",wandb_args=wandb_args,KNOCKOUT_ARGS=KNOCKOUT_ARGS)

		
		
		if LOSS_ARGS["layers"] is not None:
			self.LOSS_FUNC_LAYERS = LOSS_ARGS["layers"]
		else:
			self.LOSS_FUNC_LAYERS = ["decoded"]*len(LOSS_FUNC_STR)
		
		# LOSS_FUNC_CHANNELS = 
		if LOSS_ARGS["channels"] is not None:
			assert len(LOSS_ARGS["channels"])==self.OBS_CHANNELS, "LOSS_FUNC_CHANNELS should be same length as number of observable channels"
		elif LOSS_ARGS["channels"] is None:
			LOSS_ARGS["channels"] = jnp.ones((self.OBS_CHANNELS,),dtype=jnp.int32)*-1
		self.LOSS_FUNC_CHANNELS = LOSS_ARGS["channels"]
		
		REG_FUNCS = {
			"intermediate_state":regularisers.intermediate_reg,
			"boundary":regularisers.boundary_regulariser,
			"contiguous_growth":regularisers.contiguous_growth_regulariser,
			"update_sensitivity":regularisers.update_sensitivity_regulariser,
			"perturbation_conservation":regularisers.perturbation_conservation_regulariser,
			"latent_channel_match":regularisers.latent_channel_match_regulariser,
			"latent_size":regularisers.latent_size_regulariser
		}
		

		# Filter REG_FUNCS to the same set (optional but keeps things consistent)
		REGULARISER_COEFFS = {name:REGULARISER_COEFFS[name] for name in REGULARISER_COEFFS.keys() if REGULARISER_COEFFS[name]!=0.0}
		REG_FUNCS = {name: REG_FUNCS[name] for name in REGULARISER_COEFFS.keys()}
		#@partial(eqx.filter_jit,donate="all-except-first")
		@eqx.filter_jit
		def make_step(nca,x_latent,y_proc,t,opt_state,key):
			"""
			

			Parameters
			----------
			nca : object callable - (float32 [N_CHANNELS,_,_],PRNGKey) -> (float32 [N_CHANNELS,_,_])
				the NCA object to train
			x_latent : float32 array [BATCHES,N,CHANNELS,_,_]
				NCA latent state
			y_proc : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				processed true data
			t : int
				number of NCA timesteps between x_latent[N] and x_latent[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jr.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
				the NCA object with updated parameters
			x_latent : float32 array [BATCHES,N,CHANNELS,_,_]
				NCA latent state
			y_proc : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				processed true data
			t : int	
				number of NCA timesteps between x_latent[N] and x_latent[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			key : jr.PRNGKey
				Jax random number key
			mean_loss : float
				Mean loss across batch and time for this step
			log_dict : dict
				Dictionary of values to log, including at least "loss", and optionally "x_latent", "x_processed", "losses", and any regulariser losses under their own keys.

			"""

			def apply_intermediate_regs(reg_logs,x_latent,x_new_latent,x_proc,x_new_proc,vv_nca,key):
				aux = {
					"BOUNDARY_CALLBACK": self.BOUNDARY_CALLBACK, 
					"OBS_CHANNELS": self.OBS_CHANNELS,
					"REAL_TO_LATENT": self.NCA_model.real_to_latent,
					}
				for name in REGULARISER_COEFFS.keys():
					reg_logs[name]+=REG_FUNCS[name](x_latent,x_new_latent,x_proc,x_new_proc,vv_nca,aux,key)
				return reg_logs
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff,nca_static,x_latent,y_proc,t,key):
				# Gradient and values of loss function computed here
				_nca = eqx.combine(nca_diff,nca_static)
				v_nca = jax.vmap(_nca,in_axes=(0,None,0),out_axes=0,axis_name="N") # boundary is independant of time N
				vv_nca = lambda x,callback,key_array: jtu.tree_map(v_nca,x,callback,key_array)  # noqa: E731
				# provide a batched processor that maps model.latent_to_real over the batch/tree
				v_latent_to_real = jax.vmap(lambda model_x: _nca.latent_to_real(model_x), in_axes=0, out_axes=0)
				vv_latent_to_real = lambda x: jtu.tree_map(v_latent_to_real, x)
				v_real_to_latent = jax.vmap(lambda model_x: _nca.real_to_latent(model_x), in_axes=0, out_axes=0)
				vv_real_to_latent = lambda x: jtu.tree_map(v_real_to_latent, x)
				# Set up internal logs for regularisers
				reg_logs_internal = {name: jnp.zeros(len(x_latent),dtype=LOSS_DTYPE) for name in REGULARISER_COEFFS.keys()}
				state_shape = x_latent[0].shape[0] # Assumes the same number of outer timesteps in each batch.

				# Structuring this as function and lax.scan speeds up jit compile a lot
				def nca_step(carry,j): # function of type a,b -> a
					key,x_latent,x_proc,reg_logs_internal = carry
					# Apply NCA update step
					key = jr.fold_in(key,j)
					key_array = key_pytree_gen(key,(len(x_latent),state_shape))
					x_new_latent = vv_nca(x_latent,self.BOUNDARY_CALLBACK,key_array)
					x_new_proc = vv_latent_to_real(x_new_latent)
					reg_logs_internal = apply_intermediate_regs(reg_logs_internal,x_latent,x_new_latent,x_proc,x_new_proc,vv_nca,key)

					return (key,x_new_latent,x_new_proc,reg_logs_internal),None
				(key,x_latent,x_proc,reg_logs_internal),_ = eqx.internal.scan(nca_step,(key,x_latent,vv_latent_to_real(x_latent),reg_logs_internal),
					xs=jnp.arange(t),
					kind=LOOP_AUTODIFF  # type: ignore
				)

				loss_key = key_pytree_gen(key, (len(x_latent),))
				y_latent = vv_real_to_latent(y_proc)
				losses = jnp.array(jtu.tree_map(
					self.loss_func,
					x_proc,
					y_proc,
					x_latent,
					y_latent,
					self.LOSS_TIME_CHANNEL_MASK,
					self.LOSS_CACHE,
					loss_key
					))
				reg_loss_internal = {name: REGULARISER_COEFFS[name]*jnp.mean(reg_logs_internal[name])/t for name in REGULARISER_COEFFS.keys()}
				mean_loss = jnp.mean(losses) + jnp.sum(jnp.array(list(reg_loss_internal.values())))
				return mean_loss, (x_latent,x_proc,losses,reg_loss_internal)

			nca_diff,nca_static = nca.partition()
			loss_x,grads = compute_loss(nca_diff,nca_static,x,y,t,key)  # type: ignore
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			(mean_loss,(x_latent,x_proc,losses,reg_loss)) = loss_x
			log_dict = {
				"loss": mean_loss,
				"x_latent": x_latent,
				"x_processed": x_proc,
				"losses": losses,
				**reg_loss
			}
			return nca,x_latent,y_proc,t,opt_state,key,mean_loss,log_dict

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
		# x = jtu.tree_map(self.NCA_model.real_to_latent, x)
		print(f"Initial x shape: {jnp.array(x).shape}, y shape: {jnp.array(y).shape}",flush=True)
		
		# If we are using a loss function that needs to initialise some cache based on the data, do that here and add to LOSS_ARGS
		loss_initialiser = build_loss_initialiser(LOSS_FUNC_STR,LOSS_ARGS)
		if loss_initialiser is not None:
			vgg_target_cache_decoded = loss_initialiser(y,key,self.LOSS_TIME_CHANNEL_MASK) # dict of {"vgg_params": ..., "target_feats": List[Batches] of arrays [N, ...]}
			v_real_to_latent = jax.vmap(lambda model_x: self.NCA_model.real_to_latent(model_x), in_axes=0, out_axes=0)
			vv_real_to_latent = lambda x: jtu.tree_map(v_real_to_latent, x)
			_y_latent = vv_real_to_latent(y)
			vgg_target_cache_latent = loss_initialiser(_y_latent,key,self.LOSS_TIME_CHANNEL_MASK)

			LOSS_ARGS = {**LOSS_ARGS, "vgg_params": vgg_target_cache_decoded["vgg_params"]} # Pre-trained VGG parameters for perceptual loss, if needed. Does not need batched.
			if not LOSS_ARGS.get("random_crop", False):
				# If we are not randomly re-cropping and sampling VGG target features,
				# just compute them once and store in LOSS_CACHE.
				# self.LOSS_CACHE = {
				# 	"decoded": vgg_target_cache_decoded["target_feats"],
				# 	"latent": vgg_target_cache_latent["target_feats"]
				# }
				self.LOSS_CACHE = [{
					"decoded": vgg_target_cache_decoded["target_feats"][b],
					"latent": vgg_target_cache_latent["target_feats"][b]
					} for b in range(len(x))]
			else:
				# self.LOSS_CACHE = {
				# 	"decoded":[None]*len(x), # If using random cropping, can't use precomputed cache of target features as different crops each time
				# 	"latent":[None]*len(x)
				# }
				self.LOSS_CACHE = [{
					"decoded": None,
					"latent": None
					} for b in range(len(x))]
				
		else:
			# self.LOSS_CACHE = {
			# 		"decoded":[None]*len(x), # If using random cropping, can't use precomputed cache of target features as different crops each time
			# 		"latent":[None]*len(x)
			# 	}
			self.LOSS_CACHE = [{
				"decoded": None,
				"latent": None
				} for b in range(len(x))]
			
		self._loss_func = build_loss_functions(LOSS_FUNC_STR,LOSS_ARGS)	
		best_loss = 100000000
		loss_thresh = 1e16 # If loss exceeds this, training is diverging to NaN
		model_saved = False
		loss_diff = 0
		#prev_loss = 0
		mean_loss = 0
		loss_diff_thresh = 1e-2 # How much the loss needs to improve by to trigger a data update.
		error = 0
		error_at = 0
		# SPARSITY = jnp.concat((jnp.zeros(WARMUP),jnp.linspace(0,TARGET_SPARSITY,iters-WARMUP)))
		
		pbar = tqdm(range(iters))
		#--- Do training run ---
		for i in pbar:
			#prev_loss = mean_loss
			if i%CLEAR_CACHE_EVERY==0:
				#print(f"Clearing cache at step {i}")
				jax.clear_caches()
			key = jr.fold_in(key,i)
			
			nca,x_new,y_new,t,opt_state,key,mean_loss,log_dict = make_step(nca, x, y, t, opt_state,key)  # type: ignore
			loss_diff = mean_loss - best_loss

			log_dict["best_loss"] = best_loss
			# print_dict = {k: v if isinstance(v, (int, float)) else str(v.shape) for k, v in log_dict.items()}
			print_dict = {k:v for k,v in log_dict.items() if k not in ['x_latent','x_processed']}
			pbar.set_postfix(print_dict)

			# if SPARSE_PRUNING:
				
			# 	if i>WARMUP:

			# 		ws,_ = nca.get_weights()
			# 		sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=SPARSITY[i])
			# 		pruner = jaxpruner.MagnitudePruning(
			# 			sparsity_distribution_fn=sparsity_distribution,
			# 			skip_gradients=True)
			# 		ws = pruner.instant_sparsify(ws)[0]
			# 		nca.set_weights(ws)

			
			if self.IS_LOGGING:
				# log_x = jtu.tree_map(self.NCA_model.latent_to_real, x_new)
				self.LOGGER.tb_training_loop_log_sequence(log_dict, i, nca,write_images=WRITE_IMAGES,LOG_EVERY=LOG_EVERY)
			
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
				# if (loss_diff<loss_diff_thresh or i<WARMUP):
					# x_for_callback = log_dict.get("x_processed", x_new)
				x, y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, key)
					# x = jtu.tree_map(self.NCA_model.real_to_latent, x_aug)
					# y = y_aug
				
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.NCA_model = nca
						self.NCA_model.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						
		
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
			
			self.LOGGER.tb_training_end_log(
				self.NCA_model,
				self.DATA_AUGMENTER,
				t=t,
				boundary_callback=self.BOUNDARY_CALLBACK,
				SAVE_TRAJECTORY=False)
		self.LOGGER.finish()
