from typing import Dict
import os
from pathlib import Path

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
from NCA.trainer.tensorboard_log import (
	NCA_Train_log,
	mNCA_Train_log,
	aNCA_Train_log,
	NCA_knockout_Train_log,
)
from NCA.trainer.kan_tensorboard_log import (
	kaNCA_Train_log,
	uses_fast_kan_diagnostics,
)
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


def resolve_loss_component_weights(weights, loss_count):
	"""Validate and normalise configuration shape for loss-component weights."""
	if weights is None:
		return jnp.ones((loss_count,), dtype=LOSS_DTYPE)
	weights = list(weights)
	if len(weights) != loss_count:
		raise ValueError(
			"loss.component_weights must have one value per configured loss "
			f"({loss_count} expected, got {len(weights)})"
		)
	if any(float(weight) < 0 for weight in weights):
		raise ValueError("loss.component_weights cannot contain negative weights")
	if not any(float(weight) > 0 for weight in weights):
		raise ValueError("loss.component_weights must contain at least one positive weight")
	return jnp.asarray(weights, dtype=LOSS_DTYPE)


def combine_loss_components(losses, weights):
	"""Return a normalized weighted mean over the leading loss-component axis."""
	losses = jnp.stack(losses)
	weights = jnp.asarray(weights, dtype=losses.dtype)
	return jnp.sum(losses * weights[:, None], axis=0) / jnp.sum(weights)


def select_wandb_train_logger_class(model, knockout_time=None):
	if knockout_time is not None:
		return NCA_knockout_Train_log
	if uses_fast_kan_diagnostics(model):
		return kaNCA_Train_log
	return NCA_Train_log

def maybe_save_gpu_profile(step):
	if os.getenv("PROFILE_GPU", "0") != "1":
		return

	profile_step = int(os.getenv("PROFILE_GPU_STEP", "0"))
	if step != profile_step:
		return

	task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
	profile_dir_env = os.getenv("PROFILE_GPU_DIR") or os.getenv("RUN_CONFIG_PROFILE_DIR")
	if profile_dir_env is None:
		job_id = os.getenv("SLURM_JOB_ID", "manual")
		root = Path(os.getenv("SLURM_IO_ROOT", "output"))
		profile_dir = root / "profiles" / f"{job_id}_{task_id}"
	else:
		profile_dir = Path(profile_dir_env)
	profile_dir.mkdir(parents=True, exist_ok=True)
	profile_path = profile_dir / f"train_step_{step}_device_memory.prof"

	try:
		jax.block_until_ready(jax.device_put(0))
		jax.profiler.save_device_memory_profile(str(profile_path))
		print(f"Writing train-step device memory profile to: {profile_path}", flush=True)
	except Exception as exc:
		error_path = profile_dir / f"train_step_{step}_device_memory_error.txt"
		error_path.write_text(f"{exc!r}\n")
		print(f"Warning: train-step device memory profile failed: {exc!r}", flush=True)


def start_jax_training_trace(profile_dir):
	"""Start a compact device trace without recording every Python call."""
	profile_dir = Path(profile_dir)
	profile_dir.mkdir(parents=True, exist_ok=True)
	kwargs = {}
	profile_options_cls = getattr(jax.profiler, "ProfileOptions", None)
	if profile_options_cls is not None:
		profile_options = profile_options_cls()
		profile_options.python_tracer_level = 0
		profile_options.host_tracer_level = 2
		kwargs["profiler_options"] = profile_options
	else:
		print(
			"JAX ProfileOptions is unavailable; tracing the short training "
			"window with default host/Python settings.",
			flush=True,
		)

	jax.block_until_ready(jax.device_put(0))
	try:
		jax.profiler.start_trace(str(profile_dir), **kwargs)
	except TypeError:
		# JAX 0.5 deployments may expose ProfileOptions in jaxlib without
		# accepting profiler_options in the public start_trace signature.
		print(
			"This JAX start_trace API does not accept profiler_options; "
			"using default settings for the short training window.",
			flush=True,
		)
		jax.profiler.start_trace(str(profile_dir))
	print(f"Started JAX training trace: {profile_dir}", flush=True)


def stop_jax_training_trace(outputs):
	"""Synchronize the final captured step before flushing its device trace."""
	jax.block_until_ready(outputs)
	jax.profiler.stop_trace()
	print("Finished JAX training trace", flush=True)


def compile_and_time(jitted_function, *args):
	"""Compile a jitted function explicitly and return wall-clock seconds."""
	start = time.perf_counter()
	compiled_function = jitted_function.lower(*args).compile()
	return compiled_function, time.perf_counter() - start


def call_and_time(compiled_function, *args):
	"""Run a compiled function and include asynchronous device work in timing."""
	start = time.perf_counter()
	outputs = compiled_function(*args)
	jax.block_until_ready(outputs)
	return outputs, time.perf_counter() - start


class PoolAdmissionController:
	"""Tracks whether a rollout should be admitted into the recurrent state pool."""

	def __init__(
		self,
		enabled=True,
		relative_threshold=1.25,
		previous_relative_threshold=1.10,
		absolute_threshold=None,
		ema_decay=0.95,
		warmup=0,
	):
		self.enabled = enabled
		self.relative_threshold = relative_threshold
		self.previous_relative_threshold = previous_relative_threshold
		self.absolute_threshold = absolute_threshold
		self.ema_decay = ema_decay
		self.warmup = warmup
		self.loss_ema = None
		self.previous_admitted_loss = None
		self.admit_count = 0
		self.reject_count = 0

	def decide(self, loss_value, step, cache_clear_step, error=0):
		loss_ref = loss_value if self.loss_ema is None else self.loss_ema
		loss_ratio = loss_value / max(loss_ref, 1e-12)
		previous_loss_ref = loss_value if self.previous_admitted_loss is None else self.previous_admitted_loss
		previous_loss_ratio = loss_value / max(previous_loss_ref, 1e-12)
		check_loss_spike = self.enabled and self.loss_ema is not None and step >= self.warmup
		reject_cache_clear = bool(cache_clear_step)
		reject_relative = check_loss_spike and loss_ratio > self.relative_threshold
		reject_previous_relative = (
			self.enabled
			and self.previous_admitted_loss is not None
			and step >= self.warmup
			and previous_loss_ratio > self.previous_relative_threshold
		)
		reject_absolute = (
			check_loss_spike
			and self.absolute_threshold is not None
			and loss_value > loss_ref + self.absolute_threshold
		)
		admit = (
			error == 0
			and not reject_cache_clear
			and not reject_relative
			and not reject_previous_relative
			and not reject_absolute
		)
		return {
			"admit": admit,
			"reject_cache_clear": reject_cache_clear,
			"reject_relative": reject_relative,
			"reject_previous_relative": reject_previous_relative,
			"reject_absolute": reject_absolute,
			"loss_ref": loss_ref,
			"loss_ratio": loss_ratio,
			"previous_loss_ref": previous_loss_ref,
			"previous_loss_ratio": previous_loss_ratio,
		}

	def update(self, decision, loss_value):
		if decision["admit"]:
			self.admit_count += 1
			self.previous_admitted_loss = loss_value
			if self.enabled:
				if self.loss_ema is None:
					self.loss_ema = loss_value
				else:
					self.loss_ema = (
						self.ema_decay * self.loss_ema
						+ (1 - self.ema_decay) * loss_value
					)
		else:
			self.reject_count += 1

	def log_dict(self, decision):
		return {
			"pool/admit": int(decision["admit"]),
			"pool/reject": int(not decision["admit"]),
			"pool/reject_cache_clear": int(decision["reject_cache_clear"]),
			"pool/reject_relative": int(decision["reject_relative"]),
			"pool/reject_previous_relative": int(decision["reject_previous_relative"]),
			"pool/reject_absolute": int(decision["reject_absolute"]),
			"pool/loss_ref": decision["loss_ref"],
			"pool/loss_ratio": decision["loss_ratio"],
			"pool/previous_loss_ref": decision["previous_loss_ref"],
			"pool/previous_loss_ratio": decision["previous_loss_ratio"],
			"pool/admit_count": self.admit_count,
			"pool/reject_count": self.reject_count,
		}


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
				 CHANNEL_NAMES = None,
				 TIMEPOINT_NAMES = None,
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
		self.CHANNEL_NAMES = CHANNEL_NAMES
		self.TIMEPOINT_NAMES = TIMEPOINT_NAMES
		self.DIAGNOSTIC_BOUNDARY_MASK = BOUNDARY_MASK
		
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
		
	def setup_logging(self,BACKEND,wandb_args,KNOCKOUT_ARGS,SINGULAR_VALUE_LOGGING_CONFIG=None):
		# Set logging behvaiour based on provided filename
		print(f"Raw data shape: {jnp.array(self._data_raw).shape}")
		if self.model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			if BACKEND=="tensorboard":
				self.IS_LOGGING = True
				self.LOG_DIR = self._LOG_DIRECTORY+self.model_filename+"/train"
				if isinstance(self.NCA_model ,kaNCA) or uses_fast_kan_diagnostics(self.NCA_model):
					self.LOGGER = kaNCA_Train_log(self.LOG_DIR,self._data_raw)
				elif isinstance(self.NCA_model , mNCA):
					self.LOGGER = mNCA_Train_log(self.LOG_DIR,self._data_raw)
				elif isinstance(self.NCA_model , aNCA):
					self.LOGGER = aNCA_Train_log(self.LOG_DIR,self._data_raw)
				# elif isinstance(self.NCA_model, uNCA):
					# self.LOGGER = uNCA_Train_log(self.LOG_DIR, self._data_raw)
				else:
					self.LOGGER = NCA_Train_log(
						self.LOG_DIR,
						self._data_raw,
						singular_value_config=SINGULAR_VALUE_LOGGING_CONFIG,
					)
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
						boundary_mask=self.DIAGNOSTIC_BOUNDARY_MASK,
						channel_names=self.CHANNEL_NAMES,
						timepoint_names=self.TIMEPOINT_NAMES,
						data_augmenter=self.DATA_AUGMENTER,
						knockout_time=KNOCKOUT_ARGS["time"],
						knockout_channel=KNOCKOUT_ARGS["channel"],
						singular_value_config=SINGULAR_VALUE_LOGGING_CONFIG)
				else:
					logger_class = select_wandb_train_logger_class(self.NCA_model)
					self.LOGGER = logger_class(
						data=self._data_raw,
						wandb_config=wandb_args,
						boundary_mask=self.DIAGNOSTIC_BOUNDARY_MASK,
						channel_names=self.CHANNEL_NAMES,
						timepoint_names=self.TIMEPOINT_NAMES,
						data_augmenter=self.DATA_AUGMENTER,
						singular_value_config=SINGULAR_VALUE_LOGGING_CONFIG,
					)
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
				component_losses = [f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None))]
			elif self.LOSS_FUNC_LAYERS[idx]=="latent":
				component_losses = [f(x_lat_obs, y_lat_obs, key, loss_mask, loss_cache.get("latent",None))]
			elif self.LOSS_FUNC_LAYERS[idx]=="both":
				component_losses = [
					f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None)),
					f(x_lat_obs, y_lat_obs, key, loss_mask, loss_cache.get("latent",None)),
				]
			else:
				print(f"Warning: LOSS_FUNC_LAYERS[{idx}] is {self.LOSS_FUNC_LAYERS[idx]}, but should be either 'decoded' or 'latent'. Defaulting to 'decoded'.")
				component_losses = [f(x_proc_obs, y_proc_obs, key, loss_mask, loss_cache.get("decoded",None))]
			losses.append(jnp.mean(jnp.stack(component_losses), axis=0))
						
		return combine_loss_components(losses, self.LOSS_COMPONENT_WEIGHTS)
	
	def grad_loss_helper(self,x):
		v_perception = jax.vmap(self.NCA_model.perception,in_axes=0,out_axes=0)
		base_channels = x.shape[1]
		x = v_perception(x)
		x = x.at[:,base_channels:].set(0.1*x[:,base_channels:])
		return x

	def _make_batched_nca(self, nca):
		"""Build the established vmap/tree-map NCA application path.

		Accelerator-specific trainers may override this hook without adding
		backend flags or batching branches to the core training loop.
		"""
		v_nca = jax.vmap(nca, in_axes=(0, None, 0), out_axes=0, axis_name="N")
		return lambda x, callback, key_array: jtu.tree_map(
			v_nca, x, callback, key_array
		)
	
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
			  POOL_ADMISSION_CONFIG = None,
			  SINGULAR_VALUE_LOGGING_CONFIG = None,
			  LOOP_AUTODIFF = "checkpointed",
			  SPARSE_PRUNING = False,
			  TARGET_SPARSITY = 0.5,
			  wandb_args={"project":"NCA",
					  "group":"group_1",
					  "tags":["training"]},
			  LEARNING_RATE_SCHEDULE=None,
			  JAX_TRACE=False,
			  key=None):
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

		if key is None:
			key = jr.PRNGKey(int(time.time()))
		pool_admission_config = {
			"enabled": True,
			"relative_threshold": 1.25,
			"previous_relative_threshold": 1.10,
			"absolute_threshold": None,
			"ema_decay": 0.95,
			"warmup": None,
		}
		if POOL_ADMISSION_CONFIG is not None:
			pool_admission_config.update(POOL_ADMISSION_CONFIG)
		singular_value_logging_config = {
			"enabled": False,
			"plot_spectra": True,
			"epsilon": 1e-8,
		}
		if SINGULAR_VALUE_LOGGING_CONFIG is not None:
			for config_key in singular_value_logging_config:
				try:
					value = SINGULAR_VALUE_LOGGING_CONFIG.get(config_key)
				except AttributeError:
					value = SINGULAR_VALUE_LOGGING_CONFIG[config_key]
				if value is not None:
					singular_value_logging_config[config_key] = value
		singular_value_logging_config["enabled"] = bool(singular_value_logging_config["enabled"])
		singular_value_logging_config["plot_spectra"] = bool(singular_value_logging_config["plot_spectra"])
		singular_value_logging_config["epsilon"] = float(singular_value_logging_config["epsilon"])
		loss_func_count = 1 if isinstance(LOSS_FUNC_STR, str) else len(LOSS_FUNC_STR)
		self.LOSS_COMPONENT_WEIGHTS = resolve_loss_component_weights(
			LOSS_ARGS.get("component_weights", None), loss_func_count
		)

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
			"LOSS_COMPONENT_WEIGHTS":[float(weight) for weight in self.LOSS_COMPONENT_WEIGHTS],
			"CHANNEL_IMPORTANCE":LOSS_ARGS.get("channel_importance", None),
			"POOL_ADMISSION_CONFIG":pool_admission_config,
			"SINGULAR_VALUE_LOGGING_CONFIG":singular_value_logging_config,
			"LOOP_AUTODIFF":LOOP_AUTODIFF,
			"SPARSE_PRUNING":SPARSE_PRUNING,
			"TARGET_SPARSITY":TARGET_SPARSITY,
		}
		
		self.setup_logging(
			"wandb",
			wandb_args=wandb_args,
			KNOCKOUT_ARGS=KNOCKOUT_ARGS,
			SINGULAR_VALUE_LOGGING_CONFIG=singular_value_logging_config,
		)

		
		
		def resolve_loss_layers(layers, loss_count):
			if layers is None:
				return ["decoded"]*loss_count
			if isinstance(layers, str):
				layers = [layers]
			else:
				layers = list(layers)
			if len(layers)==0:
				return ["decoded"]*loss_count
			if len(layers)<loss_count:
				layers = layers + [layers[-1]]*(loss_count-len(layers))
			return layers[:loss_count]
		self.LOSS_FUNC_LAYERS = resolve_loss_layers(LOSS_ARGS["layers"], loss_func_count)
		
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
				vv_nca = self._make_batched_nca(_nca)
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
			y_decoded_obs = jtu.tree_map(lambda yi: yi[:, :self.DATA_CHANNELS], y)
			vgg_target_cache_decoded = loss_initialiser(y_decoded_obs,key,self.LOSS_TIME_CHANNEL_MASK) # dict of {"vgg_params": ..., "target_feats": List[Batches] of arrays [N, ...]}
			v_real_to_latent = jax.vmap(lambda model_x: self.NCA_model.real_to_latent(model_x), in_axes=0, out_axes=0)
			vv_real_to_latent = lambda x: jtu.tree_map(v_real_to_latent, x)
			_y_latent = vv_real_to_latent(y_decoded_obs)
			vgg_target_cache_latent = loss_initialiser(_y_latent,key,self.LOSS_TIME_CHANNEL_MASK)

			LOSS_ARGS = {**LOSS_ARGS, "vgg_params": vgg_target_cache_decoded["vgg_params"]} # Pre-trained VGG parameters for perceptual loss, if needed. Does not need batched.
			use_cached_vgg_targets = (
				not LOSS_ARGS.get("random_crop", False)
				and not LOSS_ARGS.get("random_channel_shuffle", False)
			)
			if use_cached_vgg_targets:
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
		compiled_make_step, initial_compile_seconds = compile_and_time(
			make_step,
			nca,
			x,
			y,
			t,
			opt_state,
			key,
		)
		runtime_tracker = {
			"jit_compile_seconds": initial_compile_seconds,
			"total_compile_seconds": initial_compile_seconds,
			"compile_count": 1,
			"first_execution_seconds": None,
			"step_compute_seconds": None,
			"step_compute_per_second": None,
			"steady_step_mean_seconds": None,
			"steady_step_mean_per_second": None,
			"steady_step_count": 0,
			"iteration_excluding_logging_seconds": None,
			"steady_iteration_mean_seconds": None,
			"steady_iteration_mean_per_second": None,
			"steady_iteration_count": 0,
			"_steady_step_seconds_total": 0.0,
			"_steady_iteration_seconds_total": 0.0,
		}
		print(
			f"Initial JIT compile time: {initial_compile_seconds:.6f} seconds",
			flush=True,
		)
		best_loss = 100000000
		loss_thresh = 1e16 # If loss exceeds this, training is diverging to NaN
		model_saved = False
		#prev_loss = 0
		mean_loss = 0
		error = 0
		error_at = 0
		pool_admission = PoolAdmissionController(
			enabled=pool_admission_config["enabled"],
			relative_threshold=pool_admission_config["relative_threshold"],
			previous_relative_threshold=pool_admission_config["previous_relative_threshold"],
			absolute_threshold=pool_admission_config["absolute_threshold"],
			ema_decay=pool_admission_config["ema_decay"],
			warmup=WARMUP if pool_admission_config["warmup"] is None else pool_admission_config["warmup"],
		)
		# SPARSITY = jnp.concat((jnp.zeros(WARMUP),jnp.linspace(0,TARGET_SPARSITY,iters-WARMUP)))
		
		pbar = tqdm(range(iters))
		trace_start_step = min(5, max(0, iters - 1))
		trace_stop_step = min(trace_start_step + 4, iters - 1)
		trace_active = False
		trace_dir = os.getenv("PROFILE_GPU_DIR", "output/jax-training-trace")
		#--- Do training run ---
		for i in pbar:
			iteration_start = time.perf_counter()
			#prev_loss = mean_loss
			key = jr.fold_in(key,i)
			CLEAR_CACHE_STEP = (
				CLEAR_CACHE_EVERY is not None 
				and CLEAR_CACHE_EVERY>0
				and i>0
				and i%CLEAR_CACHE_EVERY==0
			)
			if CLEAR_CACHE_STEP:
				print(f"Clearing cache at step {i}")
				jax.block_until_ready((x, y, opt_state))
				jax.clear_caches()
				compiled_make_step, latest_compile_seconds = compile_and_time(
					make_step,
					nca,
					x,
					y,
					t,
					opt_state,
					key,
				)
				runtime_tracker["compile_count"] += 1
				runtime_tracker["total_compile_seconds"] += latest_compile_seconds
				dry_outputs, _ = call_and_time(
					compiled_make_step, nca, x, y, t, opt_state, key
				)

				_, dry_x_new, dry_y_new, _, _, _, _, _ = dry_outputs
				key, dry_callback_key = jr.split(key)
				dry_callback_outputs = self.DATA_AUGMENTER.data_callback(
					dry_x_new,
					dry_y_new,
					i,
					dry_callback_key,
				)
				jax.block_until_ready(dry_callback_outputs)

				del dry_outputs
				del dry_callback_outputs

			if JAX_TRACE and i == trace_start_step:
				start_jax_training_trace(trace_dir)
				trace_active = True

			if trace_active:
				with jax.profiler.StepTraceAnnotation("train", step_num=i):
					step_outputs, step_compute_seconds = call_and_time(
						compiled_make_step, nca, x, y, t, opt_state, key
					)
			else:
				step_outputs, step_compute_seconds = call_and_time(
					compiled_make_step, nca, x, y, t, opt_state, key
				)

			if trace_active and i == trace_stop_step:
				stop_jax_training_trace(step_outputs)
				trace_active = False
			nca,x_new,y_new,t,opt_state,key,mean_loss,log_dict = step_outputs  # type: ignore
			maybe_save_gpu_profile(i)
			mean_loss_value = float(jax.device_get(mean_loss))
			if runtime_tracker["first_execution_seconds"] is None:
				runtime_tracker["first_execution_seconds"] = step_compute_seconds
			elif not CLEAR_CACHE_STEP:
				runtime_tracker["_steady_step_seconds_total"] += step_compute_seconds
				runtime_tracker["steady_step_count"] += 1

			runtime_tracker["step_compute_seconds"] = step_compute_seconds
			runtime_tracker["step_compute_per_second"] = 1.0 / max(
				step_compute_seconds, 1e-12
			)
			if runtime_tracker["steady_step_count"] > 0:
				runtime_tracker["steady_step_mean_seconds"] = (
					runtime_tracker["_steady_step_seconds_total"]
					/ runtime_tracker["steady_step_count"]
				)
				runtime_tracker["steady_step_mean_per_second"] = (
					1.0
					/ max(runtime_tracker["steady_step_mean_seconds"], 1e-12)
				)
			if LEARNING_RATE_SCHEDULE is not None:
				log_dict["learning_rate"] = float(
					jax.device_get(LEARNING_RATE_SCHEDULE(i))
				)

			log_dict["best_loss"] = best_loss

			# if SPARSE_PRUNING:
				
			# 	if i>WARMUP:

			# 		ws,_ = nca.get_weights()
			# 		sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=SPARSITY[i])
			# 		pruner = jaxpruner.MagnitudePruning(
			# 			sparsity_distribution_fn=sparsity_distribution,
			# 			skip_gradients=True)
			# 		ws = pruner.instant_sparsify(ws)[0]
			# 		nca.set_weights(ws)

			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x_new))):
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			
			pool_decision = pool_admission.decide(
				loss_value=mean_loss_value,
				step=i,
				cache_clear_step=CLEAR_CACHE_STEP,
				error=error,
			)
			if pool_decision["admit"]:
				key, callback_key = jr.split(key)
				x, y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, callback_key)
			pool_admission.update(pool_decision, mean_loss_value)
			log_dict.update(pool_admission.log_dict(pool_decision))
			runtime_tracker["iteration_excluding_logging_seconds"] = (
				time.perf_counter() - iteration_start
			)
			if i > 0 and not CLEAR_CACHE_STEP:
				runtime_tracker["_steady_iteration_seconds_total"] += (
					runtime_tracker["iteration_excluding_logging_seconds"]
				)
				runtime_tracker["steady_iteration_count"] += 1
				runtime_tracker["steady_iteration_mean_seconds"] = (
					runtime_tracker["_steady_iteration_seconds_total"]
					/ runtime_tracker["steady_iteration_count"]
				)
				runtime_tracker["steady_iteration_mean_per_second"] = (
					1.0
					/ max(runtime_tracker["steady_iteration_mean_seconds"], 1e-12)
				)
			log_dict.update({
				f"runtime/{name}": value
				for name, value in runtime_tracker.items()
				if not name.startswith("_") and value is not None
			})

			# print_dict = {k: v if isinstance(v, (int, float)) else str(v.shape) for k, v in log_dict.items()}
			print_dict = {
				k:v for k,v in log_dict.items()
				if k not in ['x_latent','x_processed']
				and not k.startswith(("pool/", "runtime/"))
			}
			pbar.set_postfix(print_dict)
			
			# Save model whenever mean_loss beats the previous best loss
			if i>WARMUP:
				if mean_loss < best_loss:
					model_saved=True
					self.NCA_model = nca
					self.NCA_model.save(self.MODEL_PATH,overwrite=True)
					best_loss = mean_loss

			if self.IS_LOGGING:
				# log_x = jtu.tree_map(self.NCA_model.latent_to_real, x_new)
				self.LOGGER.tb_training_loop_log_sequence(log_dict, i, nca,write_images=WRITE_IMAGES,LOG_EVERY=LOG_EVERY)
						
		
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
