import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from functools import partial
from NCA.trainer.tensorboard_log import NCA_Train_log, kaNCA_Train_log, mNCA_Train_log
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.data_augmenter_nca import DataAugmenter
from einops import repeat
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary
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
				 SHARDING = None, 
				 GRAD_LOSS = True,
				 OBS_CHANNELS = None,
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
		self.SHARDING = SHARDING
		self.GRAD_LOSS = GRAD_LOSS
		self.LOSS_TIME_CHANNEL_MASK = LOSS_TIME_CHANNEL_MASK
		
		# Set up partial mask of channels / timesteps
		if self.LOSS_TIME_CHANNEL_MASK is not None:
			_model_kernel_length = len(self.NCA_model.KERNEL_STR)
			if "GRAD" in self.NCA_model.KERNEL_STR:
				_model_kernel_length+=1
			if GRAD_LOSS:
				self.LOSS_TIME_CHANNEL_MASK = repeat(self.LOSS_TIME_CHANNEL_MASK,"n c -> n (gc c) () ()",gc=_model_kernel_length)
				print("Timestep / Channel mask: ")
				print(self.LOSS_TIME_CHANNEL_MASK[:,:,0,0])



		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data,self.CHANNELS-self.OBS_CHANNELS)
		self.DATA_AUGMENTER.data_init(self.SHARDING)
		self.data = self.DATA_AUGMENTER.return_saved_data()
		self.BATCHES = len(self.data)
		print("Batches = "+str(self.BATCHES))
		# Set up boundary augmenter class
		# length of BOUNDARY_MASK PyTree should be same as number of batches
		
		self.BOUNDARY_CALLBACK = []
		for b in range(self.BATCHES):
			if BOUNDARY_MASK is not None:
			
				self.BOUNDARY_CALLBACK.append(model_boundary(BOUNDARY_MASK[b]))
			else:
				self.BOUNDARY_CALLBACK.append(model_boundary(None))
		
		#print(jax.tree_util.tree_structure(self.BOUNDARY_CALLBACK))
		# Set logging behvaiour based on provided filename
		if model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			self.model_filename = model_filename
			self.IS_LOGGING = True
			self.LOG_DIR = LOG_DIRECTORY+self.model_filename+"/train"
			if isinstance(self.NCA_model ,kaNCA):
				self.LOGGER = kaNCA_Train_log(self.LOG_DIR,data)
			elif isinstance(self.NCA_model , mNCA):
				self.LOGGER = mNCA_Train_log(self.LOG_DIR,data)
			else:
				self.LOGGER = NCA_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		
		self.MODEL_PATH = MODEL_DIRECTORY+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
		
	@eqx.filter_jit	
	def loss_func(self,
			   	  x:Float[Array, "N CHANNELS x y"],
				  y:Float[Array, "N CHANNELS x y"],
				  key: Key,
				  SAMPLES)->Float[Array, "N"]:
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			NCA state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		key : jax.random.PRNGKey
			Jax random number key. Only useful for loss functions that are stochastic (i.e. subsampled).
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		x_obs = x[:,:self.OBS_CHANNELS]
		y_obs = y[:,:self.OBS_CHANNELS]
		if self.GRAD_LOSS:
			v_perception = jax.vmap(self.NCA_model.perception,in_axes=0,out_axes=0)
			x_obs = v_perception(x_obs)
			y_obs = v_perception(y_obs)
			x_obs = x_obs.at[:,self.OBS_CHANNELS:].set(0.1*x_obs[:,self.OBS_CHANNELS:])
			y_obs = y_obs.at[:,self.OBS_CHANNELS:].set(0.1*y_obs[:,self.OBS_CHANNELS:])
		return self._loss_func(x_obs,y_obs,key,self.LOSS_TIME_CHANNEL_MASK)
		#return loss.vgg(x_obs,y_obs,key)
		#return loss.l2(x_obs,y_obs)
		#return loss.random_sampled_euclidean(x_obs, y_obs, key, SAMPLES=SAMPLES)
	@eqx.filter_jit
	def intermediate_reg(self,x,full=True):
		"""
		Intermediate state regulariser - tracks how much of x is outwith [0,1]
		
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH

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
		if not full:
			x = x[:,:self.OBS_CHANNELS]
		return jnp.mean(jnp.abs(x)+jnp.abs(x-1)-1)
	
	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  STATE_REGULARISER=1.0,
			  WARMUP=64,
			  LOSS_SAMPLING = 64,
			  LOG_EVERY=40,
			  CLEAR_CACHE_EVERY=100,
			  WRITE_IMAGES=True,
			  UPDATE_DATA_EVERY=1,
			  LOSS_FUNC_STR = "euclidean",
			  LOOP_AUTODIFF = "checkpointed",
			  SPARSE_PRUNING = False,
			  TARGET_SPARSITY = 0.5,
			  key=jax.random.PRNGKey(int(time.time()))):
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
		STATE_REGULARISER : float optional
			Strength of intermediate state regulariser. Defaults to 1.0
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
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).
		Returns
		-------
		None
		"""
		
		if LOSS_FUNC_STR=="l2":
			self._loss_func = loss.l2
		elif LOSS_FUNC_STR=="l1":
			self._loss_func = loss.l1
		elif LOSS_FUNC_STR=="vgg":
			self._loss_func = loss.vgg
		elif LOSS_FUNC_STR=="euclidean":
			self._loss_func = loss.euclidean
		elif LOSS_FUNC_STR=="spectral":
			self._loss_func = loss.spectral
		elif LOSS_FUNC_STR=="spectral_full":
			self._loss_func = loss.spectral_weighted
		elif LOSS_FUNC_STR=="rand_euclidean":
			#def _loss_func(self,x,y,dummy_key):
			#	return loss.random_sampled_euclidean(x,y,key)
			self._loss_func = lambda x,y,dummy_key:loss.random_sampled_euclidean(x,y,key=key)




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
			key : jax.random.PRNGKey, optional
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
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff,nca_static,x,y,t,key):
				# Gradient and values of loss function computed here
				_nca = eqx.combine(nca_diff,nca_static)
				v_nca = jax.vmap(_nca,in_axes=(0,None,0),out_axes=0,axis_name="N") # boundary is independant of time N
				vv_nca = lambda x,callback,key_array:jax.tree_util.tree_map(v_nca,x,callback,key_array)
				reg_log = jnp.zeros(len(x))
				v_intermediate_reg = lambda x:jax.numpy.array(jax.tree_util.tree_map(self.intermediate_reg,x))
				_loss_func = lambda x,y,key:self.loss_func(x,y,key,SAMPLES=LOSS_SAMPLING)
				v_loss_func = lambda x,y,key_array:jax.numpy.array(jax.tree_util.tree_map(_loss_func,x,y,key_array))				
				
				# Structuring this as function and lax.scan speeds up jit compile a lot

				def nca_step(carry,j): # function of type a,b -> a
					key,x,reg_log = carry
					key = jax.random.fold_in(key,j)
					key_array = key_pytree_gen(key,(len(x),x[0].shape[0]))
					x = vv_nca(x,self.BOUNDARY_CALLBACK,key_array)				
					reg_log+=v_intermediate_reg(x)
					return (key,x,reg_log),None

				#(key,x,reg_log),_ = jax.lax.scan(nca_step,(key,x,reg_log),xs=jnp.arange(t))
				(key,x,reg_log),_ = eqx.internal.scan(nca_step,(key,x,reg_log),xs=jnp.arange(t),kind=LOOP_AUTODIFF)
				
				loss_key = key_pytree_gen(key, (len(x),))
				losses = v_loss_func(x, y, loss_key)
				mean_loss = jnp.mean(losses)+STATE_REGULARISER*(jnp.mean(reg_log)/t)
				return mean_loss,(x,losses)
			
			nca_diff,nca_static = nca.partition()
			loss_x,grads = compute_loss(nca_diff,nca_static,x,y,t,key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			(mean_loss,(x,losses)) = loss_x
			return nca,x,y,t,opt_state,key,mean_loss,losses
		
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
				print(f"Clearing cache at step {i}")
				jax.clear_caches()
			key = jax.random.fold_in(key,i)

			#nca,opt_state,(mean_loss,(x,losses)) = make_step(nca, x, y, t, opt_state,key)
			nca,x_new,y_new,t,opt_state,key,mean_loss,losses = make_step(nca, x, y, t, opt_state,key)
			loss_diff = mean_loss - best_loss


			pbar.set_postfix({'loss': mean_loss,'best loss': best_loss,'loss diff':loss_diff})

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
				self.LOGGER.tb_training_loop_log_sequence(losses, x_new, i, nca,write_images=WRITE_IMAGES,LOG_EVERY=LOG_EVERY)
			
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
				if loss_diff>loss_diff_thresh or i<WARMUP:
					x,y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, key)
				
				
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.NCA_model = nca
						self.NCA_model.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
		
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
			self.LOGGER.tb_training_end_log(self.NCA_model,x,t*2*x[0].shape[0],self.BOUNDARY_CALLBACK)