import jax
import json
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import time
import jaxpruner
from PDE.trainer.data_augmenter_pde import DataAugmenter
import Common.trainer.loss as loss
from Common.model.boundary import model_boundary
from Common.trainer.custom_functions import check_training_diverged
from PDE.trainer.tensorboard_log import PDE_Train_log
from PDE.trainer.optimiser import non_negative_diffusion_chemotaxis
from PDE.model.solver.semidiscrete_solver import PDE_solver,save,load
from functools import partial
from Common.model.spatial_operators import Ops
from jaxtyping import Float, Array, Int, Scalar, Key
import diffrax
from tqdm import tqdm
from einops import repeat,rearrange
class PDE_Trainer(object):
	
	
	def __init__(self,
			     PDE_solver,
				 PDE_HYPERPARAMETERS,
				 data: Float[Array, "Batches T C W H"],
				 Ts: Float[Array, "Batches T"],
				 model_filename=None,
				 DATA_AUGMENTER = DataAugmenter,
				 BOUNDARY_MASK = None,
				 directory="models/"):
		"""
		

		Parameters
		----------
		
		PDE_solver : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [N_CHANNELS,_,_])
			PDE solver that returns T timesteps of integrated parameterised PDE model. Parameters are to be trained
		
		NCA_model : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
			trained NCA object
			
		data : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
			set of trajectories (initial conditions) to train PDE to NCA on
		
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
		#self.NCA_model = NCA_model
		self.PDE_solver = PDE_solver
		self.PDE_HYPERPARAMETERS = PDE_HYPERPARAMETERS
		# Set up variables 

		self.OBS_CHANNELS = data[0].shape[1]
		self.CHANNELS = self.PDE_solver.func.N_CHANNELS
		
		self._op = Ops(PADDING=PDE_solver.func.PADDING,
				       dx=PDE_solver.func.dx)
		
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data_true=data,
									    	 Ts=Ts,
									         hidden_channels=self.CHANNELS-self.OBS_CHANNELS)
		self.DATA_AUGMENTER.data_init()
		self.BATCHES = len(data)
		self.TRAJECTORY_LENGTH = data.shape[1]
		print("Batches = "+str(self.BATCHES))
		print(f"Observable channels: {self.OBS_CHANNELS}")
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
			self.LOG_DIR = "logs/"+self.model_filename+"/train"
			self.LOGGER = PDE_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		self.directory = directory
		self.MODEL_PATH = directory+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
	
	
	def spatial_loss_gradients(self,X: Float[Array, "T C W H"])->Float[Array, "T s W H"]:
		def _spatial(X: Float[Array, "C W H"]):
			_grad = self._op.Grad(X)
			_gx = _grad[0]
			_gy = _grad[1]
			_lap = self._op.Lap(X)
			return rearrange([_gx,_gy,_lap],"b C x y -> (b C) x y")
		return jax.vmap(_spatial,in_axes=0,out_axes=0)(X)
	@eqx.filter_jit	
	def loss_func(self,
			   	  x: Float[Array, "T C W H"],
				  y: Float[Array, "T C_OBS W H"])->Float[Array,"T"]:
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH
	
		
		Parameters
		----------
		x : float32 array [T,CHANNELS,_,_]
			NCA state
		y : float32 array [T,OBS_CHANNELS,_,_]
			data
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		x_obs = x[:,:self.OBS_CHANNELS]
		y_obs = y[:,:self.OBS_CHANNELS]
		if self.LOSS_PARAMS["LOSS_TRAJECTORY_FULL"]:
			x_obs = x_obs[::self.LOSS_PARAMS["LOSS_SAMPLING"]]
			y_obs = y_obs[::self.LOSS_PARAMS["LOSS_SAMPLING"]]
		else:
			x_obs = x_obs[-1:]
			y_obs = y_obs[-1:]
			
		#L = self._loss(x_obs,y_obs)
		L = self.LOSS_PARAMS["LOSS_FUNC"](x_obs,y_obs)
		if self.LOSS_PARAMS["GRAD_LOSS"]:
			x_obs_spatial = self.spatial_loss_gradients(x_obs)
			y_obs_spatial = self.spatial_loss_gradients(y_obs)
			L += 0.1*self.LOSS_PARAMS["LOSS_FUNC"](x_obs_spatial,y_obs_spatial)
		return L
	
	def train(self,
		      SUBTRAJECTORY_LENGTH: Int[Scalar, ""],
			  TRAINING_ITERATIONS: Int[Scalar, ""],
			  OPTIMISER=None,  
			  WARMUP=64,
			  LOG_EVERY=10,
			  LOSS_PARAMS = {"LOSS_FUNC":loss.euclidean,
							 "GRAD_LOSS":True,
							 "LOSS_SAMPLING":1,
							 "LOSS_TRAJECTORY_END":False},
			  PRUNING = {"PRUNE":False,
						 "TARGET_SPARSITY":0.9},
			  UPDATE_X0_PARAMS = {"iters":32,
						 		  "update_every":10,
								  "optimiser":optax.nadam,
								  "learn_rate":1e-3,
								  "verbose":False},
			  key=jax.random.PRNGKey(int(time.time()))):
		"""
		At each training iteration, select a random subsequence of length t to train to

		Parameters
		----------
		t : Int
			Length of sub-sequence of full data trajectory to fit PDE to
		iters : Int
			Number of training iterations.
		optimiser : optax.GradientTransformation
			the optax optimiser to use when applying gradient updates to model parameters.
			if None, constructs adamw with exponential learning rate schedule
		
		WARMUP : int optional
			Number of iterations to wait for until starting model checkpointing. Default is 64
		SAMPLING : TYPE, optional
			DESCRIPTION. The default is 8.
		key : jax.random.PRNGKey, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).


		"""
		
		self.LOSS_PARAMS = LOSS_PARAMS
		
		@eqx.filter_jit
		def make_step(pde,
					  x: Float[Array,"Batches T C W H"],
					  y: Float[Array,"Batches T C W H"],
					  ts: Float[Array,"Batches T"],
					  opt_state,
					  target_sparsity,
					  key: Key):	
			"""
			

			Parameters
			----------
			pde : object callable - (float32 [T], float32 [N_CHANNELS,_,_]) -> (float32 [T], float32 [T,N_CHANNELS,_,_])
				the PDE solver to train
			x : float32 array [BATCHES,N,CHANNELS,_,_]
				input state
			y : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				true predictions (time offset with respect to input axes)
			t : int
				number of PDE timesteps to predict - mapping X[:,i]->X[:,i+1:i+t]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jax.random.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			pde : object callable - (float32 [T], float32 [N_CHANNELS,_,_]) -> (float32 [T], float32 [T,N_CHANNELS,_,_])
				the PDE solver with updated parameters
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [BATCHES,N,CHANNELS,_,_], float32 array [BATCHES,N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated PDE state after t iterations

			"""
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(pde_diff,pde_static,x,y,ts,key):
				_pde = eqx.combine(pde_diff,pde_static)
				#v_pde = jax.vmap(lambda x:_pde(jnp.linspace([0,t,t+1]),x)[1][1:],in_axes=0,out_axes=0,axis_name="N")
				#v_pde = lambda x:_pde(jnp.linspace(0,t,t+1),x)[1][1:] # Don't need to vmap over N
				#vv_pde= lambda x: jax.tree_util.tree_map(v_pde,x) # different data batches can have different sizes
				
				v_pde = lambda x0,ts: _pde(ts,x0)[1][1:]
				vv_pde= lambda x0,ts: jax.tree_util.tree_map(v_pde,x0,ts) # different data batches can have different sizes
				v_loss_func = lambda x,y: jnp.array(jax.tree_util.tree_map(self.loss_func,x,y))
				y_pred = vv_pde(x,ts)
				losses = v_loss_func(y_pred,y)
				mean_loss = jnp.mean(losses)
				return mean_loss,(y_pred,losses)
			#jax.debug.print("Outer loop batch number: {}",len(x))
			#jax.debug.print("Outer loop X shape: {}",x[0].shape)
			pde_diff,pde_static=pde.partition()
			loss_y,grads = compute_loss(pde_diff, pde_static, x, y, ts, key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, pde_diff)
			pde = eqx.apply_updates(pde,updates)
			(mean_loss,(y,losses)) = loss_y

			if PRUNING["PRUNE"]:
				ws,tree_def = pde.get_weights()
				_,pde_static = pde.partition()
				sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=target_sparsity)
				pruner = jaxpruner.MagnitudePruning(
					sparsity_distribution_fn=sparsity_distribution,
					skip_gradients=True)
				ws = pruner.instant_sparsify(ws)[0]
				pde_diff = pde.set_weights(tree_def,ws)
				pde.combine(pde_static,pde_diff)



			return pde,x,y,ts,opt_state,mean_loss,losses,key
		
		
		###--- Initialise optimiser
		pde = self.PDE_solver
		pde_diff,pde_static = pde.partition()
		if OPTIMISER is None:
			schedule = optax.exponential_decay(1e-4, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
			self.OPTIMISER = non_negative_diffusion_chemotaxis(schedule)
		else:
			self.OPTIMISER = OPTIMISER
		opt_state = self.OPTIMISER.init(pde_diff)
		
		
		data_steps = self.DATA_AUGMENTER.data_saved[0].shape[0]
		x,y,ts = self.DATA_AUGMENTER.data_load(L=SUBTRAJECTORY_LENGTH,key=key)
		# x: Float[Array, "Batches 1 C W H"]
		# y: Float[Array, "Batches SUBTRAJECTORY_LENGTH C W H"]
		# ts: Float[Array, "Batches SUBTRAJECTORY_LENGTH"]


		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		error = 0
		error_at = 0
		if PRUNING["PRUNE"]:
			SPARSITY_SCHEDULE = jnp.concat((jnp.zeros(WARMUP),jnp.linspace(0,PRUNING["TARGET_SPARSITY"],TRAINING_ITERATIONS-WARMUP)))
		else:
			SPARSITY_SCHEDULE = jnp.zeros(TRAINING_ITERATIONS)
		#x0 = self.DATA_AUGMENTER.data_saved[0][0]
		

		print("Training nPDE with: ")
		#print(self.PDE_HYPERPARAMETERS)
		print(json.dumps(self.PDE_HYPERPARAMETERS,sort_keys=True, indent=4))

		for i in tqdm(range(TRAINING_ITERATIONS)):
			key = jax.random.fold_in(key,i)
			#print(self.DATA_AUGMENTER.data_saved[0].shape)
			"""
			# 	y output is predicted state at each timestep,
			#	x output is passed through UNCHANGED for argument donation
			# 	pde output is updated model
			# 	opt_state output is updated optimiser state
			# 	mean_loss output is	average loss over all batches
			# 	losses output is loss for each batch
			# 	key output is UNCHANGED random key, passe through for argument donation
			"""
			pde,x,y,ts,opt_state,mean_loss,losses,key = make_step(pde, x, y, ts, opt_state,SPARSITY_SCHEDULE[i],key)

			# if PRUNING["PRUNE"]:
			# 	ws,tree_def = pde.get_weights()
			# 	_,pde_static = pde.partition()
			# 	sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=SPARSITY_SCHEDULE[i])
			# 	pruner = jaxpruner.MagnitudePruning(
			# 		sparsity_distribution_fn=sparsity_distribution,
			# 		skip_gradients=True)
			# 	ws = pruner.instant_sparsify(ws)[0]
			# 	pde_diff = pde.set_weights(tree_def,ws)
			# 	pde.combine(pde_static,pde_diff)
			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, y, i, pde,LOG_EVERY=LOG_EVERY)
			
			
			# Check if training has crashed or diverged yet
			error = check_training_diverged(mean_loss,x,i)
			if error==0:
				# Do data augmentation update
				x,y,ts = self.DATA_AUGMENTER.data_callback(x=x, 
														   y=y,
														   ts=ts,
														   i=i, 
														   L=SUBTRAJECTORY_LENGTH, 
														   key=key)
				
				# Update x0 parameters
				#if i%UPDATE_X0_PARAMS["update_every"]==0 and i>WARMUP:
				#	self.DATA_AUGMENTER.update_initial_condition_hidden_channels(pde,i,UPDATE_X0_PARAMS)
									
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.PDE_solver = pde
						#self.PDE_solver.save(self.MODEL_PATH,overwrite=True)
						save(self.MODEL_PATH,self.PDE_HYPERPARAMETERS,self.PDE_solver)
						best_loss = mean_loss
						tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
			else:
				break
		
		if error==0:
			print("Training completed successfully")
		#assert model_saved, "|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|"
		if error!=0 and model_saved==False:
			print("|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|")
		elif self.IS_LOGGING and model_saved:
			x,y = self.DATA_AUGMENTER.split_x_y(1)
			self.LOGGER.tb_training_end_log(self.PDE_solver,x,self.DATA_AUGMENTER.Ts[0],self.BOUNDARY_CALLBACK)