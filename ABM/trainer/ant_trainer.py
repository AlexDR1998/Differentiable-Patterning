import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import time
from ABM.trainer.data_augmenter_abm import DataAugmenter
import Common.trainer.loss as loss
from ABM.trainer.tensorboard_log import ABM_Train_log
from Common.utils import key_array_gen
from tqdm import tqdm

class AntTrainer(object):
	"""

	Train ant colony to match point cloud mnist dataset
	"""
	def __init__(self,
			     nslime,
				 BATCHES,
				 N_agents,
				 model_filename,
				 DATA_AUGMENTER = DataAugmenter,
				 int_list = [0,1],
				 alpha = 1.0,
				 directory = "models/"):
		self.nslime = nslime
		self.OBS_CHANNELS = 1#
		self.CHANNELS = self.nslime.N_CHANNELS
		self.alpha = alpha
		#self.OBS_CHANNELS = data[0].shape[1]
		print("Observable Channels: "+str(self.OBS_CHANNELS))
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(int_list,
									   		 BATCHES,
											 lattice_size=self.nslime.GRID_SIZE,
											 N_agents=N_agents,
											 channels=self.CHANNELS)
		#self.DATA_AUGMENTER.data_init()
		data = self.DATA_AUGMENTER.return_saved_data()
		
		self.BATCHES = BATCHES
		print("Batches = "+str(self.BATCHES))
		if model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			self.model_filename = model_filename
			self.IS_LOGGING = True
			self.LOG_DIR = "logs/"+self.model_filename+"/train"
			self.LOGGER = ABM_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		self.directory = directory
		self.MODEL_PATH = directory+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
		
	def loss_func(self,X,Y,key=jax.random.PRNGKey(int(time.time()))):
		"""
		
	
		Parameters
		----------
		X : (([B,N,N_agents,2],[B,N,N_agents,2]),[B,N,CHANNELS,_,_])
			agent positions, velocities and pheremone lattice
		y :  (([B,N,N_agents,2],[B,N,N_agents,2]),[B,N,CHANNELS,_,_])
			Data to train to
		Returns
		-------
		loss : float32 
			loss 
		"""
		self.alpha = 0.1
		((p_x,v_x),ph_x) = X
		((p_y,v_y),ph_y) = Y
		
		v_pos_loss = jax.vmap(loss.sinkhorn_divergence_loss,in_axes=(0,0),out_axes=0)
		vv_pos_loss = jax.vmap(v_pos_loss,in_axes=(0,0),out_axes=0)

		v_ph_loss = jax.vmap(lambda x,y:loss.euclidean(x[:self.OBS_CHANNELS],y[:self.OBS_CHANNELS],key),in_axes=(0,0),out_axes=0)
		vv_ph_loss = jax.vmap(v_ph_loss,in_axes=(0,0),out_axes=0)
		
		pos_loss = vv_pos_loss(p_x,p_y)
		ph_loss = vv_ph_loss(ph_x,ph_y)
		#print(pos_loss.shape)
		#print(ph_loss.shape)

		return pos_loss*self.alpha+ph_loss*(1-self.alpha)
		#return vv_loss_func(p_x,p_y)
		#def _loss_func(x,y,key):
		#	x_obs = x[:,:self.OBS_CHANNELS]
		#	y_obs = y[:,:self.OBS_CHANNELS]
		#	#x_obs = x_obs[jnp.newaxis]
			#y_obs = y_obs[jnp.newaxis]
		#	return loss.vgg(x_obs,y_obs,key)
		#v_loss_func = jax.vmap(lambda x,y:_loss_func(x,y,key),in_axes=(0,0),out_axes=0)
		#vv_loss_func = jax.vmap(v_loss_func,in_axes=(0,0),out_axes=0)
		#return v_loss_func(x,y)
			
		# def _loss_func(x,y):
		# 	x_obs = x[:self.OBS_CHANNELS]
		# 	y_obs = y[:self.OBS_CHANNELS]
		# 	return loss.euclidean(x_obs,y_obs)

		
		# return vv_loss_func(x,y)
		
	

	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  WARMUP=64,
			  loop_autodiff="checkpointed",
			  key=jax.random.PRNGKey(int(time.time()))):
		"""Iterate the training loop of the model

		Args:
			t (int): Number of timesteps for the agent based model
			iters (int): Number of training iterations
			optimiser (optax GradientTransformation, optional): Which optimiser to use?. Defaults to None, in which case adam is used
			WARMUP (int, optional): _description_. Defaults to 64.
			loop_autodiff (str, "lax" or "checkpointed"): How to handle reverse-mode autodiff through loops. "lax" is a bit faster but with much larger memory use. Defaults to "checkpointed".
			key (jax PRNG, optional): Random number key. Defaults to jax.random.PRNGKey(int(time.time())).


		"""
		@eqx.filter_jit
		def make_step(nslime,Y,X,t,opt_state,key,i):
			"""_summary_

			Args:
				nslime: eqx.Module callable ((agent_pos,agent_vel),pheremones)->((agent_pos,agent_vel),pheremones)) 
					Ant colony model that updates agent positions, velocities and the underlying pheremone lattice
				Y: ((agent_pos,agent_vel),pheremones)
					True data that nslime is being trained to reproduce. 
					Each element of Y is vmapped over 2 additional axes:
						agent_pos:  f32[B,N_steps,2,N_points]
						agent_vel:  f32[B,N_steps,2,N_points]
						pheremones: f32[B,N_steps,channels,grid_size,grid_size]


				X: ((agent_pos,agent_vel),pheremones)
					State of ant colony model after previous training iteration.
					Each element of X is vmapped over 2 additional axes:
						agent_pos:  f32[B,N_steps,2,N_agents]
						agent_vel:  f32[B,N_steps,2,N_agents]
						pheremones: f32[B,N_steps,channels,grid_size,grid_size]
					
						NOTE: N_agents isn't the same as N_points. Using OT losses means
						that the number of agents doesn't have to match that of data, and
						in the datasets being used, the number of points varies

				t: int
					Number of update steps between each timestep of model
				opt_state (_type_): _description_
				key (_type_): _description_

			Returns:
				_type_: _description_
			"""
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nslime_diff,nslime_static,Y,X,t,key):
				nslime = eqx.combine(nslime_diff,nslime_static)
				
				def _nslime_run_wrapper(X,nslime):
					def _nslime_step_wrapper(X,j):# function of type a,b->a
						X = nslime(X)
						X = self.DATA_AUGMENTER.callback_training(X,i)
						return X,None
					#X,_=jax.lax.scan(_nslime_step_wrapper,X,xs=jnp.arange(t))
					X,_= eqx.internal.scan(_nslime_step_wrapper,X,xs=jnp.arange(t),kind=loop_autodiff)
					#X,_= eqx.internal.scan(_nslime_step_wrapper,X,xs=jnp.arange(t),kind="lax")
					return X
				
				#v_init_nslime = jax.vmap(lambda key:nslime.init_state(key,zero_pheremone=True),in_axes=(0),out_axes=(0,0))
				#vv_init_nslime = jax.vmap(v_init_nslime,in_axes=(0),out_axes=(0,0))
		
				v_run = jax.vmap(_nslime_run_wrapper,in_axes=(0,None),out_axes=(0,0))
				vv_run = jax.vmap(v_run,in_axes=(0,None),out_axes=(0,0))
				
				#keys = key_array_gen(key,(len(X),X[0].shape[0]))
				
				#X = vv_init_nslime(keys)
				X = vv_run(X,nslime)
				def _norm(X):
					((p,v),ph) = X
					return ((p/float(nslime.GRID_SIZE),v/float(nslime.GRID_SIZE)),ph)	
				losses = self.loss_func(_norm(X),_norm(Y))
				mean_loss = jnp.mean(jnp.array(losses))
				return mean_loss,(X,losses)
			
			nslime_diff,nslime_static = nslime.partition()
			loss_x,grads = compute_loss(nslime_diff,nslime_static,Y,X,t,key)
			
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nslime_diff)
			nslime = eqx.apply_updates(nslime,updates)
			return nslime,opt_state,loss_x
		
		# Initialise Training
		nslime = self.nslime
		nslime_diff,nslime_static = nslime.partition()
		if optimiser is None:
			schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
			self.OPTIMISER = optax.noisy_sgd(schedule)
			#self.OPTIMISER = non_negative_diffusion(learn_rate=1e-2,iters=iters)
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(nslime_diff)
		

		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		error = 0
		error_at = 0
		#Y = self.DATA_AUGMENTER.return_saved_data()
		X,Y = self.DATA_AUGMENTER.initialize_pool()
		#print(X[1].shape)
		#print(X[0][0].shape)
		
		
		#print("Data format: ")
		#print("Number of batches: "+str(len(Y)))
		#print("Batch structure:"+str(Y[0].shape))
		for i in tqdm(range(iters)):
			key = jax.random.fold_in(key,i)
			nslime,opt_state,(mean_loss,(X,losses))= make_step(nslime,Y,X,t,opt_state,key,i)
			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(jnp.array([losses]), X, i, nslime)
			
			
			
			
			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda X: jnp.any(jnp.isnan(X)), X[1]))): # Only check for NaNs in pheremone lattice
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			
			if error==0:
				X,Y = self.DATA_AUGMENTER.advance_pool(X,Y,i,key)
				if i>WARMUP:
					if mean_loss<best_loss:
						model_saved=True
						self.nslime = nslime
						self.nslime.save(self.MODEL_PATH,overwrite=True)
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
			
			
			
			
			