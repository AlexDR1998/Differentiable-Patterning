from einops import rearrange,repeat
from NCA.NCA_visualiser import plot_weight_matrices,plot_weight_kernel_boxplot
import numpy as np
from Common.utils import squarish
from tqdm import tqdm
from jaxtyping import Float,Array,Key,PyTree
import os
import jax.random as jr
import time
from dotenv import load_dotenv
load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")
LOG_BACKEND = os.environ.get("LOG_BACKEND", "wandb")
# PVC_PATH = "mnt/ceph/ar-dp/"  # Path to the PVC where the data is stored
#if LOG_BACKEND=="wandb":
from Common.utils import get_jax_memory_stats
from Common.trainer.abstract_wandb_log import Train_log
#elif LOG_BACKEND=="tensorboard":
#	from Common.trainer.abstract_tensorboard_log import Train_log

class NCA_Train_log(Train_log):
	"""
		Class for logging training behaviour of NCA_Trainer classes
	"""

	def log_model_parameters(self,nca,i):  # type: ignore
		"""Log model parameters

		Args:
			nca : nca model class (PyTree)
			i : training step
		"""
		
		for idx, w in enumerate(nca.get_weights()):
			w = np.squeeze(w)
			self.log_histogram(f"Train/weight_{idx}", w, step=i)
			# print("Weight shape ",w.shape)
			if len(w.shape) == 2:
				w = repeat(w,"W H -> W H 3")
				self.log_image(f"Train/weight_image_{idx}", self.normalise_images(w), step=i)
			

	def log_model_outputs(self,x,i):
		"""
			x: Dict {"x_latent": PyTree[Float[Array, "N CHANNELS x y"], "B"],
					"x_processed": PyTree[Float[Array, "N CHANNELS x y"], "B"]}
			i: training step
		"""
		memory_stats = get_jax_memory_stats()
		for key in memory_stats:
			self.log_scalar(f"Memory/{key}",memory_stats[key],step=i)
		x_latent = x["x_latent"]
		x_processed = x["x_processed"]
		BATCHES = len(x_latent)
		for b in range(BATCHES):
			self.log_image(
				'Train/processed_batch_'+str(b),
				self.normalise_images(rearrange(x_processed[b][:,:3,...],"Batch Channel x y -> Batch x y Channel")),
				step=i)

			self.log_image(
				'Train/latent_batch_'+str(b),
				self.normalise_images(rearrange(x_latent[b][:,:3,...],"Batch Channel x y -> Batch x y Channel")),
				step=i)
			
		if x_latent[0].shape[1] > 3:
			b=0
			hidden_channels = x_latent[b][:,3:]
			extra_zeros = (-hidden_channels.shape[1])%3
			hidden_channels = np.pad(hidden_channels,((0,0),(0,extra_zeros),(0,0),(0,0)))
			_cy,_cx = squarish(hidden_channels.shape[1]//3) # type: ignore
			hidden_channels_r = rearrange(hidden_channels,"Batch (cx cy C) x y -> Batch (cx x) (cy y) C",C=3,cy=_cy,cx=_cx)
			hidden_channels_r = (np.tanh(hidden_channels_r)+1.0)/2.0
			self.log_image(
				f'Train/latent_batch_{b}_hidden_channels',
				hidden_channels_r,
				step=i)
	
	def tb_training_loop_log_sequence(self,log_dict,i,model,write_images=True,LOG_EVERY=10):
		
		for name in log_dict.keys():
			if name not in ["x_latent", "x_processed"]:
				self.log_scalar(f"Train/{name}",log_dict[name],step=i)
		if i%LOG_EVERY==0 and i>0:
			self.log_model_parameters(model,i)
			if write_images:
				self.log_model_outputs(log_dict,i)

	
	def tb_training_end_log(self, # type: ignore
						 	nca,
							# x: PyTree[Float[Array, "N CHANNELS x y"], "B"],  # noqa: F722, F821
							DATA_AUGMENTER,
							t,
								boundary_callback,
								SAVE_TRAJECTORY=False,
								write_images=True,
								key=None):
		"""
			Log trained NCA model trajectory after training

		"""
		if key is None:
			key = jr.PRNGKey(int(time.time()))
		x,y = DATA_AUGMENTER.split_x_y(1)
		x,y = DATA_AUGMENTER.data_callback(x,y,0,key)
		NUMBER_OF_IMAGES=x[0].shape[0]
		# Log true data for side by side comparison
		true_data = DATA_AUGMENTER.return_true_data()[0]
		true_data = true_data[:,:DATA_AUGMENTER.OBS_CHANNELS]
		true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
		true_data = repeat(true_data,"x y -> x y 3")
		self.log_image(
			'Evaluation/true_data',
			true_data,
			step=None
		)
		BATCHES = 1#len(x)
		CHANNELS = x[0].shape[1]

		print("Running final trained model for "+str(t)+" steps")
		
		SNAPSHOTS = []
		for b in tqdm(range(BATCHES)):
			T, latents = nca.run(t*NUMBER_OF_IMAGES, x[b][0], boundary_callback[b], SAVE_LATENTS=True)  # Shape T C x y
			self.log_video("Evaluation/trajectory",T[:,:3],step=None)
			T_snapshot = T[::t,:DATA_AUGMENTER.OBS_CHANNELS]
			T_snapshot = rearrange(T_snapshot,"Time C x y -> (C x) (Time y)")
			T_snapshot = repeat(T_snapshot,"x y -> x y 3")
			SNAPSHOTS.append(T_snapshot)
			
			if SAVE_TRAJECTORY:
				np.save(f"{PVC_PATH}output/{self.wandb_config['name']}_trajectory_{b}.npy",T[::t,:3])  # type: ignore

			extra_zeros = (-latents.shape[1])%3
			latents = np.pad(latents,((0,0),(0,extra_zeros),(0,0),(0,0)))
			_cy,_cx = squarish(latents.shape[1]//3)
			latents = rearrange(latents,"Time (cx cy C) x y  -> Time C (cx x) (cy y)",C=3,cy=_cy,cx=_cx)
			latents = (np.tanh(latents)+1.0)/2.0
			self.log_video("Evaluation/latent_trajectory",latents,step=None)

		SNAPSHOTS = np.array(SNAPSHOTS)
		self.log_image(
			'Evaluation/trajectory_snapshot',
			SNAPSHOTS,
			step=None
		)

class NCA_knockout_Train_log(NCA_Train_log):

	def __init__(
        self,
        data,
        wandb_config=None,
		knockout_time=None,
		knockout_channel=None,
    ):
		super().__init__(data, wandb_config)
		assert knockout_time is not None, "knockout_time must be provided for NCA_knockout_Train_log"
		assert knockout_channel is not None, "knockout_channel must be provided for NCA_knockout_Train_log"
		self.knockout_time = knockout_time
		self.knockout_channel = knockout_channel

	def tb_training_end_log(self,
						 	nca,
							# x: PyTree[Float[Array, "N CHANNELS x y"], "B"],  # noqa: F722, F821
							DATA_AUGMENTER,
							t,
								boundary_callback,
								SAVE_TRAJECTORY=False,
								write_images=True,
								key=None):
		"""
		

			Log trained NCA model trajectory after training

		"""
		if key is None:
			key = jr.PRNGKey(int(time.time()))
		x,y = DATA_AUGMENTER.split_x_y(1)
		x,y = DATA_AUGMENTER.data_callback(x,y,0,key)
		NUMBER_OF_IMAGES=x[0].shape[0]
		# Log true data for side by side comparison
		true_data = DATA_AUGMENTER.return_true_data()[0]
		true_data = true_data[:,:DATA_AUGMENTER.OBS_CHANNELS]
		true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
		true_data = repeat(true_data,"x y -> x y 3")
		self.log_image(
			'Evaluation/true_data',
			true_data,
			step=None
		)
		BATCHES = 1#len(x)
		CHANNELS = x[0].shape[1]

		print("Running final trained model for "+str(t)+" steps")
		
		SNAPSHOTS = []
		for b in tqdm(range(BATCHES)):

			# T =nca.run(t*NUMBER_OF_IMAGES,x[b][0],boundary_callback[b]) # Shape T C x y
			T = []
			xb = x[b][0] # C x y
			
			for step in range(t*NUMBER_OF_IMAGES):
				key = jr.fold_in(key,step)
				if step/t >= self.knockout_time:
					xb = xb.at[self.knockout_channel].set(0.0) # Set nodal channel to 0 at and after knockout time
				xb = nca(xb,boundary_callback[b],key)
				T.append(xb)
			T = np.array(T) # Shape T C x y
			
			self.log_video("Evaluation/trajectory_comp",rearrange(T[:,:9],"T (cx cy) X Y -> T cx X (cy Y)",cx=3,cy=3),step=None) # type: ignore
			_T_mono = rearrange(T[:,:9],"T (cx cy) X Y -> T () (cx X) (cy Y)",cx=3,cy=3)
			_T_mono = repeat(_T_mono,"T () x y -> T 3 x y")
			self.log_video("Evaluation/trajectory_monochrome",_T_mono,step=None) # type: ignore
			T_snapshot = T[::t,:DATA_AUGMENTER.OBS_CHANNELS]
			T_snapshot = rearrange(T_snapshot,"Time C x y -> (C x) (Time y)")
			T_snapshot = repeat(T_snapshot,"x y -> x y 3")
			SNAPSHOTS.append(T_snapshot)
			
			if SAVE_TRAJECTORY:
				np.save(f"{PVC_PATH}output/{self.wandb_config['name']}_trajectory_{b}.npy",T[::t,:3]) # type: ignore

		SNAPSHOTS = np.array(SNAPSHOTS)
		self.log_image(
			'Evaluation/trajectory_snapshot',
			SNAPSHOTS,
			step=None
		)



class aNCA_Train_log(NCA_Train_log):
	def log_model_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		
		pass
			

class kaNCA_Train_log(NCA_Train_log):
	def log_model_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		w1,w2 = nca.get_weights()		
		self.log_histogram('Input layer weights',w1,step=i)
		self.log_histogram('Output layer weights',w2,step=i)
		


class kaNCA_Train_pde_log(kaNCA_Train_log):
	def log_model_outputs(self, x, i):
		pass # Saving the trajectory outputs during training generates far too many images


# class uNCA_Train_log(NCA_Train_log):
# 	def log_model_parameters(self, nca, i):
# 		# uNCA exposes additional trainable arrays; log all weights generically.
# 		for idx, w in enumerate(nca.get_weights()):
# 			self.log_histogram(f"Train/weight_{idx}", np.squeeze(w), step=i)




class mNCA_Train_log(NCA_Train_log):
	
	def log_model_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		
		for scale,W in enumerate(nca.get_weights()):
			w1,w2,b2 = W
			w1 = np.squeeze(w1)
			w2 = np.squeeze(w1)
			b2 = np.squeeze(b2)		
			self.log_histogram(f'Input layer weights, scale {scale}',w1,step=i)
			self.log_histogram(f'Output layer weights, scale {scale}',w2,step=i)
			self.log_histogram(f'Output layer bias, scale {scale}',b2,step=i)				
			weight_matrix_figs = plot_weight_matrices(nca.subNCAs[scale])
			self.log_image(f"Weight matrices, scale {scale}",np.array(weight_matrix_figs)[:,0],step=i)
					
			kernel_weight_figs = plot_weight_kernel_boxplot(nca.subNCAs[scale])
			self.log_image(f"Input weights per kernel, scale {scale}",np.array(kernel_weight_figs)[:,0],step=i)
