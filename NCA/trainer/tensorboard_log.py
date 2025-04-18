from einops import rearrange
from NCA.NCA_visualiser import plot_weight_matrices,plot_weight_kernel_boxplot
import numpy as np
from Common.utils import squarish
from tqdm import tqdm
from jaxtyping import Float,Array,Key,PyTree
import os
LOG_BACKEND = os.environ.get("LOG_BACKEND", "wandb")
#if LOG_BACKEND=="wandb":
from Common.trainer.abstract_wandb_log import Train_log
#elif LOG_BACKEND=="tensorboard":
#	from Common.trainer.abstract_tensorboard_log import Train_log

class NCA_Train_log(Train_log):
	"""
		Class for logging training behaviour of NCA_Trainer classes
	"""

	def log_model_parameters(self,nca,i):
		"""Log model parameters

		Args:
			nca : nca model class (PyTree)
			i : training step
		"""
		
		
		w1,w2,b2 = nca.get_weights()
		w1 = np.squeeze(w1)
		w2 = np.squeeze(w1)
		b2 = np.squeeze(b2)		
		self.log_histogram('Train/input_layer_weights',w1,step=i)
		self.log_histogram('Train/output_layer_weights',w2,step=i)
		self.log_histogram('Train/output_layer_bias',b2,step=i)				
		weight_matrix_figs = plot_weight_matrices(nca)
		self.log_image("Train/weight_matrices",np.array(weight_matrix_figs)[:,0],step=i)
				
		kernel_weight_figs = plot_weight_kernel_boxplot(nca)
		self.log_image("Train/input_weights_per_kernel",np.array(kernel_weight_figs)[:,0],step=i)

	def log_model_outputs(self,
					      x: PyTree[Float[Array, "N CHANNELS x y"], "B"], # type: ignore
						  i):
		
		BATCHES = len(x)
		for b in range(BATCHES):
			self.log_image(
				'Train/trajectory_batch_'+str(b),
				self.normalise_images(rearrange(x[b][:,:3,...],"Batch Channel x y -> Batch x y Channel")),
				step=i)
			
		if x[0].shape[1] > 3:
			b=0
			hidden_channels = x[b][:,3:]
			extra_zeros = (-hidden_channels.shape[1])%3
			hidden_channels = np.pad(hidden_channels,((0,0),(0,extra_zeros),(0,0),(0,0)))
			_cy,_cx = squarish(hidden_channels.shape[1]//3)
			hidden_channels_r = rearrange(hidden_channels,"Batch (cx cy C) x y -> Batch (cx x) (cy y) C",C=3,cy=_cy,cx=_cx)
			self.log_image(
				f'Train/trajectory_batch_{b}_hidden_channels',
				hidden_channels_r,
				step=i)
	
	def tb_training_loop_log_sequence(self,losses,x,i,model,write_images=True,LOG_EVERY=10):
		
		self.log_histogram("Train/loss",losses,step=i)
		self.log_scalar("Train/mean_loss",np.mean(losses),step=i)

		if i%LOG_EVERY==0:
			self.log_model_parameters(model,i)
			if write_images:
				self.log_model_outputs(x,i)

	
	def tb_training_end_log(self,
						 	nca,
							x: PyTree[Float[Array, "N CHANNELS x y"], "B"],  # noqa: F722, F821
							t,
							boundary_callback,
							write_images=True):
		"""
		

			Log trained NCA model trajectory after training

		"""
		BATCHES = 1#len(x)
		CHANNELS = x[0].shape[1]
		print("Running final trained model for "+str(t)+" steps")
		
		for b in tqdm(range(BATCHES)):
			T =nca.run(t,x[b][0],boundary_callback[b])
			self.log_video("Evaluation/trajectory",T[:,:3],step=None)

			if CHANNELS>4:
				t_h = T[:,:,:,4:]
				extra_zeros = (-t_h.shape[1])%3
				t_h = np.pad(t_h,((0,0),(0,extra_zeros),(0,0),(0,0)))
				_cy,_cx = squarish(t_h.shape[1]//3)
				T_h = rearrange(t_h,"Time (cx cy C) x y  -> Time C (cx x) (cy y)",C=3,cy=_cy,cx=_cx)
				self.log_video("Evaluation/trajectory_hidden_channels",T_h,step=None)
			
		self.finish()
				


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