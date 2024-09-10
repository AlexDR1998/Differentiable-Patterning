import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import numpy as np
from PDE.model.reaction_diffusion_advection.visualize import plot_weight_kernel_boxplot,plot_weight_matrices
from Common.trainer.abstract_tensorboard_log import Train_log
from einops import rearrange



class PDE_Train_log(Train_log):
	"""
		Class for logging training behaviour of PDE_Trainer classes, using tensorboard
	"""
	def __init__(self,log_dir,data,RGB_mode="RGB"):
		"""
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""

		self.LOG_DIR = log_dir
		self.RGB_mode = RGB_mode


		train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)

		#--- Log the target image and initial condtions
		with train_summary_writer.as_default():
			for i in range(len(data[0])):
				outputs = [im[i,:1] for im in data]
				tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",outputs),step=i,max_outputs=len(outputs))
				#for b in range(len(data)):
				#    if self.RGB_mode=="RGB":
						#tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[0,:,:3,...]),step=0,max_outputs=data.shape[0])
						#tf.summary.image('True sequence RGB',np.einsum("ncxy->nxyc",data[b][:,:3,...]),step=b,max_outputs=data[b].shape[0])
					
				#        tf.summary.image('True sequence RGB, batch '+str(b),np.einsum("ncxy->nxyc",data[b][i][np.newaxis,:3,...]),step=i,max_outputs=data[b].shape[0])
				#elif self.RGB_mode=="RGBA":
				#    #tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[0,:,:4,...]),step=0,max_outputs=data.shape[0])
				#    tf.summary.image('True sequence RGBA',np.einsum("ncxy->nxyc",data[b][:,:4,...]),step=b,max_outputs=data[b].shape[0])
			
		self.train_summary_writer = train_summary_writer

	def log_model_parameters(self, model, i):
		with self.train_summary_writer.as_default():
			weight_matrix_figs = plot_weight_matrices(model)
			tf.summary.image("Weight matrices",np.array(weight_matrix_figs)[:,0],step=i,max_outputs=len(weight_matrix_figs))
			
			kernel_weight_figs = plot_weight_kernel_boxplot(model)
			tf.summary.image("Input weights per channel",np.array(kernel_weight_figs)[:,0],step=i,max_outputs=len(kernel_weight_figs))
				

	def log_model_outputs(self, x, i):


		with self.train_summary_writer.as_default():
			BATCHES = len(x)
			tf.summary.image('Training outputs ',rearrange(x,"b n c x y ->b n x y c")[:,-1,:,:,:3],step=i,max_outputs=BATCHES)
			
			if x[0].shape[1] > 4:
				hidden_channels = []
				for b in range(BATCHES):
					h = x[b][-1,3:]
					extra_zeros = (-h.shape[0])%3
					hidden_channels.append(np.pad(h,((0,extra_zeros),(0,0),(0,0))))
				tf.summary.image('Training outputs hidden channels',rearrange(hidden_channels, "B (Z C) X Y ->B (Z X) Y C",C=3),step=i,max_outputs=BATCHES)

	
	def tb_training_end_log(self,pde,x,ts,boundary_callback,write_images=True):
		"""
		

			Log trained NCA model trajectory after training

		"""

		#print(nca)
		t_max = ts[-1]
		t_len = len(ts)
		ts = np.linspace(0,2*t_max,2*t_len)
		with self.train_summary_writer.as_default():
			trs = []
			trs_h = []
			CHANNELS = x[0].shape[1]
			for b in range(len(x)):
				
				_,Y =pde(ts,x[b][0])
				trs.append(Y)
				Y_h = []
				
				if CHANNELS>4:
					for i in range(len(ts)):
						y_h = Y[i][4:]
						extra_zeros = (-y_h.shape[0])%3
						y_h = np.pad(y_h,((0,extra_zeros),(0,0),(0,0)))
						y_h = np.reshape(y_h,(3,-1,y_h.shape[-1]))
						Y_h.append(y_h)
					#print(t_h.shape)
					trs_h.append(Y_h)
			trs = np.array(trs)
			if CHANNELS > 4:
				trs_h = np.array(trs_h)
			for i in range(len(ts)):
				tf.summary.image("Final PDE trajectory",rearrange(trs,"B N C X Y->B N X Y C")[:,i,:,:,:3],step=i)
				if CHANNELS > 4:
					tf.summary.image("Final PDE trajectory hidden channels",rearrange(trs_h,"B N C X Y->B N X Y C")[:,i],step=i)
					
				