import numpy as np
from PDE.model.reaction_diffusion_advection.visualize import plot_weight_kernel_boxplot, plot_weight_matrices
from Common.trainer.abstract_wandb_log import Train_log
from einops import rearrange


class PDE_Train_log(Train_log):
	"""
	Class for logging training behaviour of PDE_Trainer classes using the wandb backend.
	Public API is preserved so callers that used the TensorBoard-backed class can continue
	to call the same methods (names and signatures unchanged).
	"""

	# def __init__(self, log_dir, data, RGB_mode="RGB"):
	# 	"""
	# 	Keep the same initializer signature as the old TensorBoard-backed class
	# 	(log_dir, data, RGB_mode). Internally we initialize the wandb-based
	# 	Train_log and pass a minimal wandb_config so wandb stores run files in
	# 	the requested directory.
	# 	"""

	# 	# keep the attribute names expected by external callers
	# 	self.LOG_DIR = log_dir
	# 	self.RGB_mode = RGB_mode

	# 	# Initialize the wandb Train_log. Pass the data so the base class can
	# 	# perform its initial data logging. Use 'dir' so wandb writes run
	# 	# outputs into the requested log directory.
	# 	wandb_config = {"dir": self.LOG_DIR, "reinit": True}
	# 	super().__init__(data, wandb_config=wandb_config)

	def log_model_parameters(self, model, i):
		"""Log model parameter visualizations (weight matrices and boxplots).
		Uses the base class image/histogram helpers to send data to wandb.
		"""
		weight_matrix_figs = plot_weight_matrices(model)
		try:
			imgs = np.array(weight_matrix_figs)[:, 0]
		except Exception:
			imgs = np.array(weight_matrix_figs)
		# normalize images before logging
		imgs = self.normalise_images(imgs)
		self.log_image("Weight matrices", imgs, step=i)

		kernel_weight_figs = plot_weight_kernel_boxplot(model)
		try:
			kimgs = np.array(kernel_weight_figs)[:, 0]
		except Exception:
			kimgs = np.array(kernel_weight_figs)
		kimgs = self.normalise_images(kimgs)
		self.log_image("Input weights per channel", kimgs, step=i)

	def log_model_outputs(self, x, i):
		"""Log model outputs (images and hidden channels if present)."""
		BATCHES = len(x)
		# rearrange to [Batch, Time, X, Y, C] and take last timestep and first 3 channels
		outputs = rearrange(x, "b n c x y -> b n x y c")[:, -1, :, :, :3]
		outputs = np.array(outputs)
		outputs = self.normalise_images(outputs)
		# wandb accepts batches so we can log the whole batch at once
		self.log_image("Training outputs", outputs, step=i)

		# If hidden channels exist (channels > 4), pack them into RGB triplets and log
		if x[0].shape[1] > 4:
			hidden_channels = []
			for b in range(BATCHES):
				h = x[b][-1, 3:]
				extra_zeros = (-h.shape[0]) % 3
				hidden_channels.append(np.pad(h, ((0, extra_zeros), (0, 0), (0, 0))))
			# reshape to make RGB triplets: [B, (Z*C), X, Y] -> [B, Z, X, Y, C]
			hidden_imgs = rearrange(hidden_channels, "B (Z C) X Y -> B (Z X) Y C", C=3)
			hidden_imgs = np.array(hidden_imgs)
			hidden_imgs = self.normalise_images(hidden_imgs)
			self.log_image("Training outputs hidden channels", hidden_imgs, step=i)

	def tb_training_end_log(self, pde, x, ts, boundary_callback, write_images=True):
		"""Log the final PDE trajectory after training.

		Keeps the same signature as before. For wandb we upload the final
		trajectories as videos (one video per batch). Each video has shape
		(T, C, X, Y) where C is 1 or 3.
		"""
		t_max = ts[-1]
		t_len = len(ts)
		ts = np.linspace(0, 2 * t_max, 2 * t_len)

		trs = []
		trs_h = []
		CHANNELS = x[0].shape[1]
		for b in range(len(x)):
			_, Y = pde(ts, x[b][0])
			trs.append(Y)
			if CHANNELS > 4:
				Y_h = []
				for i in range(len(ts)):
					y_h = Y[i][4:]
					extra_zeros = (-y_h.shape[0]) % 3
					y_h = np.pad(y_h, ((0, extra_zeros), (0, 0), (0, 0)))
					y_h = np.reshape(y_h, (3, -1, y_h.shape[-1]))
					Y_h.append(y_h)
				trs_h.append(Y_h)

		trs = np.array(trs)
		if CHANNELS > 4:
			trs_h = np.array(trs_h)

		# log one video per batch for the full trajectory (RGB channels)
		# trs: [B, N, C, X, Y] -> for each b -> [N, C, X, Y]
		for b in range(trs.shape[0]):
			video = np.array(trs[b])[:, :3, :, :]
			# ensure shape (T, C, X, Y) and values in [0,1]
			video = self.normalise_images(video)
			# wandb expects (T, C, X, Y)
			self.log_video(f"Final PDE trajectory batch_{b}", video, step=None)

		# if hidden channels exist, log them too
		if CHANNELS > 4:
			for b in range(trs_h.shape[0]):
				# trs_h[b] has shape [N, 3, something, X, Y]? We built Y_h as list of (3, Z, X, Y?)
				# reshape to (N, C, X, Y)
				video_h = np.array(trs_h[b])
				# video_h currently is [N, 3, Z, X]? attempt to reshape/trust earlier packing
				# Ensure final axis order is (N, C, X, Y)
				if video_h.ndim == 4:
					# Already (N, C, X, Y)
					pass
				else:
					# try to collapse any extra dims to (N, C, X, Y)
					video_h = video_h.reshape(video_h.shape[0], 3, video_h.shape[-2], video_h.shape[-1])
				video_h = self.normalise_images(video_h)
				self.log_video(f"Final PDE trajectory hidden channels batch_{b}", video_h, step=None)
					
				