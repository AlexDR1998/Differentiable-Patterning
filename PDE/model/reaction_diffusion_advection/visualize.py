import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import io



def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)
	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=4)
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return image



def plot_weight_matrices(pde):
	"""
	Plots heatmaps of NCA layer weights

	Parameters
	----------
	pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
		the PDE solver object to plot parameters of

	Returns
	-------
	figs : list of images
		a list of images

		

	"""
	
	w_v = []
	w_d = []
	w_r = []
	w_r_p = []
	w_r_d = []

	for i in range(pde.func.N_LAYERS+1):
		if "advection" in pde.func.TERMS:
			w_v.append(pde.func.f_v.layers[2*i].weight[:,:,0,0])
		if "diffusion" in pde.func.TERMS:
			w_d.append(pde.func.f_d.layers[2*i].weight[:,:,0,0])
		if "reaction_split" in pde.func.TERMS:
			w_r_p.append(pde.func.f_r.production_layers[2*i].weight[:,:,0,0])
			w_r_d.append(pde.func.f_r.decay_layers[2*i].weight[:,:,0,0])
		if "reaction_pure" in pde.func.TERMS:
			w_r.append(pde.func.f_r.layers[2*i].weight[:,:,0,0])


	figs = []
	
	if "advection" in pde.func.TERMS:
		for i in range(pde.func.N_LAYERS+1):
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w_v[i]),-np.min(w_v[i]))
			plt.imshow(w_v[i],cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			plt.xlabel("Input")
			plt.title(f"Advection layer {i}")
			figs.append(plot_to_image(figure))
	if "diffusion" in pde.func.TERMS:
		for i in range(pde.func.N_LAYERS+1):
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w_d[i]),-np.min(w_d[i]))
			plt.imshow(w_d[i],cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			plt.xlabel("Input")
			plt.title(f"Diffusion layer {i}")
			figs.append(plot_to_image(figure))
	if "reaction_split" in pde.func.TERMS:
		for i in range(pde.func.N_LAYERS+1):
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w_r_p[i]),-np.min(w_r_p[i]))
			plt.imshow(w_r_p[i],cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			plt.xlabel("Input")
			plt.title(f"Reaction production layer {i}")
			figs.append(plot_to_image(figure))
		for i in range(pde.func.N_LAYERS+1): 
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w_r_d[i]),-np.min(w_r_d[i]))
			plt.imshow(w_r_d[i],cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			plt.xlabel("Input")
			plt.title(f"Reaction decay layer {i}")
			figs.append(plot_to_image(figure))
	if "reaction_pure" in pde.func.TERMS:
		for i in range(pde.func.N_LAYERS+1):
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w_r[i]),-np.min(w_r[i]))
			plt.imshow(w_r[i],cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			plt.xlabel("Input")
			plt.title(f"Reaction layer {i}")
			figs.append(plot_to_image(figure))
	return figs

def plot_weight_kernel_boxplot(pde):
	"""
	Plots boxplots of PDE 1st layer weights sorted by which channel they correspond to

	Parameters
	----------
	pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
		the PDE solver object to plot parameters of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	figs = []
	if "advection" in pde.func.TERMS:
		w1_v = pde.func.f_v.layers[0].weight[:,:,0,0]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w1_v.T)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title("Advection 1st layer")
		figs.append(plot_to_image(figure))
	
	if "diffusion" in pde.func.TERMS:
		w1_d = pde.func.f_d.layers[0].weight[:,:,0,0]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w1_d.T)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title("Diffusion 1st layer")
		figs.append(plot_to_image(figure))

	if "reaction_split" in pde.func.TERMS:
		w1_r_p = pde.func.f_r.production_layers[0].weight[:,:,0,0]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w1_r_p.T)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title("Reaction production 1st layer")
		figs.append(plot_to_image(figure))

		w1_r_d = pde.func.f_r.decay_layers[0].weight[:,:,0,0]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w1_r_d.T)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title("Reaction decay 1st layer")
		figs.append(plot_to_image(figure))
	
	if "reaction_pure" in pde.func.TERMS:
		w1_r = pde.func.f_r.layers[0].weight[:,:,0,0]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w1_r.T)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title("Reaction 1st layer")
		figs.append(plot_to_image(figure))

	return figs