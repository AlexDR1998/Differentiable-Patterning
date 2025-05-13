import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange
import io
from PIL import Image



def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	buf = io.BytesIO()
	figure.savefig(buf, format='png')
	plt.close(figure)
	buf.seek(0)
	image = Image.open(buf)
	#print("Image size: ",image.size)
	image = np.array(image)
	#print("Image array shape: ",image.shape)
	image = rearrange(image, "x y c -> () x y c")
	return image
	# # Save the plot to a PNG in memory.
	# buf = io.BytesIO()
	# plt.savefig(buf, format='png')
	# # Closing the figure prevents it from being displayed directly inside
	# # the notebook.
	# plt.close(figure)
	# buf.seek(0)
	# # Convert PNG buffer to TF image
	# image = tf.image.decode_png(buf.getvalue(), channels=4)
	# # Add the batch dimension
	# image = tf.expand_dims(image, 0)
	# return image


def plot_weight_matrices(nca):
	"""
	Plots heatmaps of NCA layer weights

	Parameters
	----------
	nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
		the NCA object to plot weights of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	
	
	#w1 = nca.layers[0].weight[:,:,0,0]
	#w2 = nca.layers[2].weight[:,:,0,0]
	figs = []
	ws = nca.get_weights()
	for i,w in enumerate(ws):
		w = np.squeeze(w)
		if w.ndim==2:
			figure = plt.figure(figsize=(5,5))
			col_range = max(np.max(w),-np.min(w))
			plt.imshow(w,cmap="seismic",vmax=col_range,vmin=-col_range)
			plt.ylabel("Output")
			if i==0:
				plt.xlabel(r"N_CHANNELS$\star$ KERNELS")
			else:
				plt.xlabel("Input")
			
			figs.append(plot_to_image(figure))
			# figs.append(w)
		
		# figure = plt.figure(figsize=(5,5))
		# col_range = max(np.max(w2),-np.min(w2))
		# plt.imshow(w2,cmap="seismic",vmax=col_range,vmin=-col_range)
		# plt.xlabel("Input from previous layer")
		# plt.ylabel("NCA state increments")
		# figs.append(plot_to_image(figure))
	return figs

def plot_weight_kernel_boxplot(nca):
	"""
	Plots boxplots of NCA 1st layer weights per kernel, sorted by which channel they correspond to

	Parameters
	----------
	nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
		the NCA object to plot weights of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	#w = nca.layers[0].weight[:,:,0,0]
	w = nca.get_weights()[0]
	w = np.squeeze(w)
	N_KERNELS = nca.N_FEATURES // nca.N_CHANNELS
	N_CHANNELS = nca.N_CHANNELS
	K_STR = nca.KERNEL_STR.copy()
	if "GRAD" in K_STR:
		for i in range(len(K_STR)):
			if K_STR[i]=="GRAD":
				K_STR[i]="GRAD X"
				K_STR.insert(i,"GRAD Y")
	
	#weights_split = []
	figs = []
	for k in range(N_KERNELS):
		#w_k = w[:,k::N_KERNELS]
		w_k = w[:,k*N_CHANNELS:(k+1)*N_CHANNELS]
		figure = plt.figure(figsize=(5,5))
		plt.boxplot(w_k)
		plt.xlabel("Channels")
		plt.ylabel("Weights")
		plt.title(K_STR[k]+" kernel weights")
		#plt.plot()
		figs.append(plot_to_image(figure))
	return figs


def plot_weight_matrices_show(nca):
	"""
	Plots heatmaps of NCA layer weights. Returns fig,ax objects with subplots

	Parameters
	----------
	nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
		the NCA object to plot weights of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	
	
	w1 = nca.layers[0].weight[:,:,0,0]
	w2 = nca.layers[2].weight[:,:,0,0]
	
	fig, ax = plt.subplots(1, 2,sharey=True,figsize=(14, 6),squeeze=True)
	
	
	col_range = max(np.max(w1),-np.min(w1))
	ax[0].imshow(w1,cmap="seismic",vmax=col_range,vmin=-col_range)
	ax[0].set_ylabel("Output")
	ax[0].set_xlabel(r"N_CHANNELS$\star$ KERNELS")
	
	

	
	#figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w2),-np.min(w2))
	ax[1].imshow(w2.T,cmap="seismic",vmax=col_range,vmin=-col_range)
	ax[1].set_ylabel("Input from previous layer")
	ax[1].set_xlabel("NCA state increments")
	
	return fig,ax


def plot_weight_kernel_boxplot_show(nca):
	"""
	Plots boxplots of NCA 1st layer weights per kernel, sorted by which channel they correspond to

	Parameters
	----------
	nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
		the NCA object to plot weights of

	Returns
	-------
	figs : list of images
		a list of images

	"""
	w = nca.layers[0].weight[:,:,0,0]
	N_KERNELS = nca.N_FEATURES // nca.N_CHANNELS
	K_STR = nca.KERNEL_STR.copy()
	if "DIFF" in K_STR:
		for i in range(len(K_STR)):
			if K_STR[i]=="DIFF":
				K_STR[i]="DIFF X"
				K_STR.insert(i,"DIFF Y")
	
	#weights_split = []
	#figs = []
	fig,ax = plt.subplots(1,N_KERNELS,sharey=True,figsize=(18,6))
	for k in range(N_KERNELS):
		w_k = w[:,k::N_KERNELS]
		
		
		ax[k].boxplot(w_k.T)
		ax[k].set_xlabel("Channels")
		ax[k].set_ylabel("Weights")
		ax[k].set_title(K_STR[k]+" kernel weights")
		#plt.plot()
		#figs.append(plot_to_image(figure))
	return fig,ax

def plot_weight_matrix_kernel_subplots(nca):
	ws = nca.get_weights()
	w_in = nca.layers[0].weight[:,:,0,0]
	w_out= nca.layers[2].weight[:,:,0,0]
	print(len(ws))
	N_KERNELS = nca.N_FEATURES // nca.N_CHANNELS
	K_STR = nca.KERNEL_STR.copy()
	K_STR = sort_kstr(K_STR)
	if "GRAD" in K_STR:
		for i in range(len(K_STR)):
			if K_STR[i]=="GRAD":
				K_STR[i]="GRAD X"
				K_STR.insert(i,"GRAD Y")
	
	#weights_split = []
	#figs = []
	fig,ax = plt.subplots(1,N_KERNELS + 1,sharey=True,figsize=(12,6))
	for k in range(N_KERNELS):
		#w_k = w_in[:,k::N_KERNELS]
		w_k = w_in[:,k*nca.N_CHANNELS:(k+1)*nca.N_CHANNELS]
		col_range = max(np.max(w_k),-np.min(w_k))
		ax[k].imshow(w_k,cmap="seismic",vmax=col_range,vmin=-col_range)
		ax[k].set_xlabel("Channel inputs")
		ax[k].set_ylabel("Outputs")
		ax[k].set_title(K_STR[k])
		#plt.plot()
		#figs.append(plot_to_image(figure))
	
	col_range = max(np.max(w_out),-np.min(w_out))
	ax[-1].imshow(w_out.T,cmap="seismic",vmax=col_range,vmin=-col_range)
	ax[-1].set_ylabel("Input from previous layer")
	ax[-1].set_xlabel("NCA state increments")
	ax[-1].set_title("Output layer")
	return fig,ax



def my_animate(img,clip=True):
	"""
	Boilerplate code to produce matplotlib animation
	Parameters
	----------
	img : float32 or int array [N,rgb,_,_]
		img must be float in range [0,1] 
	"""
	if clip:
		im_min = 0
		im_max = 1
		img = np.clip(img,im_min,im_max)
	else:
		im_min = np.min(img)
		im_max = np.max(img)

	
	
	img = np.einsum("ncxy->nxyc",img)
	frames = [] # for storing the generated images
	fig = plt.figure()
	for i in range(img.shape[0]):
		
		frames.append([plt.imshow(img[i],vmin=im_min,vmax=im_max,animated=True)])
		
	ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=0)
	plt.show()


def sort_kstr(K_STR):
	# Want list of kernel types in a specific order for plotting things correctly
	K_SORTED = []
	for s in ["ID","DIFF","GRAD","AV","LAP"]:
		if s in K_STR:
			K_SORTED.append(s)
	return K_SORTED