import numpy as np
import skimage.io as sio








def load_emoji_sequence(filename_sequence,impath_emojis="../Data/Emojis/",downsample=2,crop_square=False):
	"""
		Loads a sequence of images in impath_emojis
		Parameters
		----------
		filename_sequence : list of strings
			List of names of files to load
		downsample : int
			How much to downsample the resolution - highres takes ages
	
		Returns
		-------
		images : float32 array [1,T,C,size,size]
			Timesteps of T RGB/RGBA images. Dummy index of 1 for number of batches
	"""
	images = []
	for filename in filename_sequence:
		im = sio.imread(impath_emojis+filename)[::downsample,::downsample]
		if crop_square:
			s= min(im.shape[0],im.shape[1])
			im = im[:s,:s]
		#im = im[np.newaxis] / 255.0
		im = im/255.0
		images.append(im)
	data = np.array(images)
	data = data[np.newaxis]
	data = np.einsum("btxyc->btcxy",data)
	return data