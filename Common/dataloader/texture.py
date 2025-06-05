import numpy as np
import skimage.io as sio


def load_textures(filename_sequence,impath_textures="../Data/dtd/images/",downsample=2,crop_square=False,crop_factor=1):
  images = []
  sizes = []
  for filename in filename_sequence:
    im = sio.imread(impath_textures+filename)[::downsample,::downsample]
    if crop_square:
      s= int(min(im.shape[0],im.shape[1])/crop_factor)
      im = im[:s,:s]
      sizes.append(s)
      #im = im[np.newaxis] / 255.0

    im = im/255.0
    images.append(im)
  if crop_square:
    min_s = min(sizes)
    for i,im in enumerate(images):
      images[i] = im[:min_s,:min_s]
  data = np.array(images)
  data = data[np.newaxis]
  data = np.einsum("btxyc->btcxy",data)
  return data    