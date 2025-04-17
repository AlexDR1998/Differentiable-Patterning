import time
import jax
from math import floor,ceil
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
#import pandas as pd
import skimage
from pprint import pprint
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import os
import scipy as sp
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm
from tensorflow.python.framework import tensor_util
from pathlib import Path
from typing import Union
import pickle
import tensorflow as tf
from einops import reduce,rearrange,repeat
from scipy.ndimage import shift, center_of_mass
# Some convenient helper functions
#@jax.jit


def squarish(H):
  a = int(H**0.5)
  while a > 0:
    if H % a == 0:
      return a, H // a
    a -= 1



def save_pickle(data, path: Union[str, Path], overwrite: bool = False):
    """
    Taken from https://github.com/google/jax/issues/2116

    Parameters
    ----------
    path : Union[str, Path]
        path to filename.
    overwrite : bool, optional
        Overwrite existing filename. The default is False.

    Raises
    ------
    RuntimeError
        file already exists.

    Returns
    -------
    None.

"""
    suffix = ".pickle"
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    
def load_pickle(path: Union[str, Path]):
    
    suffix = '.pickle'
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def key_array_gen(key,shape):
	"""
	Parameters
	----------
	key : jax.random.PRNGKey, 
		Jax random number key.
	shape : tuple of ints
		Shape to broadcast to

	Returns
	-------
	key_array : uint32[shape,2]
		array of random keys
	"""
	shape = list(shape)
	shape.append(2)
	key_array = jax.random.randint(key,shape=shape,minval=0,maxval=2_147_483_647,dtype="uint32")
	return key_array

def key_pytree_gen(key,shape):
	"""
	
	
	Parameters
	----------
	key : jax.random.PRNGKey, 
		Jax random number key.
	shape : tuple of ints
		Shape to broadcast to

	Returns
	-------
	key_array : uint32[shape,2]
		array of random keys
	"""
	#print(shape)
	shape = list(shape)
	shape.append(2)
	key_array = jax.random.randint(key,shape=shape,minval=0,maxval=2_147_483_647,dtype="uint32")
	key_array = list(key_array)
	return key_array

#def key_array_gen_pytree(key,BATCHES,N):
#	key_array = []
#	for i in range(BATCHES):		

def grad_norm(grad):
	"""
	Normalises each vector/matrix in grad 

	Parameters
	----------
	grad : NCA/pytree

	Returns
	-------
	grad : NCA/pytree

	"""
	w_where = lambda l: l.weight
	b_where = lambda l: l.bias
	w1 = grad.layers[3].weight/(jnp.linalg.norm(grad.layers[3].weight)+1e-8)
	w2 = grad.layers[5].weight/(jnp.linalg.norm(grad.layers[5].weight)+1e-8)
	b2 = grad.layers[5].bias/(jnp.linalg.norm(grad.layers[5].bias)+1e-8)
	grad.layers[3] = eqx.tree_at(w_where,grad.layers[3],w1)
	grad.layers[5] = eqx.tree_at(w_where,grad.layers[5],w2)
	grad.layers[5] = eqx.tree_at(b_where,grad.layers[5],b2)
	return grad

def load_textures(filename_sequence,impath_textures="../Data/dtd/images/",downsample=2,crop_square=False,crop_factor=1):
  images = []
  sizes = []
  for filename in filename_sequence:
    im = skimage.io.imread(impath_textures+filename)[::downsample,::downsample]
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
		im = skimage.io.imread(impath_emojis+filename)[::downsample,::downsample]
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

def load_loss_log(summary_dir):
	"""
	Returns the loss logged in tensorboard as an array
	Parameters
	----------
	summary_dir : string
	The directory where the tensorboard log is stored
	
	Returns
	-------
	steps : array ints
	timesteps
	losses : array float32
	losses at timesteps
	"""
	def my_summary_iterator(path):
		for r in tf_record.tf_record_iterator(path):
			yield event_pb2.Event.FromString(r)
	steps = []
	losses =[]
	for filename in os.listdir(summary_dir):
		if filename!="plugins":
			path = os.path.join(summary_dir, filename)
			for event in my_summary_iterator(path):
				for value in event.summary.value:
					if value.tag=="Mean Loss":
						t = tensor_util.MakeNdarray(value.tensor)
						steps.append(event.step)
						losses.append(t)
	return steps,losses

def load_trajectory_log(summary_dir):
  """
    Returns the NCA states at target times

    Parameters
    ----------
    summary_dir : string
      The directory where the tensorboard log is stored

    Returns
    -------
      steps : array ints
        timesteps
      losses : array float32
        losses at timesteps
  """
  def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
      yield event_pb2.Event.FromString(r)
  steps = []
  trajectory =[]
  
  
  
  for filename in os.listdir(summary_dir):
    if filename!="plugins":
      path = os.path.join(summary_dir, filename)
      for event in my_summary_iterator(path):
        for value in event.summary.value:
          if value.tag=="Trained NCA dynamics RGBA":
            t = tensor_util.MakeNdarray(value.tensor)
            steps.append(event.step)
            trajectory.append(t)
  return steps,trajectory

def load_micropattern_time_series_nodal_lef_cer(
      impath,
      downsample=4,
      BATCH_AVERAGE=False,
      VERBOSE=False,
      EXP_MODES=[1]
      ):
  
  """
  Experiment layout was as follows:
  -------------------------------------------------
  | 1         | 2         | 3         | 4         |
  | 0µM CHIR  | 1µM CHIR  | 2µM CHIR  | 3µM CHIR  |
  |           |           |           |           |
  -------------------------------------------------
  | 5         | 6         | 7         |           |
  | 4µM CHIR  | SB/LDN    | SB/LDN    |           |
  |           | @0h       | @24h      |           |

  so we don't have the same subdirectories for each timepoint
  """
  
  
  filenames = glob.glob(impath,recursive=True)
  filenames = list(sorted(filenames))
  is_tif = lambda x: ".tif" in x
  filenames = list(filter(is_tif,filenames))
  where_func = lambda filenames,label:label in filenames
  filenames_0h = list(filter(lambda x:where_func(x,"/0h"),filenames))
  filenames_6h = list(filter(lambda x:where_func(x,"/6h"),filenames))
  filenames_12h = list(filter(lambda x:where_func(x,"/12h"),filenames))
  filenames_24h = list(filter(lambda x:where_func(x,"/24h"),filenames))
  filenames_36h = list(filter(lambda x:where_func(x,"/36h"),filenames))
  filenames_48h = list(filter(lambda x:where_func(x,"/48h"),filenames))
  filenames_ordered = [
    filenames_0h,
    filenames_6h,
    filenames_12h,
    filenames_24h,
    filenames_36h,
    filenames_48h
  ]

  filenames_ordered = [[list(filter(lambda x:where_func(x,f"/{i}/"),F)) for i in EXP_MODES] for F in filenames_ordered]
  filenames_ordered = [[ft for ft in filename_times if ft] for filename_times in filenames_ordered]
  if VERBOSE:
    pprint(filenames_ordered)
  mean_0_std_1 = lambda arr: (arr-jnp.mean(arr,axis=(1,2),keepdims=True))/(jnp.std(arr,axis=(1,2),keepdims=True))
  map_to_0_1 = lambda arr: (arr-jnp.min(arr,axis=(1,2),keepdims=True))/(jnp.max(arr,axis=(1,2),keepdims=True)-jnp.min(arr,axis=(1,2),keepdims=True))
  saturate = lambda arr: jax.nn.sigmoid(arr)
  mult_by_lmbr = lambda arr: arr*arr[...,:1]
  def pad_to_full_width(arr):
    X_pad = (1080-arr.shape[1])/2
    Y_pad = (1080-arr.shape[2])/2
    arr = jnp.pad(arr,((0,0),(floor(X_pad),ceil(X_pad)),(floor(Y_pad),ceil(Y_pad)),(0,0)))
    return arr
  

  ims = []
  for filename_times in tqdm(filenames_ordered):
    
    ims_timestep = []
    for filename_conditions in filename_times:

      if VERBOSE:
       print(f"-------- Loading batch of {len(filename_conditions)} images ----------------")
      ims_cond = []
      for f_str in filename_conditions:
        _im = skimage.io.imread(f_str)
        ims_cond.append(_im)
        if VERBOSE:
          print(f"File {f_str} loaded with shape {_im.shape}")
      
      ims_cond = jnp.array(ims_cond,dtype="float32")
      ims_cond = align_centre_of_mass(ims_cond)
      
      if BATCH_AVERAGE:
        ims_cond = reduce(ims_cond,"BATCH C X Y ->() C X Y","mean")
      
      ims_cond = mean_0_std_1(ims_cond)
      ims_cond = saturate(ims_cond)
      ims_cond = map_to_0_1(ims_cond)
      ims_cond = mult_by_lmbr(ims_cond)
      
      ims_cond = pad_to_full_width(ims_cond)
      
      if VERBOSE:
        print(ims_cond.shape)
      ims_cond = reduce(ims_cond,"BATCH (X x2) (Y y2) C -> BATCH X Y C","mean",x2=downsample,y2=downsample)
      ims_cond = rearrange(ims_cond,"BATCH X Y C -> BATCH C X Y")
      ims_cond = jnp.array(ims_cond)

      ims_timestep.append(ims_cond)
      if VERBOSE:
        print(ims_cond.shape)
    ims.append(ims_timestep)
  return ims

def load_micropattern_time_series(impath,downsample=4,BATCH_AVERAGE=False,VERBOSE=False,times=[0,12,24,36,48,60]):
  filenames = glob.glob(impath)
  filenames = list(sorted(filenames))
  where_func = lambda filenames,label:label in filenames
  filenames_0h = list(filter(lambda x:where_func(x,"_0h"),filenames))
  filenames_12h = list(filter(lambda x:where_func(x,"_12h"),filenames))
  filenames_24h = list(filter(lambda x:where_func(x,"_24h"),filenames))
  filenames_36h = list(filter(lambda x:where_func(x,"_36h"),filenames))
  filenames_48h = list(filter(lambda x:where_func(x,"_48h"),filenames))
  filenames_60h = list(filter(lambda x:where_func(x,"_60h"),filenames))
  filenames_ordered = [filenames_0h,filenames_12h,filenames_24h,filenames_36h,filenames_48h,filenames_60h]
  # filenames_ordered = [
  #   list(filter(lambda x:where_func(x,f"_{i}h"),filenames)) for i in times
  # ]
  
  mean_0_std_1 = lambda arr: (arr-jnp.mean(arr,axis=(1,2),keepdims=True))/(jnp.std(arr,axis=(1,2),keepdims=True))
  map_to_0_1 = lambda arr: (arr-jnp.min(arr,axis=(1,2),keepdims=True))/(jnp.max(arr,axis=(1,2),keepdims=True)-jnp.min(arr,axis=(1,2),keepdims=True))
  saturate = lambda arr: jax.nn.sigmoid(arr)
  mult_by_lmbr = lambda arr: arr*arr[:,:,:,3:4]

  ims = []
  for filenames in tqdm(filenames_ordered):
    if VERBOSE:
      print(len(filenames))
    ims_timestep = []
    for f_str in filenames:
      if VERBOSE:
        print(f_str)
      ims_timestep.append(skimage.io.imread(f_str))
    
    ims_timestep = jnp.array(ims_timestep,dtype="float32")
    if BATCH_AVERAGE:
      ims_timestep = reduce(ims_timestep,"BATCH (X x2) (Y y2) C -> () X Y C","mean",x2=downsample,y2=downsample)
    else:
      ims_timestep = reduce(ims_timestep,"BATCH (X x2) (Y y2) C -> BATCH X Y C","mean",x2=downsample,y2=downsample)
    ims_timestep = mean_0_std_1(ims_timestep)
    ims_timestep = saturate(ims_timestep)
    ims_timestep = map_to_0_1(ims_timestep)
    ims_timestep = mult_by_lmbr(ims_timestep)
    ims_timestep = rearrange(ims_timestep,"BATCH X Y C -> BATCH C X Y")
    ims.append(ims_timestep)
    if VERBOSE:
      print(ims_timestep.shape)
    
  return ims




def load_micropattern_smad23_lef1(impath,downsample=4,VERBOSE=False,BATCH_AVERAGE=False):
  filenames = glob.glob(impath)
  filenames = list(sorted(filenames))
  where_func = lambda filenames,label:label in filenames
  #filenames_label = list(filter(lambda x:where_func(x,label),filenames))
  filenames_0h = list(filter(lambda x:where_func(x,"_0h"),filenames))
  filenames_6h = list(filter(lambda x:where_func(x,"_6h"),filenames))
  filenames_12h = list(filter(lambda x:where_func(x,"_12h"),filenames))
  filenames_24h = list(filter(lambda x:where_func(x,"_24h"),filenames))
  filenames_36h = list(filter(lambda x:where_func(x,"_36h"),filenames))
  filenames_48h = list(filter(lambda x:where_func(x,"_48h"),filenames))
  filenames_ordered = [filenames_0h,filenames_6h,filenames_12h,filenames_24h,filenames_36h,filenames_48h]

  
  mean_0_std_1 = lambda arr: (arr-jnp.mean(arr,axis=(1,2),keepdims=True))/(jnp.std(arr,axis=(1,2),keepdims=True))
  map_to_0_1 = lambda arr: (arr-jnp.min(arr,axis=(1,2),keepdims=True))/(jnp.max(arr,axis=(1,2),keepdims=True)-jnp.min(arr,axis=(1,2),keepdims=True))
  saturate = lambda arr: jax.nn.sigmoid(arr)
  mult_by_lmbr = lambda arr: arr*arr[...,1:2]
  reshape = lambda arr:rearrange(arr,"BATCH X Y C ->BATCH C X Y")

  ims = []
  for filenames in tqdm(filenames_ordered):
    if VERBOSE:
      print(len(filenames))
    ims_timestep = []
    for f_str in filenames:
      _im = skimage.io.imread(f_str)
      ims_timestep.append(_im)
      if VERBOSE:
        print(_im.shape,f_str)
    
    ims_timestep = jnp.array(ims_timestep,dtype="float32")
    if BATCH_AVERAGE:
      ims_timestep = reduce(ims_timestep,"BATCH (X x2) (Y y2) C -> () X Y C","mean",x2=downsample,y2=downsample)
    else:
      ims_timestep = reduce(ims_timestep,"BATCH (X x2) (Y y2) C -> BATCH X Y C","mean",x2=downsample,y2=downsample)
    ims_timestep = mean_0_std_1(ims_timestep)
    ims_timestep = saturate(ims_timestep)
    ims_timestep = map_to_0_1(ims_timestep)
    ims_timestep = mult_by_lmbr(ims_timestep)
    ims_timestep = reshape(ims_timestep)
    ims.append(ims_timestep)
    if VERBOSE:
      print(ims_timestep.shape)
  return ims


   

def load_micropattern_radii(impath):
	filenames = glob.glob(impath)
	filenames = list(sorted(filenames))
	#print(sorted(filenames))
	ims = []
	for f_str in filenames:
		ims.append(skimage.io.imread(f_str))
	#print(jax.tree_util.tree_structure(ims))
	
	normalise = lambda arr : arr/np.max(arr,axis=(0,1))
	pad = lambda arr : np.pad(arr,((10,10),(10,10),(0,0)))
	mask_out = lambda arr,mask : np.where(np.repeat(mask[0][:,:,np.newaxis],4,axis=-1),arr,np.zeros_like(arr))
	reshape = lambda arr:np.einsum("xyc->cxy",arr)
	just_mask = lambda mask:mask[0][np.newaxis]
	shapes = lambda arr: arr.shape[-1]
	def stack_x0(arr,mask):
		x0 = np.zeros_like(arr).astype(float)
		masked_arr = np.ma.array(arr,mask=~np.repeat(mask[0][np.newaxis],4,axis=0).astype(bool))
		#print(masked_arr)
		x0[1] = mask[0].astype(x0.dtype)#*masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
		x0[3] = mask[0].astype(x0.dtype)#*masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
		x0[1]*= masked_arr[1].mean()
		x0[3]*= masked_arr[3].mean()
		return np.stack((x0,arr),axis=0)
	ims = list(map(lambda x: pad(normalise(x)),ims))
	masks = list(map(adhesion_mask_convex_hull_circle,tqdm(ims)))
	ims = list(map(mask_out,ims,masks))
	ims = list(map(reshape,ims))
	ims = list(map(stack_x0,ims,masks))
	masks = list(map(just_mask,masks))
	shapes = list(map(shapes,ims))
	
	
	#ims = jax.tree_util.treedef_tuple(ims)
	#print(jnp.mean(ims))
	return ims,masks,shapes


def downsample_padder(arr,downsample):
  """Pads arrays with extra zeros if needed such that it can be properly downsampled by downsample

    Assumes array in shape X Y C
  Args:
      arr (_type_): _description_
      downsample (_type_): _description_
  """
  #print(arr.shape)
  if arr.shape[0] % downsample != 0:
    arr = np.pad(arr,((0,downsample-(arr.shape[0] % downsample)),(0,0),(0,0)))
  if arr.shape[1] % downsample != 0:
    arr = np.pad(arr,((0,0),(0,downsample-(arr.shape[1] % downsample)),(0,0)))
  
  return arr

def pad_to_biggest(ims):
  """takes a list of images [array[X Y C]] and pads the X and Y dimensions to that of the biggest one
  """
  # Determine the maximum height and width among all images
  max_height = max(im.shape[0] for im in ims)
  max_width = max(im.shape[1] for im in ims)
  print("Max height:",max_height)
  print("Max width:",max_width)
  padded = []
  for im in ims:
    h, w, _ = im.shape
    pad_top = (max_height - h) // 2
    pad_bottom = max_height - h - pad_top
    pad_left = (max_width - w) // 2
    pad_right = max_width - w - pad_left
    # Pad only the spatial dimensions (channels remain unchanged)
    padded_im = np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right),(0, 0)), mode="constant")
    padded.append(padded_im)
  return padded

def load_micropattern_ellipse(impath,DOWNSAMPLE,BATCH_AVERAGE=False):
  filenames = glob.glob(impath)
  filenames = list(sorted(filenames))
  #print(sorted(filenames))
  ims = []
  for f_str in filenames:
    ims.append(skimage.io.imread(f_str))
  #print(jax.tree_util.tree_structure(ims))

  downsample = lambda arr : reduce(downsample_padder(arr,DOWNSAMPLE),"(X x) (Y y) C -> X Y C","mean",x=DOWNSAMPLE,y=DOWNSAMPLE)  # noqa: E731
  
  normalise = lambda arr : arr/np.max(arr,axis=(0,1))  # noqa: E731
  mask_out = lambda arr,mask : np.where(np.repeat(mask[0][:,:,np.newaxis],4,axis=-1),arr,np.zeros_like(arr))  # noqa: E731
  reshape = lambda arr:np.einsum("xyc->cxy",arr)  # noqa: E731
  just_mask = lambda mask:mask[0][np.newaxis]  # noqa: E731
  shapes = lambda arr: arr.shape[-1]  # noqa: E731
  def stack_x0(arr,mask):
    x0 = np.zeros_like(arr).astype(float)
    masked_arr = np.ma.array(arr,mask=~np.repeat(mask[0][np.newaxis],4,axis=0).astype(bool))
    #print(masked_arr)
    x0[1] = mask[0].astype(x0.dtype)#*masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
    x0[3] = mask[0].astype(x0.dtype)#*masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
    x0[1]*= masked_arr[1].mean()
    x0[3]*= masked_arr[3].mean()
    return np.stack((x0,arr),axis=0)
  ims = list(map(normalise,ims))
  ims = pad_to_biggest(ims)
  #ims = np.array(ims)
  ims = np.array([downsample_padder(im,DOWNSAMPLE) for im in ims])
  if BATCH_AVERAGE:
    ims = reduce(ims,"B (X x) (Y y) C -> () X Y C","mean",x=DOWNSAMPLE,y=DOWNSAMPLE)
  else:
    ims = reduce(ims,"B (X x) (Y y) C -> B X Y C","mean",x=DOWNSAMPLE,y=DOWNSAMPLE)
  ims = list(ims)
  masks = list(map(adhesion_mask_convex_hull_ellipse,tqdm(ims)))
  ims = list(map(mask_out,ims,masks))
  ims = list(map(reshape,ims))
  ims = list(map(stack_x0,ims,masks))
  masks = list(map(just_mask,masks))
  shapes = list(map(shapes,ims))

  return ims,masks,shapes


def load_micropattern_shape_array(impath,DOWNSAMPLE,BATCH_AVERAGE=False):
  """_summary_

  Args:
      impath (string): path to files
      DOWNSAMPLE (int): downsampling ratio
      BATCH_AVERAGE (bool, optional): Average data across batches. Defaults to False.

  Returns:
      Array [BATCH, X, Y, C]: _description_
  """
  filenames = glob.glob(impath)
  filenames = list(sorted(filenames))
  ims = []
  for f_str in filenames:
    ims.append(skimage.io.imread(f_str))
  mean_0_std_1 = lambda arr: (arr-jnp.mean(arr,axis=(1,2),keepdims=True))/(jnp.std(arr,axis=(1,2),keepdims=True))
  map_to_0_1 = lambda arr: (arr-jnp.min(arr,axis=(1,2),keepdims=True))/(jnp.max(arr,axis=(1,2),keepdims=True)-jnp.min(arr,axis=(1,2),keepdims=True))
  saturate = lambda arr: jax.nn.sigmoid(arr)
  mult_by_lmbr = lambda arr: arr*arr[:,:,:,3:4]
  ims = pad_to_biggest(ims)
  ims = list(map(lambda a:downsample_padder(a,DOWNSAMPLE),ims))
  ims = jnp.array(ims,dtype="float32")
  
  if BATCH_AVERAGE:
    ims = reduce(ims,"BATCH (X x2) (Y y2) C -> () X Y C","mean",x2=DOWNSAMPLE,y2=DOWNSAMPLE)
  else:
    ims = reduce(ims,"BATCH (X x2) (Y y2) C -> BATCH X Y C","mean",x2=DOWNSAMPLE,y2=DOWNSAMPLE)
  ims = mean_0_std_1(ims)
  ims = saturate(ims)
  ims = map_to_0_1(ims)
  ims = mult_by_lmbr(ims)
  ims = rearrange(ims,"BATCH X Y C -> BATCH C X Y")
  return ims

def load_micropattern_shape_sequence(impath,DOWNSAMPLE,BATCH_AVERAGE):
  CHANNELS=[
     "FOXA2",
     "SOX17",
     "TBXT",
     "LMBR",
     "CER",
     "LEF1",
     "NODAL",
     "LEF1",
     #"SMAD23"
  ]
  true_data = load_micropattern_shape_array(impath,DOWNSAMPLE,BATCH_AVERAGE)
  masks = adhesion_mask_convex_hull(rearrange(true_data,"B C X Y -> X Y B C"))

  key = jr.PRNGKey(int(time.time()))
  n_channels = len(CHANNELS)
  # Expand the mask to have one channel per synthetic condition.
  mask_expanded = repeat(masks, "X Y -> C X Y", C=n_channels)
  # Create synthetic initial conditions by sampling random values where the mask is True, zero elsewhere.
  synthetic_initial_conditions = jnp.where(
    mask_expanded,
    jax.random.uniform(key, shape=(n_channels,) + masks.shape),
    0.0
  )

  return true_data,masks,synthetic_initial_conditions





def load_micropattern_triangle(impath):
  filenames = glob.glob(impath)
  filenames = list(sorted(filenames))
  #print(sorted(filenames))
  ims = []
  for f_str in filenames:
    ims.append(skimage.io.imread(f_str))
  #print(jax.tree_util.tree_structure(ims))

  normalise = lambda arr : arr/np.max(arr,axis=(0,1))
  pad = lambda arr : np.pad(arr,((10,10),(10,10),(0,0)))
  mask_out = lambda arr,mask : np.where(np.repeat(mask[:,:,np.newaxis],4,axis=-1),arr,np.zeros_like(arr))
  reshape = lambda arr:np.einsum("xyc->cxy",arr)
  just_mask = lambda mask:mask[0][np.newaxis]
  shapes = lambda arr: arr.shape[-1]
  def stack_x0(arr,mask):
    x0 = np.zeros_like(arr).astype(float)
    masked_arr = np.ma.array(arr,mask=~np.repeat(mask[np.newaxis],4,axis=0).astype(bool))
    #print(masked_arr)
    x0[1] = mask.astype(x0.dtype)#*masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
    x0[3] = mask.astype(x0.dtype)#*masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
    x0[1]*= masked_arr[1].mean()
    x0[3]*= masked_arr[3].mean()
    return np.stack((x0,arr),axis=0)
  ims = list(map(lambda x: pad(normalise(x)),ims))
  masks = list(map(adhesion_mask_convex_hull,tqdm(ims)))
  ims = list(map(mask_out,ims,masks))
  ims = list(map(reshape,ims))
  ims = list(map(stack_x0,ims,masks))
  masks = list(map(just_mask,masks))
  shapes = list(map(shapes,ims))


  #ims = jax.tree_util.treedef_tuple(ims)
  #print(jnp.mean(ims))
  return ims,masks,shapes

def load_sequence_A(name,impath):
  """
    Loads a single batch of timesteps of experimental data

    Parameters
    ----------
    name : string
      the name of the file

    Returns
    -------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches
  """


  I_0h = skimage.io.imread(impath+"0h/"+name+"_0h.ome.tiff")#[::2,::2]
  I_24h = skimage.io.imread(impath+"24h/A/"+name+"_24h.ome.tiff")#[::2,::2]
  I_36h = skimage.io.imread(impath+"36h/A/"+name+"_36h.ome.tiff")#[::2,::2]
  I_48h = skimage.io.imread(impath+"48h/A/"+name+"_48h.ome.tiff")#[::2,::2]
  #I_60h = skimage.io.imread(impath+name+"_60h.ome.tiff")#[::2,::2]
  I_0h = I_0h[np.newaxis]/np.max(I_0h,axis=(0,1))
  I_24h = I_24h[np.newaxis]/np.max(I_24h,axis=(0,1))
  I_36h = I_36h[np.newaxis]/np.max(I_36h,axis=(0,1))
  I_48h = I_48h[np.newaxis]/np.max(I_48h,axis=(0,1))
  #I_60h = I_60h[np.newaxis]/np.max(I_60h,axis=(0,1))
  data = np.stack((I_0h,I_24h,I_36h,I_48h))
  return data

def load_sequence_B(name,impath):
  """
    Loads a single batch of timesteps of experimental data. Same as load_sequence_A,
    except the B dataset has different (less) time slices

    Parameters
    ----------
    name : string
      the name of the file

    Returns
    -------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches
  """


  I_24h = skimage.io.imread(impath+name+"_24h.ome.tiff")#[::2,::2]
  I_36h = skimage.io.imread(impath+name+"_36h.ome.tiff")#[::2,::2]
  I_48h = skimage.io.imread(impath+name+"_48h.ome.tiff")#[::2,::2]
  
  
  I_24h = I_24h[np.newaxis]/np.max(I_24h)
  I_36h = I_36h[np.newaxis]/np.max(I_36h)
  I_48h = I_48h[np.newaxis]/np.max(I_48h)
  data = np.stack((I_24h,I_36h,I_48h))
  return data

def load_sequence_batch(N_BATCHES):

  """
    Loads a randomly selected batch of image sequences

    Parameters
    ----------
    N_BATCHES : int
      How many batches of image sequences to load

    Returns
    -------
    data_batches : flaot32 array [T,N_BATCHES,size,size,4]
      Batches of image sequences, stacked along dimension 1
  """

  names_A = ["A1_F1","A1_F2","A1_F3","A1_F4","A1_F5",
           "A1_F6","A1_F7","A1_F8","A1_F9","A1_F10",
           "A1_F11","A1_F12","A1_F13","A1_F14","A1_F15"]
  names_selected = np.random.choice(names_A,N_BATCHES,replace=False)
  data_batches = load_sequence_A(names_selected[0])
  for i in range(1,N_BATCHES):
    data_batches = np.concatenate((data_batches,load_sequence_A(names_selected[i])),axis=1)
  return data_batches

def load_sequence_ensemble_average(impath,masked=True,rscale=1.0):
  """
    Loads all image sequences and averages across them,
    to create image sequence of ensemble averages

    Parameters
    ----------
    masked : boolean
      controls whether to apply the adhesion mask to the data

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

    Returns
    -------
    data : float32 array [T,size,size,4]

    mask : boolean array [size,size]
  """

  """
  names_A = ["A1_F1","A1_F2","A1_F3","A1_F4","A1_F5",
           "A1_F6","A1_F7","A1_F8","A1_F9","A1_F10",
           "A1_F11","A1_F12","A1_F13","A1_F14","A1_F15"]
  data_batches = load_sequence_A(names_A[0])
  for i in range(1,len(names_A)):
    data_batches = np.concatenate((data_batches,load_sequence_A(names_A[i])),axis=1)
  data = np.mean(data_batches,axis=1,keepdims=True)
  """

  I_0h = skimage.io.imread(impath+"ensemble_averages/A/AVG_0h.tif")
  I_24h = skimage.io.imread(impath+"ensemble_averages/A/AVG_24h.tif")
  I_36h = skimage.io.imread(impath+"ensemble_averages/A/AVG_36h.tif")
  I_48h = skimage.io.imread(impath+"ensemble_averages/A/AVG_48h.tif")

  I_0h = I_0h[np.newaxis]/np.max(I_0h,axis=(0,1))
  I_24h = I_24h[np.newaxis]/np.max(I_24h,axis=(0,1))
  I_36h = I_36h[np.newaxis]/np.max(I_36h,axis=(0,1))
  I_48h = I_48h[np.newaxis]/np.max(I_48h,axis=(0,1))
  
  data = np.stack((I_0h,I_24h,I_36h,I_48h))
  if masked:
    mask = adhesion_mask(data,rscale)
    zs = np.zeros(data.shape)
    data = np.where(np.repeat(np.repeat(mask[np.newaxis],4,axis=0)[:,:,:,:,np.newaxis],4,axis=-1),data,zs)
    return data, mask
  else:
    return data


def adhesion_mask(data,rscale=1.0):
  """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere
    
    Parameters
    ----------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

    Returns
    -------
    mask : boolean array [1,size,size]
      Array with circle of 1/0 indicating likely presence/lack of adhesive surface in micropattern
  """

  #thresh = data[...,-1]# use LMBR staining?
  
  thresh = np.mean(data,axis=-1) 
  thresh = sp.ndimage.gaussian_filter(thresh,5)
  
  
  k = thresh>np.mean(thresh)
  
  regions = skimage.measure.regionprops(skimage.measure.label(k))
  cell_culture = regions[0]

  x0, y0 = cell_culture.centroid
  r = cell_culture.major_axis_length / 2.

  def cost(params):
      x0, y0, r = params
      coords = skimage.draw.disk((y0, x0), r, shape=k.shape)
      template = np.zeros_like(k)
      template[coords] = 1
      return -np.sum(template == k)

  x0, y0, r = sp.optimize.fmin(cost, (x0, y0, r))
  mask = np.zeros(k.shape,dtype="float32")
  
  r*=rscale
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      mask[i,j] = (i-x0)**2+(j-y0)**2<r**2
  print(mask.shape)
  return mask,x0,y0,r

def adhesion_mask_convex_hull_circle(data,rscale=1.0):
  """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere.
    
    Parameters
    ----------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

      
    Returns
    -------
    mask : boolean array [1,size,size]
      Array with circle of 1/0 indicating likely presence/lack of adhesive surface in micropattern
  """
  
  thresh = np.mean(data,axis=-1) 
  thresh = sp.ndimage.gaussian_filter(thresh,1)  
  
  k = thresh>np.mean(thresh)
  k = skimage.morphology.convex_hull_image(k,tolerance=0.1)
  
  regions = skimage.measure.regionprops(skimage.measure.label(k))
  cell_culture = regions[0]

  x0, y0 = cell_culture.centroid
  r = cell_culture.major_axis_length / 2.

  def cost(params):
      x0, y0, r = params
      coords = skimage.draw.disk((y0, x0), r, shape=k.shape)
      template = np.zeros_like(k)
      template[coords] = 1
      return -np.sum(template == k)

  x0, y0, r = sp.optimize.fmin(cost, (x0, y0, r),disp=False)
  mask = np.zeros(k.shape,dtype="float32")
  
  r*=rscale
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      mask[i,j] = (i-y0)**2+(j-x0)**2<r**2
  #print(mask.shape)
  return mask,x0,y0,r,k
  #return mask




def adhesion_mask_convex_hull(data):
  """
    Given data output from load_sequence_*, returns a binary mask representing the convex hull where cells can adhere.
    
    Parameters
    ----------
    data : float32 array [X,Y,...]
      data to be masked.

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

      
    Returns
    -------
    mask : boolean array [X,Y]
      Array with circle of 1/0 indicating likely presence/lack of adhesive surface in micropattern
  """
  
  if len(data.shape) == 3:
    thresh = reduce(data,"X Y C -> X Y","mean")
  elif len(data.shape) == 4:
    thresh = reduce(data,"X Y B C -> X Y","mean")
  elif len(data.shape) == 5:
    thresh = reduce(data,"X Y B T C -> X Y","mean")
  else:
    raise ValueError("Data must be 3,4 or 5 dimensional")
  thresh = sp.ndimage.gaussian_filter(thresh,1)  
  print(thresh.shape)
  k = thresh>np.mean(thresh)
  print(k.shape)
  k = skimage.morphology.convex_hull_image(k,tolerance=0.9)
  
  
  return k


def adhesion_mask_convex_hull_ellipse(data,angle=0.4):
    """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere.

    Parameters
    ----------
    data : float32 array [T,1,size,size,4]
        timesteps (T) of RGBA images. Dummy index of 1 for number of batches

    rscale : float32
        scales how much bigger or smaller the radius of the mask is

    Returns
    -------
    mask : boolean array [1,size,size]
        Array with circle of 1/0 indicating likely presence/lack of adhesive surface in micropattern
    """

    thresh = np.mean(data,axis=-1) 
    thresh = sp.ndimage.gaussian_filter(thresh,1)  

    k = thresh>np.mean(thresh)
    k = skimage.morphology.convex_hull_image(k,tolerance=0.1)

    regions = skimage.measure.regionprops(skimage.measure.label(k))
    cell_culture = regions[0]

    x0, y0 = cell_culture.centroid
    r = cell_culture.major_axis_length / 2.
    r0 = r
    r1 = r
    def cost(params):
        x0, y0, r0, r1, angle = params
        coords = skimage.draw.ellipse(y0, x0, r0, r1, shape=k.shape,rotation = angle)
        template = np.zeros_like(k)
        template[coords] = 1
        return -np.sum(template == k)**2

    x0, y0, r0, r1, angle= sp.optimize.fmin(cost, (x0, y0, r0,r1, angle),disp=False)
    mask = np.zeros(k.shape,dtype="float32")
    
    coords = skimage.draw.ellipse(y0, x0, r0, r1, shape=k.shape,rotation = angle)
    
    mask[coords] = 1
    
    return mask,x0,y0,(r0,r1),k

def adhesion_mask_batch(data):
  """
    Applies ashesion_mask but to a batch of different initial conditions
  
    Parameters
    ----------
    data : float32 array [T,N_BATCHES,size,size,4]
      Batch of N_BATCHES image sequences

    Returns
    -------
    masks : boolean array [N_BATCHES,size,size]
      Batch of adhesion masks corresponding to each image sequence

  """



  N_BATCHES = data.shape[1]
  mask0 = adhesion_mask(data[:,0:1])
  print(mask0.shape)
  masks = np.repeat(mask0,N_BATCHES,axis=0)#np.zeros((N_BATCHES,mask0.shape[0],mask0.shape[1]))
  print(masks.shape)
  for i in range(1,N_BATCHES):
    masks[i] = adhesion_mask(data[:,i:i+1])
  return masks





def shift_image(img, shift_val):
  """
  Shift a 2D image by a fractional amount using bilinear interpolation with periodic boundaries.
  
  Parameters
  ----------
  img : numpy.ndarray
    2D array with shape (H, W)
  shift_val : tuple of float
    Shift (row_shift, col_shift)
    
  Returns
  -------
  shifted_img : numpy.ndarray
    Shifted image (H, W)
  """
  H, W = img.shape
  row_shift, col_shift = shift_val

  # Create meshgrid of coordinates
  rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
  # Compute floating source coordinates
  src_rows = rows - row_shift
  src_cols = cols - col_shift

  # Compute indices for bilinear interpolation
  r0 = np.floor(src_rows).astype(int)
  c0 = np.floor(src_cols).astype(int)
  r1 = r0 + 1
  c1 = c0 + 1

  # Compute interpolation weights
  dr = src_rows - r0
  dc = src_cols - c0

  # Apply periodic boundary conditions by wrapping indices
  r0_mod = np.mod(r0, H)
  r1_mod = np.mod(r1, H)
  c0_mod = np.mod(c0, W)
  c1_mod = np.mod(c1, W)

  # Get pixel values using periodic indices
  I00 = img[r0_mod, c0_mod]
  I01 = img[r0_mod, c1_mod]
  I10 = img[r1_mod, c0_mod]
  I11 = img[r1_mod, c1_mod]

  shifted = (1 - dr) * (1 - dc) * I00 + (1 - dr) * dc * I01 + dr * (1 - dc) * I10 + dr * dc * I11
  return shifted

def align_centre_of_mass(img_stack):
  """
  Given a stack of images with shape (N, C, H, W) where the spatial structure
  is roughly circular, shift each image so that the center of mass (computed from
  the sum over channels) aligns with the center of the image.
  
  Parameters
  ----------
  img_stack : numpy.ndarray
    Stack of images with shape (N, H, W, C).

  Returns
  -------
  aligned_stack : numpy.ndarray
    Stack of aligned images with the same shape.
  """
  aligned_stack = img_stack.copy()
  N, H, W, C = aligned_stack.shape
  target = (H / 2.0, W / 2.0)
  
  # Precompute coordinate grid for center of mass calculation
  rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

  for i in range(N):
    img = aligned_stack[i]
    # Compute weight by summing over channels
    weight = img.sum(axis=-1)
    total = np.sum(weight)
    # If total weight is 0, keep image unchanged
    if total == 0:
      com = target
    else:
      com_row = np.sum(rows * weight) / total
      com_col = np.sum(cols * weight) / total
      com = (com_row, com_col)

    # Calculate shift needed: positive shift moves image content downward/right.
    shift_val = (target[0] - com[0], target[1] - com[1])
    
    # Shift each channel separately using bilinear interpolation
    for c in range(C):
      #aligned_stack[i, c] = shift_image(img[c], shift_val)
      aligned_stack = aligned_stack.at[i,:,:,c].set(shift_image(img[:,:,c], shift_val))
          
  return aligned_stack
  

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