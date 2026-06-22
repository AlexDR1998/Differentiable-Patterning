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
#from tensorflow.core.util import event_pb2
#from tensorflow.python.lib.io import tf_record
import os
import scipy as sp
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm
#from tensorflow.python.framework import tensor_util
from pathlib import Path
from typing import Union
import pickle
#import tensorflow as tf
from einops import reduce,rearrange,repeat
from scipy.ndimage import shift, center_of_mass
import itertools
# Some convenient helper functions

def get_jax_memory_stats():
  """
  Returns dict of memory usage stats for each jax device.
  """
  stats = {}
  for d in jax.devices():
    if hasattr(d, "memory_stats"):
      s = d.memory_stats() or {}
      for k, v in s.items():
        if isinstance(v, (int, float)):
          stats[f"memory/device_{d.id}/{k}"] = v
  return stats


def squarish(H):
  a = int(H**0.5)
  while a > 0:
    if H % a == 0:
      return a, H // a
    a -= 1

def index_to_param_list(index,n_processes,full_hyperparameters):
  """
    Take a Dict of arrays of hyperparameters, and return a list of n_processes dicts of hyperparameters,
    such that all hyperparameter combinations are enumerated and split over the n_processes.
    index selects which of the n_processes to return.
  """
  
  keys = list(full_hyperparameters.keys())
  values = [full_hyperparameters[k] for k in keys]
  all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
  return all_combinations[index::n_processes]


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