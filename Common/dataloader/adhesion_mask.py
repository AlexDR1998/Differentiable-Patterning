import numpy as np
#import pandas as pd
import skimage

#from tensorflow.core.util import event_pb2
#from tensorflow.python.lib.io import tf_record

import scipy as sp

#import tensorflow as tf
from einops import reduce,rearrange,repeat
from scipy.ndimage import shift, center_of_mass
import itertools









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
  mask0 = adhesion_mask(data[:,0:1])[0]
  print(mask0.shape)
  masks = np.repeat(mask0,N_BATCHES,axis=0)#np.zeros((N_BATCHES,mask0.shape[0],mask0.shape[1]))
  print(masks.shape)
  for i in range(1,N_BATCHES):
    masks[i] = adhesion_mask(data[:,i:i+1])
  return masks


