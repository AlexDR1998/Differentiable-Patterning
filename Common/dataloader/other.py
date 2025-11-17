import time
import glob
import jax
import jax.numpy as jnp
import skimage.io
from einops import reduce, rearrange, repeat
from .micropattern import process_data

def load_data_for_tilo(
    impath="../Data/video_for_tilo/*tif",
    DOWNSAMPLE=2,
    HIST_EQS=(5,95),
    PROCESSING_MODES=["hist_eq","map_to_0_1"]
    ):

    filename = glob.glob(impath)[0] # should just be one file
    data = skimage.io.imread(filename)
    data = rearrange(data,"T C H W -> T () H W C")

    data, aux = process_data(data,
        LMBR_CHANNEL=0,
        BATCH_AVERAGE=False,
        DOWNSAMPLE=DOWNSAMPLE,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS)
    data = rearrange(data,"T B H W C -> B T C H W")
    return data