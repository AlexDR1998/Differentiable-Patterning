import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import numpy as onp
from einops import repeat, rearrange

from jaxtyping import Float,Array,Key
from Common.model.abstract_model import AbstractModel
class perturbation(AbstractModel):
    """
        Equinox module that applies a learned perturbation to the NCA state.
        The perturbation can be local, global, or flat; and can affect all channels, only observed channels, only hidden channels, or a single observed channel.
        The perturbation expects the NCA state in the shape (Batch, C, X, Y), and applies the same perturbation across all batch elements.
    """
    location: Array
    values: Array
    mode: dict
    OBS_CHANNELS: int
    CHANNELS: int
    WIDTH: float

    def __init__(self,mode,CHANNELS,OBS_CHANNELS,x,WIDTH,key):
        # Location is normalised (0 to 1) coordinates
        self.WIDTH = WIDTH
        self.location = jr.uniform(key,(2,),minval=0,maxval=1)
        self.mode=mode
        self.OBS_CHANNELS = OBS_CHANNELS
        self.CHANNELS = CHANNELS
        CH = {
            "all":np.s_[:1,:self.CHANNELS],
            "obs":np.s_[:1,:self.OBS_CHANNELS],
            "hidden":np.s_[:1,self.OBS_CHANNELS:],
            "single":np.s_[:1,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.mode['channel']]
        SP = {
            "global":np.s_[:,:],
            "local":np.s_[:,:],
            "flat":np.s_[:1,:1],
        }[self.mode['spatial']]
        inds = CH + SP
        
        self.values = np.zeros_like(x[inds])
    def __call__(self,x):
        CH = {
            "all":np.s_[:,:self.CHANNELS],
            "obs":np.s_[:,:self.OBS_CHANNELS],
            "hidden":np.s_[:,self.OBS_CHANNELS:],
            "single":np.s_[:,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.mode['channel']]

        values = self._spatial_mask(jax.nn.sigmoid(self.location),np.array([self.WIDTH,self.WIDTH]),self.values)
        x = x.at[CH].set(x[CH] + values)
        return x
    @eqx.filter_jit
    def _spatial_mask(self,centers,scales,x):
        if self.mode['spatial'] in ["global","flat"]:
            return x
        else:
            xs = np.linspace(0,1, x.shape[-2])
            ys = np.linspace(0,1, x.shape[-1])
            grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1)  # Shape: (H, W, 2)
            centers = rearrange(centers, 'Dim -> 1 1 Dim')
            scales = rearrange(scales, 'Dim -> 1 1 Dim')
            m = np.exp(-0.5 * (((grid - centers)**2) / scales**2))
            m = m[:,:,0] * m[:,:,1]
            m = rearrange(m, 'H W -> 1 1 H W')
            return m*x
    
    def get_values(self):
        return self.values
    
    def get_location(self):
        return self.location
    
    def regulariser(self,x,REG_FUNCS):
        reg_loss = 0.0
        for name in REG_FUNCS.keys():
            reg_loss += REG_FUNCS[name](x,self.values)
        return reg_loss