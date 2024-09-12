import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops
class F(eqx.Module):
    ops: Ops
    c: float
    alpha: float
    D: float
    epsilon: float
    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE,
                 c=3.0,
                 alpha=0.01,
                 D=1.0,
                 epsilon=0.01):
        """Implementation of basic pattern formation model from figure 4 in Hillen & Painter "A users guide to PDE models for chemotaxis"

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): _description_
            logistic_growth_rate (float, optional): _description_. Defaults to 0.1.
            gamma (float, optional): _description_. Defaults to 10.0.
            alpha (float, optional): _description_. Defaults to 0.5.
            chi (float, optional): _description_. Defaults to 5.0.
            D (float, optional): _description_. Defaults to 0.1.
        """
        
        self.c = c
        self.alpha=alpha
        
        
        self.D=D
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)
        self.epsilon = epsilon

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:
        
        cells = X[0:1]
        signal= X[1:2]
        
        chemotactic_term = self.c*cells/(1 + cells**2)
        dcells = self.ops.Lap(cells) - self.ops.NonlinearDiffusion(chemotactic_term,signal)+cells*(1-cells)-self.epsilon*cells**3
        dsignal = self.D*self.ops.Lap(signal) + cells - self.alpha*signal
        #chemotactic_term = (cells*self.chi*(1-cells/self.gamma))/((1+self.alpha*signal)**2)

        #_a = self.ops.NonlinearDiffusion(self.D*cells,cells)
        #_b = self.ops.NonlinearDiffusion(chemotactic_term,signal)

        #dcells = _a - _b + self.logistic_growth_rate*cells*(1-cells)
        #dsignal = self.ops.Lap(signal) + cells/(1+self.phi*cells) - signal
        return jnp.concatenate((dcells,dsignal),axis=0)
    
