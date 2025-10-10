import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange,repeat

from Common.model.spatial_operators import Ops
class F(eqx.Module):
    ops: Ops
    gamma: float
    alpha: float
    DA: float
    DB: float
    epsilon: float
    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 DA=0.1,
                 DB=0.05,
                 alpha=jnp.linspace(0.062,0.063,10),
                 gamma=jnp.linspace(0.062,0.063,10)):
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
        
        self.gamma=repeat(gamma,"a -> a b () () ()",b=len(alpha))
        self.alpha=repeat(alpha,"b -> a b () () ()",a=len(gamma))
        self.DA = DA
        self.DB = DB        
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)
        self.epsilon = 1e-4

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:
        v_lap = eqx.filter_vmap(self.ops.Lap,in_axes=0,out_axes=0)
        vv_lap = eqx.filter_vmap(v_lap,in_axes=0,out_axes=0)

        A = X[:,:,0:1]
        B = X[:,:,1:2]
        dA = self.DA*vv_lap(A) - A*B*B + self.alpha*(1-A)
        dB = self.DB*vv_lap(B) + A*B*B - (self.gamma + self.alpha)*B
        
        return jnp.concatenate((dA,dB),axis=2)
        
        
    
