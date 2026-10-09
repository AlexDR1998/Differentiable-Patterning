import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar
from einops import repeat

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    gamma: Float
    alpha: Float
    DA: float
    DB: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 DA=0.1,
                 DB=0.05,
                 alpha=jnp.linspace(0.062,0.063,10),
                 gamma=jnp.linspace(0.062,0.063,10)):
        """Gray-Scott model over a grid of (gamma, alpha) values, solved in one call.

        The state has shape [len(gamma), len(alpha), 2, x, y]; see update_gray_scott.py
        for the equations.

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): grid spacing
            KERNEL_SCALE (int, optional): spatial operator kernel size. Defaults to 1.
            DA, DB (float, optional): diffusion rates of A and B.
            alpha (array, optional): feed rates.
            gamma (array, optional): kill rates.
        """
        self.gamma=repeat(gamma,"a -> a b () () ()",b=len(alpha))
        self.alpha=repeat(alpha,"b -> a b () () ()",a=len(gamma))
        self.DA = DA
        self.DB = DB
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"a b 2 x y"],
                 args)->Float[Scalar, "a b 2 x y"]:
        v_lap = eqx.filter_vmap(self.ops.Lap,in_axes=0,out_axes=0)
        vv_lap = eqx.filter_vmap(v_lap,in_axes=0,out_axes=0)

        A = X[:,:,0:1]
        B = X[:,:,1:2]
        dA = self.DA*vv_lap(A) - A*B*B + self.alpha*(1-A)
        dB = self.DB*vv_lap(B) + A*B*B - (self.gamma + self.alpha)*B

        return jnp.concatenate((dA,dB),axis=2)
