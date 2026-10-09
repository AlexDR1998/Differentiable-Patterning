import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    gamma: float
    alpha: float
    DA: float
    DB: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 DA=0.1,
                 DB=0.05,
                 alpha=0.06230,
                 gamma=0.06268):
        """Gray-Scott reaction-diffusion model.

        dA = DA*Lap(A) - A*B^2 + alpha*(1-A)
        dB = DB*Lap(B) + A*B^2 - (gamma+alpha)*B

        Parameters
        ----------
        PADDING : str
            Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
        dx : float
            Grid spacing
        KERNEL_SCALE : int, optional
            Spatial operator kernel size. Defaults to 1.
        DA, DB : float, optional
            Diffusion rates of A and B.
        alpha : float, optional
            Feed rate (often called F).
        gamma : float, optional
            Kill rate (often called k).
        """
        self.gamma = gamma
        self.alpha = alpha
        self.DA = DA
        self.DB = DB
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:

        A = X[0:1]
        B = X[1:2]

        dA = self.DA*self.ops.Lap(A) - A*B*B + self.alpha*(1-A)
        dB = self.DB*self.ops.Lap(B) + A*B*B - (self.gamma + self.alpha)*B

        return jnp.concatenate((dA,dB),axis=0)
