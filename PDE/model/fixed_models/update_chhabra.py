import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    KI: float
    KdA: float
    KdI: float
    SA: float
    SI: float
    DA: float
    DI: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 KI=1.0,
                 KdA=0.001,
                 KdI=0.008,
                 SA=0.01,
                 SI=0.01,
                 DA=0.0025, # 0.0025 or 0.014
                 DI=0.4
                 ):
        """Activator-inhibitor pattern formation model from Chhabra et al 2019.

        dA = DA*Lap(A) + SA*A^2/(KI+I) - KdA*A
        dI = DI*Lap(I) + SI*A^2 - KdI*I

        Parameters
        ----------
        PADDING : str
            Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
        dx : float
            Grid spacing
        KERNEL_SCALE : int, optional
            Spatial operator kernel size. Defaults to 1.
        KI : float, optional
            Inhibition constant.
        KdA, KdI : float, optional
            Decay rates of activator and inhibitor.
        SA, SI : float, optional
            Production rates of activator and inhibitor.
        DA, DI : float, optional
            Diffusion rates of activator and inhibitor.
        """
        self.KI = KI
        self.KdA = KdA
        self.KdI = KdI
        self.SA = SA
        self.SI = SI
        self.DA = DA
        self.DI = DI
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:

        A = X[0:1]
        I = X[1:2]

        dA = self.DA*self.ops.Lap(A) + self.SA*A*A / (self.KI + I) - self.KdA*A
        dI = self.DI*self.ops.Lap(I) + self.SI*A*A - self.KdI*I

        return jnp.concatenate((dA,dI),axis=0)
