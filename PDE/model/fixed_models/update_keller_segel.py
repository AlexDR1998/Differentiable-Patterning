import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar

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
                 KERNEL_SCALE=1,
                 c=3.0,
                 alpha=0.01,
                 D=1.0,
                 epsilon=0.01):
        """Keller-Segel chemotaxis model with saturating sensitivity and logistic growth.

        du = Lap(u) - div(c*u/(1+u^2) * grad(v)) + u*(1-u) - epsilon*u^3
        dv = D*Lap(v) + u - alpha*v

        Parameters
        ----------
        PADDING : str
            Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
        dx : float
            Grid spacing
        KERNEL_SCALE : int, optional
            Spatial operator kernel size. Defaults to 1.
        c : float, optional
            Chemotactic sensitivity. Defaults to 3.0.
        alpha : float, optional
            Signal decay. Defaults to 0.01.
        D : float, optional
            Signal diffusion. Defaults to 1.0.
        epsilon : float, optional
            Cubic cell death. Defaults to 0.01.
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
        return jnp.concatenate((dcells,dsignal),axis=0)
