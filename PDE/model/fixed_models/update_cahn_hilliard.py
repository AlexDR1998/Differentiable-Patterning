import equinox as eqx
from jaxtyping import Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    gamma: float
    D: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 gamma = 1.0,
                 D = 0.1
                 ):
        """Cahn-Hilliard phase separation model.

        mu = X^3 - X - gamma*Lap(X)
        dX = D*Lap(mu)

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): grid spacing
            KERNEL_SCALE (int, optional): spatial operator kernel size. Defaults to 1.
            gamma (float, optional): structure lengthscale. Defaults to 1.0.
            D (float, optional): Diffusion strength. Defaults to 0.1.
        """
        self.gamma = gamma
        self.D = D
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"1 x y"],
                 args)->Float[Scalar, "1 x y"]:

        mu = X**3 - X - self.gamma*self.ops.Lap(X)
        return self.D*self.ops.Lap(mu)
