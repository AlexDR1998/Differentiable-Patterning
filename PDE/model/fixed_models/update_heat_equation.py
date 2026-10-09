import equinox as eqx
from jaxtyping import Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    D: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 D = 0.1
                 ):
        """Heat equation: dX = D*Lap(X)

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): grid spacing
            KERNEL_SCALE (int, optional): spatial operator kernel size. Defaults to 1.
            D (float, optional): Diffusion strength. Defaults to 0.1.
        """
        self.D = D
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"1 x y"],
                 args)->Float[Scalar, "1 x y"]:

        return self.D*self.ops.Lap(X)
