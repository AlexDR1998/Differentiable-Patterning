import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops

class F(eqx.Module):
    ops: Ops
    a: float
    b: float
    D: float
    epsilon: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 a=0.2,
                 b=0.8,
                 D=50.0):
        """
        Schnakenberg reaction-diffusion model:
          ∂U/∂t = Lap(U) + a - U + U^2 V
          ∂V/∂t = D⋅Lap(V) + b - U^2 V

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE', or 'CIRCULAR'
            dx (float): Grid spacing
            KERNEL_SCALE (int): Scale factor for convolution kernel
            a (float): Feed rate parameter
            b (float): Removal rate parameter
            D (float): Diffusion coefficient for V
        """
        self.a = a
        self.b = b
        self.D = D
        self.ops = Ops(PADDING, dx, KERNEL_SCALE)
        self.epsilon = 1e-4

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar, "2 x y"],
                 args) -> Float[Scalar, "2 x y"]:
        # Split concentration fields
        U = X[0:1]
        V = X[1:2]

        # Reaction-diffusion kinetics
        dU = self.ops.Lap(U) + self.a - U + U**2 * V
        dV = self.D * self.ops.Lap(V) + self.b - U**2 * V

        # Concatenate back to a 2-channel field
        return jnp.concatenate((dU, dV), axis=0)
