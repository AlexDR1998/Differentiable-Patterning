import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar
from Common.model.spatial_operators import Ops

class F(eqx.Module):
    ops: Ops
    D: float
    eps_v: float
    a_v: float
    a_z: float

    def __init__(
        self,
        PADDING: str,
        dx: float,
        KERNEL_SCALE: int = 1,
        D: float = 0.1,
        eps_v: float = 0.01,
        a_v: float = 0.5,
        a_z: float = 0.1,
    ):
        """
        FitzHugh–Nagumo reaction-diffusion:

            ∂u/∂t =  ∇²u + u − u³ − v
            ∂v/∂t = D ∇²v + ε_v (u − a_v v − a_z)

        Args:
            PADDING (str): 'ZEROS', 'REFLECT', 'REPLICATE', or 'CIRCULAR'
            dx (float): spatial grid spacing
            KERNEL_SCALE (int): convolution kernel scale
            D (float): diffusion coefficient for v
            eps_v (float): time‐scale separation for v
            a_v (float): linear recovery rate for v
            a_z (float): constant offset in v‐equation
        """
        self.ops  = Ops(PADDING, dx, KERNEL_SCALE)
        self.D    = D
        self.eps_v = eps_v
        self.a_v  = a_v
        self.a_z  = a_z

    def __call__(
        self,
        t: Float[Scalar, ""],
        X: Float[Scalar, "2 x y"],
        args
    ) -> Float[Scalar, "2 x y"]:
        # Split the two fields
        u = X[0:1]   # activator
        v = X[1:2]   # inhibitor

        # diffusion
        lap_u = self.ops.Lap(u)
        lap_v = self.ops.Lap(v)

        # kinetics
        du = lap_u + u - u**3 - v
        dv = self.D * lap_v + self.eps_v * (u - self.a_v * v - self.a_z)

        return jnp.concatenate([du, dv], axis=0)
