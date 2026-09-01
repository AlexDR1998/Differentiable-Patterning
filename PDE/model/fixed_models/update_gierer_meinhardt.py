# update_gierer_meinhardt.py
import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    a: float
    b: float
    c: float
    D: float
    epsilon: float

    def __init__(
        self,
        PADDING: str,
        dx: float,
        KERNEL_SCALE: int = 1,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 6.1,
        D: float = 100.0,
    ):
        """
        Gierer–Meinhardt activator–inhibitor model:
          ∂u/∂t = ∇²u + a + u²/v − b·u
          ∂v/∂t = D∇²v + u² − c·v

        Args:
            PADDING (str): 'ZEROS', 'REFLECT', 'REPLICATE', or 'CIRCULAR'
            dx      (float): Grid spacing
            KERNEL_SCALE (int): Scale factor for convolution kernel
            a, b, c (float): kinetic parameters
            D       (float): diffusion coefficient of v
        """
        self.a, self.b, self.c, self.D = a, b, c, D
        self.ops = Ops(PADDING, dx, KERNEL_SCALE)
        self.epsilon = 1e-4          # avoids divide-by-zero when v≈0

    # -----------------------------------------------------------------
    def __call__(
        self,
        t: Float[Scalar, ""],
        X: Float[Array, "2 x y"],
        args=None,                    # kept for API compatibility
    ) -> Float[Array, "2 x y"]:
        # -----------------------------------------------------------------
        # 1. split activator (u) and inhibitor (v)
        u = X[0:1]                    # shape (1,H,W)
        v = X[1:2]

        # 2. diffusion terms
        lap_u = self.ops.Lap(u)
        lap_v = self.ops.Lap(v)

        # 3. reaction kinetics
        v_pos = jnp.clip(v, 1e-3, None)             # keep inhibitor positive
        f_u = lap_u + self.a + u**2 / v_pos - self.b * u
        f_v = self.D * lap_v + u**2 - self.c * v_pos

        # -------- numerical guard ----------------------------------
        f_u = jnp.nan_to_num(f_u, nan=0.0, posinf=0.0, neginf=0.0)
        f_v = jnp.nan_to_num(f_v, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. concatenate back to (2,H,W)
        return jnp.concatenate((f_u, f_v), axis=0)
