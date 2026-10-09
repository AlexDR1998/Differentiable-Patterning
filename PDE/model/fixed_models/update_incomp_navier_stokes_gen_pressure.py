import equinox as eqx
from jaxtyping import Array, Float, PyTree

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    rho: float
    nu: float
    M: float
    D: float
    forcing: Array

    def __init__(self,
                 PADDING,
                 dx,
                 forcing,
                 rho=1.0,
                 nu=0.1,
                 M = 1.0,
                 D = 1.0,
                 KERNEL_SCALE=1,
                 ):
        """Near-incompressible Navier-Stokes (artificial compressibility), advecting a passive scalar S.

        The state is the tuple (V, P, S), with V of shape [2, C, x, y]:
        dV = -(V.grad)V - grad(P)/rho + nu*Lap(V) + forcing
        dP = nu*Lap(P) + div(V)/M^2
        dS = D*Lap(S) - V.grad(S)

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): grid spacing
            forcing (array): body force, broadcastable to V
            rho (float, optional): density. Defaults to 1.0.
            nu (float, optional): viscosity. Defaults to 0.1.
            M (float, optional): artificial Mach number; smaller is closer to incompressible. Defaults to 1.0.
            D (float, optional): scalar diffusion. Defaults to 1.0.
            KERNEL_SCALE (int, optional): spatial operator kernel size. Defaults to 1.
        """
        self.ops = Ops(PADDING,dx,KERNEL_SCALE,SMOOTHING=1)
        self.forcing = forcing
        self.rho = rho
        self.nu = nu
        self.M = M
        self.D = D

    def __call__(self,
                t: Float,
                X: PyTree,
                args)->PyTree:

        (V,P,S)=X

        dV = (-self.ops.VectorMatDiff(V,V)
              - self.ops.Grad(P)/self.rho
              + self.nu*self.ops.VectorLaplacian(V)
              + self.forcing)

        dP = self.nu*self.ops.Lap(P) + self.ops.Div(V)/(self.M**2)
        dS = self.D*self.ops.Lap(S) - self.ops.MatDiff(V,S)

        return (dV,dP,dS)
