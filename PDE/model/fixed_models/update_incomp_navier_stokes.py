import jax
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange,reduce
from Common.model.spatial_operators import Ops

#TODO: Return to this once a gauss-seidel or relaxation solver is added to lineax

class F(eqx.Module):
    ops: Ops
    rho: Float
    nu: Float
    dt: Float
    forcing: Array
    
    def __init__(self,
                 PADDING,
                 dx,
                 forcing,
                 rho=1.0,
                 nu=0.1,
                 dt = 0.1,
                 KERNEL_SCALE=1,
                
                 ):
        """Implementation of incompressible navier stokes, with relaxation solver of poissons equation for pressure

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): _description_
            
        """
        
      
        self.ops = Ops(PADDING,dx,KERNEL_SCALE,LAP_MODE=1)
        self.forcing = forcing
        self.rho = rho
        self.nu = nu
        self.dt = dt
    
    def __call__(self,
                t: Float[Scalar, ""],
                X: Float[Scalar,"2 C x y"],
                args)->Float[Scalar, "2 C x y"]:
        
        # Partial computation of velocity field
        partial = - self.ops.VectorMatDiff(X,X) + self.nu*self.ops.VectorLaplacian(X) + self.forcing
        B = self.rho*self.ops.Div(partial)/self.dt

        # Iterate jacobi relaxation to get pressure from velocity field
        # def poisson_jacobi_step(phi,j):
        #     phi = (self.ops.LapInv(phi)-B*self.ops.dx**2)/4.0
        #     return phi,None
        # phi,_ = jax.lax.scan(poisson_jacobi_step,phi,xs=jnp.arange(1000))
        phi = calc_pressure(B,B,self.ops,1000)


        # Compute final velocity field with updated pressure
        return partial - self.ops.Grad(phi)/self.rho
    





def calc_pressure(init_guess,B,ops,steps):
    def poisson_jacobi_step(phi,j):
        phi = (ops.LapInv(phi)-B*ops.dx**2)/4.0
        return phi,None
    phi,_ = jax.lax.scan(poisson_jacobi_step,init_guess,xs=jnp.arange(steps))
    return phi


def calc_pressure_init(X,PADDING,dx,rho,nu,dt,forcing):
    ops = Ops(PADDING,dx,LAP_MODE=1)
    partial = - ops.VectorMatDiff(X,X) + nu*ops.VectorLaplacian(X) + forcing
    B = rho*ops.Div(partial)/dt
    phi = calc_pressure(B,B,ops,10000)
    return phi