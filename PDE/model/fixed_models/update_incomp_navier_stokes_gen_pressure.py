import equinox as eqx

from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange,reduce
from Common.model.spatial_operators import Ops

#TODO: Return to this once a gauss-seidel or relaxation solver is added to lineax

class F(eqx.Module):
    ops: Ops
    rho: Float
    nu: Float
    M: Float
    D: Float
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
        """Implementation of incompressible navier stokes, with generalised pressure solver

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): _description_
            
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