import jax
import equinox as eqx
import time
import json
import jax.random as jr
import jax.numpy as jnp
import diffrax
from PDE.model.reaction_diffusion_advection.update import F
from jaxtyping import PyTree,Array,Scalar,Float
from Common.model.spatial_operators import Ops


class NavierStokesSplittingEuler(eqx.Module):
    ops:Ops
    d:dict
    params:dict
    forcing:Array
    def __init__(self,
                 PADDING,
                 d = {"dx":1.0,"dt":0.1,"beta":0.5},
                 params={"rho":1.0,"nu":0.1},
                 forcing=0):
        """Implementation of incompressible navier stokes, with relaxation solver of poissons equation for pressure
        """
        self.ops = Ops(PADDING,d["dx"],1,SMOOTHING=0)
        self.d = d
        self.params = params
        self.forcing = forcing



    @eqx.filter_jit
    def step(self,
            V: Float[Array,"2 C x y"],
            P: Float[Array,"C x y"],
        ):
        """ Step for euler method with embedded jacobi relaxation for pressure poisson equation
        """
        #control = terms.contr(t0, t1)
        

        # Partial computation of velocity field
        V_star = V + self.d["dt"]*( - self.ops.VectorMatDiff(V,V) 
                                   - self.d["beta"]*self.ops.Grad(P)/self.params["rho"] 
                                   + self.params["nu"]*self.ops.VectorLaplacian(V) 
                                   + self.forcing)
                    
        Phi = self.calc_pressure(P,V_star,1000)
        V = V_star - self.d["dt"]*self.ops.Grad(Phi)/self.params["rho"]
        P = Phi + self.d["beta"]*P
        return V,P

    def solve(self,V_0,P_0,steps):
        if P_0 is None:
            P_0 = self.calc_pressure_init(V_0)
            print(P_0.shape)
        def _step(carry,j):
            V,P = carry
            V,P = self.step(V,P)
            Vorticity = self.ops.Curl(V)
            return (V,P),(V,P,Vorticity)
        _,trajectory = jax.lax.scan(_step,(V_0,P_0),xs=jnp.arange(steps))
        return trajectory



    @eqx.filter_jit
    def calc_pressure(self,init_guess,V_star,steps):
        """ Jacobi relaxation to get pressure from velocity field
        """
        B = self.params["rho"]*self.ops.Div(V_star)/self.d["dt"]
        phi = init_guess
        #print(phi.shape)
        def poisson_jacobi_step(phi,j):
            phi = (self.ops.LapInv(phi)-B*self.ops.dx**2)/4.0
            #print(phi.shape)
            return phi,None
        phi,_ = jax.lax.scan(poisson_jacobi_step,init_guess,xs=jnp.arange(steps))
        return phi
    
    def calc_pressure_init(self,V):
        V_star = V + self.d["dt"]*( - self.ops.VectorMatDiff(V,V) 
                                    + self.params["nu"]*self.ops.VectorLaplacian(V) 
                                    + self.forcing
                                    )
        return self.calc_pressure(jnp.zeros_like(self.ops.Div(V)),V_star,10000)