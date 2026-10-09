import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Scalar

from Common.model.spatial_operators import Ops


class F(eqx.Module):
    ops: Ops
    logistic_growth_rate: float
    gamma: float
    alpha: float
    chi: float
    phi: float
    D: float

    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE=1,
                 logistic_growth_rate=0.1,
                 gamma=10.0,
                 alpha=0.5,
                 chi=5.0,
                 phi=0.8, # Paper says 1, but gives different patterns to this
                 D=0.1):
        """Chemotaxis pattern formation model from figure 4 in Hillen & Painter "A users guide to PDE models for chemotaxis"

        du = div(D*u*grad(u)) - div(chi*u*(1-u/gamma)/(1+alpha*v)^2 * grad(v)) + r*u*(1-u)
        dv = Lap(v) + u/(1+phi*u) - v

        Parameters
        ----------
        PADDING : str
            Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
        dx : float
            Grid spacing
        KERNEL_SCALE : int, optional
            Spatial operator kernel size. Defaults to 1.
        logistic_growth_rate : float, optional
            Logistic growth rate r. Defaults to 0.1.
        gamma : float, optional
            Cell crowding capacity. Defaults to 10.0.
        alpha : float, optional
            Receptor saturation. Defaults to 0.5.
        chi : float, optional
            Chemotactic sensitivity. Defaults to 5.0.
        phi : float, optional
            Signal production saturation. Defaults to 0.8.
        D : float, optional
            Cell diffusion. Defaults to 0.1.
        """
        self.logistic_growth_rate=logistic_growth_rate
        self.gamma=gamma
        self.alpha=alpha
        self.chi=chi
        self.phi=phi
        self.D=D
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:

        cells = X[0:1]
        signal= X[1:2]
        chemotactic_term = (cells*self.chi*(1-cells/self.gamma))/((1+self.alpha*signal)**2)

        _a = self.ops.NonlinearDiffusion(self.D*cells,cells)
        _b = self.ops.NonlinearDiffusion(chemotactic_term,signal)

        dcells = _a - _b + self.logistic_growth_rate*cells*(1-cells)
        dsignal = self.ops.Lap(signal) + cells/(1+self.phi*cells) - signal
        return jnp.concatenate((dcells,dsignal),axis=0)
