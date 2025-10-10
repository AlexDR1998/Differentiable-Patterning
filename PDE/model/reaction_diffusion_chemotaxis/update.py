import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from PDE.model.reaction_diffusion_chemotaxis.cell_motility import Cell_motility
from PDE.model.reaction_diffusion_chemotaxis.cell_reaction import Cell_reaction
from PDE.model.reaction_diffusion_chemotaxis.signal_diffusion import Signal_diffusion
from PDE.model.reaction_diffusion_chemotaxis.signal_reaction import Signal_reaction


class F(eqx.Module):
    cell_motility: Cell_motility
    cell_reaction: Cell_reaction
    signal_diffusion: Signal_diffusion
    signal_reaction: Signal_reaction
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    N_CHANNELS: int
    PADDING: str
    dx: float

    def __init__(self,
                 CELL_CHANNELS,
                 SIGNAL_CHANNELS,
                 PADDING,
                 dx,
                 INTERNAL_ACTIVATION=jax.nn.relu,
                 OUTER_ACTIVATION=jax.nn.tanh,
                 INIT_SCALE=0.01,
                 USE_BIAS=True,
                 STABILITY_FACTOR=0.01,
                 key=jax.random.PRNGKey(int(time.time()))):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.N_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        self.PADDING = PADDING
        key1,key2,key3,key4 = jax.random.split(key,4)
        self.dx = dx
        self.cell_motility = Cell_motility(
            CELL_CHANNELS,
            SIGNAL_CHANNELS,
            PADDING,
            dx,
            INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
            OUTER_ACTIVATION=OUTER_ACTIVATION,
            INIT_SCALE=INIT_SCALE,
            USE_BIAS=USE_BIAS,
            key=key1)
        self.cell_reaction = Cell_reaction(
            CELL_CHANNELS,
            SIGNAL_CHANNELS,
            INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
            OUTER_ACTIVATION=OUTER_ACTIVATION,
            INIT_SCALE=INIT_SCALE,
            USE_BIAS=USE_BIAS,
            STABILITY_FACTOR=STABILITY_FACTOR,
            key=key2)
        self.signal_diffusion= Signal_diffusion(
            CELL_CHANNELS,
            SIGNAL_CHANNELS,
            PADDING,
            dx,
            key=key3)
        self.signal_reaction = Signal_reaction(
            CELL_CHANNELS,
            SIGNAL_CHANNELS,
            INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
            OUTER_ACTIVATION=jax.nn.relu,
            INIT_SCALE=INIT_SCALE,
            USE_BIAS=USE_BIAS,
            STABILITY_FACTOR=STABILITY_FACTOR,
            key=key4)
    
    @eqx.filter_jit
    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Array,"{self.CELL_CHANNELS}+{self.SIGNAL_CHANNELS} x y"],
                 args)-> Float[Array,"{self.CELL_CHANNELS}+{self.SIGNAL_CHANNELS} x y"]:
        dcell = self.cell_motility(X) + self.cell_reaction(X)
        dsignal= self.signal_diffusion(X) + self.signal_reaction(X)
        return jnp.concatenate((dcell,dsignal),axis=0)
    
    
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        
        cm_diff,cm_static = self.cell_motility.partition()
        cr_diff,cr_static = self.cell_reaction.partition()
        sd_diff,sd_static = self.signal_diffusion.partition()
        sr_diff,sr_static = self.signal_reaction.partition()
    
        where_cm = lambda m:m.cell_motility
        where_cr = lambda m:m.cell_reaction
        where_sd = lambda m:m.signal_diffusion
        where_sr = lambda m:m.signal_reaction
        
        total_diff = eqx.tree_at(where_cm,total_diff,cm_diff)
        total_diff = eqx.tree_at(where_cr,total_diff,cr_diff)
        total_diff = eqx.tree_at(where_sd,total_diff,sd_diff)
        total_diff = eqx.tree_at(where_sr,total_diff,sr_diff)
        
        total_static = eqx.tree_at(where_cm,total_static,cm_static)
        total_static = eqx.tree_at(where_cr,total_static,cr_static)
        total_static = eqx.tree_at(where_sd,total_static,sd_static)
        total_static = eqx.tree_at(where_sr,total_static,sr_static)

        return total_diff,total_static

    def combine(self,diff,static):
        self = eqx.combine(diff,static)