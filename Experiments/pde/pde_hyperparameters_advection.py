import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from PDE.trainer.optimiser import non_negative_diffusion
from einops import repeat
from PDE.model.reaction_diffusion_advection.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
from Common.eddie_indexer import index_to_pde_advection_hyperparameters
from Common.model.spatial_operators import Ops
from einops import rearrange
import time
import sys

index=int(sys.argv[1])-1


PARAMS = index_to_pde_advection_hyperparameters(index)
INIT_SCALE = 0.5
STABILITY_FACTOR = 0.01
REACTION_INIT_SCALE_RATIO = 0.01

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

CHANNELS = 12
SIZE = 64
BATCHES = 4
PADDING = "CIRCULAR"

if PARAMS["EQUATION_INDEX"]==0:
    PDE_STR = "gray_scott"
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
    v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(5):
        x0 = v_av(x0)
    x0 = x0.at[:,0].set(jnp.where(x0[:,0]>0.51,1.0,0.0))
    x0 = x0.at[:,1].set(1-x0[:,0])
    func = F_gray_scott(PADDING=PADDING,dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,10000,101),y0=x0)

elif PARAMS["EQUATION_INDEX"]==1:
    PDE_STR = "chhabra"
    scale=0.5
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
    func = F_chhabra(PADDING=PADDING,dx=0.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,5000,101),y0=x0)

elif PARAMS["EQUATION_INDEX"]==2:
    PDE_STR = "hillen_painter"
    scale=0.1  
    x0 = jnp.ones((BATCHES,2,SIZE,SIZE))
    x0 = x0.at[:,1].set((1-scale)*x0[:,0]+jr.uniform(key,shape=(BATCHES,SIZE,SIZE))*scale)
    func = F_hillen_painter(PADDING=PADDING,
                            dx=0.25,
                            KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.01)
    T,Y = solver(ts=jnp.linspace(0,100,101),y0=x0)
elif PARAMS["EQUATION_INDEX"]==3:
    PDE_STR = "cahn_hilliard"
    scale=2.0
    x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))*scale - 1
    func = F_cahn_hilliard(PADDING=PADDING,dx=1.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.5)
    T,Y = solver(ts=jnp.linspace(0,20000,101),y0=x0)

Y = rearrange(Y,"T B C X Y -> B T C X Y")
Y = Y[:,:,:1] # Only include main channel, not inhibitor/other chemical
Y = 2*(Y-jnp.min(Y))/(jnp.max(Y)-jnp.min(Y)) - 1
#Y = jnp.pad(Y,((0,0),(0,0),(0,CHANNELS-1),(0,0),(0,0)),mode="constant")
print(Y.shape)


# Define PDE model
func = F(CHANNELS,
         PADDING=PADDING,
         dx=1.0,
         INTERNAL_ACTIVATION=PARAMS["INTERNAL_ACTIVATIONS"],
         ADVECTION_OUTER_ACTIVATION=PARAMS["ADVECTION_OUTER_ACTIVATIONS"],
         INIT_SCALE=INIT_SCALE,
         STABILITY_FACTOR=STABILITY_FACTOR,
         USE_BIAS=True,
         ZERO_INIT=False,
         REACTION_INIT_SCALE_RATIO=REACTION_INIT_SCALE_RATIO,
         key=key)
pde = PDE_solver(func,dt=0.1)


# Define optimiser and lr schedule
iters = 2000
schedule = optax.exponential_decay(PARAMS["LEARN_RATE"], transition_steps=iters, decay_rate=0.99)
#opt = non_negative_diffusion(schedule,optimiser=OPTIMISER)
opt = optax.chain(optax.scale_by_param_block_norm(),
			PARAMS["OPTIMISER"](schedule))

trainer = PDE_Trainer(pde,
                      Y,
                      #model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca_2/init_scale_"+str(INIT_SCALE)+"_stability_factor_"+str(STABILITY_FACTOR)+"act_"+INTERNAL_TEXT+"_"+OUTER_TEXT)
                      model_filename="pde_hyperparameters_advreacdiff/cubic_stablised_nonnegative_diffusion_"+PDE_STR+"_act_"+PARAMS["INTERNAL_TEXT"]+"_"+PARAMS["OUTER_TEXT"]+"_opt_"+PARAMS["OPTIMISER_TEXT"]+"_lr_"+PARAMS["LEARN_RATE_TEXT"]+"_tl_"+str(PARAMS["TRAJECTORY_LENGTH"]))
trainer.train(PARAMS["TRAJECTORY_LENGTH"],iters,optimiser=opt,LOG_EVERY=100)