import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys
from einops import rearrange

from Common.model.spatial_operators import Ops
from Common.eddie_indexer import index_to_kaNCA_pde_parameters

from NCA.model.NCA_KAN_model import kaNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter # Use different data augmenter to handle PDE mini-batching

from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
from PDE.model.solver.semidiscrete_solver import PDE_solver


index=int(sys.argv[1])-1
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)


#MINIBATCHES = 32
#EQUATION_INDEX = 0
#NCA_STEPS_PER_PDE_STEP = 1
EQUATION_INDEX,TIME_SAMPLING = index_to_kaNCA_pde_parameters(index)


ITERS = 8000
CHANNELS = 8
BASIS_FUNCS = 25
SIZE = 32
BATCHES = 2

#class data_augmenter_subclass(DataAugmenter):
#    def __init__(self,data_true,hidden_channels):
#        super().__init__(data_true,hidden_channels,MINIBATCHES=MINIBATCHES)
    


if EQUATION_INDEX==0:
    PDE_STR = "gray_scott"
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
    v_av = jax.vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(5):
        x0 = v_av(x0)
    x0 = x0.at[:,0].set(jnp.where(x0[:,0]>0.51,0.0,1.0))
    x0 = v_av(x0)
    x0 = x0.at[:,1].set(1-x0[:,0])
    func = F_gray_scott(PADDING="REFLECT",dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,10000,513),y0=x0)

elif EQUATION_INDEX==1:
    PDE_STR = "chhabra"
    scale=0.5
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
    func = F_chhabra(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,5000,513),y0=x0)

elif EQUATION_INDEX==2:
    PDE_STR = "hillen_painter"
    scale=0.1  
    x0 = jnp.ones((BATCHES,2,SIZE,SIZE))
    x0 = x0.at[:,1].set((1-scale)*x0[:,0]+jr.uniform(key,shape=(BATCHES,SIZE,SIZE))*scale)
    func = F_hillen_painter(PADDING="CIRCULAR",
                            dx=0.25,
                            KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.01)
    T,Y = solver(ts=jnp.linspace(0,100,513),y0=x0)
elif EQUATION_INDEX==3:
    PDE_STR = "cahn_hilliard"
    scale=2.0
    x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))*scale - 1
    func = F_cahn_hilliard(PADDING="CIRCULAR",dx=1.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.5)
    T,Y = solver(ts=jnp.linspace(0,20000,513),y0=x0)

Y = rearrange(Y,"T B C X Y -> B T C X Y")
Y = Y[:,:,:1] # Only include main channel, not inhibitor/other chemical
Y = 2*(Y-jnp.min(Y))/(jnp.max(Y)-jnp.min(Y)) - 1
Y = Y[:,::TIME_SAMPLING]

print("Downsampled PDE trajectory shape: "+str(Y.shape))

schedule = optax.exponential_decay(1e-4, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))




nca = kaNCA(CHANNELS,
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=1.0,
            PADDING="CIRCULAR",
            BASIS_FUNCS=BASIS_FUNCS,
            BASIS_WIDTH=4,
            INIT_SCALE=0.1,
            key=key)


opt = NCA_Trainer(nca,
                  Y,
                  model_filename="kaNCA_pde/debug_channels_"+str(CHANNELS)+"_res_"+str(BASIS_FUNCS)+"_time_sampling_"+str(TIME_SAMPLING)+"_"+PDE_STR,
                  DATA_AUGMENTER=DataAugmenter,
                  GRAD_LOSS=True)

opt.train(TIME_SAMPLING,
          ITERS,
          WARMUP=50,
          optimiser=optimiser,
          LOSS_FUNC_STR="euclidean",
          LOOP_AUTODIFF="checkpointed",
          LOG_EVERY=50,
          key=key)