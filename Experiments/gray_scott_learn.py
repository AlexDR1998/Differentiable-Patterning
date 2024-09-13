import jax
#jax.config.update("jax_enable_x64", True)
import jaxpruner
from functools import partial
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import optax
from PDE.trainer.optimiser import multi_learnrate
from einops import repeat
from PDE.model.reaction_diffusion_advection.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
# from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.fixed_models.update_keller_segel import F as F_keller_segel
from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
#from Common.eddie_indexer import index_to_pde_gray_scott_hyperparameters
from Common.eddie_indexer import index_to_pde_gray_scott_rda
from Common.model.spatial_operators import Ops
from einops import rearrange
import time
import sys

index=int(sys.argv[1])-1
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)


PARAMS = index_to_pde_gray_scott_rda(index)
INIT_SCALE = {"reaction":0.01,"advection":0.01,"diffusion":0.01}
STABILITY_FACTOR = 0.01
CHANNELS = PARAMS["CHANNELS"]
ITERS = 2001
SIZE = 64
BATCHES = 8
PADDING = "CIRCULAR"
TRAJECTORY_LENGTH = PARAMS["TRAJECTORY_LENGTH"]
dt = 1.0

MODEL_FILENAME = "pde_hyperparameters_reacdiff/2_"+PARAMS["PDE_STR"]+"_"+PARAMS["PDE_SOLVER"]+"_tl_"+str(TRAJECTORY_LENGTH)+"_"+PARAMS["TRAJECTORY_TYPE"]+"_res_"+str(PARAMS["TIME_RESOLUTION"])+"_ch_"+str(CHANNELS)+"_ord_"+str(PARAMS["ORDER"])+"_act_"+PARAMS["INTERNAL_ACTIVATION"]+"_l_"+str(PARAMS["N_LAYERS"])+"_"+"_".join(PARAMS["TERMS"])+PARAMS["TEXT_LABEL"]

pde_hyperparameters = {"N_CHANNELS":CHANNELS,
                       "PADDING":PADDING,
                       "INTERNAL_ACTIVATION":PARAMS["INTERNAL_ACTIVATION"],
                       "dx":1.0,
                       "TERMS":PARAMS["TERMS"],
                       "ADVECTION_OUTER_ACTIVATION":"relu",
                       "INIT_SCALE":INIT_SCALE,
                       "INIT_TYPE":{"reaction":"orthogonal","advection":"orthogonal","diffusion":"orthogonal"},
                       "STABILITY_FACTOR":STABILITY_FACTOR,
                       "USE_BIAS":True,
                       "ORDER":PARAMS["ORDER"],
                       "N_LAYERS":PARAMS["N_LAYERS"],
                       "ZERO_INIT":{"reaction":False,"advection":False,"diffusion":False}}

solver_hyperparameters = {"dt":dt,
                          "SOLVER":PARAMS["PDE_SOLVER"],
                          "rtol":1e-3,
                          "DTYPE":"float32",
                          "atol":1e-3,
                          "ADAPTIVE":PARAMS["PDE_SOLVER_ADAPTIVE"]}

true_solver_hyperparameters = {"dt":dt,
                               "SOLVER":"heun",
                               "rtol":1e-3,
                               "DTYPE":"float32",
                               "atol":1e-3,
                               "ADAPTIVE":True}
hyperparameters = {"pde":pde_hyperparameters,
                   "solver":solver_hyperparameters}


# ----------------- Define data -----------------


if PARAMS["PDE_STR"]=="gray_scott":
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    op = Ops(PADDING=PADDING,dx=1.0,KERNEL_SCALE=2)
    v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(1):
        x0 = v_av(x0)
    x0 = x0.at[:,1].set(np.where(x0[:,1]>0.55,1.0,0.0))

    x0 = x0.at[:,1,:SIZE//4].set(0)
    x0 = x0.at[:,1,:,:SIZE//4].set(0)
    x0 = x0.at[:,1,-SIZE//4:].set(0)
    x0 = x0.at[:,1,:,-SIZE//4:].set(0)
    for i in range(1):
        x0 = v_av(x0)
    x0 = x0.at[:,0].set(1-x0[:,1])

    func = F_gray_scott(PADDING=PADDING,dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,**true_solver_hyperparameters)
    T,Y = solver(ts=np.linspace(0,2000,PARAMS["TIME_RESOLUTION"]),y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")
    Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
    Y = Y.at[:,:,1].set(2*(Y[:,:,1]-np.min(Y[:,:,1]))/(np.max(Y[:,:,1])-np.min(Y[:,:,1])) - 1)

elif PARAMS["PDE_STR"]=="cahn_hilliard":
    scale=2.0
    x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))*scale - 1
    func = F_cahn_hilliard(PADDING=PADDING,dx=1,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)

    solver = PDE_solver(v_func,**true_solver_hyperparameters)
    T,Y = solver(np.linspace(0,10000,PARAMS["TIME_RESOLUTION"]),x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")
    Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
    

elif PARAMS["PDE_STR"]=="keller_segel":
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    x0 = x0*0.1
    x0 = x0.at[:,1].set(0.0)

    func = F_keller_segel(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1,alpha=0.01,c=3.8,D=0.8,epsilon=0.1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,**true_solver_hyperparameters)
    T,Y = solver(ts=np.linspace(0,200,PARAMS["TIME_RESOLUTION"]),y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")
    Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
    Y = Y.at[:,:,1].set(2*(Y[:,:,1]-np.min(Y[:,:,1]))/(np.max(Y[:,:,1])-np.min(Y[:,:,1])) - 1)


# ----------------- Define model -----------------
func = F(key=key,**hyperparameters["pde"])
pde = PDE_solver(func,**hyperparameters["solver"])


#------------------- Define optimiser and lr schedule -----------------

schedule = optax.exponential_decay(5e-4, transition_steps=ITERS, decay_rate=0.99)
opt = optax.chain(PARAMS["OPTIMISER_PRE_PROCESS"],optax.apply_if_finite(PARAMS["OPTIMISER"](schedule),max_consecutive_errors=8))


#-------------------- Define trainer object -----------------

trainer = PDE_Trainer(PDE_solver=pde,
                      PDE_HYPERPARAMETERS=hyperparameters,
                      data=Y,
                      Ts=T,
                      model_filename=MODEL_FILENAME)


#-------------------- Train model -----------------

trainer.train(SUBTRAJECTORY_LENGTH=TRAJECTORY_LENGTH,
              TRAINING_ITERATIONS=ITERS,
              OPTIMISER=opt,
              LOG_EVERY=50,
              WARMUP=32,
              PRUNING={"PRUNE":False,"TARGET_SPARSITY":0},
              LOSS_PARAMS = {"LOSS_FUNC":PARAMS["LOSS_FUNCTION"],
							 "GRAD_LOSS":True,
							 "LOSS_SAMPLING":PARAMS["LOSS_TIME_SAMPLING"],
							 "LOSS_TRAJECTORY_FULL":PARAMS["TRAJECTORY_FULL"]})
              
              #UPDATE_X0_PARAMS=UPDATE_X0_PARAMS)