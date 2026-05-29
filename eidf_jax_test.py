import jax
import jax.numpy as np
import jax.random as jr
import optax
import equinox as eqx
from einops import rearrange
from Common.model.spatial_operators import Ops
from Common.trainer.loss import euclidean
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.model.reaction_diffusion_advection.update import F
from PDE.trainer.data_augmenter_pde import DataAugmenter
from PDE.trainer.PDE_trainer import PDE_Trainer

BATCHES = 4
key = jr.PRNGKey(0)
PADDING = "CIRCULAR"
SIZE=32

print(jax.default_backend())
print(jax.devices())



MODEL_FILENAME = "pde_gray_scott_eidf_test_1"

pde_hyperparameters = {"N_CHANNELS":16,
                       "PADDING":PADDING,
                       "INTERNAL_ACTIVATION":"relu",
                       "dx":1.0,
                       "TERMS":["reaction_pure","diffusion"],
                       "ADVECTION_OUTER_ACTIVATION":"relu",
                       "INIT_SCALE":{"reaction":0.001,"advection":0.01,"diffusion":0.01},
                       "INIT_TYPE":{"reaction":"orthogonal","advection":"orthogonal","diffusion":"orthogonal"},
                       "STABILITY_FACTOR":0.001,
                       "USE_BIAS":True,
                       "ORDER":1,
                       "N_LAYERS":4,
                       "ZERO_INIT":{"reaction":False,"advection":False,"diffusion":False}}


true_solver_hyperparameters = {"dt":0.1,
                               "SOLVER":"heun",
                               "rtol":1e-3,
                               "DTYPE":"float32",
                               "atol":1e-3,
                               "ADAPTIVE":True}

model_hyperparameters = {"pde":pde_hyperparameters,
                         "solver":true_solver_hyperparameters}




trainer_hyperparameters = {"OPTIMISER":optax.nadam,
                          "OPTIMISER_PRE_PROCESS":optax.scale_by_param_block_norm(),
                          "NOISE_FRACTION":0.0,
                          "LOSS_FUNCTION":euclidean,
                          "LOSS_SAMPLING":1,
                          "TRAJECTORY_FULL":True,
                          "TRAJECTORY_LENGTH":32,
                          "TRAINING_ITERATIONS":4000,
                          }

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
T,Y = solver(ts=np.linspace(0,2000,100),y0=x0)
Y = rearrange(Y,"T B C X Y -> B T C X Y")
Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
Y = Y.at[:,:,1].set(2*(Y[:,:,1]-np.min(Y[:,:,1]))/(np.max(Y[:,:,1])-np.min(Y[:,:,1])) - 1)



func = F(key=key,**model_hyperparameters["pde"])
pde = PDE_solver(func,**model_hyperparameters["solver"])




#------------------- Define optimiser and lr schedule -----------------

schedule = optax.exponential_decay(5e-4, transition_steps=trainer_hyperparameters["TRAINING_ITERATIONS"], decay_rate=0.99)
opt = optax.chain(trainer_hyperparameters["OPTIMISER_PRE_PROCESS"],optax.apply_if_finite(trainer_hyperparameters["OPTIMISER"](schedule),max_consecutive_errors=8))


# ------------------ Define data augmenter parameters -----------------

class data_augmenter_subclass(DataAugmenter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.CALLBACK_PARAMS = {"INCREMENT_PROB":0.1,
                                "RESET_PROB":0.001,
                                "OVERWRITE_OBS_CHANNELS":False,
                                "NOISE_FRACTION":trainer_hyperparameters["NOISE_FRACTION"]}



#-------------------- Define trainer object -----------------

trainer = PDE_Trainer(PDE_solver=pde,
                      PDE_HYPERPARAMETERS=model_hyperparameters,
                      data=Y,
                      Ts=T,
                      model_filename=MODEL_FILENAME,
                      DATA_AUGMENTER=data_augmenter_subclass)


#-------------------- Train model -----------------

trainer.train(SUBTRAJECTORY_LENGTH=trainer_hyperparameters["TRAJECTORY_LENGTH"],
              TRAINING_ITERATIONS=trainer_hyperparameters["TRAINING_ITERATIONS"],
              OPTIMISER=opt,
              LOG_EVERY=50,
              WARMUP=32,
              PRUNING={"PRUNE":False,"TARGET_SPARSITY":0},
              LOSS_PARAMS = {"LOSS_FUNC":trainer_hyperparameters["LOSS_FUNCTION"],
							 "GRAD_LOSS":True,
							 "LOSS_SAMPLING":trainer_hyperparameters["LOSS_SAMPLING"],
							 "LOSS_TRAJECTORY_FULL":trainer_hyperparameters["TRAJECTORY_FULL"]})
              
              #UPDATE_X0_PARAMS=UPDATE_X0_PARAMS)