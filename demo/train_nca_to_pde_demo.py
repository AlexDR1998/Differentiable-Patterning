import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys
sys.path.append('..')
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA



#--- Set model and training parameters
ITERS = 8000        # Training iterations
CHANNELS = 16        # NCA channels
SIZE = 32           # Grid size
BATCHES = 2         # Data batches
TIME_SAMPLING = 32  # Timesteps between data snapshots
LEARN_RATE = 1e-4   # Learn rate for gradient optimiser

#--- Set up true PDE trajectory
key = jax.random.PRNGKey(int(time.time()))
scale=0.5
x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
func = F_chhabra(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
solver = PDE_solver(v_func,dt=0.1)
T,Y = solver(ts=jnp.linspace(0,5000,TIME_SAMPLING*8),y0=x0)
Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
Y = Y[:,:,:1]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
Y = (Y-jnp.min(Y))/(jnp.max(Y)-jnp.min(Y))                      # Rescale data between 0 and 1
Y = Y[:,::TIME_SAMPLING]                                        # Downsample along time axis

# Define NCA model
nca = NCA(N_CHANNELS=CHANNELS,          # An important NCA hyperparameter - how many channels
          KERNEL_STR=["ID","LAP"],      # What spatial derivatives/kernels to use? For this PDE we should only need Identity and Laplacian
          ACTIVATION=jax.nn.relu,       # What nonlinear activation function to use? Must be of form F: x -> x
          FIRE_RATE=1.0,                # Probability that each pixel gets updated at each timestep - for PDE training set this to 1
          key=key)                      # JAX PRNGKey for initialisation

# Define NCA trainer
# Y must either be a PyTree (list, dict etc) of arrays, or a full array
# Y must have shape [Batches,Timesteps,Observable channels, Height, Width]
#      Batch axis allows us to train 1 model on several different trajectories at once

opt = NCA_Trainer(nca,
                  Y,                                                
                  model_filename="demo/nca_pde_chhabra_example",    # Where to save model and tensorboard log files
                  DATA_AUGMENTER=DataAugmenter,                     # DataAugmenter class handles important data augmentation - create subclass of this to modify
                  GRAD_LOSS=True)                                   # Use spatial gradients of the cell states in the loss function as well?

# Define optax.GradientTransformation() object to handle training
schedule = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

# Do the training
opt.train(TIME_SAMPLING,                # How many NCA timesteps between each data step?
          ITERS,                        # How many training iterations?
          WARMUP=50,                    # How many training iterations to wait before saving models?
          optimiser=optimiser,          # Pass the optax.GradientTransformation() in here
          LOSS_FUNC_STR="euclidean",    # String identifying which loss function to use
          LOOP_AUTODIFF="lax",          # How to handle gradient checkpointing - "lax" is faster but uses more memory, "checkpointed" is slower but uses less memory
          LOG_EVERY=50,                 # How frequently to log model outputs to tensorboard?
          key=key)                      # JAX PRNGKey