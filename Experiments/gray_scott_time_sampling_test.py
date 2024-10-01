import jax
import time
from tqdm.auto import tqdm
import jax.numpy as np
from Common.trainer.loss import euclidean,spectral
import jax.random as jr
import equinox as eqx
import glob
from einops import rearrange,repeat,reduce
import matplotlib.pyplot as plt
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.reaction_diffusion_advection.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver,load,save

#--------------------------------


BATCHES = 4
MODEL_DUPLICATES = 8
SIZE = 64
N_CHANNELS = 32
PADDING = "CIRCULAR"
true_solver_hyperparameters = {"dt":0.1,
                               "SOLVER":"heun",
                               "rtol":1e-3,
                               "DTYPE":"float32",
                               "atol":1e-3,
                               "ADAPTIVE":True}
TIME_RESOLUTION = 100



#--------------------------------


key = jr.PRNGKey(int(time.time()))
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
ts = np.linspace(0,1000,TIME_RESOLUTION)
ts = repeat(ts,"T -> B T",B=BATCHES)
func = F_gray_scott(PADDING=PADDING,dx=1.0,KERNEL_SCALE=1)
solver = PDE_solver(func,**true_solver_hyperparameters)
v_solver = eqx.filter_vmap(solver,in_axes=(0,0),out_axes=(0,0))
T,Y = v_solver(ts,x0)
Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
Y = Y.at[:,:,1].set(2*(Y[:,:,1]-np.min(Y[:,:,1]))/(np.max(Y[:,:,1])-np.min(Y[:,:,1])) - 1)
x0_model = Y[:,0]
x0_augmented = np.pad(x0_model,((0,0),(0,N_CHANNELS-2),(0,0),(0,0)))



#--------------------------------
filename_base = "models/pde_hyperparameters_reacdiff/7*instance*"
filenames = glob.glob(filename_base)
filenames = list(sorted(filenames))
print(len(filenames))
filenames_short = ["4","8","12","16","24","32"]
models = []
for f in filenames:

    models.append(load(f))
models_reshaped = []
filenames_reshaped = []
for i in range(6):
    models_reshaped.append(models[i*MODEL_DUPLICATES:(i+1)*MODEL_DUPLICATES])
    
    filenames_reshaped.append(filenames[i*MODEL_DUPLICATES:(i+1)*MODEL_DUPLICATES])



#--------------------------------
Y_ENSEMBLE = []
for i in range(6):
    Y_t = []
    for m in tqdm(models_reshaped[i]):
        #m = m.at["PDE_solver"].
        v_solver = eqx.filter_vmap(m,in_axes=(0,0),out_axes=(0,0))
    
        _,y_pred = v_solver(ts,x0_augmented)
        Y_t.append(y_pred)
    Y_ENSEMBLE.append(Y_t)


#--------------------------------
Y_ARR = np.array(Y_ENSEMBLE)
np.save("data/gray_scott_time_sampling_test.npy",Y_ARR)
