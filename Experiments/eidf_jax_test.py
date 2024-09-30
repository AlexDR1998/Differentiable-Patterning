import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.solver.semidiscrete_solver import PDE_solver

BATCHES = 8
key = jr.PRNGKey(0)
PADDING = "CIRCULAR"
SIZE=64

print(jax.default_backend())
print(jax.devices())

true_solver_hyperparameters = {"dt":0.1,
                               "SOLVER":"heun",
                               "rtol":1e-3,
                               "DTYPE":"float32",
                               "atol":1e-3,
                               "ADAPTIVE":True}


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