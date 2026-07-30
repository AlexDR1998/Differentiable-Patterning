import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import time
import diffrax
import matplotlib.pyplot as plt

from PDE.model.reaction_diffusion_chemotaxis.update import F
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.NCA_visualiser import my_animate
from PDE.trainer.optimiser import non_negative_diffusion_chemotaxis
from Common.utils import load_emoji_sequence
from Common.model.spatial_operators import Ops

from NCA.trainer.data_augmenter_nca import DataAugmenter as DataAugmenterNCA
#from PDE.trainer.PDE_trainer import PDE_Trainer
from NCA.model.NCA_model import NCA






#CELL_CHANNELS = 3
#SIGNAL_CHANNELS=13
#CHANNELS = CELL_CHANNELS+SIGNAL_CHANNELS
BATCHES = 2
SIZE=32

#key=jax.random.PRNGKey(int(time.time()*1000))

#t = 0
# scale=0.1  
# x0 = jnp.ones((BATCHES,2,SIZE,SIZE))
# x0 = x0.at[:,1].set((1-scale)*x0[:,0]+jr.uniform(key,shape=(BATCHES,SIZE,SIZE))*scale)
# func = F_hillen_painter(PADDING="CIRCULAR",
#                         dx=0.25,
#                         KERNEL_SCALE=1)
# v_func = jax.vmap(func,in_axes=(None,0,None),out_axes=0)
# solver = PDE_solver(v_func,dt=0.01)

# t0 = time.time()

# T,Y = solver(ts=jnp.linspace(0,100,101),y0=x0)
# #T,Y = solver(ts=jnp.linspace(0,10000,101),y0=x0)
# t1 = time.time()
# print(t1-t0)
# print(Y.shape)
# print(x0.shape)
# my_animate(Y[:,0,0:1]/jnp.max(Y[:,0]),clip=False)
# my_animate(Y[:,1,0:1]/jnp.max(Y[:,0]),clip=False)
# #my_animate(Y[:,1:2]/jnp.max(Y[:,1]),clip=True)
# plt.plot(jnp.mean(Y[:,:,0],axis=(-1,-2)))
# plt.show()



#key=jr.PRNGKey(4)
#key=jr.PRNGKey(14)
key = jr.PRNGKey(int(time.time()))

x0 = jr.uniform(key,shape=(6,SIZE,SIZE))
#x0 = jr.uniform(key,shape=(4,SIZE,SIZE))
op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)

for i in range(5):
    x0 = op.Average(x0)
x0 = x0.at[0].set(jnp.where(x0[0]>0.51,1.0,0.0))
x0 = op.Average(x0)
x0 = x0.at[1].set(1-x0[0])

nfunc = F(CELL_CHANNELS=1,
         SIGNAL_CHANNELS=5,
         PADDING="CIRCULAR",
         dx=1.0,
         INTERNAL_ACTIVATION=jax.nn.tanh,
         OUTER_ACTIVATION=jax.nn.celu,
         INIT_SCALE=0.5,
         STABILITY_FACTOR=0.5,
         USE_BIAS=True,
         key=key)
pde = PDE_solver(nfunc,dt=0.1,rtol=0.01,atol=0.01)
t0 = time.time()
T,Y = pde(ts=jnp.linspace(0,50,101),y0=x0)
t1 = time.time()
print(t1-t0)
print(Y.shape)




my_animate(Y[:,:1],clip=False)
my_animate(Y[:,1:2],clip=False)
my_animate(Y[:,2:3],clip=False)
my_animate(Y[:,3:4],clip=False)
my_animate(Y[:,4:5],clip=False)
my_animate(Y[:,5:6],clip=False)
plt.plot(jnp.mean(Y,axis=(-1,-2)))
plt.show()
#my_animate(Y[:,3:6])
#plt.imshow(Y[-1,0])
#plt.show()
#print(T)


#print(update.f_v)