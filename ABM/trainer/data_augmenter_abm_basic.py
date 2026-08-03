from Common.trainer.abstract_data_augmenter_array import DataAugmenterAbstract
from Common.mnist_reader import MnistDataloader
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import einops
import time
import numpy as np
import pandas as pd
from os.path  import join
from scipy import ndimage

class DataAugmenter(DataAugmenterAbstract):
    # Data format is (((B,N,N_agents,2),(B,N,N_agents,2)),(B,N,CHANNELS,width,height))
    #   ((agent_position,agent_velocity),pheremone_lattice)

    def __init__(self,int_list,BATCHES,lattice_size,N_agents,channels=4,key=jr.PRNGKey(int(time.time()))):
        
        
        
        def gaussian(sigma,res=100):

            xmax = res//2

            xmax = np.ceil(max(1, xmax))
            ymax = res//2

            ymax = np.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = np.meshgrid(np.arange(ymin, ymax), np.arange(xmin, xmax))

            gb = np.exp(
                -0.5 * (x**2 / sigma**2 + y**2 / sigma**2)
            )
            return gb / np.sum(gb)
        
        self.OBS_CHANNELS = channels
        self.LATTICE_SIZE = lattice_size
        self.key = key
        #ph = jnp.pad(ph,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0)))
        pos_0 = np.random.uniform(0,lattice_size,size=(BATCHES,1,2,N_agents))
        pos_1 = np.random.normal(loc=lattice_size//2,scale=lattice_size//10,size=(BATCHES,1,2,N_agents))
        #pos = einops.rearrange([pos_0,pos_1],"BATCHES N N_agents ndim -> BATCHES N N_agents ndim")
        pos = np.concatenate([pos_0,pos_1],axis=1)
        vel = np.random.uniform(-1,1,size=(BATCHES,2,2,N_agents))

        ph_0 = np.ones((BATCHES,1,channels,lattice_size,lattice_size))
        ph_0[:,:,1:] = 0
        ph_0[:,:,0] = ph_0[:,:,0]#/np.sum(ph_0[:,:,0])
        ph_1 = gaussian(lattice_size*0.1,lattice_size)
        ph_1 = ph_1/np.max(ph_1)
        ph_1 = einops.repeat(ph_1,"x y -> b n c x y",b=BATCHES,n=1,c=channels)
        ph_1[:,:,1:] = 0
        #ph =  einops.rearrange([ph_1,ph_1],"BATCHES N N_agents ndim -> BATCHES N N_agents ndim")
        ph = np.concatenate([ph_0,ph_1],axis=1)
        data_true = ((pos,vel),ph)
        print(ph.shape)
        print(pos.shape)
        self.data_true = data_true
        self.data_saved = data_true

    def data_init(self):
        data = self.return_saved_data()
        #data = self.duplicate_batches(data, 4)
        self.save_data(data)
        
        return None
    def initialize_pool(self):
        return self.split_x_y()


    def callback_training(self,data,i):
        # called in the ABM iteration loop - so that parts of X are always fixed
        # Tuned such that it decays to identity transform after M epochs
        # NOTE this acts on un vmapped model
        ((a_p,a_v),ph) = data
        
        # if i<100:
        #     _a = jnp.linspace(0,1,self.LATTICE_SIZE)
        #     alpha = (100-i)/100.0
        #     ph_x_grad = einops.repeat(_a,"h -> h w", w=self.LATTICE_SIZE)
        #     ph_y_grad = einops.repeat(_a,"w -> h w", h=self.LATTICE_SIZE)
            
        #     ph = ph.at[-1].set(alpha*ph_x_grad + (1-alpha)*ph[-1])
        #     ph = ph.at[-2].set(alpha*ph_y_grad + (1-alpha)*ph[-2])
            

        return ((a_p,a_v),ph)
    
    def duplicate_batches(self,data,B):
        ((a_p,a_v),ph) = data
        a_p = super.duplicate_batches(a_p,B)
        a_v = super.duplicate_batches(a_v,B)
        ph = super.duplicate_batches(ph,B)
        return ((a_p,a_v),ph)
    
    def split_x_y(self, N_steps=1):
        ((a_p,a_v),ph) = self.data_saved
        X = ((a_p[:,:-N_steps],a_v[:,:-N_steps]),ph[:,:-N_steps])
        Y = ((a_p[:,N_steps:],a_v[:,N_steps:]),ph[:,N_steps:])
        return X,Y
    @eqx.filter_jit
    def advance_pool(self, x, y, i,key):
        """Re-initialise X0 to uniform distribtion

        Args:
            x: Initial conditions
            y: Target conditions
            i: iteration number


        """

        #self.key = jr.fold_in(self.key,i)
        key1,key2 = jr.split(key)
        ((x_p,x_v),x_ph) = x
        #((y_p,y_v),y_ph) = y

        #x_p = x_p.at[:,1:].set(x_p[:,:-1]) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        #x_v = x_v.at[:,1:].set(x_v[:,:-1])
        #x_ph = x_ph.at[:,1:].set(x_ph[:,:-1])
        ((x_p_true,x_v_true),x_ph_true),_ =self.split_x_y(1)
        #x_p = x_p.at[:,0].set(x_p_true[:,0])
        
        x_v = x_v.at[:,0].set(jr.uniform(key=key1,shape=x_v[:,0].shape,minval=-1,maxval=1))
        x_p = x_p.at[:,0].set(jr.uniform(key=key2,shape=x_p[:,0].shape,minval=0,maxval=self.LATTICE_SIZE))
        
        x_ph = x_ph.at[:,0].set(x_ph_true[:,0])
        return ((x_p,x_v),x_ph),y