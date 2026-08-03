from ABM.trainer.data_augmenter_abm import DataAugmenter
import jax.numpy as jnp
import jax.random as jr
import time
import numpy as np
import pandas as pd

class DataAugmenter(DataAugmenter):
    # Data format is (((B,N,N_agents,2),(B,N,N_agents,2)),(B,N,CHANNELS,width,height))
    #   ((agent_position,agent_velocity),pheremone_lattice)

    def __init__(self,int_list,BATCHES,lattice_size,N_agents,channels=4):
        df = pd.read_csv("../Data/mnist_point_cloud/train.csv")
        df_sampled = df.groupby("label").sample(BATCHES)
        df_sampled = df_sampled[df_sampled["label"].isin(int_list)]
        key=jr.PRNGKey(int(time.time()))
        self.key = key
        number_of_points = df_sampled.filter(regex="y").gt(-1).sum(axis=1)
        min_number = number_of_points.min()
        self.N_agents = N_agents
        self.lattice_size = lattice_size
        #print("Minimum number of points: "+str(min_number))

        weights = df_sampled.filter(regex="v")
        weights = weights[weights>0]
        weights = weights.fillna(0)
        weights = weights.div(weights.sum(axis=1),axis=0)

        def sample_by_min(df,weights,i,N):
            #print(weights.iloc[i].to_numpy().shape)
            indexes = np.random.choice(np.arange(351),N,replace=False,p=weights.iloc[i])
            X = df.iloc[i].filter(regex="x")
            Y = df.iloc[i].filter(regex="y")
            X = X.iloc[indexes].to_numpy()
            Y = Y.iloc[indexes].to_numpy()
            
            return jnp.array([X,Y])

        
        pos = []
        for i in range(BATCHES):
            data_batch = []
            for n in range(len(int_list)):
                
                data_batch.append(sample_by_min(df_sampled,weights,i+n*BATCHES,min_number))
            pos.append(data_batch)
        
        pos = jnp.array(pos)
        #print("Agent positions: "+str(pos.shape))
        pos = pos* (lattice_size/27.0)
        vel = jr.uniform(key=key,shape=pos.shape,minval=-1,maxval=1)
        ph = jnp.zeros((BATCHES,len(int_list),channels,lattice_size,lattice_size))
        

        self.OBS_CHANNELS = channels
        #ph = jnp.pad(ph,((0,0),(0,0),(0,hidden_channels),(0,0),(0,0)))
        
        data_true = ((pos,vel),ph)
        self.data_true = data_true
        self.data_saved = data_true

        
        return None
    def initialize_pool(self):
        self.key = jr.fold_in(self.key,1)
        key1,key2,key3 = jr.split(self.key,3)
        x0,y0 = self.split_x_y()
        ((x_p,x_v),x_ph) = x0
        x_p = jr.uniform(key=key1,shape=x_p.shape,minval=0,maxval=self.lattice_size)
        x_v = jr.uniform(key=key2,shape=x_v.shape,minval=-1,maxval=1)
        x_ph = x_ph.at[:,0].set(jr.uniform(key=key3,shape=x_ph[:,0].shape,minval=0,maxval=1))
        
        return ((x_p,x_v),x_ph),y0


    def advance_pool(self, x, y, i,key):
        self.key = jr.fold_in(self.key,i)
        key1,key2,key3 = jr.split(self.key,3)
        ((x_p,x_v),x_ph) = x
        #((y_p,y_v),y_ph) = y

        x_p = x_p.at[:,1:].set(x_p[:,:-1]) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        x_v = x_v.at[:,1:].set(x_v[:,:-1])
        x_ph = x_ph.at[:,1:].set(x_ph[:,:-1])
        ((x_p_true,x_v_true),x_ph_true),_ =self.split_x_y(1)
        x_p = x_p.at[:,0].set(jr.uniform(key=key1,shape=(x_p[:,0].shape[0]),minval=0,maxval=self.lattice_size))
        x_v = x_v.at[:,0].set(jr.uniform(key=key2,shape=(x_v[:,0].shape[0]),minval=-1,maxval=1))
        x_ph = x_ph.at[:,0].set(jr.uniform(key=key3,shape=x_ph[:,0].shape,minval=0,maxval=1))

        return ((x_p,x_v),x_ph),y