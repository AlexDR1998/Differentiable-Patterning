from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import jax
import time
class DataAugmenterSubsampleNoiseTexture(DataAugmenterAbstract):
    def __init__(self, data_true, hidden_channels=0):
        super().__init__(data_true, hidden_channels)
        self.sample_size = 32
        self.resample_freq = 8

    def data_init(self,*args):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 4)
        self.save_data(data)


    def data_load(self):
        x0,y0 = self.split_x_y(1)
        init_sample = lambda x:x[:,:,:self.sample_size,:self.sample_size]
        x0 = jax.tree_util.tree_map(init_sample,x0)
        y0 = jax.tree_util.tree_map(init_sample,y0)
        print("X0 shape: "+str(x0[0].shape))
        
        return x0,y0
    def split_x_y(self, N_steps=1,key=jax.random.PRNGKey(int(time.time()))):
        x,y = super().split_x_y(N_steps)
        set_x0_noise = lambda x:x.at[0].set(jax.random.uniform(key,shape=x[0].shape,minval=0,maxval=0.1))
        x = jax.tree_util.tree_map(set_x0_noise,x)
        return x,y
        
    def data_callback(self, x, y, i):
        """
        Every self.sample_freq calls, reset x and y to a different sampled patch from self.split_x_y()
        Otherwise do the usual propagation of intermediate states. Set X0 to noise

        Args:
            x (_type_): _description_
            y (_type_): _description_
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        if hasattr(self, "PREVIOUS_KEY"):
            key = jax.random.fold_in(self.PREVIOUS_KEY,i)
            self.PREVIOUS_KEY = key
        else:
            key=jax.random.PRNGKey(int(time.time()))
            self.PREVIOUS_KEY = key
        if i%self.resample_freq==0:
            #Reset the spatial subsample every "resample_freq" steps
            self.SPATIAL_KEY = jax.random.fold_in(self.PREVIOUS_KEY,i)
            x_inds = []
            y_inds = []
            for j in range(len(x)):
                x_inds.append(jax.random.randint(jax.random.fold_in(self.SPATIAL_KEY,j),shape=(1,),minval=0,maxval=x[j].shape[-2]-self.sample_size)[0])
                y_inds.append(jax.random.randint(jax.random.fold_in(self.SPATIAL_KEY,j),shape=(1,),minval=0,maxval=x[j].shape[-1]-self.sample_size)[0])
            sample = lambda x,xi,yi: x[:,:,xi:xi+self.sample_size,yi:yi+self.sample_size]
            x_true,y_true =self.split_x_y(1,key=key)

            x = jax.tree_util.tree_map(sample,x_true,x_inds,y_inds)
            y = jax.tree_util.tree_map(sample,y_true,x_inds,y_inds)
            
        else:
            propagate_xn = lambda x:x.at[1:].set(x[:-1])
            set_x0_noise = lambda x:x.at[0].set(jax.random.uniform(key,shape=x[0].shape,minval=0,maxval=0.1))
            x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
            x = jax.tree_util.tree_map(set_x0_noise,x) 
        
        key = jax.random.fold_in(key,i)
        x = self.noise(x,0.005,key=key) 
        return x,y