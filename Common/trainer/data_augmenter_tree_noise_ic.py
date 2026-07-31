from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Common.utils import key_pytree_gen
import jax
import time

class DataAugmenterNoise(DataAugmenterAbstract):
    """Sets initial condition x[0] to noise at each timestep, and randomly subsample target images

    Args:
        DataAugmenterAbstract (_type_): _description_
    """
    def data_init(self,*args):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 4)
        key = jax.random.PRNGKey(int(time.time()))
        #keys = jax.random.split(key,len(data))
        keys = key_pytree_gen(key,[len(data),])
        set_x0_noise = lambda x,key:x.at[0].set(jax.random.uniform(key,shape=x[0].shape,minval=0,maxval=1))	
        data = jax.tree_util.tree_map(set_x0_noise,data,keys)
        self.save_data(data)
        return None
    

    
    def advance_pool(self, x, y, i):
        """
		Called after every training iteration to perform data augmentation and processing		

		Parameters
		----------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states
		i : int
			Current training iteration - useful for scheduling mid-training data augmentation

		Returns
		-------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""
        if hasattr(self, "PREVIOUS_KEY"):
            self.PREVIOUS_KEY = jax.random.fold_in(self.PREVIOUS_KEY,i)
            key = self.PREVIOUS_KEY
        else:
            key = jax.random.PRNGKey(int(time.time()))
            self.PREVIOUS_KEY = key
        x_true,y_true =self.split_x_y(1)
        propagate_xn = lambda x:x.at[1:].set(x[:-1])
        set_x0_noise = lambda x:x.at[0].set(jax.random.uniform(key,shape=x[0].shape,minval=0,maxval=1))	
        x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        x = jax.tree_util.tree_map(set_x0_noise,x) 
        if i < 500:		
            for b in range(len(x)//2):
                x[b*2] = x[b*2].at[1:,:self.OBS_CHANNELS].set(x_true[b*2][1:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
        key = jax.random.fold_in(key,i)
        x = self.noise(x,0.005,key=key) # Add little noise to next steps for stability
        
        
        return x,y