import jax.tree_util as jtu
from NCA.trainer.data_augmenter_nca_basic import (
    DataAugmenter as DataAugmenterBasic,
    jittable_callback_bit,
)


class DataAugmenter(DataAugmenterBasic):
    def split_x_y(self, N_steps=1):
        """
        Splits 4-channel group-A data into initial and target states.

        This keeps the simple 4-channel task direct, without the 12-to-9
        duplicate-colony reduction used by the full micropattern dataset.
        """
        if self.batch_mode == "array":
            x = self.map_batches(self.real_to_latent, self.data_saved[:, :-N_steps])
            y = self.data_saved[:, N_steps:]
        else:
            x = jtu.tree_map(lambda data: data[:-N_steps], self.data_saved)
            y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
            x = jtu.tree_map(lambda x: self.real_to_latent(x), x)
        return x, y
    
    def data_callback(self,x,y,i,key):
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

        
        x_true,_ =self.split_x_y(1)
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS,key)
        x = self.noise(x,0.01,key=key)
        self.PREVIOUS_KEY = key
        return x,y
