import jax.numpy as jnp
import jax.tree_util as jtu
from Common.dataloader.micropattern_schemas import MICROPATTERN_4CH_SCHEMA
from NCA.trainer.data_augmenter_nca_basic import (
    DataAugmenter as DataAugmenterBasic,
    jittable_callback_bit,
)


class DataAugmenter(DataAugmenterBasic):
    schema = MICROPATTERN_4CH_SCHEMA

    def __init__(self, *args, **kwargs):
        model = kwargs.get("nca_model")
        self.channels = (
            model.N_CHANNELS
            if model is not None
            else self.schema.n_state_channels + kwargs.get("hidden_channels", 0)
        )
        if self.channels < self.schema.n_state_channels:
            raise ValueError("NCA has fewer channels than the 4-channel schema")
        super().__init__(*args, **kwargs)
        self.OBS_CHANNELS = self.schema.n_state_channels

    def _to_state(self, data):
        return jnp.pad(
            data,
            ((0, 0), (0, self.channels - data.shape[1]), (0, 0), (0, 0)),
        )

    def split_x_y(self, N_steps=1):
        """
        Splits 4-channel group-A data into initial and target states.

        This keeps the simple 4-channel task direct, without the 12-to-9
        duplicate-colony reduction used by the full micropattern dataset.
        """
        if self.batch_mode == "array":
            x = self.map_batches(
                lambda data: self.real_to_latent(self._to_state(data[:-N_steps])),
                self.data_saved,
            )
            y = self.data_saved[:, N_steps:]
        else:
            x = jtu.tree_map(
                lambda data: self.real_to_latent(self._to_state(data[:-N_steps])),
                self.data_saved,
            )
            y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
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
