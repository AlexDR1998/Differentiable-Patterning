import jax

#from NCA.model.NCA_model import NCA
#from NCA.model.NCA_gated_model import gNCA
#from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
#from Common.eddie_indexer import index_to_data_nca_type
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax
import sys

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 4)
        data = self.pad(data, 10) 		
        self.save_data(data)
        return None

CHANNELS=32
DOWNSAMPLE = 1
t=64
iters=8000

key = jax.random.PRNGKey(int(time.time()))
#key = jax.random.fold_in(key,index)

data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
data_filename = "cr_mi_al"


schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))


mnca = mNCA(N_CHANNELS=CHANNELS,
            SCALES=[1,4],
            GATED = False,
            KERNEL_STR=["ID","LAP","GRAD"],
            ACTIVATION=jax.nn.relu, 
            PADDING="REPLICATE", 
            FIRE_RATE=0.5, 
            key=key)

trainer = NCA_Trainer(mnca,
                      data,
                      model_filename="multiscale_nca_"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True)
trainer.train(t,
              iters,
              WARMUP=10,
              optimiser=optimiser,
              LOSS_FUNC_STR="euclidean")