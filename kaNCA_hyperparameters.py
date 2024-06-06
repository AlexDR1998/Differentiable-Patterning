from NCA.model.NCA_KAN_model import kaNCA
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_kaNCA_hyperparameters
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.trainer.NCA_trainer import NCA_Trainer
import time
import optax
import sys
import jax.random as jr
import jax

index=int(sys.argv[1])-1


key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)

BASIS_FUNCS,BASIS_WIDTH=index_to_kaNCA_hyperparameters(index)

CHANNELS=8
DOWNSAMPLE=3
T=32
ITERATIONS=2000

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 2)
        data = self.pad(data, 10) 		
        self.save_data(data)
        return None


data = load_emoji_sequence(["crab.png","microbe.png","microbe.png"],downsample=DOWNSAMPLE)
data_filename = "cr_mi"

nca = kaNCA(CHANNELS,
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=0.5,
            PADDING="REPLICATE",
            BASIS_FUNCS=BASIS_FUNCS,
            BASIS_WIDTH=BASIS_WIDTH,
            key=key)

opt = NCA_Trainer(nca,
                  data,
                  model_filename="kaNCA_hyperparameters_res_"+str(BASIS_FUNCS)+"_width_"+str(BASIS_WIDTH)+"_data_"+data_filename,
                  DATA_AUGMENTER=data_augmenter_subclass,
                  GRAD_LOSS=True)

schedule = optax.exponential_decay(1e-3, transition_steps=ITERATIONS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

opt.train(T,
        ITERATIONS,
        WARMUP=10,
        optimiser=optimiser,
        LOSS_FUNC_STR="euclidean",
        LOOP_AUTODIFF="checkpointed",
        key=key)