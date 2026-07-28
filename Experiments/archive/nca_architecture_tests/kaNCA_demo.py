from NCA.model.NCA_KAN_model import kaNCA
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_kaNCA_hyperparameters
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.trainer.NCA_trainer import NCA_Trainer
import time
import optax
import jax
import jax.random as jr
import sys

data_index=int(sys.argv[1])-1
#data_index,nca_type_index = index_to_data_nca_type(index)

CHANNELS=12
BASIS_FUNCS = 16
DOWNSAMPLE = 2
T=32
iters=8000

key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,data_index)

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 2)
        data = self.pad(data, 10) 		
        self.save_data(data)
        return None

if data_index == 0:
    data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi_al"
if data_index == 1:
    data = load_emoji_sequence(["microbe.png","avocado_1f951.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "mi_av_al"
if data_index == 2:
    data = load_emoji_sequence(["avocado_1f951.png","mushroom_1f344.png","lizard_1f98e.png","lizard_1f98e.png"],downsample=DOWNSAMPLE)
    data_filename = "av_mu_li"
if data_index == 3:
    data = load_emoji_sequence(["mushroom_1f344.png","alien_monster.png","rooster_1f413.png","rooster_1f413.png"],downsample=DOWNSAMPLE)
    data_filename = "mu_al_ro"
if data_index == 4:
    data = load_emoji_sequence(["alien_monster.png","mushroom_1f344.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "al_mu_al"
if data_index == 5:
    data = load_emoji_sequence(["butterfly.png","eye.png","lizard_1f98e.png","lizard_1f98e.png"],downsample=DOWNSAMPLE)
    data_filename = "bu_ey_li"
if data_index == 6:
    data = load_emoji_sequence(["rooster_1f413.png","butterfly.png","eye.png","eye.png"],downsample=DOWNSAMPLE)
    data_filename = "ro_bu_ey"
if data_index == 7:
    data = load_emoji_sequence(["lizard_1f98e.png","microbe.png","anatomical_heart.png","anatomical_heart.png"],downsample=DOWNSAMPLE)
    data_filename = "li_mi_an"



schedule = optax.exponential_decay(1e-4, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))



nca = kaNCA(CHANNELS,
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=0.5,
            PADDING="REPLICATE",
            BASIS_FUNCS=BASIS_FUNCS,
            BASIS_WIDTH=4,
            INIT_SCALE=0.1,
            key=key)


				  	    
opt = NCA_Trainer(nca,
                  data,
                  model_filename="kaNCA_demo/channels_"+str(CHANNELS)+"_res_"+str(BASIS_FUNCS)+"_data_"+data_filename,
                  DATA_AUGMENTER=data_augmenter_subclass,
                  GRAD_LOSS=True)

opt.train(T,
        iters,
        WARMUP=10,
        optimiser=optimiser,
        LOSS_FUNC_STR="euclidean",
        LOOP_AUTODIFF="checkpointed",
        LOG_EVERY=50,
        key=key)