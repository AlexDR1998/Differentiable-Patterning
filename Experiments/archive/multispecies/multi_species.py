from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_data_nca_type_multi_species
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from einops import rearrange,repeat
import time
import jax
import jax.numpy as np
import optax
import matplotlib.pyplot as plt
import sys


CHANNELS = 32           # How many channels to use in the model
TRAINING_STEPS = 8000   # How many steps to train for
DOWNSAMPLE = 1          # How much to downsample the image by
NCA_STEPS = 64          # How many NCA steps between each image in the data sequence


index=int(sys.argv[1])-1
data_index,model_index = index_to_data_nca_type_multi_species(index)

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        #data = self.duplicate_batches(data,2)
        data = self.pad(data, 10) 		
        self.save_data(data)
        return None
    

if data_index == 0:
    data = load_emoji_sequence(["crab.png","microbe.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi"
if data_index == 1:
    data = load_emoji_sequence(["microbe.png","avocado_1f951.png"],downsample=DOWNSAMPLE)
    data_filename = "mi_av"
if data_index == 2:
    data = load_emoji_sequence(["avocado_1f951.png","crab.png"],downsample=DOWNSAMPLE)
    data_filename = "av_cr"
if data_index == 3:
    data = load_emoji_sequence(["crab.png","microbe.png","avocado_1f951.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi_av"
if data_index == 4:
    data = load_emoji_sequence(["crab.png","microbe.png","avocado_1f951.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi_av_al"


data = rearrange(data,"B T C W H -> T B C W H")

data = repeat(data,"B T C W H -> (B b) T C W H",b=2)

initial_condition = np.array(data)

W = initial_condition.shape[-2]
H = initial_condition.shape[-1]

initial_condition = initial_condition.at[:,:,:,:W//2-3].set(0)
initial_condition = initial_condition.at[:,:,:,W//2+2:].set(0)
initial_condition = initial_condition.at[:,:,:,:,:H//2-3].set(0)
initial_condition = initial_condition.at[:,:,:,:,H//2+2:].set(0)
data = np.concatenate([initial_condition,data,data],axis=1) # Join initial condition and data along the time axis
print("(Batch, Time, Channels, Width, Height): "+str(data.shape))


if model_index == 0:
    print("Training NCA model on "+data_filename)
    model = NCA(N_CHANNELS=CHANNELS,
                KERNEL_STR=["ID","GRAD","LAP"],
                ACTIVATION=jax.nn.relu,
                PADDING="CIRCULAR",
                FIRE_RATE=0.5)
    nca_filename = "nca"
    
if model_index==1:
    print("Training gated NCA model on "+data_filename)
    model = gNCA(N_CHANNELS=CHANNELS,
                KERNEL_STR=["ID","GRAD","LAP"],
                ACTIVATION=jax.nn.relu,
                PADDING="CIRCULAR",
                FIRE_RATE=0.5)
    nca_filename = "gnca"

trainer = NCA_Trainer(model,
                      data,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      model_filename="multi_species_stable_"+nca_filename+"_"+data_filename)

schedule = optax.exponential_decay(1e-3, transition_steps=TRAINING_STEPS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

try:
    trainer.train(t=NCA_STEPS,iters=TRAINING_STEPS,LOOP_AUTODIFF="lax",optimiser=optimiser)
except:
    print("Not enough memory, falling back to checkpointed gradients")
    trainer.train(t=NCA_STEPS,iters=TRAINING_STEPS,LOOP_AUTODIFF="checkpointed",optimiser=optimiser)
