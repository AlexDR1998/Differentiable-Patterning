import jax
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax
import sys
from einops import repeat, rearrange

PVC_PATH = "/mnt/ceph/ar-dp/"

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        W = data[0].shape[-1]
        H = data[0].shape[-2]
        padwidth = W%16 + 16
        print(W)
        print("Padding width: ",padwidth)
        data = self.duplicate_batches(data, 4)
        data = self.pad(data, padwidth//2) 		
        self.save_data(data)
        return None

CHANNELS=24
DOWNSAMPLE = 1
UPSCALE = 4
t=128
iters=8000

key = jax.random.PRNGKey(int(time.time()))
#key = jax.random.fold_in(key,index)

data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png"],impath_emojis=PVC_PATH+"Data/Emojis/",downsample=DOWNSAMPLE)
data_filename = "cr_mi_al"

data = repeat(data, "b t c x y -> b t c (x h) (y w)", h=UPSCALE, w=UPSCALE)



schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))


mnca = mNCA(N_CHANNELS=CHANNELS,
            SCALES=[UPSCALE,2,1],
            GATED = False,
            KERNEL_STR=["ID","LAP","GRAD"],
            ACTIVATION=jax.nn.relu, 
            PADDING="REPLICATE", 
            FIRE_RATE=0.5, 
            key=key)

trainer = NCA_Trainer(mnca,
                      data,
                      model_filename="multiscale_nca"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True,
                      MODEL_DIRECTORY=PVC_PATH+"models/",
                      LOG_DIRECTORY=PVC_PATH+"logs/")
trainer.train(t,
              iters,
              WARMUP=10,
              optimiser=optimiser,
              wandb_args={
                "project":"nca-experiments",
                "group":"multiscale_image_morph",
                "tags":["training","mNCA"]},
              LOSS_FUNC_STR="euclidean")