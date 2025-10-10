from NCA.model.NCA_gated_model import gNCA
#from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer_multiscale_loss import NCA_Trainer_multiscale_loss
import jax
import jax.numpy as jnp
import jax.random as jr
from Common.utils import load_micropattern_time_series
from Common.trainer.custom_functions import multi_channel_perlin_noise
from NCA.trainer.data_augmenter_nca_texture import DataAugmenter
import time
from einops import rearrange
import optax
import sys


PVC_PATH = "/mnt/ceph_rbd/ar-dp/"

index = int(sys.argv[1])
# Index hyperparameters
#CHANNELS = [8,16,24,32][index]
#BATCH_SIZE = [8,8,4,4][index]
CHANNELS = 16
BATCH_SIZE = 8
LOSS_SCALES = [[1],
               [1,2],
               [1,2,4],
               [1,2,4,8]
               [1,2,4,8,16]][index]

print(f"Running with {CHANNELS} channels and batch size {BATCH_SIZE}")


class data_augmenter_subclass(DataAugmenter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.OVERWRITE_OBS_CHANNELS = False
        self.NOISE_CUTOFF = 4
    
    def data_init(self,SHARDING = None,key=jax.random.PRNGKey(int(time.time()))):
        """
        Data must be normalised to [-1,1] range for LPIPS to work. Set the initial conditions to perlin noise

        """
        
        map_to_m1p1 = lambda x:2*(x - jnp.min(x)) / (jnp.max(x) - jnp.min(x)) -1
        data = self.return_saved_data()
        data = jax.tree_util.tree_map(map_to_m1p1,data)
        for i,d in enumerate(data):
            key = jr.fold_in(key,i)
            data[i] = data[i].at[0].set(multi_channel_perlin_noise(data[i].shape[2],data[i].shape[1],self.NOISE_CUTOFF,key)) 
        self.save_data(data)
        return None




#BATCH_SIZE = 8
t=64
iters=1000
DOWNSAMPLE = 1
key = jax.random.PRNGKey(int(time.time()))
key = jr.fold_in(key,index+1234)
impath = PVC_PATH+"Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
data = load_micropattern_time_series(impath,downsample=DOWNSAMPLE)
data_selection_key_base = jr.fold_in(key,1)

#for CHANNELS in [8,16,24]:
data_selection_key = data_selection_key_base
for I in range(5):
    print(f"Training timestep {I}")
    key = jr.fold_in(key,I)
    data_array = []

    # Do an extra sample of t0, as this will be overwritten by the random noise initialisation for texture pretraining
    #key = jr.fold_in(key,I)
    sampled_slices = jr.choice(data_selection_key, data[0].shape[0], shape=(BATCH_SIZE,), replace=False)
    data_array.append(data[0][sampled_slices])


    #for array in data:
    data_selection_key = jr.fold_in(data_selection_key,I)
    sampled_slices = jr.choice(data_selection_key, data[I].shape[0], shape=(BATCH_SIZE,), replace=False)
    data_array.append(data[I][sampled_slices])


    data_array = jnp.array(data_array)
    print(data_array.shape)
    S = data_array.shape[-1]
    data_array = data_array[:,:,:,S//4:S//4+512,S//4:S//4+512]
    data_array = rearrange(data_array,"t b c x y -> b t c x y")
    print(data_array.shape)
    boundary_mask = jnp.ones((data_array.shape[0],1,data_array.shape[-2],data_array.shape[-1]))
    print(boundary_mask.shape)




    #print(data.shape)
    schedule = optax.exponential_decay(5e-3, transition_steps=iters, decay_rate=0.98)
    optimiser = optax.chain(optax.scale_by_param_block_norm(),
                            optax.nadam(schedule))


    nca = gNCA(N_CHANNELS=CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PADDING="REPLICATE",key=key)

    opt = NCA_Trainer_multiscale_loss(
                    LOSS_SCALES=LOSS_SCALES,
                    NCA_model=nca,
                    data=data_array,
                    model_filename=f"FOXA2_SOX17_TBXT_LMBR_texture_pretrained_iso_gnca_boundary_masked_ch{CHANNELS}_bs{BATCH_SIZE}_timestep_{I}_run3_loss_scale_{LOSS_SCALES[-1]}",
                    DATA_AUGMENTER=data_augmenter_subclass,
                    BOUNDARY_MASK=boundary_mask,
                    MODEL_DIRECTORY=PVC_PATH+"models/",
                    LOG_DIRECTORY=PVC_PATH+"logs/")
                    
    opt.train(t,
            iters,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="vgg",
            UPDATE_DATA_EVERY=10,
            LOOP_AUTODIFF="checkpointed")#checkpointed