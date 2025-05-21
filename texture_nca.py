from NCA.model.NCA_gated_model import gNCA
#from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer_multiscale_loss import NCA_Trainer_multiscale_loss
import jax
from Common.utils import load_textures
from NCA.trainer.data_augmenter_nca_texture import DataAugmenter
import time
import optax


PVC_PATH = "/mnt/ceph_rbd/ar-dp/"

CHANNELS=16
t=64
iters=1000
DOWNSAMPLE = 1
key = jax.random.PRNGKey(int(time.time()))

#data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
#data = load_textures(["banded/banded_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg"],downsample=DOWNSAMPLE,crop_square=True,crop_factor=1.5)
#data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg","dotted/dotted_0109.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)
#data = load_textures(["scaly/scaly_0136.jpg","scaly/scaly_0136.jpg","scaly/scaly_0136.jpg","honeycombed/honeycombed_0078.jpg","honeycombed/honeycombed_0078.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)


data = load_textures(["banded/banded_0041.jpg","banded/banded_0041.jpg","banded/banded_0041.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)
data = data[:,:,:,:512,:512]
print(data.shape)
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.98)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))


nca = gNCA(N_CHANNELS=CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PADDING="REPLICATE",key=key)
# nca = mNCA(N_CHANNELS=CHANNELS,
#             SCALES=[1,8,32,64],
#             GATED = False,
#             KERNEL_STR=["ID","LAP","GRAD"],
#             ACTIVATION=jax.nn.relu, 
#             PADDING="REPLICATE", 
#             FIRE_RATE=0.5, 
#             key=key)
#print(nca)
#print(nca.partition())

opt = NCA_Trainer_multiscale_loss(
                  LOSS_SCALES=[1,4,16],
                  NCA_model=nca,
				  data=data,
				  model_filename="texture_gated_nca_multiscale_loss_iso_scale_honey_1",
				  DATA_AUGMENTER=DataAugmenter,
                  MODEL_DIRECTORY=PVC_PATH+"models/",
                  LOG_DIRECTORY=PVC_PATH+"logs/")
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="vgg",
          UPDATE_DATA_EVERY=10,
          LOOP_AUTODIFF="checkpointed")