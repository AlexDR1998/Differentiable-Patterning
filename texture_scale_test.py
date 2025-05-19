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
data = load_textures(["scaly/scaly_0136.jpg","scaly/scaly_0136.jpg","scaly/scaly_0136.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)
#data = load_textures(["honeycombed/honeycombed_0078.jpg","honeycombed/honeycombed_0078.jpg","honeycombed/honeycombed_0078.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)

#data = load_textures(["banded/banded_0041.jpg","banded/banded_0041.jpg","banded/banded_0041.jpg"],impath_textures=PVC_PATH+"Data/dtd/images/",downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)
data = data[:,:,:,:256,:256]
print(data.shape)
schedule = optax.exponential_decay(5e-3, transition_steps=iters, decay_rate=0.97)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))



loss_scales_list = [
    [1, ],
    [1, 2],
    [1, 2, 4],
    [1, 2, 4, 8],
    [1, 2, 4, 8, 16],
    [1, 2, 4, 8, 16, 32],
    # [1, 4, 16],
    # [1, 8, 64],
    # [2, 4, 8],
    # [4, 8, 16],
    # [8, 16, 32],
    # [16, 32, 64],
]

for scales in loss_scales_list:
    jax.clear_caches()
    try:
        print(f"Starting training with LOSS_SCALES: {scales}")
        # Reinitialize a new model for each run using a fresh random key
        
        key, subkey = jax.random.split(key)
        nca_model = NCA(
            N_CHANNELS=CHANNELS,
            KERNEL_STR=["ID", "LAP", "DIFF"],
            FIRE_RATE=0.5,
            PADDING="REPLICATE",
            key=subkey
        )
        trainer = NCA_Trainer_multiscale_loss(
            LOSS_SCALES=scales,
            NCA_model=nca_model,
            data=data,
            model_filename=f"texture_normal_nca_multiscale_vgg_loss_iso_scale_{scales[-1]}",
            DATA_AUGMENTER=DataAugmenter,
            MODEL_DIRECTORY=PVC_PATH + "models/",
            LOG_DIRECTORY=PVC_PATH + "logs/"
        )
        trainer.train(
            t,
            iters,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="vgg",
            UPDATE_DATA_EVERY=10,
            LOOP_AUTODIFF="checkpointed"
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        continue
