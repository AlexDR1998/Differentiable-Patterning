import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import sys
import time
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_KAN_model import kaNCA
from NCA.trainer.NCA_trainer_multiscale_loss import NCA_Trainer_multiscale_loss
from NCA.trainer.data_augmenter_nca_texture import DataAugmenter
from Common.utils import load_textures
key = jax.random.PRNGKey(int(time.time()))

PVC_PATH = "/mnt/ceph_rbd/ar-dp/"

index = int(sys.argv[1])
#index = 2
# Hyperparameters
CHANNELS = 8
t = 64
iters = 1000
DOWNSAMPLE = 2
LOSS_SCALES = [1,2,4,8,16]
BASIS_FUNCS = [4,8,16,32][index]

print(f"Starting kaNCA texture synthesis training sweep over multiple images, with {BASIS_FUNCS} basis functions")

# List of texture image filenames (relative to the texture directory)
texture_files = [
    "dotted/dotted_0109.jpg",
    "dotted/dotted_0106.jpg",
    "dotted/dotted_0188.jpg",
    "crystalline/crystalline_0204.jpg",
    "bumpy/bumpy_0059.jpg",
    "bumpy/bumpy_0163.jpg",
    "bumpy/bumpy_0107.jpg",
    "meshed/meshed_0098.jpg",
    "paisley/paisley_0050.jpg",
    "paisley/paisley_0122.jpg",
    "stained/stained_0061.jpg",
    "marbled/marbled_0150.jpg",
    "marbled/marbled_0060.jpg",
    "cracked/cracked_0004.jpg",
    "cracked/cracked_0064.jpg",
    "freckled/freckled_0142.jpg",
    "checkered/checkered_0017.jpg",
    "interlaced/interlaced_0045.jpg",
    "veined/veined_0093.jpg",
    "freckled/freckled_0095.jpg",
    "stratified/stratified_0067.jpg",
    "stratified/stratified_0148.jpg",
    "lined/lined_0170.jpg",
    "smeared/smeared_0096.jpg",
    "smeared/smeared_0129.jpg",
    "smeared/smeared_0139.jpg",
    "blotchy/blotchy_0060.jpg",
    "perforated/perforated_0106.jpg",
    "striped/striped_0011.jpg",
    "striped/striped_0083.jpg",
    "striped/striped_0099.jpg",
    "honeycombed/honeycombed_0078.jpg",
    "scaly/scaly_0136.jpg",
    "bubbly/bubbly_0101.jpg",
    "banded/banded_0041.jpg",
    "grid/grid_0002.jpg",
    "braided/braided_0107.jpg",    
]

# Number of training runs per texture (to train many models)
runs_per_texture = 1

for texture in texture_files:
    
    for run in range(runs_per_texture):
        key = jr.fold_in(key,1)
        jax.clear_caches()
        try:
            print(f"Training gNCA model on texture: {texture} (run {run+1}/{runs_per_texture})")
            
            # Load the texture data for the current image.
            data = load_textures([texture,texture,texture],
                                impath_textures=PVC_PATH + "Data/dtd/images/",
                                downsample=DOWNSAMPLE,
                                crop_square=True,
                                crop_factor=1)
            # Crop the loaded texture to a fixed size (e.g., 256x256)
            imsize = data.shape[-1]
            # largest_power = 1
            # while largest_power * 2 < imsize:
            #     largest_power *= 2
            largest_power = 128
            print(f"Cropping texture to {largest_power}x{largest_power}")
            data = data[:,:,:,:largest_power,:largest_power]
            
            jax.clear_caches()
            key, subkey = jax.random.split(key)
            
            # Initialize the gNCA model for texture synthesis
            # nca_model = gNCA(
            #     N_CHANNELS=CHANNELS,
            #     KERNEL_STR=["ID", "LAP", "GRAD"],
            #     FIRE_RATE=0.5,
            #     PADDING="REPLICATE",
            #     key=subkey
            # )

            nca_model = kaNCA(N_CHANNELS=CHANNELS,
                KERNEL_STR=["ID","LAP","GRAD"],
                FIRE_RATE=0.5,
                PADDING="REPLICATE",
                BASIS_FUNCS=BASIS_FUNCS,
                BASIS_WIDTH=4,
                INIT_SCALE=0.0001,
                key=subkey)
            
            schedule = optax.exponential_decay(1e-4, transition_steps=iters, decay_rate=0.99)
            optimiser = optax.chain(
                optax.scale_by_param_block_norm(),
                optax.nadam(schedule)
            )
            
            trainer = NCA_Trainer_multiscale_loss(
                LOSS_SCALES=LOSS_SCALES,
                NCA_model=nca_model,
                data=data,
                model_filename=f"texture_ka_nca_bf{BASIS_FUNCS}_grad_{texture.split('/')[-1].split('.')[0]}_run{run+1}_multiscale_euclidean_{LOSS_SCALES[-1]}",
                DATA_AUGMENTER=DataAugmenter,
                MODEL_DIRECTORY=PVC_PATH + "models/",
                LOG_DIRECTORY=PVC_PATH + "logs/"
            )
            
            trainer.train(
                t,
                iters,
                WARMUP=10,
                optimiser=optimiser,
                LOG_EVERY=100,
                LOSS_FUNC_STR="euclidean",
                UPDATE_DATA_EVERY=10,
                LOOP_AUTODIFF="checkpointed"
            )
        except Exception as e:
            print(f"Error training model on texture {texture} (run {run+1}): {e}")
            continue
