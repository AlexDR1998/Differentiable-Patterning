#from NCA.model.NCA_model import NCA
#from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_smooth_model import cNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_textures
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
from Common.trainer.data_augmenter_tree_subsample import DataAugmenterSubsampleNoiseTexture
import time
import optax



CHANNELS=16
t=64
iters=2000


data = load_textures(["dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adamw(schedule))

nca = cNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],KERNEL_SCALE=3,FIRE_RATE=0.5,PERIODIC=True)
print(nca)
opt = NCA_Trainer(nca,
				  data,
				  model_filename="texture_smooth_nca_test_1",
				  DATA_AUGMENTER=DataAugmenterSubsampleNoiseTexture)
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="vgg")