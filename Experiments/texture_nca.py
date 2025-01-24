#from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_multi_scale import mNCA
#from NCA.model.NCA_smooth_model import cNCA
#from NCA.model.NCA_smooth_gated_model import gcNCA
#from NCA.model.NCA_gated_bounded_model import gbNCA
#from NCA.model.NCA_bounded_model import bNCA
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
import jax
from Common.utils import load_textures
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
#from Common.trainer.data_augmenter_tree_subsample_noise import DataAugmenterSubsampleNoiseTexture
from NCA.trainer.data_augmenter_nca_texture import DataAugmenter
import time
import optax



CHANNELS=16
t=64
iters=2000
DOWNSAMPLE = 1


#data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
#data = load_textures(["banded/banded_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg"],downsample=DOWNSAMPLE,crop_square=True,crop_factor=1.5)
data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg"],downsample=DOWNSAMPLE,crop_square=True,crop_factor=1)
schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))


nca = NCA(N_CHANNELS=CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5)#,PADDING="CIRCULAR")
print(nca)
print(nca.partition())

opt = NCA_Trainer(nca,
				  data,
				  model_filename="texture_perlin_nca_test_1",
				  DATA_AUGMENTER=DataAugmenter)
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="vgg",
          LOOP_AUTODIFF="checkpointed")