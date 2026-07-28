from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_SAE_trainer_v3 import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
from NCA.trainer.data_augmenter_nca import DataAugmenter
from Common.utils import index_to_param_list
from NCA.analysis.optimiser import unitary_decoder_transform
import jax
import jax.numpy as np
from einops import rearrange, repeat
import optax
from Common.utils import load_emoji_sequence
import jax.random as jr
import time
import optax
import sys

index = int(sys.argv[1])
#--- Static hyperparameters ----#
TOTAL_NUM_GPUS=1
PVC_PATH = "/mnt/ceph/ar-dp/"
CHANNELS = 48
STEPS_BETWEEN_IMAGES = 128
TRAINING_STEPS = 3000
DOWNSAMPLE = 1  # How much to downsample the image by

#--- Dynamic hyperparameters ----#
FULL_HYPERPARAMETERS = {
    "SPARSITY_PARAM": [256],
    "HIDDEN_DIM": [1024],
    "SPARSITY_STRENGTH": [0.01],
    "OPTIMISER_MODE": ["grad_norm_nadam_norm"],
    "SAE_ACTIVATION": ["topk","relu"],
    "TARGET_LAYER": ["linear_hidden","activation","linear_output"],
    "LOSS_FUNCTION": ["l2","spectral"],
}

HYPERPARAMETER_LIST = index_to_param_list(index,n_processes=TOTAL_NUM_GPUS,full_hyperparameters=FULL_HYPERPARAMETERS)



# --- Prepare Data --- #
class data_augmenter_subclass(DataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.pad(data, 10)
        self.save_data(data)
        return None
    

data = load_emoji_sequence(
    [
        "crab.png",
        "microbe.png",
        "avocado.png",
        "alien_monster.png",
        "butterfly.png",
        "lizard.png",
        "mushroom.png",
    ],
    downsample=DOWNSAMPLE,
    impath_emojis=PVC_PATH + "Data/Emojis/",
)
data_filename = "cr_mi_av_al_bt_li_mu"

data = rearrange(data, "B T C W H -> T B C W H")

data = repeat(data, "B T C W H -> (B b) T C W H", b=1)

initial_condition = np.array(data)

W = initial_condition.shape[-2]
H = initial_condition.shape[-1]

initial_condition = initial_condition.at[:, :, :, : W // 2 - 6].set(0)
initial_condition = initial_condition.at[:, :, :, W // 2 + 5 :].set(0)
initial_condition = initial_condition.at[:, :, :, :, : H // 2 - 6].set(0)
initial_condition = initial_condition.at[:, :, :, :, H // 2 + 5 :].set(0)
data = np.concatenate(
    [initial_condition, data, data], axis=1
)  # Join initial condition and data along the time axis
print("(Batch, Time, Channels, Width, Height): " + str(data.shape))


#---- Load NCA  ----#
nca = gNCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "GRAD", "LAP"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5,
    key=jr.PRNGKey(int(time.time())),
)
nca = nca.load(
    PVC_PATH+f"models/multi_species_stable_gnca_grad_{CHANNELS}ch_cr_mi_av_al_bt_li_mu_ds_1_long.eqx"
)

key = jr.fold_in(jr.PRNGKey(int(1000*time.time())), index)


for HPAR in HYPERPARAMETER_LIST:
    jax.clear_caches()
    key = jr.fold_in(key, index)
    SPARSITY_PARAM = HPAR["SPARSITY_PARAM"]
    HIDDEN_DIM = HPAR["HIDDEN_DIM"]
    SPARSITY_STRENGTH = HPAR["SPARSITY_STRENGTH"]
    TARGET_LAYER = HPAR["TARGET_LAYER"]
    LOSS_FUNCTION = HPAR["LOSS_FUNCTION"]
    OPTIMISER_MODE = HPAR["OPTIMISER_MODE"]
    SAE_ACTIVATION = HPAR["SAE_ACTIVATION"]
    
    FILENAME = f"SAE_{TARGET_LAYER}_k{SPARSITY_PARAM}_hd{HIDDEN_DIM}_sp{SPARSITY_STRENGTH}_act_{SAE_ACTIVATION}_emoji_multi_species_gated_nca_grad_{CHANNELS}ch_opt_{OPTIMISER_MODE}_loss_{LOSS_FUNCTION}"
    wandb_config = {"project":"multi_species_patterning",
                    "name":FILENAME,
                    "group":"SAE trajectory hyperparameter sweep - no sparsity",
                    "tags":[
                        "multi_species",
                        "gated",
                        "grad",
                        "SAE path",
                        OPTIMISER_MODE,
                        f"hidden dim {HIDDEN_DIM}",
                        f"sparsity param {SPARSITY_PARAM}"
                        ]}
    
    #--- Optimiser ----#
    warmup_steps = 100  # number of steps for warmup
    init_lr = 1e-6      # starting learning rate
    target_lr = 1e-4    # learning rate after warmup

    warmup_fn = optax.linear_schedule(
        init_value=init_lr,
        end_value=target_lr,
        transition_steps=warmup_steps,
    )
    decay_fn = optax.exponential_decay(
        init_value=target_lr,
        transition_steps=TRAINING_STEPS,
        decay_rate=0.98,
    )
    schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )
    if OPTIMISER_MODE == "grad_clip_nadam_norm":
        optimiser = optax.chain(
            optax.adaptive_grad_clip(clipping=1), 
            optax.nadam(schedule),
            unitary_decoder_transform(norm=1.0,eps=1e-8,axis=1)
        )
    elif OPTIMISER_MODE == "grad_norm_nadam_norm":
        optimiser = optax.chain(
            optax.scale_by_param_block_norm(), 
            optax.nadam(schedule),
            unitary_decoder_transform(norm=1.0,eps=1e-8,axis=1)
        )
    elif OPTIMISER_MODE == "nadam_norm":
        optimiser = optax.chain(
            optax.nadam(schedule),
            unitary_decoder_transform(norm=1.0,eps=1e-8,axis=1)
        )
    TRAINING_STEPS+= warmup_steps
    #--- SAE paramaeters ----#

    SAE = SparseAutoencoder(
        TARGET_LAYER=TARGET_LAYER,
        N_CHANNELS=CHANNELS,
        N_KERNELS=4,
        ACTIVATION=SAE_ACTIVATION,
        GATED=True,
        latent_dim=HIDDEN_DIM,
        TIED_INIT=False,
        sparsity_param=SPARSITY_PARAM,
        key=key,
    )

   

    #--- Setup trainer class
    trainer = NCA_SAE_Trainer(
        NCA_model=nca,
        SAE=SAE,
        data=data,
        DATA_AUGMENTER=data_augmenter_subclass,
        model_filename=FILENAME,
        MODEL_DIRECTORY="models/",
        LOG_DIRECTORY="logs/"
    )


    #--- Do Training ----#
    # try:
    trainer.train(
        t=STEPS_BETWEEN_IMAGES,
        iters=TRAINING_STEPS,
        optimiser=optimiser,
        SPARSITY_STRENGTH=SPARSITY_STRENGTH,
        WARMUP=warmup_steps,
        CLEAR_CACHE_EVERY=4000,
        LOG_EVERY=500,
        LOSS_FUNC_STR=LOSS_FUNCTION,
        LOOP_AUTODIFF="checkpoint",
        wandb_args=wandb_config,
        )
    # except Exception as e:
    #     print(f"Error with hyperparameters {HPAR}: {e}")