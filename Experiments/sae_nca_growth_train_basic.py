from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Emoji
from NCA.analysis.NCA_SAE_trainer import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
import jax
import jax.random as jr
import time
import optax
import sys
from Common.utils import index_to_param_list
from NCA.analysis.optimiser import unitary_decoder_transform

TOTAL_NUM_GPUS = 1
index = int(sys.argv[1])
PVC_PATH = "/mnt/ceph/ar-dp/"
#PVC_PATH = ""

DOWNSAMPLE = 1
STEPS_BETWEEN_IMAGES = 128
CHANNELS = 48
TRAINING_STEPS = 10000



key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key, index)

nca = gNCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "GRAD", "LAP"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5,
    key=key,
)
nca = nca.load(
    PVC_PATH+f"models/multi_species_stable_gnca_grad_{CHANNELS}ch_cr_mi_av_al_bt_li_mu_ds_1_long.eqx"
)


FULL_HYPERPARAMETERS = {
    "SPARSITY_STRENGTH": [0,0.001,0.01,0.1,1],
    "SPARSITY_PARAM": [1024],
    "HIDDEN_DIM": [8096],
    "SAE_ACTIVATION": ["topk","relu"],
    "TARGET_LAYER": ["linear_hidden", "activation", "linear_output"],
    "LOSS_FUNCTION": ["l2", "cosine"],
    "OPTIMISER_MODE": ["nadam_norm"],
    "BATCHES": [1,2,4],
}

HYPERPARAMETER_LIST = index_to_param_list(index,n_processes=TOTAL_NUM_GPUS,full_hyperparameters=FULL_HYPERPARAMETERS)

for HPAR in HYPERPARAMETER_LIST:
    jax.clear_caches()
    key = jr.fold_in(key, index)
    SPARSITY_STRENGTH = HPAR["SPARSITY_STRENGTH"]
    SPARSITY_PARAM = HPAR["SPARSITY_PARAM"]
    HIDDEN_DIM = HPAR["HIDDEN_DIM"]
    TARGET_LAYER = HPAR["TARGET_LAYER"]
    LOSS_FUNCTION = HPAR["LOSS_FUNCTION"]
    SAE_ACTIVATION = HPAR["SAE_ACTIVATION"]
    OPTIMISER_MODE = HPAR["OPTIMISER_MODE"]
    BATCHES = HPAR["BATCHES"]
    wandb_config = {
                "project": "multi_species_patterning", 
                "name": f"SAE_{TARGET_LAYER}_hd{HIDDEN_DIM}_k{SPARSITY_PARAM}_sparsity{SPARSITY_STRENGTH}_{SAE_ACTIVATION}_nca_gated_grad_{CHANNELS}ch_optimiser_{OPTIMISER_MODE}_{LOSS_FUNCTION}_batches_{BATCHES}",
                "group":"spacetime_independent_sae_hyperparameters_2",
                "tags": ["multi_species", "gated", "grad","SAE basic"],}

    print(f"Training with sparsity strength: {SPARSITY_STRENGTH}, hidden dim: {HIDDEN_DIM}, sparsity param: {SPARSITY_PARAM}")
    FE = NCA_Feature_Extractor_Emoji([nca], BOUNDARY_MASKS=None, GATED=True)
    SAE = SparseAutoencoder(
        TARGET_LAYER=TARGET_LAYER,
        N_CHANNELS=CHANNELS,
        N_KERNELS=4,
        ACTIVATION=SAE_ACTIVATION,
        GATED=True,
        TIED_INIT=False,
        latent_dim=HIDDEN_DIM,
        sparsity_param=SPARSITY_PARAM,
        key=key,
    )
    trainer = NCA_SAE_Trainer(
        FE,
        SAE,
        filename=f"SAE_{TARGET_LAYER}_hd{HIDDEN_DIM}_k{SPARSITY_PARAM}_emoji_multi_species_gated_nca_grad_32ch_cr_mi_av_al_bt_li_mu_ds_1_long",
        model_directory=PVC_PATH+"models/",
        log_directory=PVC_PATH+"logs/",
    )
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
    FE_params = {
        "t0": 0,
        "t1": STEPS_BETWEEN_IMAGES,
        "BATCH_SIZE": BATCHES,
        "SIZE": None,
        "key": jr.fold_in(key, 2),
    }

    #try:
    trainer.train(
        iters=TRAINING_STEPS, 
        Sparsity=SPARSITY_STRENGTH,
        optimiser=optimiser, 
        FE_params=FE_params,
        LOSS=LOSS_FUNCTION,
        MINIBATCH_SIZE=8192,
        REGENERATE_EVERY=512,
        LOG_EVERY=512,
        wandb_config=wandb_config,

        key=jr.fold_in(key, 3))
    # except Exception as e:
    #     print(f"Error: {e}")
    #     continue