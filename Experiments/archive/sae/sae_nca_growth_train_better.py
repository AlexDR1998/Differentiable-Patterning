from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Emoji
from NCA.analysis.NCA_SAE_trainer_better import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
from Common.utils import index_to_param_list
from NCA.analysis.optimiser import unitary_decoder_transform
import jax
import jax.random as jr
import time
import optax
import sys

index = int(sys.argv[1])
TOTAL_NUM_GPUS=7
PVC_PATH = "/mnt/ceph/ar-dp/"
CHANNELS = 48
STEPS_BETWEEN_IMAGES = 128
TRAINING_STEPS = 3000


FULL_HYPERPARAMETERS = {
    "PATH_LENGTH": [16],
    "SPARSITY_PARAM": [1024],
    "HIDDEN_DIM": [8094],
    "SPARSITY_STRENGTH": [0],
    "MINIBATCH_SIZE": [1,2,4],
    "OPTIMISER_MODE": ["nadam_norm"],
    "NORMALISE_MODE": ["none","stepwise","pathwise"],
    "LOSS_CHANNELS": ["obs"],
    "SAE_ACTIVATION": ["topk","relu"],
    "PATH_LOSS_SAMPLING":["uniform","endpoint"],#["endpoint","uniform","geometric_forward"],
    "TARGET_LAYER": ["linear_hidden","activation","linear_output"],
    "LOSS_FUNCTION": ["l2","spectral","cosine"],
    "PROPAGATE_SAE_PATHS": [0],
}

HYPERPARAMETER_LIST = index_to_param_list(index,n_processes=TOTAL_NUM_GPUS,full_hyperparameters=FULL_HYPERPARAMETERS)


#---- Load NCA and initial condition ----#
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
    PATH_LENGTH = HPAR["PATH_LENGTH"]
    SPARSITY_PARAM = HPAR["SPARSITY_PARAM"]
    HIDDEN_DIM = HPAR["HIDDEN_DIM"]
    SPARSITY_STRENGTH = HPAR["SPARSITY_STRENGTH"]
    NORMALISE_MODE = HPAR["NORMALISE_MODE"]
    TARGET_LAYER = HPAR["TARGET_LAYER"]
    LOSS_FUNCTION = HPAR["LOSS_FUNCTION"]
    LOSS_CHANNELS = HPAR["LOSS_CHANNELS"]
    PROPAGATE_SAE_PATHS = HPAR["PROPAGATE_SAE_PATHS"]
    PATH_LOSS_SAMPLING = HPAR["PATH_LOSS_SAMPLING"]
    OPTIMISER_MODE = HPAR["OPTIMISER_MODE"]
    SAE_ACTIVATION = HPAR["SAE_ACTIVATION"]
    MINIBATCH_SIZE = HPAR["MINIBATCH_SIZE"]
    FILENAME = f"SAE_{TARGET_LAYER}_k{SPARSITY_PARAM}_hd{HIDDEN_DIM}_sp{SPARSITY_STRENGTH}_act_{SAE_ACTIVATION}_norm_{NORMALISE_MODE}_pl{PATH_LENGTH}_{PATH_LOSS_SAMPLING}_pathprop{PROPAGATE_SAE_PATHS}_emoji_multi_species_gated_nca_grad_{CHANNELS}ch_incremental_reset_mb{MINIBATCH_SIZE}_opt_{OPTIMISER_MODE}_loss_{LOSS_FUNCTION}_{LOSS_CHANNELS}"
    wandb_config = {"project":"multi_species_patterning",
                    "name":FILENAME,
                    "group":"path length SAE hyperparameter sweep - no sparsity - reduced batches",
                    "tags":[
                        "multi_species",
                        "gated",
                        "grad",
                        "SAE path",
                        OPTIMISER_MODE,
                        NORMALISE_MODE,
                        f"path length {PATH_LENGTH}",
                        f"propagating sae path {PROPAGATE_SAE_PATHS}",
                        f"hidden dim {HIDDEN_DIM}",
                        f"sparsity param {SPARSITY_PARAM}"
                        ]}
    
    FE = NCA_Feature_Extractor_Emoji([nca], BOUNDARY_MASKS=None, GATED=True)
    X0_true = FE.initial_condition(BATCH_SIZE = MINIBATCH_SIZE)
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
        SAE,
        filename=FILENAME,
        model_directory=PVC_PATH+"models/",
        log_directory=PVC_PATH+"logs/",
    )


    #--- Do Training ----#
    try:
        trainer.train(
            NCA=nca,
            X0=X0_true,
            ITERS=TRAINING_STEPS,
            SPARSITY=SPARSITY_STRENGTH,
            optimiser=optimiser,
            MINIBATCH_SIZE=MINIBATCH_SIZE,
            CLEAR_CACHE_EVERY=4000,
            NCA_TIMESTEPS=STEPS_BETWEEN_IMAGES*2,
            PROPAGATE_SAE_PATHS=PROPAGATE_SAE_PATHS, # Must be less than minibatch size
            PATH_LENGTH=PATH_LENGTH,
            PATH_LOSS_SAMPLING=PATH_LOSS_SAMPLING,
            RESAMPLE_X0_EVERY=1,
            NORMALISE_MODE=NORMALISE_MODE, # "none", "stepwise", "pathwise"
            LOG_EVERY=500,
            LOOP_AUTODIFF="checkpoint",
            wandb_config=wandb_config,
            )
    except Exception as e:
        print(f"Error with hyperparameters {HPAR}: {e}")