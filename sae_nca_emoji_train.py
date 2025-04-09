from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Emoji
from NCA.analysis.NCA_SAE_trainer import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
import jax
import jax.random as jr
import time
import optax


# PVC_PATH = "/mnt/ceph_rbd/ar-dp/"
PVC_PATH = ""

DOWNSAMPLE = 1
BATCHES = 2
STEPS_BETWEEN_IMAGES = 128
CHANNELS = 48
ITERS = 1000

key = jr.PRNGKey(int(time.time()))

nca = gNCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "GRAD", "LAP"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5,
    key=key,
)
nca = nca.load(
    "models/eidf_runs/multi_species_stable_gnca_grad_48ch_cr_mi_av_al_bt_li_mu_ds_1_long.eqx"
)

for TARGET_LAYER in [
    "perception",
    "linear_hidden",
    "activation",
    "linear_output",
    "gate_func",
]:
    key = jr.fold_in(key, 1)
    # filename = f"models/texture_gated_nca_grad_{texture.split('/')[-1].split('.')[0]}_run1_multiscale_vgg_16"

    # nca = nca_base.load(PVC_PATH+filename)
    FE = NCA_Feature_Extractor_Emoji([nca], BOUNDARY_MASKS=None, GATED=True)
    SAE = SparseAutoencoder(
        TARGET_LAYER=TARGET_LAYER,
        N_CHANNELS=CHANNELS,
        N_KERNELS=3,
        ACTIVATION="topk",
        GATED=True,
        hidden_dim=1024,
        sparsity_param=50,
        key=key,
    )
    trainer = NCA_SAE_Trainer(
        FE,
        SAE,
        1.0,
        filename=f"SAE_{TARGET_LAYER}_emoji_multi_species_gated_nca_grad_48ch_cr_mi_av_al_bt_li_mu_ds_1_long",
        model_directory=PVC_PATH+"models/",
        log_directory=PVC_PATH+"logs/",
    )
    optimiser = optax.nadam(
        optax.exponential_decay(1e-3, transition_steps=ITERS, decay_rate=0.99)
    )
    FE_params = {
        "t0": 0,
        "t1": STEPS_BETWEEN_IMAGES,
        "BATCH_SIZE": BATCHES,
        "SIZE": None,
        "key": jr.fold_in(key, 2),
    }

    _, _ = trainer.train(ITERS, optimiser, FE_params=FE_params, key=jr.fold_in(key, 3))
