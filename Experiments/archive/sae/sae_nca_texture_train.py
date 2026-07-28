from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Texture
from NCA.analysis.NCA_SAE_trainer import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
import jax
import jax.random as jr
import time
import optax

PVC_PATH = "/mnt/ceph_rbd/ar-dp/"


DOWNSAMPLE = 1
STEPS_BETWEEN_IMAGES = 64
CHANNELS = 16
SIZE = 128
BATCHES = 1
ITERS = 1000
key = jr.PRNGKey(int(time.time()))

NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"REPLICATE",
                    "key":jr.PRNGKey(int(time.time()))}
nca_base = gNCA(**NCA_hyperparameters)



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


for texture in texture_files:
    for TARGET_LAYER in ["perception","linear_hidden","activation","linear_output","gate_func"]:
        key = jr.fold_in(key,1)
        filename = f"models/texture_gated_nca_grad_{texture.split('/')[-1].split('.')[0]}_run1_multiscale_vgg_16"
        
        nca = nca_base.load(PVC_PATH+filename)
        FE = NCA_Feature_Extractor_Texture([nca],BOUNDARY_MASKS=None,GATED=True)
        SAE = SparseAutoencoder(TARGET_LAYER=TARGET_LAYER,
                                N_CHANNELS=CHANNELS,
                                N_KERNELS=3,
                                ACTIVATION="topk",
                                GATED=True,
                                hidden_dim=1024,
                                sparsity_param=50,
                                key=key)
        trainer = NCA_SAE_Trainer(FE,
                                  SAE,
                                  1.0,
                                  filename=f"models/SAE_{TARGET_LAYER}_texture_gated_nca_grad_{texture.split('/')[-1].split('.')[0]}_run1_multiscale_vgg_16",
                                  file_path=PVC_PATH)
        optimiser = optax.nadam(optax.exponential_decay(1e-3, transition_steps=ITERS, decay_rate=0.99))
        FE_params={"t0":0,
                   "t1":STEPS_BETWEEN_IMAGES,
                   "BATCH_SIZE":BATCHES,
                   "SIZE":SIZE,
                   "key":jr.fold_in(key,2)}
        
        _, _ = trainer.train(ITERS,
                            optimiser,
                            FE_params=FE_params,
                            key=jr.fold_in(key,3))