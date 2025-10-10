from NCA.model.NCA_gated_model import gNCA
from NCA.analysis.NCA_feature_extractor import NCA_Feature_Extractor_Texture
from NCA.analysis.NCA_SAE_trainer import NCA_SAE_Trainer
from NCA.analysis.NCA_SAE_class import SparseAutoencoder
from tqdm import tqdm
import jax.numpy as jnp
import jax
import jax.random as jr
import time
import optax

PVC_PATH = "/mnt/ceph_rbd/ar-dp/"


DOWNSAMPLE = 1
STEPS_BETWEEN_IMAGES = 96
CHANNELS = 16
SIZE = 128
BATCHES = 1
MODEL_BATCH_SIZE = 8
MINIBATCH_SIZE = 4096
REGENERATE_EVERY = 256
HIDDEN_DIM = 8192
#TOP_K_VALUE = 64
ACTIVATION = "topk"
ITERS = 10000
key = jr.PRNGKey(int(time.time()))

NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","GRAD"],
                    "FIRE_RATE":0.5,
                    "PADDING":"REPLICATE",
                    "key":jr.PRNGKey(int(time.time()))}
nca_base = gNCA(**NCA_hyperparameters)


texture_files = [
    "dotted/dotted_0109.jpg",
    "dotted/dotted_0106.jpg",
    #"dotted/dotted_0188.jpg",
    "crystalline/crystalline_0204.jpg",
    #"bumpy/bumpy_0059.jpg",
    #"bumpy/bumpy_0163.jpg",
    #"bumpy/bumpy_0107.jpg",
    "meshed/meshed_0098.jpg",
    "paisley/paisley_0050.jpg",
    "paisley/paisley_0122.jpg",
    "stained/stained_0061.jpg",
    "marbled/marbled_0150.jpg",
    #"marbled/marbled_0060.jpg",
    "cracked/cracked_0004.jpg",
    "cracked/cracked_0064.jpg",
    #"freckled/freckled_0142.jpg",
    #"checkered/checkered_0017.jpg",
    "interlaced/interlaced_0045.jpg",
    #"veined/veined_0093.jpg",
    #"freckled/freckled_0095.jpg",
    #"stratified/stratified_0067.jpg",
    "stratified/stratified_0148.jpg",
    #"lined/lined_0170.jpg",
    #"smeared/smeared_0096.jpg",
    "smeared/smeared_0129.jpg",
    #"smeared/smeared_0139.jpg",
    #"blotchy/blotchy_0060.jpg",
    "perforated/perforated_0106.jpg",
    "striped/striped_0011.jpg",
    "striped/striped_0083.jpg",
    "striped/striped_0099.jpg",
    #"honeycombed/honeycombed_0078.jpg",
    "scaly/scaly_0136.jpg",
    #"bubbly/bubbly_0101.jpg",
    "banded/banded_0041.jpg",
    "grid/grid_0002.jpg",
    "braided/braided_0107.jpg",    
]


ncas = []
for texture in texture_files:
    try:
        filename = f"models/texture_gated_nca_grad_{texture.split('/')[-1].split('.')[0]}_run1_multiscale_vgg_16.eqx"
        ncas.append(nca_base.load(PVC_PATH+filename))
    except Exception as e:
        print(f"Failed to load NCA for texture {texture}")
        print(e)
        continue

#HIDDEN_DIM = 8192
#for TOP_K_VALUE in [64,]:
TOP_K_VALUE = 64
for ACTIVATION in ["topk","relu"]:
    for TARGET_LAYER in ["perception","linear_hidden","activation","linear_output","gate_func"]:
        try:
            jax.clear_caches()
            print(f"Training SAE on target layer {TARGET_LAYER} with hidden dim {HIDDEN_DIM} and top k value {TOP_K_VALUE}")
            key = jr.fold_in(key,1)
            filename = f"models/texture_gated_nca_grad_{texture.split('/')[-1].split('.')[0]}_run1_multiscale_vgg_16.eqx"
            
            
            FE = NCA_Feature_Extractor_Texture(ncas,
                                            BOUNDARY_MASKS=None,
                                            GATED=True,
                                            MODEL_BATCH_SIZE=MODEL_BATCH_SIZE)
            SAE = SparseAutoencoder(TARGET_LAYER=TARGET_LAYER,
                                    N_CHANNELS=CHANNELS,
                                    N_KERNELS=4,
                                    ACTIVATION=ACTIVATION,
                                    GATED=True,
                                    hidden_dim=HIDDEN_DIM,
                                    sparsity_param=TOP_K_VALUE,
                                    key=key)
            trainer = NCA_SAE_Trainer(FE,
                                    SAE,
                                    1.0,
                                    filename=f"SAE_{TARGET_LAYER}_dim_{HIDDEN_DIM}_act_{ACTIVATION}_k_{TOP_K_VALUE}_model_batch_{MODEL_BATCH_SIZE}_texture_gated_nca_grad_multitexture_run1_multiscale_vgg_16",
                                    model_directory=PVC_PATH+"models/",
                                    log_directory=PVC_PATH+"logs/")
            optimiser = optax.nadam(optax.exponential_decay(5e-4, transition_steps=ITERS, decay_rate=0.98))
            FE_params={"t0":0,
                    "t1":STEPS_BETWEEN_IMAGES,
                    "BATCH_SIZE":BATCHES,
                    "SIZE":SIZE,
                    "key":jr.fold_in(key,2)}
            
            SAE_trained = trainer.train(ITERS,
                                optimiser,
                                FE_params=FE_params,
                                MINIBATCH_SIZE=MINIBATCH_SIZE,
                                REGENERATE_EVERY=REGENERATE_EVERY,
                                key=jr.fold_in(key,3))
            
            del FE
            del SAE
            del trainer
        
        except Exception as e:
            print(f"Failed to train SAE at layer {TARGET_LAYER} with hidden dim {HIDDEN_DIM} and top k value {TOP_K_VALUE}")
            print(e)
            