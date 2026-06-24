from NCA.analysis.NCA_channel_extractor import NCA_channel_extractor
from NCA.analysis.NCA_channel_map import NCA_channel_map_fully_connected_local, NCA_channel_map_conv, NCA_channel_map_linear
from NCA.analysis.NCA_channel_map_trainer import NCA_channel_map_trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from Common.model.boundary import model_boundary
from Common.utils import index_to_param_list#,#print_hparams_per_process
import optax
import jax
import jax.random as jr
import equinox as eqx
import time
import jax.numpy as np
import sys
from pprint import pprint

index = int(sys.argv[1])
key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)
FULL_HYPERPARAMETERS = {
    "CM_MODEL":[("conv",NCA_channel_map_conv),
                ("linear",NCA_channel_map_linear),
                ("fully_connected_local",NCA_channel_map_fully_connected_local)],
    "TARGET_CHANNEL":[0,1,2,3,4,5,6,7],
}
HPARAMS = index_to_param_list(index,6,FULL_HYPERPARAMETERS)
#print_hparams_per_process(index,HPARAMS)

PVC_PATH = "/mnt/ceph/ar-dp/"
DOWNSAMPLE = 4
BATCHES = 1
CHANNELS = 32
STEPS_BETWEEN_IMAGES = 64
TRAIN_ITERS = 40000


#--- Load trained NCA model ---
NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"CIRCULAR",
                    "key":jr.PRNGKey(int(time.time()))}

nca = NCA(**NCA_hyperparameters)
# nca_cell_fate = nca.load(PVC_PATH+f"models/SFTLCLNL_average_NCA_boundary_regulariser_steps_between_images_12h_start_64_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v6.eqx")
nca_cell_fate = nca.load(PVC_PATH+f"models/SFTLCLNL_average_NCA_boundary_regulariser_steps_between_images_12h_start_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v6.eqx")
data,boundary_mask,channel_names,aux = load_micropattern_circle_8ch(
    DOWNSAMPLE=DOWNSAMPLE,
    BATCHES=BATCHES,
    PVC_PATH=PVC_PATH,
    TIMESTEPS=[12,24,36,48,60],
    HIST_EQS={
            "sftl":(0.5,99.95),
            "dcln":(0.5,99.95),
            "lls":(0.5,99.95)},
    SHOW_HISTOGRAMS=False,
    PROCESSING_MODES=["hist_eq","batch_average","map_to_0_1"]  
)
data = data[:,:4]

data = np.pad(data,((0,0),(0,0),(0,CHANNELS-8),(0,0),(0,0)))
#boundary_func = model_boundary(=boundary_mask[0]) # Importantly model_boundary expects [1 X Y], not [Batches 1 X Y]
#Channel_extractor = NCA_channel_extractor(nca_cell_fate,BOUNDARY_CALLBACK=[boundary_func],GATED=True)
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=TRAIN_ITERS,
    decay_rate=0.98,
)
optimiser = optax.nadam(schedule)
for H in HPARAMS:
    TARGET_CHANNEL = H["TARGET_CHANNEL"]
    MAP_MODE = H["CM_MODEL"][0]
    CM_MODEL = H["CM_MODEL"][1]
    #try:
    NAME = f"SFTLCLNL_average_NCA_channel_map_12h_start_steps_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_model_cell_fate_ds_{DOWNSAMPLE}_v6_target_channel_{TARGET_CHANNEL}_{MAP_MODE}"
    CM_trainer = NCA_channel_map_trainer(
        nca_cell_fate,
        data=data,
        boundary_mask=boundary_mask,
        #MEASURED_CHANNELS=[8,9,10,11,12,13,14,15],
        MEASURED_CHANNELS=[n for n in range(0,CHANNELS) if n != TARGET_CHANNEL],
        TARGET_CHANNELS=[TARGET_CHANNEL],
        BATCHES=BATCHES,
        CM_MODEL=CM_MODEL,
        GATED=True)

    CM_trainer.train(
        STEPS_BETWEEN_IMAGES=STEPS_BETWEEN_IMAGES,
        ITERS=TRAIN_ITERS,
        LOG_EVERY=1000,
        optimiser=optimiser,
        FILENAME=NAME,
        wandb_config={"project":"micropattern channel map",
                    "group":"Channel map target channel mult sweep 1",
                    "name":NAME,
                    "tags":["NCA","channel map","micropattern"]}
    )
    #except Exception as e:
    #    print(f"Error in target channel {TARGET_CHANNEL}: {e}")