from NCA.analysis.NCA_channel_extractor import NCA_channel_extractor
from NCA.analysis.NCA_channel_map import NCA_channel_map,NCA_channel_map_trainer
from Common.utils import load_micropattern_circle_8ch
from NCA.model.NCA_gated_model import gNCA
from Common.model.boundary import model_boundary
import optax
import jax
import jax.random as jr
import equinox as eqx
import time
import jax.numpy as np
import sys

index = int(sys.argv[1])
PVC_PATH = "/mnt/ceph/ar-dp/"
DOWNSAMPLE = 2
BATCHES = 4
CHANNELS = 16
STEPS_BETWEEN_IMAGES = 64
TRAIN_ITERS = 40000



#--- Load trained NCA model ---
NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"circular",
                    "key":jr.PRNGKey(int(time.time()))}
nca = gNCA(**NCA_hyperparameters)
nca_cell_fate = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v3.eqx")

data,boundary_mask = load_micropattern_circle_8ch(DOWNSAMPLE=DOWNSAMPLE,BATCHES=BATCHES,PVC_PATH=PVC_PATH)

data = np.pad(data,((0,0),(0,0),(0,CHANNELS-8),(0,0),(0,0)))
#boundary_func = model_boundary(=boundary_mask[0]) # Importantly model_boundary expects [1 X Y], not [Batches 1 X Y]

#Channel_extractor = NCA_channel_extractor(nca_cell_fate,BOUNDARY_CALLBACK=[boundary_func],GATED=True)
schedule = optax.exponential_decay(
    init_value=1e-4,
    transition_steps=TRAIN_ITERS,
    decay_rate=0.98,
)
optimiser = optax.nadam(schedule)

CM_trainer = NCA_channel_map_trainer(
    nca_cell_fate,
    data=data,
    boundary_mask=boundary_mask,
    FULL_CHANNELS=np.arange(5,CHANNELS),
    TARGET_CHANNELS=[5],
    BATCHES=BATCHES,
    GATED=True)

CM_trainer.train(
    STEPS_BETWEEN_IMAGES=STEPS_BETWEEN_IMAGES,
    ITERS=TRAIN_ITERS,
    optimiser=optimiser,
    FILENAME=f"FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_channel_map_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v3_test_2",
    wandb_config={"project":"micropattern channel map",
                  "group":"Development",
                  "name":f"FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_channel_map_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v3_test_2",
                  "tags":["NCA","channel map","micropattern"]}
)