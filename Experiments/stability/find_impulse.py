import os
from dotenv import load_dotenv
load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")
DATA_PATH_BASE = os.getenv("DATA_PATH_BASE")
import sys

from time import time
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)

from NCA.trainer.NCA_impulse_optimiser import NCA_impulse_optimiser
from Common.dataloader.emoji import load_emoji_sequence
from einops import rearrange
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Experiments.multispecies.train_good import prepare_data
from Common.utils import index_to_param_list
import jax
import jax.numpy as np
import jax.random as jr
import optax
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from einops import repeat


class DA_pad(DataAugmenterAbstract):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        P = 10
        data = self.pad(data, [P, P, P, P])
        self.save_data(data)
        return None

#-------------------- TESTING CODE --------------------

def load_data_and_model(REG_MODE,DATA_AUGMENTER="regrowth",DATA_MODE="patch",DATA_TYPE=["crab","microbe"],BATCHES=1,NCA_MODEL=NCA):
    # data,_ = prepare_data(BATCHES=1,mode="patch",species=["crab","microbe"])
    # print("Prepared data shape: "+str(data.shape))

    CHANNELS = 32
    OBS_CHANNELS = 4
    key = jr.PRNGKey(0)
    # nca = gNCA(
    #     N_CHANNELS=CHANNELS,
    #     KERNEL_STR = ["ID","LAP","GRAD"],
    #     ACTIVATION=jax.nn.relu,
    #     PADDING="CIRCULAR",
    #     FIRE_RATE=0.5,
    #     key=key,)

    # nca = nca.load("models/multi_species_stable_gNCA_grad_64ch_contiguous_1.0_wide_perturbation_0.1_cr_mi_ds_1_long.eqx")

    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","GRAD"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }

    REG_STR = ["","_intermediate_reg","_intermediate_contiguous_reg"][REG_MODE]
    data, data_filename = prepare_data(BATCHES,DATA_MODE,species=DATA_TYPE)
    nca = NCA_MODEL(**NCA_hyperparameters)
    name = f"good_emoji_multi_species_{data_filename}_{nca.get_config()['MODEL']}{REG_STR}_32ch_t128_{DATA_AUGMENTER}.eqx"
    nca = nca.load(f"models/{name}")
    return nca,data

def run_training(H,key):
    DATA_AUGMENTER = H["data_augmenter"]
    CH_MODE = H["channel_mode"]
    REG_MODE = H["nca_reg_mode"]
    SP_MODE = H["spatial_mode"]
    P_WIDTH = H["spatial_width"]
    INIT_MODE = H["init_mode"]
    DX_L2 = H["dx_l2"]
    DX_L1 = H["dx_l1"]
    DX_MAX = H["dx_max"]
    DX_IN_0_1 = H["dx_in_0_1"]
    STEPS_FROM_STABLE = H["steps_from_stable"]
    
    REG_STR = ["","_intermediate_reg","_intermediate_contiguous_reg"][REG_MODE]
    NAME = f"nca_signal_impulse_NCA{REG_STR}_32ch_{DATA_AUGMENTER}_{INIT_MODE}_dx_{CH_MODE}_{SP_MODE}_l2{DX_L2}_l1{DX_L1}_max{DX_MAX}_in_0_1{DX_IN_0_1}_w{P_WIDTH}_steps{STEPS_FROM_STABLE}"
    nca,data = load_data_and_model(
        REG_MODE=REG_MODE,
        DATA_AUGMENTER=DATA_AUGMENTER,
        DATA_MODE=INIT_MODE,
        DATA_TYPE = ["crab","microbe"],
        BATCHES=1,
        NCA_MODEL=gNCA,
    )
    ITERATIONS = 100
    schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=ITERATIONS,
        decay_rate=0.99,
    )
    opt = NCA_impulse_optimiser(
        NCA_model = nca,
        data = data,
        DATA_AUGMENTER=DA_pad,
        STEPS_TO_STABLE=[128,128+128],
        STEPS_FROM_STABLE=[STEPS_FROM_STABLE,STEPS_FROM_STABLE+32],
        FILENAME=NAME,
        MODEL_DIRECTORY= "models/",
        LOG_DIRECTORY= "logs/",
        OUTPUT_DIRECTORY= "perturbations/",
        BOUNDARY_MASK = None,
        BOUNDARY_MODE = "soft",
        OBS_CHANNELS = 4, 
        wandb_args = {
            "name":NAME,
            "project":"NCA_impulse_optimiser",
            "group":"long_trajectory_regulariser_sweep_save_test",
            "tags":[f"{k}:{v}" for k,v in H.items()]
        }
    )
    opt.train(
        iters=ITERATIONS, 
        optimiser=optax.adam(schedule), 
        log_interval=100,
        perturbation_mode = {"channel":CH_MODE,"spatial":SP_MODE},
        perturbation_width=P_WIDTH,
        perturbation_reg_coeff = {
            "l2":DX_L2,
            "l1":DX_L1,
            "max":DX_MAX,
            "in_0_1":DX_IN_0_1},
        # perturbation_location=[32,32],
        key=key,
        RESAMPLE_EVERY=500,
        )
# run_training({},jr.PRNGKey(index))


def main():
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])
    key = jr.PRNGKey(int(time()))
    key = jr.fold_in(key,index)

    HPARAMS = {
        # "data_augmenter":["regrowth","standard"],
        "data_augmenter":["regrowth"],
        "channel_mode":["hidden"],
        "spatial_mode":["local"],#,"pixel","patch","flat"],
        "spatial_width":[0.2],
        "nca_reg_mode":[2],
        "init_mode":["patch"],
        "dx_l2":[0.0],
        "dx_l1":[0.0],
        "dx_max":[0.0],
        "dx_in_0_1":[0.0],
        "steps_from_stable":[256],
    }
    Hlist = index_to_param_list(index,TOTAL_JOBS,HPARAMS)
    for H in Hlist:
        jax.clear_caches() # Weird OOM errors
        key = jr.fold_in(key,1)

        print("Running with hyperparameters: "+str(H))
        # try:
        run_training(H,key)
        # except Exception as e:
            # print(f"Error occurred with hyperparameters {H}: {e}")

if __name__ == "__main__":
    main()