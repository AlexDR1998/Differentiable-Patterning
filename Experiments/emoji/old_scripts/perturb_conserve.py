import os
import sys
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/"
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)
import time
from NCA.trainer.NCA_impulse_conserve_optimiser import NCA_impulse_optimiser

from einops import rearrange
from Common.utils import index_to_param_list
import jax
import jax.numpy as np
import jax.random as jr
import optax
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_noise_model import nNCA
from NCA.model.NCA_gated_noise_model import gnNCA
from einops import repeat
from Experiments.emoji.old_scripts.local_perturbation import prepare_data
from Experiments.emoji.old_scripts.fire_rate_sweep import H_to_filename as H_to_filename_fr
from Experiments.emoji.old_scripts.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
from Experiments.emoji.old_scripts.parameter_noise_sweep import H_to_filename as H_to_filename_noise

def run(H,key):
    if "fire_rate" in H:
        H["steps_between_images"]=int(32 / H["fire_rate"])
    else:
        if H["channels"]==32:
            H["steps_between_images"]=64
        elif H["channels"]==16:
            H["steps_between_images"]=128
    data = prepare_data(H)
    if H["model"] == "NCA":
         model = NCA
    elif H["model"] == "gNCA":
         model = gNCA
    elif H["model"] == "nNCA":
         model = nNCA
    elif H["model"] == "gnNCA":
         model = gnNCA
    if H["model"] in ["nNCA","gnNCA"]:
        print(f"Using parameter noise level: {H['parameter_noise_level']}")
        nca = model(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=H['fire_rate'] if "fire_rate" in H else 0.5,
            PADDING="REPLICATE",
            PARAMETER_NOISE_LEVEL=H["parameter_noise_level"],
            key=key
        )
    else:
        nca = model(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=H['fire_rate'] if "fire_rate" in H else 0.5,
            PADDING="REPLICATE",
            key=key
        )
    # if H["regenerate"]:
    #     regen_str = "regenerate_"
    # else:
    #     regen_str = ""
    # FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_{H['model']}_{regen_str}ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}"
    FILENAME = H_to_filename_fr(H)
    # FILENAME = H_to_filename_noise(H)
    nca = nca.load(f"{CODE_PATH}/models/{FILENAME}.eqx")
    # data_augmenter = DA_pad(data,H["channels"]-4)
    optimiser = optax.adam(1e-3)

    impulse_trainer = NCA_impulse_optimiser(
        NCA_model=nca,
        data=data,
        FILENAME=f"{FILENAME}_impulse_global_{H['optimisation_mode']}_iters{H['perturb_iters']}_T{H['timesteps']}",
        OBS_CHANNELS = 4,
        OUTPUT_DIRECTORY = CODE_PATH+"/perturbations/emoji_global/",
        wandb_args={
            "project":"nca-emojis-thesis-ch1",
            "group":"preserving-perturbation-6",
            # "group":"baseline-9ch-train-1",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
    )

    impulse_trainer.train(
        iters=H["perturb_iters"],
        optimiser=optimiser,
        BATCHES=8,
        STEPS_TO_TARGET=H["steps_between_images"]*H["timesteps"],
        perturbation_mode=H["perturbation_mode"],
        optimisation_mode=H["optimisation_mode"],
        perturbation_reg_coeff={
            "l2":1.0,
            "l1":0.0,
            "max":0.0,
            "in_0_1":0.0,
        },
        log_interval = 100,
        LOSS_FUNC_STR = "l2",
        RESAMPLE_EVERY = 100,
        key=key,
    )
def main():
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])
    key = jr.PRNGKey(int(time.time()))
    key = jr.fold_in(key,index)
    # HYPERPARAMETERS = {
    #     # These specify the NCA model to load
    #     "loss_mode":["l2"],
    #     "model":["NCA","gNCA"],
    #     "channels":[32,16],
    #     "downsample":[1],
    #         # "steps_between_images":[64,128],
    #     "iters":[8000],
    #     "intermediate_growth_coeff":[0.0],
    #     "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
    #     "contiguous_growth_coeff":[0.0],
    #     "perturbation_conservation_coeff":[0.0],
    #     "update_sensitivity_coeff":[0.0],
    #     "regenerate":[False,True],
    #     # From here on are specific to perturbation optimiser
    #     "timesteps":[3],
    #     "perturbation_mode":[{"channel":"obs","spatial":"global"},],
    #     "optimisation_mode":["maximal_preservative","minimal_destructive"],
    #     "perturb_iters":[5,10,50,100,500]
    # }
    HYPERPARAMETERS = {
        # These specify the NCA model to load
        "loss_mode":["l2"],
        "model":["NCA"],
        "channels":[16],
        "downsample":[1],
            # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        # "regenerate":[False,True],
        "fire_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        "regenerate":[True],
        # "parameter_noise_level":[0.0,0.001,0.005,0.01,0.05,0.1],
        # "data_noise_level":[0.001,0.005,0.01,0.05,0.1,0.5],
        # From here on are specific to perturbation optimiser
        "timesteps":[3],
        "perturbation_mode":[{"channel":"obs","spatial":"global"},],
        "optimisation_mode":["maximal_preservative","minimal_destructive"],
        "perturb_iters":[100,500]
    }
    HPARAMS = index_to_param_list(index,TOTAL_JOBS,HYPERPARAMETERS)


    for H in HPARAMS:
        print("---------------------------------------------------")
        print(f"RUNNING WITH HYPERPARAMS:")
        pprint(H)
        
        key = jr.fold_in(key,index)
        # try:
        run(H,key)
        # except Exception as e:
            # print(f"Error during run with hyperparameters {H}: {e}")
if __name__ == "__main__":
    main()