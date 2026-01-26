import jax
import jax.random as jr
import jax.numpy as np
import numpy as onp
import equinox as eqx
import os
import sys
import time
import optax
from pprint import pprint
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/"
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_noise_model import nNCA
from NCA.model.NCA_gated_noise_model import gnNCA
# from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list

from Experiments.emoji.fire_rate_sweep import H_to_filename as H_to_filename_fr
from Experiments.emoji.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
from Experiments.emoji.parameter_noise_sweep import H_to_filename as H_to_filename_noise
# from NCA.trainer.data_augmenter_nca import DataAugmenter



# BATCHES = 4

def prepare_data(H):
    data = load_emoji_sequence(
        ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
        impath_emojis=DATA_PATH,
        downsample=H["downsample"]
    )
    data = data[0] # T C X Y
    data = np.pad(data,((0,0),(0,H["channels"]-4),(10,10),(10,10)))
    return data

def perturb_pixel(ic,coords,perturbation_mode):
    # data: C X Y
    x,y = coords
    if perturbation_mode=="invert":
        ic_perturbed = ic.at[:,x,y].set(1-ic[:,x,y])
    elif perturbation_mode=="zero":
        ic_perturbed = ic.at[:,x,y].set(0.0)
    elif perturbation_mode=="large":
        ic_perturbed = ic.at[:,x,y].set(10.0)
    
    return ic_perturbed


@eqx.filter_jit
def run_perturbed_nca(nca,ic,H,key):
    x = ic
    for i in range(H["steps_between_images"]*H["timesteps"]):
        key = jr.fold_in(key,i)
        x = nca(x,lambda x:x,key)
    return x

def numpy_image_float_to_int(array):
    array = onp.clip(array,0.0,1.0)
    array = (array * 255.0).astype(onp.uint8)
    return array

def run(H,key):
    # H[""]
    if "fire_rate" in H:
        H["steps_between_images"]=int(32 / H["fire_rate"])
    else:
        if H["channels"]==32:
            H["steps_between_images"]=64
        elif H["channels"]==16:
            H["steps_between_images"]=128

    data = prepare_data(H)  # T C X Y
    ic = data[0]  # C X Y
    _,X,Y = ic.shape
    C = 3
    # C = 1
    # X = 4
    # Y = 4
    perturbation_results = np.zeros((X,Y))
    perturbation_counts = np.zeros((X,Y))
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
    # FILENAME = H_to_filename_fr(H)
    FILENAME = H_to_filename_noise(H)
    nca = nca.load(f"{CODE_PATH}/models/{FILENAME}.eqx")
    baseline_traj = run_perturbed_nca(nca,ic,H,key) # no perturbation yet
    for x in tqdm(range(X)):
        # for c in range(C):
        for y in range(Y):
            ic_perturbed = perturb_pixel(ic,(x,y),H["perturbation_mode"])
            perturbed_traj = run_perturbed_nca(nca,ic_perturbed,H,key)
            if x % 8 ==0 and y % 8 ==0: # Only save full image every 8th pixel to save space
                perturbed_result = onp.array(perturbed_traj[:4]) # save only first 4 channels (RGB + alpha)
                perturbed_result = numpy_image_float_to_int(perturbed_result) # Reduce space required to save and transfer later
                onp.save(f"perturbations/emoji_local/{FILENAME}_coords_all_{x}_{y}_{H['perturbation_mode']}_T{H['timesteps']}.npy",perturbed_result)
            
            diff = np.abs(perturbed_traj - baseline_traj).mean()
            num_diff = np.count_nonzero(np.abs(perturbed_traj - baseline_traj) > 0.1)
            perturbation_results = perturbation_results.at[x,y].set(diff)
            perturbation_counts = perturbation_counts.at[x,y].set(num_diff)
            print(f"Perturbed pixel ({x},{y}), mean abs diff: {diff:.6f}")
    perturbation_results = onp.array(perturbation_results)
    onp.save(f"perturbations/emoji_local/{FILENAME}_full_{H['perturbation_mode']}_T{H['timesteps']}.npy",perturbation_results)
    onp.save(f"perturbations/emoji_local/{FILENAME}_counts_{H['perturbation_mode']}_T{H['timesteps']}.npy",onp.array(perturbation_counts))
    # vrun  = jax.vmap(run_perturbed_nca, in_axes=(None,0,None,None))
    # for 

def main():
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])
    key = jr.PRNGKey(int(time.time()))
    key = jr.fold_in(key,index)
    HYPERPARAMETERS = {
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        # "channels":[32,16],
        "model":["nNCA"],
        "channels":[32],
        "downsample":[1],
        # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],#,0.01,0.1,1.0],
        "timesteps":[3],
        "perturbation_mode":["large"],
        "regenerate":[False],
        # "fire_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        # "fire_rate":[0.5],
        "parameter_noise_level":[0.0,0.001,0.005,0.01,0.05,0.1],
        "data_noise_level":[0.001,0.005,0.01,0.05,0.1,0.5],
    }
    HPARAMS = index_to_param_list(index,TOTAL_JOBS,HYPERPARAMETERS)


    for H in HPARAMS:
        print("---------------------------------------------------")
        print(f"RUNNING WITH HYPERPARAMS:")
        pprint(H)
        
        key = jr.fold_in(key,index)
        run(H,key)
if __name__ == "__main__":
    main()