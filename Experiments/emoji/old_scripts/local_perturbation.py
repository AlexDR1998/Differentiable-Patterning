import jax
import jax.random as jr
import jax.numpy as np
import numpy as onp
import equinox as eqx
from einops import rearrange
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

print("Running in directory:", os.getcwd())
print("Code path:", CODE_PATH)
print("Data path:", DATA_PATH)

import matplotlib.pyplot as plt
from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_noise_model import nNCA
from NCA.model.NCA_gated_noise_model import gnNCA
# from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list
from Common.save_to_video import save_to_video_rgb

from Experiments.emoji.old_scripts.fire_rate_sweep import H_to_filename as H_to_filename_fr
from Experiments.emoji.old_scripts.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
from Experiments.emoji.old_scripts.parameter_noise_sweep import H_to_filename as H_to_filename_noise
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
        
    def nca_step(carry,j):
        key,x = carry
        key = jr.fold_in(key,j)
        # keys = jr.split(key,len(x))
        x = nca(x,lambda x:x,key)
        return (key,x),x
    (_,x),_ = eqx.internal.scan(nca_step,(key,ic),xs=np.arange(H["steps_between_images"]*H["timesteps"]),kind="lax")
    return x

@eqx.filter_jit
def run_perturbed_nca_batch(v_nca,ic,H,key):
    """
        Runs vmapped NCA on a batch of initial conditions. Also uses scan rather than loop for jit efficiency.
        ic: B C X Y
        returns: B C X Y
    """
    
    def nca_step(carry,j):
        key,x = carry
        key = jr.fold_in(key,j)
        keys = jr.split(key,len(x))
        x = v_nca(x,lambda x:x,keys)
        return (key,x),x
    (_,x),_ = eqx.internal.scan(nca_step,(key,ic),xs=np.arange(H["steps_between_images"]*H["timesteps"]),kind="lax")
    return x


def run_perturbed_nca_batch_full(v_nca,ic,H,key):
    """
        Like run_perturbed_nca_batch but returns the full trajectory rather than just the final state. 
        ic: B C X Y
        returns: T B C X Y
    """
    
    def nca_step(carry,j):
        key,x = carry
        key = jr.fold_in(key,j)
        keys = jr.split(key,len(x))
        x = v_nca(x,lambda x:x,keys)
        return (key,x),x[:,:3]
    (_,x),xs = eqx.internal.scan(nca_step,(key,ic),xs=np.arange(H["steps_between_images"]*H["timesteps"]),kind="lax")
    return xs

def numpy_image_float_to_int(array):
    array = onp.clip(array,0.0,1.0)
    array = (array * 255.0).astype(onp.uint8)
    return array

def run(
    H,
    FULL_RUN,
    key,
    NAME_FUNC=H_to_filename_noise,
    video_aux={"duration":20,"scale_up":4,"output_path":"Videos/ThesisEmojis/"}):
    # H[""]
    # if "fire_rate" in H:
        # H["steps_between_images"]=int(32 / H["fire_rate"])
    # else:
    if H["channels"]==32:
        if "fire_rate" in H:
            H["steps_between_images"]=int(32 / H["fire_rate"])
        else:
            H["steps_between_images"]=64
    elif H["channels"]==16:
        if "fire_rate" in H:
            H["steps_between_images"]=int(64 / H["fire_rate"])
        else:
            H["steps_between_images"]=128

    data = prepare_data(H)  # T C X Y
    # ic = data[0]  # C X Y
    # _,X,Y = ic.shape
    # C = 3
    # # C = 1
    # # X = 4
    # # Y = 4
    # perturbation_results = np.zeros((X,Y))
    # perturbation_counts = np.zeros((X,Y))
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
    
    FILENAME = NAME_FUNC(H)
    # FILENAME = H_to_filename_noise(H)
    nca = nca.load(f"{CODE_PATH}/models/thesis_ch1/emoji/{FILENAME}.eqx")
    
    if FULL_RUN:
        perturbation_results, perturbation_counts = iterate_over_perturbations(nca,data,H,key,FILENAME)
        perturbation_results = onp.array(perturbation_results)
        onp.save(f"perturbations/emoji_local/{FILENAME}_full_{H['perturbation_mode']}_T{H['timesteps']}.npy",perturbation_results)
        onp.save(f"perturbations/emoji_local/{FILENAME}_counts_{H['perturbation_mode']}_T{H['timesteps']}.npy",onp.array(perturbation_counts))
    else:
        coordinates = [(10*4,28*4),(30*4,14*4),(18*4,18*4)]
        save_videos_of_perturbations(nca,data,H,key,FILENAME,coordinates,video_aux=video_aux)
    # vrun  = jax.vmap(run_perturbed_nca, in_axes=(None,0,None,None))
    # for 

def save_videos_of_perturbations(nca,data,H,key,FILENAME,coordinates,video_aux={"duration":10,"scale_up":4}):
    """
        Only runs NCA for a small number of perturbations, selected by coordinates variable. 
        Saves the full videos of their trajectories as .mp4 files, based on video_aux.
    """

    ic = data[0]  # C X Y
    v_nca = jax.vmap(nca, in_axes=(0, None, 0))
    ic_batch = np.stack([perturb_pixel(ic, (x, y), H["perturbation_mode"]) for (x,y) in coordinates])
    perturbed_trajs = run_perturbed_nca_batch_full(v_nca, ic_batch, H, key)  # T B C X Y
    print("Shape of perturbed trajectories:", perturbed_trajs.shape)
    videos_normalised = []
    for i, (x,y) in enumerate(coordinates):
        # video_data = perturbed_trajs[:,i].transpose(0,2,3,1)[:,:,:,:4] # T X Y C

        video_data = rearrange(perturbed_trajs[:,i], "T C X Y -> T X Y C")[:,:,:,:3] # T X Y C

        video_data = onp.array(video_data)
        video_data = onp.clip(video_data,0.0,1.0)
        videos_normalised.append(video_data)
        print("Video data shape:", video_data.shape)
        # plt.imshow(video_data[0])
        # plt.show()

    top_left = onp.zeros_like(videos_normalised[0])
    videos_full = onp.stack([top_left]+videos_normalised,axis=0) # 4 T X Y C
    videos_full = rearrange(videos_full, "(Bx By) T X Y C -> T (Bx X) (By Y) C", Bx=2, By=2)
    video_name = f"{video_aux['output_path']}{FILENAME}_full.mp4"
    
    

    print("Saving video to:", video_name)
    save_to_video_rgb(videos_full, video_name, fps=30,duration=video_aux["duration"],SCALE_UP=video_aux["scale_up"])

def iterate_over_perturbations(nca,data,H,key,FILENAME):
    ic = data[0]  # C X Y
    _,X,Y = ic.shape
    C = 3
    # C = 1
    # X = 4
    # Y = 4
    perturbation_results = np.zeros((X,Y))
    perturbation_counts = np.zeros((X,Y))

    baseline_traj = run_perturbed_nca(nca,ic,H,key) # no perturbation yet
    batch_size = 16
    v_nca = jax.vmap(nca, in_axes=(0, None, 0))
    for x in tqdm(range(X)):
        for y0 in range(0, Y, batch_size):
            ys = list(range(y0, min(y0 + batch_size, Y)))
            # build batch of perturbed initial conditions
            ic_batch = np.stack([perturb_pixel(ic, (x, y), H["perturbation_mode"]) for y in ys])
            # derive a per-batch key to avoid reusing the same randomness
            key = jr.fold_in(key, x * Y + y0)
            # run vmapped NCA on the batch
            perturbed_trajs = run_perturbed_nca_batch(v_nca, ic_batch, H, key)  # B C X Y
            bas = baseline_traj[None, ...]  # 1 C X Y -> broadcastable to B C X Y
            diffs = np.mean(np.abs(perturbed_trajs - bas), axis=(1, 2, 3))
            num_diffs = np.sum((np.abs(perturbed_trajs - bas) > 0.1), axis=(1, 2, 3))

            for i, y in enumerate(ys):
                if x % 8 == 0 and y % 8 == 0:  # save full image every 8th pixel as before
                    perturbed_result = onp.array(perturbed_trajs[i][:4])  # first 4 channels
                    perturbed_result = numpy_image_float_to_int(perturbed_result)
                    onp.save(f"perturbations/emoji_local/{FILENAME}_coords_all_{x}_{y}_{H['perturbation_mode']}_T{H['timesteps']}.npy", perturbed_result)

                perturbation_results = perturbation_results.at[x, y].set(diffs[i])
                perturbation_counts = perturbation_counts.at[x, y].set(num_diffs[i])
                print(f"Perturbed pixel ({x},{y}), mean abs diff: {float(diffs[i]):.6f}")
    return perturbation_results, perturbation_counts

def main():
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])
    key = jr.PRNGKey(int(time.time()))
    key = jr.fold_in(key,index)
    HYPERPARAMETERS = {
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        # "channels":[32,16],
        "model":["gnNCA"],
        "channels":[16],
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
        run(H,False,key)
if __name__ == "__main__":
    main()