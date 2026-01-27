import jax
import jax.random as jr
import jax.numpy as np
import os
import sys
import time
import optax
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/"
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_KAN_model import kaNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list
from NCA.trainer.data_augmenter_nca import DataAugmenter


BATCHES = 4
# CHANNELS=32
# DOWNSAMPLE = 1
# STEPS_BETWEEN_IMAGES=64
# iters=8000



def H_to_filename(H):
    FILENAME = f"emoji_al_mi_ro_loss_{H['loss_mode']}_s{H['wasserstein_samples']}_ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}"
    return FILENAME

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, BATCHES)
        data = self.pad(data, [10,10,10,10]) 		
        self.save_data(data)
        return None


def run(H,key):
    
    loss_str = {
        "l2":["l2"],
        "l1":["l1"],
        "euclidean":["euclidean"],
        "spectral":["spectral"],
        "spectral_no_phase":["spectral_no_phase"],
        "spectral_phase":["spectral_phase"],
        "spectral_euclidean":["spectral","euclidean"],
        "sliced_wasserstein_spatial":["sliced_wasserstein_spatial"],
        "sliced_wasserstein_channel":["sliced_wasserstein_channel"],
        "sliced_wasserstein_full":["sliced_wasserstein_full"],
        "spectral_wasserstein_full":["spectral_wasserstein_full"],
        "sliced_wasserstein_rotational":["sliced_wasserstein_rotational"],
        "ott":["ott"],
        "bhattacharyya":["bhattacharyya"],
        "kl_divergence":["kl_divergence"],
        "hellinger":["hellinger"],
        "bhattacharyya_modified":["bhattacharyya","average_amplitude"],
        "hellinger_modified":["hellinger","average_amplitude"],
        "kl_divergence_modified":["kl_divergence","average_amplitude"],
        "cosine":["cosine"],
        "vgg_3ch":["vgg_3ch"],
    }[H["loss_mode"]]
    
    
    data = load_emoji_sequence(
        ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
        impath_emojis=DATA_PATH,
        downsample=H["downsample"]
    )

    schedule = optax.exponential_decay(1e-3, transition_steps=H["iters"], decay_rate=0.99)
    optimiser = optax.chain(optax.scale_by_param_block_norm(),
                            optax.nadam(schedule))


    print("Training anisotropic nca")
    nca = NCA(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=key
    )
    # FILENAME = f"emoji_al_mi_ro_loss_{H['loss_mode']}_ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}"
    FILENAME = H_to_filename(H)
    opt = NCA_Trainer(nca,
                        data,
                        model_filename=FILENAME,
                        MODEL_DIRECTORY=CODE_PATH+"models/",
                        DATA_AUGMENTER=data_augmenter_subclass,
                        GRAD_LOSS=True)

    opt.train(
        t=H["steps_between_images"],
        iters=H["iters"],
        REGULARISER_COEFFS={
            "intermediate_state":H["intermediate_growth_coeff"],
            "boundary":H["boundary_reg_coeff"],
            "contiguous_growth":H["contiguous_growth_coeff"],
        },
        WARMUP=10,
        optimiser=optimiser,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR=loss_str,
        LOSS_ARGS = {
				"channels":None,
				"experiment_groups":None,
				"S":1024,
				"K":5,
				"D":3,
				"sharpen":True,
				"epsilon":0.1,
				"internal_loss_func":"l2",
				"samples":H['wasserstein_samples'],
			  },
        # LOSS_ARGS={
        #     "channels":None,
        #     "experiment_groups":None,
        #     "S":H["ott_S"],
        #     "K":H["ott_K"],
        #     "D":H["ott_D"],
        #     "sharpen":H["ott_sharpen"],
        #     "epsilon":H["ott_epsilon"],
        #     "internal_loss_func":H["ott_internal_loss_func"],
        # },
        wandb_args={
            "project":"nca-emojis-thesis-ch1",
            "group":"loss-function-wasserstein-sample-comparisons-2",
            # "group":"baseline-9ch-train-1",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        # KNOCKOUT_ARGS=KNOCKOUT_ARGS,
        LOG_EVERY=100,
        CLEAR_CACHE_EVERY=500,
    )

def main():
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])
    HYPERPARAMETERS = {
        "loss_mode":[#"l2","l1","euclidean","spectral","spectral_no_phase","spectral_phase","spectral_euclidean",
                    "sliced_wasserstein_spatial","sliced_wasserstein_channel","spectral_wasserstein_full","sliced_wasserstein_full",
                    "sliced_wasserstein_rotational",
                    # "ott"
                    #  "bhattacharyya","kl_divergence","hellinger",
                    #  "bhattacharyya_modified","hellinger_modified","kl_divergence_modified", 
        ],
                    #"cosine"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        "steps_between_images":[64],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0],
        "contiguous_growth_coeff":[0.0],
        "wasserstein_samples":[1,4,16,32,64,128,256,512],
        # "wasserstein_samples":[64],
    }




    HPARAMS = index_to_param_list(index,TOTAL_JOBS,HYPERPARAMETERS)


    key = jr.PRNGKey(int(time.time()))
    key = jr.fold_in(key,index)

    for H in HPARAMS:
        print("---------------------------------------------------")
        print(f"RUNNING WITH HYPERPARAMS:")
        pprint(H)
        
        key = jr.fold_in(key,index)
        run(H,key)
if __name__ == "__main__":
    main()