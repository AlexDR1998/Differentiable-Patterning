from NCA.trainer.NCA_impulse_optimiser import NCA_impulse_optimiser
from Common.dataloader.emoji import load_emoji_sequence
from einops import rearrange
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import jax
import jax.numpy as np
import jax.random as jr
from NCA.model.NCA_gated_model import gNCA
from einops import repeat
#-------------------- TESTING CODE --------------------



data = load_emoji_sequence(
    [
        "crab.png",
        "microbe.png",
        # "avocado.png",
        # "alien_monster.png",
        # "butterfly.png",
        # "lizard.png",
        # "mushroom.png",
    ],
    downsample=1,
    impath_emojis="/projects/u5be/alex_data/Emojis/",
)
# data_filename = "cr_mi_av_al_bt_li_mu"
data_filename = "cr_mi"

data = rearrange(data, "B T C W H -> T B C W H")

data = repeat(data, "B T C W H -> (B b) T C W H", b=1)

initial_condition = np.array(data)

W = initial_condition.shape[-2]
H = initial_condition.shape[-1]

initial_condition = initial_condition.at[:, :, :, : W // 2 - 6].set(0)
initial_condition = initial_condition.at[:, :, :, W // 2 + 5 :].set(0)
initial_condition = initial_condition.at[:, :, :, :, : H // 2 - 6].set(0)
initial_condition = initial_condition.at[:, :, :, :, H // 2 + 5 :].set(0)
data = np.concatenate(
    [initial_condition, data, data], axis=1
)  # Join initial condition and data along the time axis
print("(Batch, Time, Channels, Width, Height): " + str(data.shape))


CHANNELS = 64
OBS_CHANNELS = 4
key = jr.PRNGKey(0)
nca = gNCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR=["ID", "GRAD", "LAP"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5,
    key=key,)

nca = nca.load("models/multi_species_stable_gNCA_grad_64ch_contiguous_1.0_wide_perturbation_0.1_cr_mi_ds_1_long.eqx")

opt = NCA_impulse_optimiser(
    NCA_model = nca,
    data = data,
    DATA_AUGMENTER=DataAugmenterAbstract,
    STEPS_TO_STABLE=128,
    FILENAME="test_impulse",
    MODEL_DIRECTORY= "models/",
    LOG_DIRECTORY= "logs/",
    BOUNDARY_MASK = None,
    BOUNDARY_MODE = "soft",
    OBS_CHANNELS = 4, # Assume first third of channels are observable
)