from Common.utils import load_micropattern_shape_sequence
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Common.model.boundary import model_boundary
from NCA.model.NCA_gated_model import gNCA
import matplotlib.pyplot as plt
import jax
import equinox as eqx
import jax.numpy as np
from einops import rearrange,repeat
import jax.random as jr
import time
import os
import glob

DOWNSAMPLE = 4
CHANNELS = 16
BATCHES = 1
STEPS_BETWEEN_IMAGES = 64
PVC_PATH = "/mnt/ceph/ar-dp/"
index = 0

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

#print(glob.glob(PVC_PATH+"Data/micropattern_shapes/Max Projections */*Triangle*"))

data_triangle,mask_triangle,X0_triangle = load_micropattern_shape_sequence(
    PVC_PATH+"Data/micropattern_shapes/Max Projections */*Triangle*",
    DOWNSAMPLE=1,
    BATCH_AVERAGE=True
    )
data_ellipse,mask_ellipse,X0_ellipse = load_micropattern_shape_sequence(
    PVC_PATH+"Data/micropattern_shapes/Max Projections */*Ellipse*",
    DOWNSAMPLE=1,
    BATCH_AVERAGE=True)


mask_ellipse = rearrange(mask_ellipse,"H W-> () H W")
mask_triangle = rearrange(mask_triangle,"H W-> () H W")

print("Triangle data shape: ",X0_triangle.shape)
print("Ellipse data shape: ",X0_ellipse.shape)

print("Triangle mask shape: ",mask_triangle.shape)
print("Ellipse mask shape: ",mask_ellipse.shape)




boundary_func_triangle = model_boundary(mask=mask_triangle)
boundary_func_ellipse = model_boundary(mask=mask_ellipse)


X0_ellipse = np.pad(X0_ellipse,((0,CHANNELS-8),(0,0),(0,0)))
X0_triangle = np.pad(X0_triangle,((0,CHANNELS-8),(0,0),(0,0)))


# boundary_mask = repeat(boundary_mask,"1 X Y -> B 1 X Y",B=BATCHES)
# data = repeat(data,"T () C X Y -> B T C X Y", B=BATCHES)
# #print("Boundary mask shape: ",boundary_mask.shape)

# data = data*rearrange(boundary_mask,"B () X Y -> B () () X Y")

# print("Channel order: Foxa2, Sox17, TbxT, Lmbr, Cer, Lefty, Nodal, Lef1")
# print(f"Total data shape: {data.shape}")

# DA = DataAugmenterAbstract(data,hidden_channels=CHANNELS-8)
# DA.data_init()
# T_true = DA.return_true_data()
# x,_ = DA.split_x_y()
# X0 = x[0][0]
# print("T_true shape: ",np.array(T_true).shape)


NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"circular",
                    "key":jr.PRNGKey(int(time.time()))}
nca = gNCA(**NCA_hyperparameters)

nca_full = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_full_ds_{DOWNSAMPLE}.eqx")
nca_cell_fate = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}.eqx")
nca_cell_fate_signal = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_and_signalling_ds_{DOWNSAMPLE}.eqx")
nca_cell_fate_effector = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_cell_fate_and_effectors_ds_{DOWNSAMPLE}.eqx")
nca_signalling_effector= nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{CHANNELS}_mode_signalling_and_effectors_ds_{DOWNSAMPLE}.eqx")

# if not os.path.exists(PVC_PATH+"output"):
#     os.makedirs(PVC_PATH+"output")
ncas = {
    "full": nca_full,
    "cell_fate": nca_cell_fate,
    "cell_fate_signal": nca_cell_fate_signal,
    "cell_fate_effector": nca_cell_fate_effector,
    "signalling_effector": nca_signalling_effector
}

def run_sparse(nca,X0,iters,SAVE_EVERY,callback,key=jr.PRNGKey(int(1000*time.time()))):

    T = []
    X = X0
    for i in range(iters):
        X = eqx.filter_jit(nca)(X,callback,key)
        key = jr.fold_in(key,i)
        if i % SAVE_EVERY == 0:
            T.append(X)

    T = np.array(T)
    print("T shape: ",T.shape)
    return T


print("--- Running on ellipses ---")

for name,nca in ncas.items():
    print(f"Running {name} on ellipse")
    T = run_sparse(nca,X0_ellipse,STEPS_BETWEEN_IMAGES*6,STEPS_BETWEEN_IMAGES,callback=boundary_func_ellipse)
    np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_ds_{DOWNSAMPLE}_T_{name}_ellipse.npy",T)
    del T


print("--- Running on triangles ---")

for name,nca in ncas.items():
    print(f"Running {name} on ellipse")
    T = run_sparse(nca,X0_triangle,STEPS_BETWEEN_IMAGES*6,STEPS_BETWEEN_IMAGES,callback=boundary_func_triangle)
    np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_ds_{DOWNSAMPLE}_T_{name}_triangle.npy",T)
    del T
