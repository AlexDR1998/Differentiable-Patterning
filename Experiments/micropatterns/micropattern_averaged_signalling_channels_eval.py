from Common.utils import load_micropattern_time_series,load_micropattern_smad23_lef1,load_micropattern_time_series_nodal_lef_cer,adhesion_mask_convex_hull_circle
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Common.model.boundary import model_boundary
from NCA.model.NCA_gated_model import gNCA
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
from einops import rearrange,repeat
import jax.random as jr
import time
import os
import sys


DOWNSAMPLE = 2
CHANNELS = 16
BATCHES = 1
STEPS_BETWEEN_IMAGES = 64
PVC_PATH = "/mnt/ceph/ar-dp/"
index = int(sys.argv[1])

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

impath_fstl = PVC_PATH+"Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"     # Foxa2, Sox17, TbxT, Lmbr
impath_nlc = PVC_PATH+"Data/Nodal_LEFTY_CER/**"                                                  # Lmbr, Cer Lefty, Nodal
impath_ls = PVC_PATH+"Data/Timecourse 60h June/Smad23_LEF 48h/Max Projections/*"            # Lef1, Lmbr, Smad23
data_fstl = load_micropattern_time_series(impath_fstl,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)               # 0h, 12h, 24h, 36h, 48h, 60h
data_nlc = load_micropattern_time_series_nodal_lef_cer(impath_nlc,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)   # 0h, 6h, 12h, 24h, 36h, 48h
data_ls = load_micropattern_smad23_lef1(impath_ls,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)                   # 0h, 6h, 12h, 24h, 36h, 48h
data_fstl = np.array(data_fstl)
data_nlc = np.array(data_nlc)[:,:,0] # select only condition 1 from data
data_ls = np.array(data_ls)



print("---- Before removing 6h and duplicate LMBR ----")
print(f"Foxa2 sox17 tbxt lmbr shape: {data_fstl.shape}")
print(f"Lmbr cer lefty nodal shape: {data_nlc.shape}")
print(f"Lef1 Lmbr smad23 shape: {data_ls.shape}")
# Data shape: (Time, batch, channels, width, height)

# Try without the 6h data first - it makes the timestepping a lot simpler
data_nlc = np.concatenate([data_nlc[:1],data_nlc[2:],np.zeros((1,*data_nlc.shape[1:]))],axis=0)
data_ls = np.concatenate([data_ls[:1],data_ls[2:],np.zeros((1,*data_ls.shape[1:]))],axis=0)

# Remove duplicates of LMBR channel
data_nlc = data_nlc[:,:,1:]
data_ls = data_ls[:,:,:1] # Also remove smad23 channel as guillaume recommended



print("---- After removing 6h and duplicate LMBR ----")
print(f"Foxa2 sox17 tbxt lmbr shape: {data_fstl.shape}")
print(f"Lmbr cer lefty nodal shape: {data_nlc.shape}")
print(f"Lef1 Lmbr smad23 shape: {data_ls.shape}")


# Combine the datasets
data = np.concatenate([data_fstl,data_nlc,data_ls],axis=2)

boundary_mask = adhesion_mask_convex_hull_circle(rearrange(data_fstl[0,0],"C X Y -> X Y C"))[0]
boundary_mask = rearrange(boundary_mask,"X Y -> 1 X Y")
print("Boundary mask shape: ",boundary_mask.shape)
boundary_func = model_boundary(boundary_mask)


boundary_mask = repeat(boundary_mask,"1 X Y -> B 1 X Y",B=BATCHES)
data = repeat(data,"T () C X Y -> B T C X Y", B=BATCHES)
#print("Boundary mask shape: ",boundary_mask.shape)

data = data*rearrange(boundary_mask,"B () X Y -> B () () X Y")

print("Channel order: Foxa2, Sox17, TbxT, Lmbr, Cer, Lefty, Nodal, Lef1")
print(f"Total data shape: {data.shape}")

DA = DataAugmenterAbstract(data,hidden_channels=CHANNELS-8)
DA.data_init()
T_true = DA.return_true_data()
x,_ = DA.split_x_y()
X0 = x[0][0]
print("T_true shape: ",np.array(T_true).shape)


NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"circular",
                    "key":jr.PRNGKey(int(time.time()))}
nca = gNCA(**NCA_hyperparameters)

nca_full = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_full_ds_{DOWNSAMPLE}_v3.eqx")
nca_cell_fate = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_cell_fate_ds_{DOWNSAMPLE}_v3.eqx")
nca_cell_fate_signal = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_cell_fate_and_signalling_ds_{DOWNSAMPLE}_v3.eqx")
nca_cell_fate_effector = nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_cell_fate_and_effectors_ds_{DOWNSAMPLE}_v3.eqx")
nca_signalling_effector= nca.load(PVC_PATH+f"models/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_steps_between_images_64_ch_{CHANNELS}_mode_signalling_and_effectors_ds_{DOWNSAMPLE}_v3.eqx")

if not os.path.exists(PVC_PATH+"output"):
    os.makedirs(PVC_PATH+"output")



T_full = nca_full.run(iters=STEPS_BETWEEN_IMAGES*6,x=X0,callback=boundary_func)
np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_T_full.npy",T_full[::STEPS_BETWEEN_IMAGES])
print("Done full trajectory")
del T_full

T_cell_fate = nca_cell_fate.run(iters=STEPS_BETWEEN_IMAGES*6,x=X0,callback=boundary_func)
np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_T_cell_fate.npy",T_cell_fate[::STEPS_BETWEEN_IMAGES])
print("Done cell fate trajectory")
del T_cell_fate

T_cell_fate_signal = nca_cell_fate_signal.run(iters=STEPS_BETWEEN_IMAGES*6,x=X0,callback=boundary_func)
np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_T_cell_fate_signal.npy",T_cell_fate_signal[::STEPS_BETWEEN_IMAGES])
print("Done cell fate signal trajectory")
del T_cell_fate_signal

T_cell_fate_effector = nca_cell_fate_effector.run(iters=STEPS_BETWEEN_IMAGES*6,x=X0,callback=boundary_func)
np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_T_cell_fate_effector.npy",T_cell_fate_effector[::STEPS_BETWEEN_IMAGES])
print("Done cell fate effector trajectory")
del T_cell_fate_effector

T_signalling_effector= nca_signalling_effector.run(iters=STEPS_BETWEEN_IMAGES*6,x=X0,callback=boundary_func)
np.save(PVC_PATH+f"output/FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_ch{CHANNELS}_boundary_reg_T_signalling_effector.npy",T_signalling_effector[::STEPS_BETWEEN_IMAGES])
print("Done signalling effector trajectory")
del T_signalling_effector