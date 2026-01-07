import time
import glob
import jax
import jax.numpy as jnp
import skimage.io
from skimage import filters, restoration, util, exposure
from skimage.measure import regionprops
from jaxtyping import Float, Array
from jax import random as jr
from einops import reduce, rearrange, repeat
from tqdm import tqdm
from Common.dataloader.adhesion_mask import (
    adhesion_mask_convex_hull,
    adhesion_mask_convex_hull_circle,
    adhesion_mask_convex_hull_ellipse,
)
import scipy.ndimage as ndi
from math import floor, ceil
import matplotlib.pyplot as plt

import equinox as eqx
import numpy as np

# import pandas as pd
import skimage
from pprint import pprint
# from tensorflow.core.util import event_pb2


def load_micropattern_nodal_lefty_cer(
    impath="../Data/Nodal_LEFTY_CER/**",
    downsample=4,
    BATCH_AVERAGE=False,
    BACKGROUND_RADIUS=50,
    TIMESTEPS=[0, 6, 12, 24, 36, 48],  # 0h, 6h, 12h, 24h, 36h, 48h
    VERBOSE=False,
    HIST_EQS=(5, 95),
    SHOW_HISTOGRAMS=False,
    PROCESSING_MODES=[
        "mean_0_std_1",
        "saturate",
        "map_to_0_1",
        "align",
        "pad_to_full_width",
        "downsample",
    ],
    EXP_MODES=[1],
):
    """
    Experiment layout was as follows:
    -------------------------------------------------
    | 1         | 2         | 3         | 4         |
    | 0µM CHIR  | 1µM CHIR  | 2µM CHIR  | 3µM CHIR  |
    |           |           |           |           |
    -------------------------------------------------
    | 5         | 6         | 7         |           |
    | 4µM CHIR  | SB/LDN    | SB/LDN    |           |
    |           | @0h       | @24h      |           |

    so we don't have the same subdirectories for each timepoint
    Channel order is:
        Dappi (like LMBR, measures if cells are present)
        Cerberus
        Lefty
        Nodal
    """

    CHANNEL_NAMES = ["Dappi", "Cerberus", "Lefty", "Nodal"]
    # TIMESTEPS = [0,6,12,24,36,48]  # 0h, 6h, 12h, 24h, 36h, 48h
    filenames = glob.glob(impath, recursive=True)
    filenames = list(sorted(filenames))
    is_tif = lambda x: ".tif" in x
    filenames = list(filter(is_tif, filenames))
    if 60 in TIMESTEPS:
        TIMESTEPS = [
            t for t in TIMESTEPS if t != 60
        ]  # Remove 60h as it is not present in the data
    # where_func = lambda filenames,label:label in filenames
    # filenames_0h = list(filter(lambda x:"/0h" in x,filenames))
    # filenames_6h = list(filter(lambda x:"/6h" in x,filenames))
    # filenames_12h = list(filter(lambda x:"/12h" in x,filenames))
    # filenames_24h = list(filter(lambda x:"/24h" in x,filenames))
    # filenames_36h = list(filter(lambda x:"/36h" in x,filenames))
    # filenames_48h = list(filter(lambda x:"/48h" in x,filenames))
    # filenames_ordered = [
    #     filenames_0h,
    #     filenames_6h,
    #     filenames_12h,
    #     filenames_24h,
    #     filenames_36h,
    #     filenames_48h
    # ]
    filenames_ordered_base = [
        list(filter(lambda x: f"/{i}h/" in x, filenames)) for i in TIMESTEPS
    ]
    filenames_ordered = [
        [list(filter(lambda x: f"/{i}/" in x, F)) for i in EXP_MODES]
        for F in filenames_ordered_base
    ]
    filenames_ordered = [
        [ft for ft in filename_times if ft] for filename_times in filenames_ordered
    ]
    if not filenames_ordered[0]:
        exp1_filenames = list(filter(lambda x: "/1/" in x, filenames_ordered_base[0]))
        if exp1_filenames:
            filenames_ordered[0] = [exp1_filenames]

    if VERBOSE:
        pprint(filenames_ordered)

    ims = []
    for filename_times in tqdm(filenames_ordered):
        # ims_timestep = []
        for filename_conditions in filename_times:
            if VERBOSE:
                print(
                    f"-------- Loading batch of {len(filename_conditions)} images ----------------"
                )
            ims_cond = []
            for f_str in filename_conditions:
                _im = skimage.io.imread(f_str)
                ims_cond.append(_im)
                if VERBOSE:
                    print(f"File {f_str} loaded with shape {_im.shape}")

            ims_cond = jnp.array(ims_cond, dtype="float32")

            if VERBOSE:
                print(ims_cond.shape)
            ims.append(ims_cond)
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Pre processing")

    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=0,
        BATCH_AVERAGE=BATCH_AVERAGE,
        DOWNSAMPLE=downsample,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=VERBOSE,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Post processing")
    return ims, aux, CHANNEL_NAMES


def load_micropattern_sox17_foxa2_tbxt_lmbr(
    impath="../Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*",
    downsample=4,
    BATCH_AVERAGE=False,
    VERBOSE=False,
    TIMESTEPS=[0, 12, 24, 36, 48, 60],  # 0h, 12h, 24h, 36h, 48h, 60h
    BACKGROUND_RADIUS=50,
    SHOW_HISTOGRAMS=False,
    HIST_EQS=(5, 95),
    PROCESSING_MODES=["mean_0_std_1", "saturate", "map_to_0_1", "mult_by_lmbr"],
):
    """
    Data is ordered as follows:
    0h, 12h, 24h, 36h, 48h, 60h
    Each timestep has 4 channels: Sox17, Foxa2, TbxT, Lmbr

    Output is either a List of arrays of shape [BATCH, X, Y, CHANNELS] or a single array of shape [T, BATCH, C, X, Y]
    """
    CHANNEL_NAMES = ["Sox17", "Foxa2", "TbxT", "Lmbr"]
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))

    where_func = lambda filenames, label: label in filenames
    # filenames_0h = list(filter(lambda x:where_func(x,"_0h"),filenames))
    # filenames_12h = list(filter(lambda x:where_func(x,"_12h"),filenames))
    # filenames_24h = list(filter(lambda x:where_func(x,"_24h"),filenames))
    # filenames_36h = list(filter(lambda x:where_func(x,"_36h"),filenames))
    # filenames_48h = list(filter(lambda x:where_func(x,"_48h"),filenames))
    # filenames_60h = list(filter(lambda x:where_func(x,"_60h"),filenames))
    filenames_ordered = [
        list(filter(lambda x: where_func(x, f"_{i}h"), filenames)) for i in TIMESTEPS
    ]
    # filenames_ordered = [filenames_0h,filenames_12h,filenames_24h,filenames_36h,filenames_48h,filenames_60h]
    # filenames_ordered = [
    #   list(filter(lambda x:where_func(x,f"_{i}h"),filenames)) for i in times
    # ]

    ims = []
    for filenames in tqdm(filenames_ordered):
        if VERBOSE:
            print(len(filenames))
        ims_timestep = []
        for f_str in filenames:
            if VERBOSE:
                print(f_str)
            ims_timestep.append(skimage.io.imread(f_str))

        ims_timestep = jnp.array(ims_timestep, dtype="float32")

        ims.append(ims_timestep)
        if VERBOSE:
            print(ims_timestep.shape)
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Pre processing")
    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=3,
        BATCH_AVERAGE=BATCH_AVERAGE,
        DOWNSAMPLE=downsample,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=VERBOSE,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Post processing")
    return ims, aux, CHANNEL_NAMES


def load_micropattern_smad23_lef1(
    impath="../Data/Timecourse 60h June/Smad23_LEF 48h/Max Projections/*",
    downsample=4,
    VERBOSE=False,
    BATCH_AVERAGE=False,
    TIMESTEPS=[0, 6, 12, 24, 36, 48],  # 0h, 6h, 12h, 24h, 36h, 48h
    BACKGROUND_RADIUS=50,
    SHOW_HISTOGRAMS=False,
    HIST_EQS=(5, 95),
    PROCESSING_MODES=["mean_0_std_1", "saturate", "map_to_0_1", "mult_by_lmbr"],
):
    CHANNEL_NAMES = ["Lef1", "Lmbr", "Smad23"]
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))
    where_func = lambda filenames, label: label in filenames
    if 60 in TIMESTEPS:
        TIMESTEPS = [
            t for t in TIMESTEPS if t != 60
        ]  # Remove 60h as it is not present in the data
    # filenames_label = list(filter(lambda x:where_func(x,label),filenames))
    # filenames_0h = list(filter(lambda x:where_func(x,"_0h"),filenames))
    # filenames_6h = list(filter(lambda x:where_func(x,"_6h"),filenames))
    # filenames_12h = list(filter(lambda x:where_func(x,"_12h"),filenames))
    # filenames_24h = list(filter(lambda x:where_func(x,"_24h"),filenames))
    # filenames_36h = list(filter(lambda x:where_func(x,"_36h"),filenames))
    # filenames_48h = list(filter(lambda x:where_func(x,"_48h"),filenames))
    # filenames_ordered = [filenames_0h,filenames_6h,filenames_12h,filenames_24h,filenames_36h,filenames_48h]
    filenames_ordered = [
        list(filter(lambda x: f"_{i}h" in x, filenames)) for i in TIMESTEPS
    ]

    ims = []
    for filenames in tqdm(filenames_ordered):
        if VERBOSE:
            print(len(filenames))
        ims_timestep = []
        for f_str in filenames:
            _im = skimage.io.imread(f_str)
            ims_timestep.append(_im)
            if VERBOSE:
                print(_im.shape, f_str)

        ims_timestep = jnp.array(ims_timestep, dtype="float32")
        ims.append(ims_timestep)
        if VERBOSE:
            print(ims_timestep.shape)
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Pre processing")

    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=1,
        BATCH_AVERAGE=BATCH_AVERAGE,
        DOWNSAMPLE=downsample,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=VERBOSE,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )

    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Post processing")
    return ims, aux, CHANNEL_NAMES


def show_histograms(data, channel_names, title="Pre processing histograms"):
    """
    Shows histograms of pixel intensities for each channel and timestep
    """

    # n_channels = #data.shape[-1]

    # n_timesteps = data.shape[0]
    n_timesteps = len(data)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(15, 4), sharex=True, sharey=True)
    for t in range(n_timesteps):
        axes[t].hist(
            rearrange(data[t], "B X Y C -> (B X Y) C"),
            bins=50,
            alpha=1.0,
            label=channel_names,
            density=True,
        )
        axes[t].set_title(f"Timestep {t}")
        axes[t].set_xlabel("Pixel intensity")
        axes[t].set_ylabel("Frequency")
    plt.tight_layout()
    plt.title(title)
    plt.legend()
    plt.show()


def process_data(
    data,
    LMBR_CHANNEL=0,
    BATCH_AVERAGE=False,
    BACKGROUND_RADIUS=50,
    DOWNSAMPLE=4,
    HIST_BINS=None,  # Useful to normalise data from histogram bins of other datasets
    HIST_EQS=(5, 95),
    VERBOSE=False,
    mode=["clip", "mean_0_std_1", "saturate", "map_to_0_1", "downsample"],
):
    """
    Expects data as a list of [T] arrays of shape [BATCH, X, Y, CHANNELS], where for each entry in the list, BATCH can be different
    Some transformations need to be applied consistently across T,

    """
    # print(f"Data structure: {len(data)} {data[0].shape}")

    # mean_0_std_1 = lambda arr: (arr - jnp.mean(arr, axis=(1, 2), keepdims=True)) / (jnp.std(arr, axis=(1, 2), keepdims=True) + 1e-8)
    # map_to_0_1 = lambda arr: (arr - jnp.min(arr, axis=(1, 2), keepdims=True)) / (jnp.max(arr, axis=(1, 2), keepdims=True) - jnp.min(arr, axis=(1, 2), keepdims=True) + 1e-8)
    # saturate = lambda arr: jax.nn.sigmoid(arr)
    # mult_by_lmbr = lambda arr: arr * arr[:,:,:,LMBR_CHANNEL:LMBR_CHANNEL + 1]

    # def saturate(data):
    #    return [jax.nn.sigmoid(timestep) for timestep in data]

    def histogram_equalise(data):
        data_flat = [rearrange(timestep, "B X Y C -> (B X Y) C") for timestep in data]
        channel_percentiles = []
        if HIST_BINS is None:
            for channel in range(data[0].shape[-1]):
                channel_data_flat = []
                for T in range(len(data_flat)):
                    channel_data_flat.append(data_flat[T][:, channel])
                    # print(data_flat[T].shape)

                channel_data_flat = np.concatenate(channel_data_flat, axis=0)
                # print("Channel data shape:", channel_data_flat.shape)
                channel_percentiles.append(
                    np.percentile(a=channel_data_flat, q=HIST_EQS)
                )
        else:
            channel_percentiles = HIST_BINS

        # print("Channel percentiles:", channel_percentiles)

        data_new = []
        for T in tqdm(range(len(data))):
            arr = np.array(data[T])
            for C in range(arr.shape[-1]):
                for B in range(arr.shape[0]):
                    # Apply histogram equalisation to each channel of the array.
                    # Expects an array of shape [BATCH, X, Y, C]
                    # arr = arr.at[B,:,:,C].set(exposure.equalize_hist(arr[B,:,:,C]))
                    arr[B, :, :, C] = exposure.rescale_intensity(
                        arr[B, :, :, C],
                        in_range=(channel_percentiles[C][0], channel_percentiles[C][1]),
                        out_range=(0, 1),
                    )

            data_new.append(jnp.array(arr))
        return data_new, channel_percentiles

    def map_to_0_1(data):
        maxs = [jnp.max(timestep, axis=(0, 1, 2), keepdims=True) for timestep in data]
        mins = [jnp.min(timestep, axis=(0, 1, 2), keepdims=True) for timestep in data]
        maxs = jnp.array(maxs)
        mins = jnp.array(mins)
        maxs = jnp.max(maxs, axis=0, keepdims=False)
        mins = jnp.min(mins, axis=0, keepdims=False)
        # print("Maxs:", maxs.shape)
        # print("Mins:", mins.shape)
        data = [(timestep - mins) / (maxs - mins + 1e-8) for timestep in data]
        return data

    def mean_0_std_1(arr):
        means = [jnp.mean(timestep, axis=(0, 1, 2), keepdims=True) for timestep in arr]
        stds = [jnp.std(timestep, axis=(0, 1, 2), keepdims=True) for timestep in arr]
        means = jnp.array(means)
        stds = jnp.array(stds)
        means = jnp.mean(means, axis=0, keepdims=False)
        stds = jnp.mean(stds, axis=0, keepdims=False)
        # print("Means:", means.shape)
        # print("Stds:", stds.shape)
        arr = [(timestep - means) / (stds + 1e-8) for timestep in arr]
        return arr

    def batch_average(arr):
        """
        Averages the data across the batch dimension.
        Expects an array of shape [BATCH, X, Y, C]
        Returns an array of shape [1, X, Y, C]
        """
        if BATCH_AVERAGE:
            return [
                reduce(timestep, "BATCH X Y C -> () X Y C", "mean") for timestep in arr
            ]
        else:
            return arr

    def downsample(arr):
        # if BATCH_AVERAGE:
        #    arr = reduce(arr,"BATCH (X x2) (Y y2) C -> () X Y C","mean",x2=DOWNSAMPLE,y2=DOWNSAMPLE)
        # else:
        # for timestep in arr:
        # print("Downsampling timestep with shape:", timestep.shape)
        arr = [downsample_padder(timestep, DOWNSAMPLE) for timestep in arr]
        return [
            reduce(
                timestep,
                "BATCH (X x2) (Y y2) C -> BATCH X Y C",
                "mean",
                x2=DOWNSAMPLE,
                y2=DOWNSAMPLE,
            )
            for timestep in arr
        ]

    def _pad_to_full_width(arr):
        arr = jnp.array(arr)
        X_pad = (1080 - arr.shape[1]) / 2
        Y_pad = (1080 - arr.shape[2]) / 2
        arr = jnp.pad(
            arr,
            ((0, 0), (floor(X_pad), ceil(X_pad)), (floor(Y_pad), ceil(Y_pad)), (0, 0)),
            mode="edge",
        )
        return arr

    def pad_to_full_width(data):
        return [_pad_to_full_width(timestep) for timestep in data]

    def remove_background(data):
        arr = []
        backgrounds = []
        for timestep in data:
            arr_t, backgrounds_t = _remove_background_tophat(timestep)
            arr.append(arr_t)
            backgrounds.append(backgrounds_t)
        return arr, backgrounds

    def _remove_background(arr):
        """
        Expects an array of shape [BATCH, X, Y, C]
        For each channel and batch, computes the background with the rolling ball algorithm and subtracts it from the image.
        """
        arr = jnp.array(arr)
        backgrounds = []
        # Apply rolling ball algorithm to each channel
        for b in tqdm(range(arr.shape[0])):
            _b = []
            for c in range(arr.shape[-1]):
                background = restoration.rolling_ball(
                    arr[b, :, :, c], radius=BACKGROUND_RADIUS
                )
                _b.append(background)
                arr = arr.at[b, :, :, c].set(arr[b, :, :, c] - background)
            backgrounds.append(jnp.stack(_b, axis=-1))
        return arr, backgrounds

    def _remove_background_tophat(arr):
        """
        Expects an array of shape [BATCH, X, Y, C]
        For each channel and batch, computes the background with the tophat algorithm and subtracts it from the image.
        """
        arr = jnp.array(arr)
        backgrounds = []
        # Apply tophat algorithm to each channel
        hat = skimage.morphology.disk(BACKGROUND_RADIUS)
        for b in tqdm(range(arr.shape[0])):
            _b = []
            for c in range(arr.shape[-1]):
                _im = skimage.morphology.white_tophat(arr[b, :, :, c], footprint=hat)
                _b.append(arr[b, :, :, c] - _im)
                arr = arr.at[b, :, :, c].set(_im)
            backgrounds.append(jnp.stack(_b, axis=-1))
        return arr, backgrounds

    def threshold(data):
        data_new = []
        for timestep in data:
            for c in range(timestep.shape[-1]):
                for b in range(timestep.shape[0]):
                    # Apply thresholding to each channel of the array.
                    # Expects an array of shape [BATCH, X, Y, C]
                    thresh = skimage.filters.threshold_otsu(timestep[b, :, :, c])
                    timestep = timestep.at[b, :, :, c].set(
                        jnp.where(
                            timestep[b, :, :, c] > thresh, timestep[b, :, :, c], 0.0
                        )
                    )
            data_new.append(timestep)
        return data_new

    def align_centre_of_mass_stack(data):
        arr = []
        foregrounds = []
        for timestep in tqdm(data):
            arr_t, foregrounds_t = align_centre_of_mass(timestep)
            arr.append(arr_t)
            foregrounds.append(foregrounds_t)
        return arr, foregrounds

    funcs = {
        "hist_eq": histogram_equalise,
        "remove_background": remove_background,
        "threshold": threshold,
        "batch_average": batch_average,
        "mean_0_std_1": mean_0_std_1,
        "map_to_0_1": map_to_0_1,
        "pad_to_full_width": pad_to_full_width,
        "downsample": downsample,
        "align": align_centre_of_mass_stack,
    }

    backgrounds = None
    foregrounds = None
    data = [jnp.array(timestep).astype(jnp.float32) for timestep in data]
    for m in mode:
        if m in funcs:
            if m == "remove_background":
                data, backgrounds = funcs[m](data)
            elif m == "align":
                data, foregrounds = funcs[m](data)
            elif m == "hist_eq":
                data, HIST_BINS = funcs[m](data)
            else:
                data = funcs[m](data)
            if VERBOSE:
                print(f"Applied {m} to data with resulting {len(data)} shapes: ")
                for timestep in data:
                    print(timestep.shape)
        else:
            raise ValueError(f"Unknown normalisation mode: {m}")

    if BATCH_AVERAGE:
        data = np.array(data)
        # data_processed = rearrange(data_processed,"T BATCH X Y C -> BATCH T C X Y")
    return data, {
        "backgrounds": backgrounds,
        "foregrounds": foregrounds,
        "HIST_BINS": HIST_BINS,
    }


def load_micropattern_circle_8ch(
    DOWNSAMPLE,
    BATCHES,
    PVC_PATH="/mnt/ceph/ar-dp/",
    BACKGROUND_RADIUS=50,
    TIMESTEPS=[0, 12, 24, 36, 48, 60],  # 0h, 6h, 12h, 24h, 36h, 48h
    HIST_EQS={"sftl": (0.5, 99.95), "dcln": (0.5, 99.95), "lls": (0.5, 99.95)},
    SHOW_HISTOGRAMS=False,
    PROCESSING_MODES=["hist_eq", "batch_average", "map_to_0_1"],
):
    """
    Loads circular micropatterns for channels: Sox17, Foxa2, TbxT, Lmbr, Cer, Lefty, Nodal, Lef1
    """
    impath_sftl = (
        PVC_PATH + "Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
    )  # Sox17, Foxa2, TbxT, Lmbr
    impath_dcln = PVC_PATH + "Data/Nodal_LEFTY_CER/**"  # Lmbr, Cer Lefty, Nodal
    impath_lls = (
        PVC_PATH + "Data/Timecourse 60h June/Smad23_LEF 48h/Max Projections/*"
    )  # Lef1, Lmbr, Smad23
    data_sftl, aux_sftl, sftl_names = load_micropattern_sox17_foxa2_tbxt_lmbr(
        impath_sftl,
        downsample=DOWNSAMPLE,
        VERBOSE=False,
        BATCH_AVERAGE=True,
        TIMESTEPS=TIMESTEPS,
        PROCESSING_MODES=["downsample"] + PROCESSING_MODES,
        HIST_EQS=HIST_EQS["sftl"],
        SHOW_HISTOGRAMS=SHOW_HISTOGRAMS,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )  # 0h, 12h, 24h, 36h, 48h, 60h
    data_dcln, aux_nlc, dcln_names = load_micropattern_nodal_lefty_cer(
        impath_dcln,
        downsample=DOWNSAMPLE,
        VERBOSE=False,
        BATCH_AVERAGE=True,
        TIMESTEPS=TIMESTEPS,
        PROCESSING_MODES=["pad_to_full_width", "downsample"] + PROCESSING_MODES,
        HIST_EQS=HIST_EQS["dcln"],
        SHOW_HISTOGRAMS=SHOW_HISTOGRAMS,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )  # 0h, 6h, 12h, 24h, 36h, 48h
    data_lls, aux_lls, lls_names = load_micropattern_smad23_lef1(
        impath_lls,
        downsample=DOWNSAMPLE,
        VERBOSE=False,
        BATCH_AVERAGE=True,
        TIMESTEPS=TIMESTEPS,
        PROCESSING_MODES=["downsample"] + PROCESSING_MODES,
        HIST_EQS=HIST_EQS["lls"],
        SHOW_HISTOGRAMS=SHOW_HISTOGRAMS,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )  # 0h, 6h, 12h, 24h, 36h, 48h

    aux = {
        "sftl": aux_sftl,
        "dcln": aux_nlc,
        "lls": aux_lls,
    }
    data_sftl = np.array(data_sftl)
    data_dcln = np.array(data_dcln)  # select only condition 1 from data
    data_lls = np.array(data_lls)

    print("---- Before removing duplicate LMBR/Dappi ----")
    print("--- (Time , batch, width, height, channels) ---")
    print(f"{' '.join(sftl_names)} shape: {data_sftl.shape}")
    print(f"{' '.join(dcln_names)} shape: {data_dcln.shape}")
    print(f"{' '.join(lls_names)} shape: {data_lls.shape}")
    # Data shape: (Time, batch, width, height, channels)

    # Try without the 6h data first - it makes the timestepping a lot simpler
    # data_dcln = np.concatenate([data_dcln[:1],data_dcln[2:],np.zeros((1,*data_dcln.shape[1:]))],axis=0)
    # data_lls = np.concatenate([data_lls[:1],data_lls[2:],np.zeros((1,*data_lls.shape[1:]))],axis=0)

    if 60 in TIMESTEPS:
        # Add zeros on final timestep for Nodal_Lefty_Cerberus and Smad23_Lef1
        data_dcln = np.concatenate(
            [data_dcln, np.zeros((1, *data_dcln.shape[1:]))], axis=0
        )
        data_lls = np.concatenate(
            [data_lls, np.zeros((1, *data_lls.shape[1:]))], axis=0
        )
    # Remove duplicates of LMBR channel
    data_dcln = data_dcln[:, :, :, :, 1:]
    dcln_names = dcln_names[1:]  # Remove LMBR channel name
    data_lls = data_lls[
        :, :, :, :, :1
    ]  # Also remove smad23 channel as guillaume recommended
    lls_names = lls_names[:1]  # Keep only Lef1 channel name

    print("---- After removing 6h and duplicate LMBR ----")
    print(f"{' '.join(sftl_names)} shape: {data_sftl.shape}")
    print(f"{' '.join(dcln_names)} shape: {data_dcln.shape}")
    print(f"{' '.join(lls_names)} shape: {data_lls.shape}")

    # Combine the datasets along channels
    data = np.concatenate([data_sftl, data_dcln, data_lls], axis=-1)
    channel_names = sftl_names + dcln_names + lls_names
    boundary_mask = adhesion_mask_convex_hull_circle(data_sftl[-1, 0])[
        0
    ]  # last timestep looks good

    boundary_mask = repeat(boundary_mask, "X Y -> B () X Y", B=BATCHES)
    data = repeat(data, "T () X Y C -> B T C X Y", B=BATCHES)
    print("Boundary mask shape: ", boundary_mask.shape)

    data = data * rearrange(boundary_mask, "B () X Y -> B () () X Y")

    print("Channel order: " + " ".join(channel_names))
    print(f"Total data shape: {data.shape}")
    return data, boundary_mask, channel_names, aux


def load_micropattern_circle_8ch_individual(
    impath="../Data/Timecourse Individual Images/*",
    DOWNSAMPLE=1,
    BATCHES=1,
    BACKGROUND_RADIUS=20,
    TIMESTEPS=[0, 12, 24, 36, 48],  # 0h, 6h, 12h, 24h, 36h, 48h
    HIST_EQS=(1.0, 95.0),
    SHOW_HISTOGRAMS=False,
    PROCESSING_MODES=["map_to_0_1"],
):
    filenames = glob.glob(impath)
    ims = []
    where_func = lambda filenames, label: label in filenames
    # TIMESTEPS = [0,12,24,36,48]
    # CHANNEL_NAMES_SORTED = ["Cer1","Foxa2","LMBR","Lefty","Nodal","Sox17","Sox2","Tbxt"] # Sorted alphabetically
    CHANNEL_NAMES_DESIRED = [
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "FOXA2",
        "Cer1",
        "Lefty2",
        "Nodal",
    ]  # Desired order
    filenames_ordered = [
        list(filter(lambda x: where_func(x, f"_{i}h"), sorted(filenames)))
        for i in TIMESTEPS
    ]

    pprint(filenames_ordered[-1])
    for f_times in filenames_ordered:
        ims_time = []
        channel_names = []
        for f_str in f_times:
            ims_time.append(skimage.io.imread(f_str))
            channel_name = f_str.split("/")[-1].split("_")[0].replace(".tif", "")
            channel_names.append(channel_name)
        print("Channel names found: ", channel_names)
        ims_time = jnp.array(ims_time)
        ims_time = rearrange(ims_time, "C X Y -> () X Y C")
        # Reorder channels
        ims_time = ims_time[:,:,:,
            [
                channel_names.index(name)
                for name in CHANNEL_NAMES_DESIRED[: len(ims_time[0, 0, 0])]
            ],
        ]  # Some timepoints have less than 8 channels]
        if ims_time.shape[-1] < 8:
            # Pad with zeros to have 8 channels
            ims_time = jnp.pad(
                ims_time,
                ((0, 0), (0, 0), (0, 0), (0, 8 - ims_time.shape[-1])),
                mode="constant",
            )
        ims.append(ims_time)
    # ims = np.array(ims)
    # print("Loaded images with shape: ",ims.shape)

    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=2,
        BATCH_AVERAGE=False,
        DOWNSAMPLE=DOWNSAMPLE,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=False,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    ims = np.array(ims)  # shape of T B X Y C
    print("Processed images with shape: ", ims.shape)
    boundary_mask = adhesion_mask_convex_hull_circle(ims[-1, 0])[
        0
    ]  # last timestep looks good
    ims = repeat(ims, "T () X Y C -> B T C X Y", B=BATCHES)
    boundary_mask = repeat(boundary_mask, "X Y -> B () X Y", B=BATCHES)

    # boundary_mask = repeat(boundary_mask,"X Y -> B () X Y",B=BATCHES)
    # data = repeat(data,"T () X Y C -> B T C X Y", B=BATCHES)

    print("Data shape after batching: ", ims.shape)
    print("Boundary mask shape: ", boundary_mask.shape)
    ims = ims * rearrange(boundary_mask, "B () X Y -> B () () X Y")
    # ims = jnp.pad(ims,((0,0),(0,0),(0,0),()))
    return ims, aux, CHANNEL_NAMES_DESIRED, boundary_mask



def load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
    impath="../Data/Timecourse Seperate Colonies/",
    DOWNSAMPLE=1,
    BATCHES=1,
    BACKGROUND_RADIUS=20,
    TIMESTEPS=[0, 12, 24, 36, 48],  # 0h, 6h, 12h, 24h, 36h, 48h
    HIST_EQS=(1.0, 95.0),
    FILTER_KN_TIME=0,
    PROCESSING_MODES=["map_to_0_1"],
):
    """
        Loads circular micropatterns for 9 channels: LMBR, TBXT, SOX17, SOX2, FOXA2, Cer1, Lefty2, Nodal, Dappi
        Data is measured from 6 separate colonies, with some duplication of channels (LMBR, TBXT, SOX17 are in both A and B)

        Depending on if FILTER_KN_TIME is None, 0 or 24, this function will load different colonies and return DIFFERENT SHAPES OF OUTPUT DATA
        1) If FILTER_KN_TIME is None, loads colonies A, B, C, D (no knockouts)
            Returns data of shape [BATCHES, TIMESTEPS, 12 CHANNELS, X, Y]
            CHANNEL NAMES:
            [
                "A-LMBR",
                "A-TBXT",
                "A-SOX17",
                "A-SOX2",
                "B-LMBR",
                "B-TBXT",
                "B-SOX17",
                "B-FOXA2",
                "C-Cer1",
                "C-Lefty2",
                "C-Nodal",
                "D-Lef1",
            ]
        2) If FILTER_KN_TIME is 0 or 24, loads colonies and D, E, F (knockout), and some extra padding channels as the NCA model expects 9 channels
            Returns data of shape [BATCHES, TIMESTEPS, 9 CHANNELS, X, Y]
            CHANNEL NAMES:
            [
                "0-LMBR",
                "E-TBXT",
                "E-SOX17",
                "E-SOX2",
                "F-FOXA2",
                "0-Cer1",
                "0-Lefty2",
                "C-Nodal",
                "D-Lef1",
            ]
        Parameters:
        ----------
        impath: str
            Path to the folder containing the data. Expects subfolders A, B, C, D, E, F for each colony.
        DOWNSAMPLE: int
            Factor to downsample the images by.
        BATCHES: int
            Number of batches to repeat the data for.
        BACKGROUND_RADIUS: int
            Radius for background subtraction.
        TIMESTEPS: list
            List of timesteps in hours to load. 
        HIST_EQS: tuple
            Percentiles for histogram equalisation.
        PROCESSING_MODES: list
            List of processing modes to apply. See process_data function for options.
        Returns:
        -------
        ims: jnp.array
            Processed images of shape [BATCHES, TIMESTEPS, CHANNELS, X, Y]
        aux: dict
            Auxiliary data from processing. Used for debugging.
        CHANNEL_NAMES_COLONIES: list
            List of channel names in the order they are loaded.
        boundary_mask: jnp.array
            Boundary mask of shape [BATCHES, 1, X, Y]. Indicates where the micropattern is adhesing to the substrate.
    """
    CHANNEL_NAMES_DESIRED = [
        ["LMBR","TBXT","SOX17","SOX2"],
        ["LMBR","TBXT","SOX17","FOXA2"],
        ["Cer1","Lefty2","Nodal",],
        ["Lef1",],
        ["TBXT","SOX17","SOX2"],
        ["FOXA2"]
        ]
    
    if FILTER_KN_TIME==None:
        CHANNEL_TIMESTEP_MASK = np.ones((len(TIMESTEPS)-1,9))
    else:
        CHANNEL_TIMESTEP_MASK = np.array([
            [0,0,0,0,0,0,0,1,1],  # 12h
            [0,0,0,0,0,0,0,1,1],  # 24h
            [0,0,0,0,0,0,0,1,1],  # 36h
            [0,1,1,1,1,0,0,1,1],  # 48h
        ])
    # CHANNEL_NAMES_COLONIES = [
    #     "A-LMBR",
    #     "A-TBXT",
    #     "A-SOX17",
    #     "A-SOX2",
    #     "B-LMBR",
    #     "B-TBXT",
    #     "B-SOX17",
    #     "B-FOXA2",
    #     "C-Cer1",
    #     "C-Lefty2",
    #     "C-Nodal",
    #     "D-Lef1",
    #     "E-TBXT",
    #     "E-SOX17",
    #     "E-SOX2",
    #     "F-FOXA2",
    # ]
    if FILTER_KN_TIME==None:
        cols = ["A","B","C","D"]
        cols_knockout = []
        colony_paths_knockout = []
        CHANNEL_NAMES_COLONIES = [
            "A-LMBR",
            "A-TBXT",
            "A-SOX17",
            "A-SOX2",
            "B-LMBR",
            "B-TBXT",
            "B-SOX17",
            "B-FOXA2",
            "C-Cer1",
            "C-Lefty2",
            "C-Nodal",
            "D-Lef1",
        ]
    else:
        cols = ["A","B","C"]
        cols_knockout = ["D","E","F"]
        colony_paths_knockout = [impath+f"{i}/*" for i in cols_knockout]
        CHANNEL_NAMES_COLONIES = [
            "0-LMBR",
            "E-TBXT",
            "E-SOX17",
            "E-SOX2",
            "0-LMBR",
            "E-TBXT",
            "E-SOX17",
            "F-FOXA2",
            "0-Cer1",
            "0-Lefty2",
            "C-Nodal",
            "D-Lef1",
        ]

    colony_paths = [impath+f"{i}/*" for i in cols]#,"E","F"]]
    
    # rearrange big list of paths into lists of filenames per colony per timepoint
    
    colony_filenames = [
        [list(filter(lambda x: f"_{i}h" in x, sorted(filenames))) for i in TIMESTEPS] 
        for filenames in [glob.glob(path) for path in colony_paths]
        ]
    # Filter condition for if kn{FILTER_KN_TIME} in x and _{time}h in x; OR if _{time}h in x for time<FILTER_KN_TIME

    filter_time_knockout = lambda x,time: (f"_{time}h_kn{FILTER_KN_TIME}" in x) if time>FILTER_KN_TIME else (f"_{time}h" in x)

    colony_filenames_knockout = [
        [list(filter(lambda x: filter_time_knockout(x, i), sorted(filenames))) for i in TIMESTEPS]
        for filenames in [glob.glob(path) for path in colony_paths_knockout]
    ]
    colony_filenames += colony_filenames_knockout
    cols+=cols_knockout
    print("Colony filenames: ")
    print(len(colony_filenames))
    for i in range(len(colony_filenames)):
        print(f"Colony {cols[i]} has {len(colony_filenames[i])} timepoints.")
        for j in range(len(colony_filenames[i])):
            print(f"  Timepoint {TIMESTEPS[j]}h has {len(colony_filenames[i][j])} channels.")
        pprint(colony_filenames[i])
    
    
    ims = []
    names = []
    for i,f_colony in enumerate(colony_filenames):
        ims_colony = [] 
        channel_names_colony = []
        print(f"Loading colony {f_colony} ...")
        #iterate over non empty lists in f_colony
        f_colony = [f_times for f_times in f_colony if len(f_times)>0]
        for f_times in f_colony:
            ims_time = []
            channel_names = []
            for f_str in f_times: # Stack up channels
                ims_time.append(skimage.io.imread(f_str))
                channel_name = f_str.split("/")[-1].split("_")[0].replace(".tif", "")
                channel_names.append(channel_name)
                print(f_str)
            print("Channel names found: ", channel_names)
            ims_time = jnp.array(ims_time)
            ims_time = rearrange(ims_time, "C X Y -> () X Y C")
            ims_time = ims_time[:,:,:,
                [
                    channel_names.index(name)
                    for name in CHANNEL_NAMES_DESIRED[i][: len(ims_time[0, 0, 0])]
                ],
            ]
            # ims_time.append(ims_time)
            ims_colony.append(ims_time)
            channel_names_colony.append(channel_names)
        ims_timestep = np.array(ims_colony)
        if ims_timestep.shape[0]==1:
            ims_timestep = np.pad(ims_timestep,((4,0),(0,0),(0,0),(0,0),(0,0)),mode="constant")
        print(f"Colony {cols[i]} loaded with shape {ims_timestep.shape}")
        ims.append(ims_timestep)
        names.append(channel_names_colony)
    print("Names of channels loaded from colonies: ", names)
    # print(len(ims))
    # print(len(ims[0]))
    ims = np.concatenate(ims,axis=-1) # Concatenate along channels
    ims = list(ims)
    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=0,
        BATCH_AVERAGE=False,
        DOWNSAMPLE=DOWNSAMPLE,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=False,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    ims = np.array(ims)  # shape of T B X Y C
    print("Processed images with shape: ", ims.shape)
    boundary_mask = adhesion_mask_convex_hull_circle(ims[-1, 0])[
        0
    ]  # last timestep looks good
    boundary_mask = repeat(boundary_mask, "X Y -> B () X Y", B=BATCHES)


    if FILTER_KN_TIME!=None: # If doing knockout, we rearrange the channels a bit
        # ims[...,0] = 0 # We have no LMBR channel in the knockout colonies
        ims[...,1:4] = ims[...,12:15] # Use TBXT, SOX17, SOX2 from knockout colonies
        # ims[...,4] = 0 # No LMBR
        ims[...,5:7] = ims[...,12:14] # Use TBXT, SOX17 from knockout colonies
        ims[...,7] = ims[...,15] # Use FOXA2 from knockout colonies

        # ims[...,8:10] = 0 # No Cer1, Lefty2 channels in knockout colonies
        
        if FILTER_KN_TIME==0: # Manually set Nodal to 0 at the timepoints where it is knocked out
            ims[...,10] = 0
        elif FILTER_KN_TIME==24:
            ims[2:,:,:,:,10]=0
        # ims[...,11] = ims[...,11] # Use Lef1 from knockout colonies
        ims = ims[...,:12]
        # ims = ims[...,:9]

    ims = repeat(ims, "T () X Y C -> B T C X Y", B=BATCHES)
    ims = ims * rearrange(boundary_mask, "B () X Y -> B () () X Y")
    return ims, aux, CHANNEL_NAMES_COLONIES, boundary_mask, CHANNEL_TIMESTEP_MASK



def load_micropattern_circle_8ch_individual_explicit_colony(
    impath="../Data/Timecourse Seperate Colonies/",
    DOWNSAMPLE=1,
    BATCHES=1,
    BACKGROUND_RADIUS=20,
    TIMESTEPS=[0, 12, 24, 36, 48],  # 0h, 6h, 12h, 24h, 36h, 48h
    HIST_EQS=(1.0, 95.0),
    PROCESSING_MODES=["map_to_0_1"],
):
    """
        Loads circular micropatterns for 8 channels: LMBR, TBXT, SOX17, SOX2, FOXA2, Cer1, Lefty2, Nodal
        Data is measured from 3 separate colonies, with some duplication of channels (LMBR, TBXT, SOX17 are in both A and B)

        Parameters:
        ----------
        impath: str
            Path to the folder containing the data. Expects subfolders A, B, C for each colony.
        DOWNSAMPLE: int
            Factor to downsample the images by.
        BATCHES: int
            Number of batches to repeat the data for.
        BACKGROUND_RADIUS: int
            Radius for background subtraction.
        TIMESTEPS: list
            List of timesteps in hours to load. 
        HIST_EQS: tuple
            Percentiles for histogram equalisation.
        PROCESSING_MODES: list
            List of processing modes to apply. See process_data function for options.
        Returns:
        -------
        ims: jnp.array
            Processed images of shape [BATCHES, TIMESTEPS, CHANNELS=11, X, Y]
        aux: dict
            Auxiliary data from processing. Used for debugging.
        CHANNEL_NAMES_COLONIES: list
            List of channel names in the order they are loaded.
        boundary_mask: jnp.array
            Boundary mask of shape [BATCHES, 1, X, Y]. Indicates where the micropattern is adhesing to the substrate.
    """
    CHANNEL_NAMES_DESIRED = [
        ["LMBR","TBXT","SOX17","SOX2"],
        ["LMBR","TBXT","SOX17","FOXA2"],
        ["Cer1","Lefty2","Nodal",],
        ]
    CHANNEL_NAMES_COLONIES = [
        "A-LMBR",
        "A-TBXT",
        "A-SOX17",
        "A-SOX2",
        "B-LMBR",
        "B-TBXT",
        "B-SOX17",
        "B-FOXA2",
        "C-Cer1",
        "C-Lefty2",
        "C-Nodal",
    ]
    colony_paths = [impath+f"{i}/*" for i in ["A","B","C"]]
    # colony_filenames = [sorted(glob.glob(path)) for path in colony_paths]
    where_func = lambda filenames, label: label in filenames
    colony_filenames = [[list(filter(lambda x: where_func(x, f"_{i}h"), sorted(filenames)))
        for i in TIMESTEPS] for filenames in [glob.glob(path) for path in colony_paths]]
    # return colony_filenames
    ims = []
    names = []
    for i,f_colony in enumerate(colony_filenames):
        ims_colony = [] 
        channel_names_colony = []
        for f_times in f_colony:
            ims_time = []
            channel_names = []
            for f_str in f_times: # Stack up channels
                ims_time.append(skimage.io.imread(f_str))
                channel_name = f_str.split("/")[-1].split("_")[0].replace(".tif", "")
                channel_names.append(channel_name)
            print("Channel names found: ", channel_names)
            ims_time = jnp.array(ims_time)
            ims_time = rearrange(ims_time, "C X Y -> () X Y C")
            ims_time = ims_time[:,:,:,
            [
                channel_names.index(name)
                for name in CHANNEL_NAMES_DESIRED[i][: len(ims_time[0, 0, 0])]
            ],
        ]
            # ims_time.append(ims_time)
            ims_colony.append(ims_time)
            channel_names_colony.append(channel_names)
        ims.append(np.array(ims_colony))
        names.append(channel_names_colony)
    
    # Ims is list of arrays
    ims = np.concatenate(ims,axis=-1) # Concatenate along channels
    ims = list(ims)
    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=0,
        BATCH_AVERAGE=False,
        DOWNSAMPLE=DOWNSAMPLE,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        VERBOSE=False,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    ims = np.array(ims)  # shape of T B X Y C
    print("Processed images with shape: ", ims.shape)
    boundary_mask = adhesion_mask_convex_hull_circle(ims[-1, 0])[
        0
    ]  # last timestep looks good
    ims = repeat(ims, "T () X Y C -> B T C X Y", B=BATCHES)
    boundary_mask = repeat(boundary_mask, "X Y -> B () X Y", B=BATCHES)

    # boundary_mask = repeat(boundary_mask,"X Y -> B () X Y",B=BATCHES)
    # data = repeat(data,"T () X Y C -> B T C X Y", B=BATCHES)

    print("Data shape after batching: ", ims.shape)
    print("Boundary mask shape: ", boundary_mask.shape)
    ims = ims * rearrange(boundary_mask, "B () X Y -> B () () X Y")
    # ims = jnp.pad(ims,((0,0),(0,0),(0,0),()))
    return ims, aux, CHANNEL_NAMES_COLONIES, boundary_mask
    

def load_micropattern_radii(impath):
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))
    # print(sorted(filenames))
    ims = []
    for f_str in filenames:
        ims.append(skimage.io.imread(f_str))
    # print(jax.tree_util.tree_structure(ims))

    normalise = lambda arr: arr / np.max(arr, axis=(0, 1))
    pad = lambda arr: np.pad(arr, ((10, 10), (10, 10), (0, 0)))
    mask_out = lambda arr, mask: np.where(
        np.repeat(mask[0][:, :, np.newaxis], 4, axis=-1), arr, np.zeros_like(arr)
    )
    reshape = lambda arr: np.einsum("xyc->cxy", arr)
    just_mask = lambda mask: mask[0][np.newaxis]
    shapes = lambda arr: arr.shape[-1]

    def stack_x0(arr, mask):
        x0 = np.zeros_like(arr).astype(float)
        masked_arr = np.ma.array(
            arr, mask=~np.repeat(mask[0][np.newaxis], 4, axis=0).astype(bool)
        )
        # print(masked_arr)
        x0[1] = mask[0].astype(
            x0.dtype
        )  # *masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
        x0[3] = mask[0].astype(
            x0.dtype
        )  # *masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
        x0[1] *= masked_arr[1].mean()
        x0[3] *= masked_arr[3].mean()
        return np.stack((x0, arr), axis=0)

    ims = list(map(lambda x: pad(normalise(x)), ims))
    masks = list(map(adhesion_mask_convex_hull_circle, tqdm(ims)))
    ims = list(map(mask_out, ims, masks))
    ims = list(map(reshape, ims))
    ims = list(map(stack_x0, ims, masks))
    masks = list(map(just_mask, masks))
    shapes = list(map(shapes, ims))

    # ims = jax.tree_util.treedef_tuple(ims)
    # print(jnp.mean(ims))
    return ims, masks, shapes


def downsample_padder(arr, downsample):
    """Pads arrays with extra zeros if needed such that it can be properly downsampled by downsample

        Assumes array in shape X Y C, _ X Y C or _ _ X Y C
    Args:
        arr (_type_): _description_
        downsample (_type_): _description_
    """
    # print(arr.shape)
    if arr.ndim == 3:
        if arr.shape[0] % downsample != 0:
            arr = jnp.pad(
                arr, ((0, downsample - (arr.shape[0] % downsample)), (0, 0), (0, 0))
            )
        if arr.shape[1] % downsample != 0:
            arr = jnp.pad(
                arr, ((0, 0), (0, downsample - (arr.shape[1] % downsample)), (0, 0))
            )
    elif arr.ndim == 4:
        if arr.shape[1] % downsample != 0:
            arr = jnp.pad(
                arr,
                ((0, 0), (0, downsample - (arr.shape[1] % downsample)), (0, 0), (0, 0)),
            )
        if arr.shape[2] % downsample != 0:
            arr = jnp.pad(
                arr,
                ((0, 0), (0, 0), (0, downsample - (arr.shape[2] % downsample)), (0, 0)),
            )
    elif arr.ndim == 5:
        if arr.shape[2] % downsample != 0:
            arr = jnp.pad(
                arr,
                (
                    (0, 0),
                    (0, 0),
                    (0, downsample - (arr.shape[2] % downsample)),
                    (0, 0),
                    (0, 0),
                ),
            )
        if arr.shape[3] % downsample != 0:
            arr = jnp.pad(
                arr,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, downsample - (arr.shape[3] % downsample)),
                    (0, 0),
                ),
            )
    # print(arr)
    return arr


def pad_to_biggest(ims):
    """takes a list of images [array[X Y C]] and pads the X and Y dimensions to that of the biggest one"""
    # Determine the maximum height and width among all images
    max_height = max(im.shape[0] for im in ims)
    max_width = max(im.shape[1] for im in ims)
    print("Max height:", max_height)
    print("Max width:", max_width)
    padded = []
    for im in ims:
        h, w, _ = im.shape
        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left
        # Pad only the spatial dimensions (channels remain unchanged)
        padded_im = jnp.pad(
            im, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant"
        )
        padded.append(padded_im)
    return padded


def load_micropattern_ellipse(impath, DOWNSAMPLE, BATCH_AVERAGE=False):
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))
    # print(sorted(filenames))
    ims = []
    for f_str in filenames:
        ims.append(skimage.io.imread(f_str))
    # print(jax.tree_util.tree_structure(ims))

    downsample = lambda arr: reduce(
        downsample_padder(arr, DOWNSAMPLE),
        "(X x) (Y y) C -> X Y C",
        "mean",
        x=DOWNSAMPLE,
        y=DOWNSAMPLE,
    )  # noqa: E731

    normalise = lambda arr: arr / np.max(arr, axis=(0, 1))  # noqa: E731
    mask_out = lambda arr, mask: np.where(
        np.repeat(mask[0][:, :, np.newaxis], 4, axis=-1), arr, np.zeros_like(arr)
    )  # noqa: E731
    reshape = lambda arr: np.einsum("xyc->cxy", arr)  # noqa: E731
    just_mask = lambda mask: mask[0][np.newaxis]  # noqa: E731
    shapes = lambda arr: arr.shape[-1]  # noqa: E731

    def stack_x0(arr, mask):
        x0 = np.zeros_like(arr).astype(float)
        masked_arr = np.ma.array(
            arr, mask=~np.repeat(mask[0][np.newaxis], 4, axis=0).astype(bool)
        )
        # print(masked_arr)
        x0[1] = mask[0].astype(
            x0.dtype
        )  # *masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
        x0[3] = mask[0].astype(
            x0.dtype
        )  # *masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
        x0[1] *= masked_arr[1].mean()
        x0[3] *= masked_arr[3].mean()
        return np.stack((x0, arr), axis=0)

    ims = list(map(normalise, ims))
    ims = pad_to_biggest(ims)
    # ims = np.array(ims)
    ims = np.array([downsample_padder(im, DOWNSAMPLE) for im in ims])
    if BATCH_AVERAGE:
        ims = reduce(
            ims, "B (X x) (Y y) C -> () X Y C", "mean", x=DOWNSAMPLE, y=DOWNSAMPLE
        )
    else:
        ims = reduce(
            ims, "B (X x) (Y y) C -> B X Y C", "mean", x=DOWNSAMPLE, y=DOWNSAMPLE
        )
    ims = list(ims)
    masks = list(map(adhesion_mask_convex_hull_ellipse, tqdm(ims)))
    ims = list(map(mask_out, ims, masks))
    ims = list(map(reshape, ims))
    ims = list(map(stack_x0, ims, masks))
    masks = list(map(just_mask, masks))
    shapes = list(map(shapes, ims))

    return ims, masks, shapes


def load_micropattern_shape_array(
    impath="../Data/micropattern_shapes/Max Projections */*Triangle*",
    DOWNSAMPLE=1,
    BATCH_AVERAGE=False,
    SHOW_HISTOGRAMS=False,
    BACKGROUND_RADIUS=50,
    HIST_EQS=(0.5, 99.95),
    HIST_BINS=None,
    PROCESSING_MODES=["align", "hist_eq", "map_to_0_1"],
):
    """_summary_

    Args:
        impath (string): path to files
        DOWNSAMPLE (int): downsampling ratio
        BATCH_AVERAGE (bool, optional): Average data across batches. Defaults to False.

    Returns:
        Array [BATCH, X, Y, C]: _description_
    """
    CHANNEL_NAMES = [
        "SOX17",
        "FOXA2",
        "TBXT",
        "LMBR",
        "CER",
        "LEFTY",
        "NODAL",
        "LEF1",
        # "SMAD23"
    ]
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))
    print("Found filenames: ", filenames)
    ims = []
    
    # for f_str in tqdm(filenames):
        # ims.append(skimage.io.imread(f_str))

    ims.append(skimage.io.imread(filenames[0])) # Try just loading 1 image as we only really need the shape

    # mean_0_std_1 = lambda arr: (arr-jnp.mean(arr,axis=(1,2),keepdims=True))/(jnp.std(arr,axis=(1,2),keepdims=True))
    # map_to_0_1 = lambda arr: (arr-jnp.min(arr,axis=(1,2),keepdims=True))/(jnp.max(arr,axis=(1,2),keepdims=True)-jnp.min(arr,axis=(1,2),keepdims=True))
    # saturate = lambda arr: jax.nn.sigmoid(arr)
    # mult_by_lmbr = lambda arr: arr*arr[:,:,:,3:4]
    #for im in ims:
        #print(im.shape)
    ims = pad_to_biggest(ims)
    ims = list(map(lambda a: downsample_padder(a, DOWNSAMPLE), ims))
    # ims = [rearrange(im,"X Y C -> () X Y C") for im in ims]  # Add time dimension

    ims = jnp.array(ims, dtype="float32")

    print("Shape of images:", ims.shape)
    ims = [ims]
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Pre processing")
    ims, aux = process_data(
        ims,
        LMBR_CHANNEL=3,
        BATCH_AVERAGE=BATCH_AVERAGE,
        DOWNSAMPLE=DOWNSAMPLE,
        mode=PROCESSING_MODES,
        HIST_EQS=HIST_EQS,
        HIST_BINS=HIST_BINS,
        BACKGROUND_RADIUS=BACKGROUND_RADIUS,
    )
    if SHOW_HISTOGRAMS:
        show_histograms(ims, CHANNEL_NAMES, title="Post processing")
    return ims, aux, CHANNEL_NAMES


def load_micropattern_shape_sequence(
    impath, 
    DOWNSAMPLE, 
    BATCH_AVERAGE, 
    CIRCLE_DATA, 
    CIRCLE_MASK, 
    CIRCLE_HIST_BINS, 
    PROCESSING_MODES,
    SHAPED_MASK = None,
):
    CHANNELS = [
        "SOX17",
        "FOXA2",
        "TBXT",
        "LMBR",
        "CER",
        "LEFTY",
        "NODAL",
        "LEF1",
        # "SMAD23"
    ]
    # CIRCLE_DATA is (B,T,CHANNELS, X, Y)
    if SHAPED_MASK is None:
        true_data = load_micropattern_shape_array(
            impath,
            DOWNSAMPLE,
            BATCH_AVERAGE,
            HIST_BINS=CIRCLE_HIST_BINS,
            PROCESSING_MODES=PROCESSING_MODES,
        )[0]
        masks = adhesion_mask_convex_hull(rearrange(true_data[0], "B X Y C -> X Y B C"))
        print(f"True data shape: {true_data[0].shape}")
    else:
        masks = SHAPED_MASK
        true_data = None
    print(f"Masks shape internal {masks.shape}")
    key = jr.PRNGKey(int(time.time()))
    n_channels = len(CHANNELS)
    # Expand the mask to have one channel per synthetic condition.
    mask_expanded = repeat(masks, "X Y -> C X Y", C=n_channels)
    # Create synthetic initial conditions by sampling random values where the mask is True, zero elsewhere.
    # unmasked_ic = jr.choice(key,)
    unmasked_ic = []
    for i in range(n_channels):
        unmasked_ic.append(
            jax.random.choice(
                key,
                shape=(masks.shape[0], masks.shape[1]),
                a=CIRCLE_DATA[0, 0, i][CIRCLE_MASK[0, 0] == 1].flatten(),
                replace=True,
            )
        )
    unmasked_ic = jnp.array(unmasked_ic)
    synthetic_initial_conditions = jnp.where(mask_expanded, unmasked_ic, 0.0)

    return true_data, masks, synthetic_initial_conditions


def load_micropattern_triangle(impath):
    filenames = glob.glob(impath)
    filenames = list(sorted(filenames))
    # print(sorted(filenames))
    ims = []
    for f_str in filenames:
        ims.append(skimage.io.imread(f_str))
    # print(jax.tree_util.tree_structure(ims))

    normalise = lambda arr: arr / np.max(arr, axis=(0, 1))
    pad = lambda arr: np.pad(arr, ((10, 10), (10, 10), (0, 0)))
    mask_out = lambda arr, mask: np.where(
        np.repeat(mask[:, :, np.newaxis], 4, axis=-1), arr, np.zeros_like(arr)
    )
    reshape = lambda arr: np.einsum("xyc->cxy", arr)
    just_mask = lambda mask: mask[0][np.newaxis]
    shapes = lambda arr: arr.shape[-1]

    def stack_x0(arr, mask):
        x0 = np.zeros_like(arr).astype(float)
        masked_arr = np.ma.array(
            arr, mask=~np.repeat(mask[np.newaxis], 4, axis=0).astype(bool)
        )
        # print(masked_arr)
        x0[1] = mask.astype(
            x0.dtype
        )  # *masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
        x0[3] = mask.astype(
            x0.dtype
        )  # *masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
        x0[1] *= masked_arr[1].mean()
        x0[3] *= masked_arr[3].mean()
        return np.stack((x0, arr), axis=0)

    ims = list(map(lambda x: pad(normalise(x)), ims))
    masks = list(map(adhesion_mask_convex_hull, tqdm(ims)))
    ims = list(map(mask_out, ims, masks))
    ims = list(map(reshape, ims))
    ims = list(map(stack_x0, ims, masks))
    masks = list(map(just_mask, masks))
    shapes = list(map(shapes, ims))

    # ims = jax.tree_util.treedef_tuple(ims)
    # print(jnp.mean(ims))
    return ims, masks, shapes


def shift_image(img, shift_val):
    """
    Shift a 2D image by a fractional amount using bilinear interpolation with periodic boundaries.

    Parameters
    ----------
    img : numpy.ndarray
        2D array with shape (H, W)
    shift_val : tuple of float
        Shift (row_shift, col_shift)

    Returns
    -------
    shifted_img : numpy.ndarray
        Shifted image (H, W)
    """
    H, W = img.shape
    row_shift, col_shift = shift_val

    # Create meshgrid of coordinates
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Compute floating source coordinates
    src_rows = rows - row_shift
    src_cols = cols - col_shift

    # Compute indices for bilinear interpolation
    r0 = np.floor(src_rows).astype(int)
    c0 = np.floor(src_cols).astype(int)
    r1 = r0 + 1
    c1 = c0 + 1

    # Compute interpolation weights
    dr = src_rows - r0
    dc = src_cols - c0

    # Apply periodic boundary conditions by wrapping indices
    r0_mod = np.mod(r0, H)
    r1_mod = np.mod(r1, H)
    c0_mod = np.mod(c0, W)
    c1_mod = np.mod(c1, W)

    # Get pixel values using periodic indices
    I00 = img[r0_mod, c0_mod]
    I01 = img[r0_mod, c1_mod]
    I10 = img[r1_mod, c0_mod]
    I11 = img[r1_mod, c1_mod]

    shifted = (
        (1 - dr) * (1 - dc) * I00
        + (1 - dr) * dc * I01
        + dr * (1 - dc) * I10
        + dr * dc * I11
    )
    return shifted


# @jax.jit
def align_centre_of_mass(img_stack):
    """
    Given a stack of images with shape (N, H, W, C) where the spatial structure
    is roughly circular, shift each image so that the center of mass (computed from
    the sum over channels) aligns with the center of the image.

    Parameters
    ----------
    img_stack : numpy.ndarray
        Stack of images with shape (N, H, W, C).

    Returns
    -------
    aligned_stack : numpy.ndarray
        Stack of aligned images with the same shape.
    """
    aligned_stack = img_stack.copy()
    N, H, W, C = aligned_stack.shape
    target = (H / 2.0, W / 2.0)

    # Precompute coordinate grid for center of mass calculation
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    foregrounds = []
    for i in range(N):
        img = aligned_stack[i]
        # thresh = np.mean(img, axis=(0,1))  # Threshold to avoid noise
        # # Compute weight by summing over channels
        # weight = np.sum(img>rearrange(thresh,"C -> () () C"), axis=-1)  # Use a threshold to avoid noise
        greyscale = np.mean(img, axis=-1)  # Convert to greyscale by averaging channels

        greyscale = ndi.gaussian_filter(
            greyscale, sigma=4.0
        )  # Smooth the image to reduce noise
        # threshold_value = filters.threshold_otsu(greyscale)
        threshold_value = np.median(greyscale)
        labeled_foreground = (greyscale > threshold_value).astype(int)
        labeled_foreground = skimage.morphology.convex_hull_image(
            labeled_foreground, tolerance=0.5
        )
        footprint = skimage.morphology.disk(32)
        labeled_foreground = skimage.morphology.binary_erosion(
            labeled_foreground, footprint=footprint
        )
        properties = regionprops(skimage.measure.label(labeled_foreground))
        # properties = regionprops(labeled_foreground, greyscale)
        com = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        # com = ndi.center_of_mass(greyscale)
        # Calculate shift needed: positive shift moves image content downward/right.
        shift_val = (target[0] - com[0], target[1] - com[1])
        # print(f"Image {i}: Center of mass at {com}, shift value: {shift_val}")
        # Shift each channel separately using bilinear interpolation
        for c in range(C):
            # aligned_stack[i, c] = shift_image(img[c], shift_val)
            aligned_stack = aligned_stack.at[i, :, :, c].set(
                shift_image(img[:, :, c], shift_val)
            )

        labeled_foreground = shift_image(labeled_foreground, shift_val)
        foregrounds.append(labeled_foreground)
    foregrounds = np.array(foregrounds)
    # print("Aligned stack shape:", aligned_stack.shape)
    return aligned_stack, foregrounds
