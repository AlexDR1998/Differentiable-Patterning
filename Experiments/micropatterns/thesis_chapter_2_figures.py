import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    # print(sys.path)
    import jax 
    import jax.numpy as np
    import numpy as onp
    import jax.random as jr
    import scipy as sp
    import equinox as eqx
    from tqdm.notebook import tqdm
    import time
    from einops import rearrange,repeat,reduce
    from NCA.model.NCA_gated_model import gNCA
    from NCA.model.NCA_model import NCA
    # from NCA.model.NCA_multi_scale import mNCA
    from NCA.model.NCA_noise_model import nNCA
    from NCA.model.NCA_gated_noise_model import gnNCA
    # from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual,load_micropattern_circle_8ch_individual_explicit_colony
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from Common.dataloader.micropattern import load_micropattern_circle_nodal_knockout_9ch_explicit_colony,normalise_micropattern_radii
    from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch,split_and_pad_by_experiment_groups_12ch
    # from Experiments.emoji.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
    # from Experiments.emoji.parameter_noise_sweep import H_to_filename as H_to_filename_noise
    # from Experiments.emoji.fire_rate_sweep import H_to_filename as H_to_filename_fr
    from Experiments.micropatterns.nodal_knockout_fine_tune import H_to_filename as H_to_filename_nodal
    # from Experiments.micropatterns.micropattern_individual_eval import calculate_radial_average
    from Common.dataloader.micropattern import load_micropattern_shape_sequence
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations,generate_hyperparameter_combinations_indexed
    import matplotlib.pyplot as plt
    from pprint import pprint
    from skimage import measure
    import matplotlib.style
    matplotlib.style.use(
        "default"
    )

    from dotenv import load_dotenv
    load_dotenv()
    import os
    PVC_PATH = os.getenv("PVC_PATH")
    DATA_PATH_BASE = os.getenv("DATA_PATH_BASE")
    DATA_PATH_INDIVIDUAL = DATA_PATH_BASE + "Timecourse Individual Images/*"
    DATA_PATH_GROUPED= DATA_PATH_BASE + "Timecourse Seperate Colonies/*"
    import scipy
    from Common.trainer.loss import build_loss_functions
    vgg_loss_dict = {
        "vgg_metric":"l2",
        "internal_loss_func":"l2",
        "epsilon":1e-10,
        "tau":None,
        "normalize":False,
        "samples":None
    }


@app.cell
def _():
    mo.md(r"""
    # Load data
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Load different micropattern sizes
    """)
    return


@app.cell
def _(DATA_9CH_CIRCULAR):
    DATA_4CH_RADII = normalise_micropattern_radii(DATA_9CH_CIRCULAR[2],"../Data/MAX_Chir_Fgf_scaling/*",percentile_thresh=[99.5,99.5,97,98.5])
    pads = [15,15,25,65,95,85,95,125]
    for i in range(len(DATA_4CH_RADII)):
        if i==0:
            # DATA_4CH_RADII[i] = onp.pad(DATA_4CH_RADII[i],((0,10),(0,0),(0,0)))
            DATA_4CH_RADII[i] = DATA_4CH_RADII[i][:-20,20:]
        DATA_4CH_RADII[i] = onp.pad(DATA_4CH_RADII[i],((pads[i],pads[i]),(pads[i],pads[i]),(0,0)))

        if i==5:
            DATA_4CH_RADII[i] = DATA_4CH_RADII[i][:,24:]
        if i==6:
            DATA_4CH_RADII[i] = DATA_4CH_RADII[i][:,24:]


        print(f"Shape after cropping or padding {i}: {DATA_4CH_RADII[i].shape}")
        # print(i)
        # for _T in _D:
        #     for _d in _T:
        #         _d = np.pad(_d,((0,0),(0,0),(0,1)))
        #         _d[:,-1] = 0
        #         print(_d.shape)
    return (DATA_4CH_RADII,)


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_4CH_RADII, size_slices):
    RADII_DATA = [rearrange(X,"X Y C-> () C X Y") for X in DATA_4CH_RADII]
    print(RADII_DATA[0].shape)
    size_slices(RADII_DATA,chnames= CHANNEL_NAMES_9CH_CIRCULAR,tres=None)
    return (RADII_DATA,)


@app.cell
def _(DATA_9CH_CIRCULAR):
    print(DATA_9CH_CIRCULAR[2].shape)
    return


@app.cell
def _(DATA_4CH_RADII):
    print(DATA_4CH_RADII[2].shape)
    # Channel order is actually SOX17 (Y) - SOX2 (C) -  TBXT (M) - LMBR (W)
    # Should now be LMBR-TBXT-SOX17-SOX2
    CH = 3
    # plt.imshow(onp.clip(DATA_4CH_RADII[5][:,:,CH],0,1),cmap="gray") 
    for _D in range(len(DATA_4CH_RADII)):
        plt.figure(figsize=(2,2),dpi=100)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        _im = reshape_cmy_cells(DATA_4CH_RADII[_D][:,:,1:4])
        print(f"Data shape: {_im.shape}")
        _w,_h = _im.shape[:2]
        # plt.imshow(_im[:100,_h//2-100:_h//2+100])
        plt.imshow(_im)
        # plt.title(f"Image {_D}")
        plt.show()
    return (CH,)


@app.cell
def _(CH, DATA_4CH_RADII, DATA_9CH_CIRCULAR):
    plt.hist(DATA_9CH_CIRCULAR[2][0,-1,CH].flatten(),bins=100,alpha=0.5,density=True,label="Training data")
    plt.hist(DATA_4CH_RADII[4][:,:,CH].flatten(),bins=100,alpha=0.5,density=True,label="Size data")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    mo.md(r"""
    ## Load training data
    """)
    return


@app.cell(hide_code=True)
def _():
    DATA_9CH_CIRCULAR = {}
    DATA_12CH_CIRCULAR = {}
    BOUNDARY_MASK_9CH_CIRCULAR = {}
    DATA_9CH_CIRCULAR_KO = {}
    DATA_12CH_CIRCULAR_KO = {}

    for _d in [1,2,4,8]:
        _dataset = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=None,
            BATCHES=1,
            DOWNSAMPLE=_d,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        _data = _dataset.data
        CHANNEL_NAMES_12CH_CIRCULAR = list(_dataset.channel_names)
        BOUNDARY_MASK_9CH_CIRCULAR[_d] = _dataset.boundary_mask
        DATA_12CH_CIRCULAR[_d] = _data
        _data_split = [_data[:,:,:4],_data[:,:,7:]]
        DATA_9CH_CIRCULAR[_d] = np.concatenate(_data_split,axis=2)

        _ko_9ch = {}
        _ko_12ch = {}
        for _kn in [0,24]:
            _ko_dataset = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
                impath=DATA_PATH_GROUPED,
                FILTER_KN_TIME=_kn,
                BATCHES=1,
                DOWNSAMPLE=_d,
                TIMESTEPS=[0,12,24,36,48],
                PROCESSING_MODES=(
                    "map_to_0_1",
                    "downsample"
                )
            )
            _ko = _ko_dataset.data
            _ko_12ch[_kn] = _ko
            _ko_9ch_split = [_ko[:,:,:4],_ko[:,:,7:]]
            _ko_9ch[_kn] = np.concatenate(_ko_9ch_split,axis=2)

        DATA_12CH_CIRCULAR_KO[_d] = _ko_12ch
        DATA_9CH_CIRCULAR_KO[_d] = _ko_9ch

    CHANNEL_NAMES_9CH_CIRCULAR = CHANNEL_NAMES_12CH_CIRCULAR[:4] + CHANNEL_NAMES_12CH_CIRCULAR[7:]
    return (
        BOUNDARY_MASK_9CH_CIRCULAR,
        CHANNEL_NAMES_12CH_CIRCULAR,
        CHANNEL_NAMES_9CH_CIRCULAR,
        DATA_12CH_CIRCULAR,
        DATA_9CH_CIRCULAR,
        DATA_9CH_CIRCULAR_KO,
    )


@app.cell(hide_code=True)
def _(DOWNSAMPLES):
    DATA_IND_TRIANGLE = {}
    X0_TRIANGLE = {}
    BOUNDARY_IND_TRIANGLE = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="triangle")
        # _,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=8,shape="triangle")
        # if _d!=8:
        #     _x0 = sp.ndimage.zoom(_x0,(1,8/_d,8/_d),order=3)
        #     _boundary = sp.ndimage.zoom(_boundary,(1,8/_d,8/_d),order=0)

        DATA_IND_TRIANGLE[_d] = _data
        BOUNDARY_IND_TRIANGLE[_d] = _boundary
        X0_TRIANGLE[_d] = _x0
    return BOUNDARY_IND_TRIANGLE, DATA_IND_TRIANGLE, X0_TRIANGLE


@app.cell(hide_code=True)
def _(DOWNSAMPLES):
    DATA_IND_ELLIPSE = {}
    X0_ELLIPSE = {}
    BOUNDARY_IND_ELLIPSE = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="ellipse")
        # _,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=8,shape="ellipse")
        # if _d!=8:
            # _x0 = sp.ndimage.zoom(_x0,(1,8/_d,8/_d),order=3)
            # _boundary = sp.ndimage.zoom(_boundary,(1,8/_d,8/_d),order=0)

        DATA_IND_ELLIPSE[_d] = _data
        BOUNDARY_IND_ELLIPSE[_d] = _boundary
        X0_ELLIPSE[_d] = _x0
    return BOUNDARY_IND_ELLIPSE, DATA_IND_ELLIPSE, X0_ELLIPSE


@app.cell(hide_code=True)
def _(DOWNSAMPLES):
    X0_NO_BOUNDARY = {}
    BOUNDARY_IND_NO_BOUNDARY = {}
    for _d in DOWNSAMPLES:
        # _data,_,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="full")
        _,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="full")
        # if _d!=4:
        #     _x0 = sp.ndimage.zoom(_x0,(1,8/_d,8/_d),order=3)
        #     _boundary = sp.ndimage.zoom(_boundary,(1,8/_d,8/_d),order=0)

        # DATA_IND_NO_BOUNDARY[_d] = _data
        BOUNDARY_IND_NO_BOUNDARY[_d] = _boundary
        X0_NO_BOUNDARY[_d] = _x0
    return BOUNDARY_IND_NO_BOUNDARY, X0_NO_BOUNDARY


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(DOWNSAMPLES, RADII):
    DATA_IND_CIRCLE_RADII = {}
    X0_CIRCLE_RADII = {}
    BOUNDARY_IND_CIRCLE_RADII = {}
    for _d in DOWNSAMPLES:
        _DR = {}
        _XR = {}
        _BR = {}
        for _r in RADII:
            _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="circle",radius=_r)
            # _,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=4,shape="circle",radius=_r)
            # if _d!=4:
            #     _x0 = sp.ndimage.zoom(_x0,(1,8/_d,8/_d),order=3)
            #     _boundary = sp.ndimage.zoom(_boundary,(1,8/_d,8/_d),order=0)
            _DR[_r] = _data
            _XR[_r] = _x0
            _BR[_r] = _boundary
        DATA_IND_CIRCLE_RADII[_d] = _DR
        X0_CIRCLE_RADII[_d] = _XR
        BOUNDARY_IND_CIRCLE_RADII[_d] = _BR
    return BOUNDARY_IND_CIRCLE_RADII, X0_CIRCLE_RADII


@app.cell
def _():
    # print(BOUNDARY_IND_TRIANGLE[8].shape)
    # print(X0_CIRCLE_RADII[8][RADII[6]].shape)
    return


@app.cell
def _():
    # print(DATA_12CH_CIRCULAR[2].shape)
    return


@app.cell
def _():
    mo.md(r"""
    # Visualise data for thesis chapter figures
    """)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR_KO):
    visualise_single_from_tensor(DATA_9CH_CIRCULAR_KO[4][0][0],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    visualise_single_from_tensor(DATA_9CH_CIRCULAR[4][0],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="Data",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_12CH_CIRCULAR, DATA_12CH_CIRCULAR):
    visualise_single_from_tensor(DATA_12CH_CIRCULAR[4][0],CHANNEL_NAMES_12CH_CIRCULAR,tres=1,title="",NCHANNELS=12)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    visualise_cell_fate(DATA_9CH_CIRCULAR[2][0],CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="signal")
    return


@app.function(hide_code=True)
def plot_data_48h_ind(trajectory,chnames,tres=1,title="",NCHANNELS=9):
    """
        Shows just 1 trajectory. Expects single trajectory as a tensor of T C X Y, and int tres
        describing how to downsample the trajectory in time.
    """
    trajectory = trajectory[-1]
    trajectory = rearrange(trajectory[:NCHANNELS],"C x y -> (x) (C y)")
    trajectory = onp.clip(trajectory,a_max=1.0,a_min=0.0)
    xw,yw = trajectory.shape

    # plt.figure(figsize=(24,2),dpi=400)
    fig, ax = plt.subplots(figsize=(2*NCHANNELS,2),dpi=400)
    ax.imshow(trajectory,cmap="gray")
    # plt.xlabel("Time (hours)")
    # plt.title(title)
    # STEPS = 5

    # plt.xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
    ax.set_yticks([])
    ax.set_xticks(np.linspace(yw/(NCHANNELS*2),yw*(NCHANNELS*2 - 1)/(NCHANNELS*2),NCHANNELS),[ch[2:] for ch in chnames],fontsize=14)
    ax.xaxis.tick_top()

    return plt.gca()


@app.function(hide_code=True)
def plot_channel_group_split(trajectory,chnames,NCHANNELS):
    trajectory = trajectory[-1]
    trajectory = rearrange(trajectory[:NCHANNELS],"C x y -> () C x y")
    if NCHANNELS==9:
        trajectory = duplicate_x_channels_9ch(trajectory)
        trajectory = split_and_pad_by_experiment_groups_12ch(trajectory)
        trajectory = onp.concatenate([trajectory[:,:6],trajectory[:,9:]],axis=1)
    if NCHANNELS==12:
        trajectory = split_and_pad_by_experiment_groups_12ch(trajectory)
    trajectory = rearrange(trajectory[0],"(Cy Cx) x y -> (Cx x) (Cy y)",Cx=3)


    trajectory = onp.clip(trajectory,a_max=1.0,a_min=0.0)
    xw,yw = trajectory.shape

    # plt.figure(figsize=(24,2),dpi=400)
    fig, ax = plt.subplots(figsize=(2*NCHANNELS,2),dpi=400)
    ax.imshow(trajectory,cmap="gray")
    # plt.xlabel("Time (hours)")
    # plt.title(title)
    # STEPS = 5

    # plt.xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xticks(np.linspace(yw/(NCHANNELS*2),yw*(NCHANNELS*2 - 1)/(NCHANNELS*2),NCHANNELS),[ch[2:] for ch in chnames],fontsize=14)
    # ax.xaxis.tick_top()

    return plt.gca()


@app.cell
def _(CHANNEL_NAMES_12CH_CIRCULAR, DATA_12CH_CIRCULAR):
    plot_channel_group_split(DATA_12CH_CIRCULAR[4][0],CHANNEL_NAMES_12CH_CIRCULAR,NCHANNELS=12)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, TS):
    plot_channel_group_split(TS[0],CHANNEL_NAMES_9CH_CIRCULAR,NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_12CH_CIRCULAR, DATA_12CH_CIRCULAR):
    plot_data_48h_ind(DATA_12CH_CIRCULAR[4][0],CHANNEL_NAMES_12CH_CIRCULAR,tres=1,title="",NCHANNELS=12)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, TS):
    plot_data_48h_ind(TS[0],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell(hide_code=True)
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    for _i in range(9):
        plt.xticks([])
        plt.yticks([])
        plt.title(CHANNEL_NAMES_9CH_CIRCULAR[_i][2:],fontsize=32)
        # plt.imshow(onp.clip(TS[0][-1,_i],0,1),cmap="gray")
        plt.imshow(onp.clip(DATA_9CH_CIRCULAR[4][0,-1,_i],0,1),cmap="gray")
        plt.show()
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise fine grained textural details
    """)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    _ds=1
    visualise_cellular_detail(DATA_9CH_CIRCULAR[_ds][0],CHANNEL_NAMES_9CH_CIRCULAR,timestep=4,ch=3,DOWNSAMPLE=_ds)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, TS):
    visualise_cellular_detail(TS[1],CHANNEL_NAMES_9CH_CIRCULAR,timestep=4,ch=3,DOWNSAMPLE=4)
    return


@app.cell
def _(CHANNEL_NAMES_12CH_CIRCULAR):
    print(CHANNEL_NAMES_12CH_CIRCULAR)
    return


@app.cell
def _():
    mo.md(r"""
    # Shape predictions
    """)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_IND_TRIANGLE):
    # print(DATA_IND_ELLIPSE[8][0].shape)
    _tri = DATA_IND_TRIANGLE[4][0]
    print(_tri.shape)
    _factors = onp.array([1.0,2,1.5,1])
    _factors = rearrange(_factors,"c -> () () () c")
    _tri = onp.clip(_tri*_factors,a_max=1.0,a_min=0.0)
    visualise_cell_fate(rearrange(_tri," T x y C -> T C x y"),CHANNEL_NAMES_9CH_CIRCULAR,title = "",cmode="shape_data_composite")
    # visualise_single_from_tensor(rearrange(DATA_IND_TRIANGLE[4][0]," T x y C -> T C x y"),CHANNEL_NAMES_9CH_CIRCULAR[:4],title = "",NCHANNELS=4)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_IND_ELLIPSE):
    visualise_cell_fate(rearrange(DATA_IND_ELLIPSE[4][0]," T x y C -> T C x y"),CHANNEL_NAMES_9CH_CIRCULAR,title = "",cmode="shape_data_composite")
    return


@app.cell
def _(HYPERPARAMETERS_9CH_CIRCULAR, NCA_MODELS, run_nca):
    best_index = 0
    best_nca = NCA_MODELS[best_index]
    _H_BEST = HYPERPARAMETERS_9CH_CIRCULAR[best_index]
    XS_TRI = run_nca(best_nca,shape="triangle",radius=1.0,key=jr.PRNGKey(42))
    XS_ELL = run_nca(best_nca,shape="ellipse",radius=1.0,key=jr.PRNGKey(42))
    XS_FULL= run_nca(best_nca,shape="full",radius=1.0,key=jr.PRNGKey(42))
    # pprint(HYPERPARAMETERS_9CH_CIRCULAR[best_index])
    # print(best_nca)
    return XS_ELL, XS_FULL, XS_TRI, best_nca


@app.cell
def _(
    BOUNDARY_MASK_9CH_CIRCULAR,
    HYPERPARAMETERS_9CH_CIRCULAR,
    NCA_MODELS,
    run_full_trajectory,
):
    def render_video(SHAPE,ncadict,H):
        _TFULL = run_full_trajectory(ncadict,shape=SHAPE,radius=1.0,key=jr.PRNGKey(42))
        _T,_Tcomp,_Tmono = reshape_for_videos(_TFULL,BOUNDARY_MASK_9CH_CIRCULAR[H["downsample"]][0])
        save_videos(_Tcomp,H,title_suffix=f"{SHAPE}_composite",name_func=H_to_filename_nodal)

    best_index = 0
    best_nca = NCA_MODELS[best_index]
    _H_BEST = HYPERPARAMETERS_9CH_CIRCULAR[best_index]

    render_video("triangle",best_nca,_H_BEST)
    render_video("ellipse",best_nca,_H_BEST)
    render_video("full",best_nca,_H_BEST)
    # _TFULL_TRI = run_full_trajectory(_ncadict,shape="triangle",radius=1.0,key=jr.PRNGKey(42))
    # _T,_Tcomp,_Tmono = reshape_for_videos(_TFULL_TRI,BOUNDARY_MASK_9CH_CIRCULAR[_H["downsample"]][0])
    return (best_nca,)


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_TRI):
    visualise_single_from_tensor(XS_TRI,CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_TRI):
    # visualise_cell_fate(XS_TRI,CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="shape_nca_composite")
    visualise_cell_fate(XS_TRI,CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="composite")
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_ELL):
    visualise_single_from_tensor(XS_ELL,CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_ELL):
    visualise_cell_fate(XS_ELL,CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="composite")
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_FULL):
    visualise_single_from_tensor(XS_FULL,CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_FULL):
    visualise_cell_fate(XS_FULL,CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="composite")
    return


@app.cell
def _():
    return


@app.cell
def _():
    mo.md(r"""
    # Radii predictions
    """)
    return


@app.cell
def _():
    RADII = onp.array([120.0,220.0,320.0,420.0,520.0,620.0,720.0,820.0])/500.0
    return (RADII,)


@app.cell
def _(RADII, best_nca, run_nca):
    XS_RADII = []
    for _r in tqdm(RADII):
        _xs = run_nca(best_nca,shape="circle",radius=_r,key=jr.PRNGKey(42))
        XS_RADII.append(_xs)
    return (XS_RADII,)


@app.cell(disabled=True)
def _(RADII_DATA, XS_RADII):

    vgg_loss,l2_loss = build_loss_functions(["vgg","l2"],loss_args=vgg_loss_dict)
    """
        Compute VGG loss directly between NCA and DATA for different micropattern sizes - doesn't work too well, quite noisy

    """
    def compare_data_nca_radii(XS,DS):
        losses = []
        losses_v = []
        losses_l = []
        for x,d in tqdm(zip(XS,DS),total=8):
            x = x[-1,:4]
            d = d[0]
            # size_ratio = x.shape[-1]/d.shape[-1]

            d = scipy.ndimage.zoom(d,(1,128/d.shape[-1],128/d.shape[-1]),order=3)
            x = scipy.ndimage.zoom(x,(1,128/x.shape[-1],128/x.shape[-1]),order=3)

            d = onp.pad(d,((0,2),(0,0),(0,0)))
            x = onp.pad(x,((0,2),(0,0),(0,0)))
            print(f"X shape: {x.shape}")
            print(f"Data shape: {d.shape}")
            # print(f"Size ratio: {size_ratio}")
            loss_v = vgg_loss(x[None],d[None],key=jr.PRNGKey(0),where=None)[0]
            loss_l = l2_loss(x[None],d[None],key=jr.PRNGKey(0),where=None)[0]
            # print(loss_v,loss_l)
            losses.append(loss_v+loss_l)
            losses_v.append(loss_v)
            losses_l.append(loss_l)
            # comb = onp.concatenate([x_zoom,d],axis=1)
            # plt.imshow(rearrange(comb,"C X Y -> X (C Y)"),cmap="gray")
            # plt.show()
        return losses,losses_v,losses_l

    RLOSSES = compare_data_nca_radii(XS_RADII,RADII_DATA)
    return (RLOSSES,)


@app.cell(disabled=True)
def _(RADII, RLOSSES):
    plt.plot(RLOSSES[0])
    plt.xticks(onp.arange(8),[str(int(500*R)) for R in RADII])
    plt.ylabel("VGG+L2 loss")
    plt.show()
    # plt.plot(RLOSSES[1])
    # plt.plot(RLOSSES[2])
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, RADII, XS_RADII):
    _r=7
    print(f"{int(RADII[_r]*500)} micron diameter")
    visualise_cell_fate(XS_RADII[_r],CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="composite")
    return


@app.cell
def _():
    return


@app.function(hide_code=True)
def calculate_radial_average(T,R_res,padding_ratio=1.0,pixel_offsets=[[0,0],[0,0]],PLOT_DEBUG=True):
    """
        Extract radial averages from trajectories.
        Parameters
            T float32 [T C X Y]
                Input data to radially average
            R_res int
                Spatial resolution of radial averaging
            padding_ratio float
                Proportion of width of circular region of interest to full square image. Defaults to 1.0
            pixel_offsets list
                Extra pixels to remove from padding on either side of width or height. Useful for if the padding around the region
                of interest is slightly assymetric
            PLOT_DEBUG bool
                Flags whether to plot a sample of the cropped trajectory to test if the padding_ratio and pixel_offsets are set correctly
        Returns
            T_rad [T C R]
                Radially averaged T. reduces X Y -> R where R=Y*padding_ratio//2
    """
    W_full = T.shape[3]
    H_full = T.shape[2]
    _x_low = int(H_full*(1-padding_ratio)+pixel_offsets[0][0])
    _x_high= int(H_full*padding_ratio+pixel_offsets[0][1])
    _y_low = int(W_full*(1-padding_ratio) + pixel_offsets[1][0])
    _y_high= int(W_full*padding_ratio + pixel_offsets[1][1])
    T_cropped = T[:,:9,_x_low:_x_high,_y_low:_y_high]
    # T_cropped = T[:,:8,16:-18,14:-18]
    if PLOT_DEBUG:
        plt.imshow(T_cropped[-1,0])
        plt.show()
    print(T_cropped.shape)
    W = T_cropped.shape[3]
    H = T_cropped.shape[2]
    def rmask(r):
        x,y = np.meshgrid(np.arange(W),np.arange(H))
        R = np.sqrt((x-W/2)**2+(y-H/2)**2)
        return (R>=r)*(R<(r+1))
    ravs = []
    radii = onp.linspace(0,W//2,R_res)
    for r in radii:
        mask = rmask(r)
        # plt.imshow(mask)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        mask = repeat(mask,"w h -> () () w h")
        rav = reduce(T_cropped*mask,"T C w h -> T C","sum")/np.count_nonzero(mask)
        # print(rav.shape)
        ravs.append(rav)
    ravs = onp.array(ravs)
    ravs = rearrange(ravs,"R T C -> T C R")
    return ravs,radii


@app.cell(hide_code=True)
def _(RADII):
    def size_slices(T,chnames,tres,padding_ratio=0.9):
        """
            Plots radial distributions of different channels at 48h for each micropattern radius
        """
        # plt.plot()
        T_slice = {}
        radii_dict = {}
        # true_radii = {
        #     0.24:220

        # }
        r_res = 200
        for i,R in enumerate(RADII):
            slice = T[i]
            T_slice[R],_rad = calculate_radial_average(
                T=slice,
                R_res=r_res,
                padding_ratio=padding_ratio,
                # padding_ratio=1.0,
                pixel_offsets=[[1,0],[1,0]],
                PLOT_DEBUG=True,
            )
            # print(_rad)
            radii_dict[R] = (_rad/_rad[-1])*R*500 / 2

            # D = slice.shape[-1]
            # T_slice[R] = slice[-1,:8,:D//2,D//2]
            # print(slice.shape)
            # print(T_slice[R].shape)
            # print(radii_dict[R].shape)
            # print(radii_dict[R])
        # plt.plot(T_slice[1.64].T)

        linestyles = ["-","--","-.","-",":","-","-","-","-"]
        # lines_to_plot = [1,2,3,4,5,6,7]
        # lines_to_plot = [1,2,3,4]
        lines_to_plot = [1,2,3]
        # cmap_cellfate = plt.cm.get_cmap("autumnn",8)
        # colors1 = plt.cm.winter(onp.linspace(0,1.0,5))
        # colors2 = plt.cm.autumn(onp.linspace(0,0.8,4))
        # colors = list(onp.vstack((colors1, colors2)))
        colors = [
            onp.array([0,0,0,1]),
            onp.array([1,0,1,1]),
            onp.array([1,0.7,0,1]),
            onp.array([0,0.8,0.8,1]),
            onp.array([0,0,0,1]),
        ]
        print(colors)
        # plt.plot(radii_dict[0.24],T_slice[0.24][-1].T,)
        # R = 0.64
        # plt.p
        fig,axs = plt.subplots(2,4,figsize=(16,8),dpi=400,sharey=True)
        for i,R in enumerate(RADII):
            for ch in lines_to_plot:
                axs[i//4,i%4].plot(
                    radii_dict[R],
                    T_slice[R][-1,ch,::-1].T, # Select last timestep and channel, flip radius so it starts from edge
                    color=colors[ch],
                    label=chnames[ch],
                    linewidth=2,
                    linestyle=linestyles[ch])
                axs[i//4,i%4].set_title(f"{int(R*500)} ")
        axs[0,0].legend(loc="upper left")
        plt.tight_layout()
        return plt.gca()

        # print(T[1.64][::90].shape)
    return (size_slices,)


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, XS_RADII, size_slices):
    size_slices(XS_RADII,chnames=CHANNEL_NAMES_9CH_CIRCULAR,tres=None)
    return


@app.function(hide_code=True)
def compute_radial_errors(T,DATA,RADII,padding_ratio=0.9):
    """
        Comapres radial average distributions of marker concentration at a single timestep between data and NCA for all MP radii
    """
    T_slice_nca = {}
    T_slice_data = {}
    radial_distances = []
    radii_dict = {}
    # true_radii = {
    #     0.24:220

    # }
    print(RADII)
    r_res = 100
    # T = T[:,-1]
    for i,R in enumerate(RADII):
        slice_nca = T[i][-1] # Seelct timestep of nca trajectory
        slice_data = DATA[i][-1]
        slice_nca = rearrange(slice_nca,"C X Y -> () C X Y")
        slice_data = rearrange(slice_data,"C X Y -> () C X Y")
        # print(slice.shape)
        T_slice_nca[R],_rad = calculate_radial_average(
            T=slice_nca,
            R_res=r_res,
            padding_ratio=padding_ratio,
            # padding_ratio=1.0,
            pixel_offsets=[[1,0],[1,0]],
            PLOT_DEBUG=False,
        )
        # radii_dict[R] = (_rad/_rad[-1])*R*500 / 2
        T_slice_data[R],_rad = calculate_radial_average(
            T=slice_data,
            R_res=r_res,
            padding_ratio=padding_ratio,
            # padding_ratio=1.0,
            pixel_offsets=[[1,0],[1,0]],
            PLOT_DEBUG=False,
        )
        def compare_radial_averages(rn,rd):
            # nca is 9 channels, data is 4
            rn = rn[0,:4] # discard batch channel
            rd = rd[0,:4]
            # return reduce((rd[:4]-rn[:4])**2," c R -> c", "mean")
            return reduce(onp.abs((rd-rn))," c R -> c", "mean")
        # print(_rad)
        # print(T_slice_nca[R].shape)
        # print(T_slice_data[R].shape)
        radial_distances.append(compare_radial_averages(T_slice_nca[R],T_slice_data[R]))
        # print(radial_distances[R].shape)
    radial_distances = onp.array(radial_distances)
    RADII_PLOT = onp.array(RADII)*500
    colors = [
        onp.array([0,0,0,1]),
        onp.array([1,0,1,1]),
        onp.array([1,0.7,0,1]),
        onp.array([0,0.8,0.8,1]),
        onp.array([0,0,0,1]),
    ]
    linestyles = ["-","--","-.","-",":","-","-","-","-"]
    plt.figure(figsize=(8,6),dpi=400)
    plt.plot(RADII_PLOT,radial_distances[:,0],label="LMBR",color=colors[0],linestyle=linestyles[0],marker="o")
    plt.plot(RADII_PLOT,radial_distances[:,1],label="TBXT",color=colors[1],linestyle=linestyles[1],marker="o")
    plt.plot(RADII_PLOT,radial_distances[:,2],label="SOX17",color=colors[2],linestyle=linestyles[2],marker="o")
    plt.plot(RADII_PLOT,radial_distances[:,3],label="SOX2",color=colors[3],linestyle=linestyles[3],marker="o")
    plt.xticks(RADII_PLOT,labels=[f"{r*500:.0f}" for r in RADII])
    plt.xlabel("Radii (microns)")
    plt.ylabel("Mean Absolute Difference")
    plt.legend()
    return plt.gca()


@app.function(hide_code=True)
def radial_average(T,ncadict,chnames,R_res=400,downsample=8):

    # ravs,radii = calculate_radial_average(T,R_res,padding_ratio=0.86)
    ravs,radii = calculate_radial_average(T,R_res,padding_ratio=0.85)

    t_cutoff = 4*ncadict["aux"]["timesteps"]
    ravs = ravs[:t_cutoff,:,::-1]
    # ravs_av_per_timestep = np.mean(ravs,axis=2,keepdims=True)
    # ravs_masked = ravs.at[ravs<ravs_av_per_timestep*0.9].set(0)
    # ravs_masked = ravs
    # ravs_pdf = ravs_masked / np.sum(ravs_masked,axis=2,keepdims=True) # Normalise each timestep to a pdf
    # print(ravs_pdf.shape)
    # radii_bc = repeat(radii*2/W,"R -> () () R")*R_res # Broadcast radii and normalise to 0-1
    # print(radii_bc.shape)
    # rmean = reduce(radii_bc*ravs_pdf,"T C R -> T C ()","sum") # Mean
    # rstd = np.sqrt(reduce(ravs_pdf*(radii_bc-rmean)**2,"T C R -> T C ()","sum")) # Standard deviation


    rmins = np.min(ravs,axis=2,keepdims=True)
    rmax = np.max(ravs,axis=2,keepdims=True)

    # ravs = (ravs + rmins) / (rmax+rmins)
    # ravs = ravs_pdf
    cscales = onp.array([1,1,1.1,0.85,1,1,1,1])
    tsteps = np.arange(t_cutoff)
    xaxs = np.arange(len(radii))
    fig,axs = plt.subplots(2,4,sharex=True,sharey=True,figsize=(10,6))
    CONTOURS = []
    for ch,ax in enumerate(axs.reshape(-1)):
        ax.imshow(ravs[:,ch])
        ax.set_title(chnames[ch])

        #--- Plot isolines on top of kymographs
        contours = measure.find_contours(
            onp.array(ravs[:,ch]),
            # onp.mean(ravs[:,ch])*cscales[ch])
            onp.median(ravs[:,ch])*cscales[ch])
        ax.set_xticks([])
        ax.set_yticks([])
        # c = contours[0]
        c = contours[onp.argmax([len(c) for c in contours])]
        CONTOURS.append(c)
        # for c in contours:
        # ax.plot(c[:,1],c[:,0],color="red")
        # ax.plot(rmean[:,i,0],tsteps,color="red")
        # ax.plot(rmean[:,i,0]+rstd[:,i,0],tsteps,color="red",alpha=0.3)
        # ax.plot(rmean[:,i,0]-rstd[:,i,0],tsteps,color="red",alpha=0.3)

    plt.tight_layout()
    plt.show()
    linestyles = ["-","-","-","--","-","-","-","-"]
    lines_to_plot = [1,2,3,4,5,6,7]
    # cmap_cellfate = plt.cm.get_cmap("autumnn",8)
    colors1 = plt.cm.winter(onp.linspace(0,1.0,5))
    colors2 = plt.cm.autumn(onp.linspace(0,0.8,3))
    colors = list(onp.vstack((colors1, colors2)))
    for ch in lines_to_plot:
        plt.plot(
            CONTOURS[ch][:,1],
            -CONTOURS[ch][:,0],
            color=colors[ch],
            label=chnames[ch],
            linewidth=2,
            linestyle=linestyles[ch])
    plt.xticks(onp.linspace(0,400,6),onp.linspace(250,0,6))
    plt.xlabel("Radial distance (microns)")
    plt.yticks(onp.linspace(0,-360,5),["0h","12h","24h","36h","48h"])
    plt.ylabel("Time")
    plt.legend()
    plt.show()


@app.cell
def _(RADII, RADII_DATA, XS_RADII):
    compute_radial_errors(XS_RADII,RADII_DATA,RADII)
    return


@app.cell
def _(NCA_MODELS, run_full_trajectory):
    T_FULL = run_full_trajectory(NCA_MODELS[0],"colony",radius=1.0,key=jr.PRNGKey(42))
    return (T_FULL,)


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, NCA_MODELS, T_FULL):
    radial_average(T=T_FULL,ncadict=NCA_MODELS[0],chnames=CHANNEL_NAMES_9CH_CIRCULAR,downsample=4)
    return


@app.cell
def _():
    return


@app.cell
def _():
    mo.md(r"""
    # No boundary prediction
    """)
    return


@app.cell
def _(NCA_MODELS, run_nca):
    T_NO_BOUND = run_nca(NCA_MODELS[0],"full",radius=1.0,key=jr.PRNGKey(1))
    return (T_NO_BOUND,)


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, T_NO_BOUND):
    visualise_cell_fate(T_NO_BOUND,chnames=CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",cmode="composite")
    return


@app.cell(column=1)
def _():
    mo.md(r"""
    # Run baseline 9ch circular models
    """)
    return


@app.cell
def _():
    DOWNSAMPLES = [4]
    return (DOWNSAMPLES,)


@app.cell
def _(DOWNSAMPLES):


    # HYPERPARAMETERS_9CH_CIRCULAR = {
        # # Common parameters
        # "model":["NCA"],
        # "optimizer":["nadam"],
        # "block_norm":[True],
        # "noise_strength":[0.005],
        # "multistep":[1],
        # "channels":[64],
        # "ott_S":[1024],
        # "ott_D":[4],
        # "learn_rate":[1e-3],
        # "downsample":[4,8],
        # "ott_sharpen":[True],
        # "ott_epsilon":[0.01],
        # "samples":[64],

        # # VGG parameters
        # "loss_mode":["vgg_grouped_and_l2","vgg_grouped"],
        # "metric":["otch","otsp","l2"],
        # "ott_internal_loss_func":["l1"],
        # "ott_K":[5],
        # "loss_normalize":[False],

        # # OTT parameters
        # # "loss_mode":["ott_grouped_and_l2","ott_grouped"],
        # # "ott_internal_loss_func":["l1","l2"],
        # # "metric":["l2"],
        # # "ott_K":[7,5,3,1],
        # # "loss_normalize":[False],

        # # Clip parameters
        # # "loss_mode":["clip_grouped_and_l2"],
        # # "ott_internal_loss_func":["l2"],
        # # "metric":["l2","l1"],
        # # "ott_K":[1],
        # # "loss_normalize":[False,True],

        # "intermediate_growth":[1.0],
        # "boundary_reg":[5.0],
        # "contiguous_growth":[0.0],
        # # Parameters variants for knockout fine tuning
        # # "TRAINING_ITERATIONS": [10,100,1000,5000],
        # # "knockout":[0,24], # or None for baseline model training
        # # "finetune_lr":[1e-4,1e-5],
        # # Parameter variants for baseline
        # "TRAINING_ITERATIONS": [8000],
        # "knockout":[None], # or None for baseline model training
        # "finetune_lr":[1e-4],
    # }

    HYPERPARAMETERS_9CH_CIRCULAR = {
        # "loss_mode":["ott_grouped_and_l2","vgg_grouped_and_l2"],

        "model":["NCA"],
        "optimizer":["nadam"],
        "block_norm":[True],
        "noise_strength":[0.005],
        "multistep":[1],
        "channels":[64],
        "ott_S":[512],
        # "ott_D":[4],
        "learn_rate":[1e-3],
        "downsample":DOWNSAMPLES,
        "ott_sharpen":[True],
        "ott_epsilon":[0.01],
        "samples":[64],
        "stepsize_scaling":["diffusive"],
        "steps_at_ds8":[32],


        "loss_mode":["vgg_grouped_and_l2"],  
        "metric":["l2"],
        "ott_internal_loss_func":["l2"],
        "ott_K":[5],
        "loss_normalize":[False],


        # "loss_mode":["ott_grouped"],
        # "ott_internal_loss_func":["l2"],
        # "ott_K":[5],
        # "metric":["l2"],
        # "loss_normalize":[False],

        # "loss_mode":["clip_grouped_and_l2","clip_grouped"],
        # "ott_internal_loss_func":["l2"],
        # "metric":["l2","l1"],
        # "ott_K":[1],
        # "loss_normalize":[False,True],

        # "loss_mode":["l2_grouped"],
        # "metric":["l2"],
        # "ott_internal_loss_func":["l2"],
        # "ott_K":[5],
        # "loss_normalize":[False],

        "intermediate_growth":[1.0],
        "boundary_reg":[5.0],
        "contiguous_growth":[0.0],
        "TRAINING_ITERATIONS": [8000],
        "knockout":[None], # or None for baseline model training
        "finetune_lr":[1e-4],

        # "TRAINING_ITERATIONS": [10,100,1000,5000],
        # "knockout":[0,24], # or None for baseline model training
        # "finetune_lr":[1e-4,1e-5],
    }

    HYPERPARAMETERS_9CH_CIRCULAR = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_9CH_CIRCULAR)
    return (HYPERPARAMETERS_9CH_CIRCULAR,)


@app.cell
def _(
    BOUNDARY_MASK_9CH_CIRCULAR,
    HYPERPARAMETERS_9CH_CIRCULAR,
    load_and_run_nca,
    run_full_trajectory,
):
    TS = []
    NCA_MODELS = []
    SAVE_VIDEOS = True
    for _H in tqdm(HYPERPARAMETERS_9CH_CIRCULAR):
        _xs,_ncadict, = load_and_run_nca(_H,jr.PRNGKey(42))
        print(_ncadict["aux"])

        TS.append(_xs)
        NCA_MODELS.append(_ncadict)
        if SAVE_VIDEOS:
            _TFULL = run_full_trajectory(_ncadict,shape="colony",radius=1.0,key=jr.PRNGKey(42))
            _T,_Tcomp,_Tmono = reshape_for_videos(_TFULL,BOUNDARY_MASK_9CH_CIRCULAR[_H["downsample"]][0])
            # save_videos(_T,_H,title_suffix="full",fps=5)
            save_videos(_Tcomp,_H,title_suffix="composite",fps=5)
            save_videos(_Tmono,_H,title_suffix="monochrome",fps=5)
    return NCA_MODELS, TS


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, HYPERPARAMETERS_9CH_CIRCULAR, TS):
    _i = 0
    pprint(HYPERPARAMETERS_9CH_CIRCULAR[_i])
    visualise_single_from_tensor(TS[_i],CHANNEL_NAMES_9CH_CIRCULAR,title="NCA model")
    # pprint(HYPERPARAMETERS_9CH_CIRCULAR[5])
    return


@app.cell
def _():
    return


@app.cell
def _(
    CHANNEL_NAMES_9CH_CIRCULAR,
    HYPERPARAMETERS_9CH_CIRCULAR,
    NCA_MODELS,
    TS,
):

    _i=0
    # pprint(HYPERPARAMETERS_9CH_CIRCULAR[_i])
    print(HYPERPARAMETERS_9CH_CIRCULAR[_i]["loss_mode"])
    print(HYPERPARAMETERS_9CH_CIRCULAR[_i]["stepsize_scaling"])
    print(HYPERPARAMETERS_9CH_CIRCULAR[_i]["steps_at_ds8"])
    print("Steps between images:", NCA_MODELS[_i]["aux"]["timesteps"])
    print("Shape of trajectory",TS[_i].shape)
    visualise_cell_fate(TS[_i],CHANNEL_NAMES_9CH_CIRCULAR,title="",cmode="composite")
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    visualise_cell_fate(DATA_9CH_CIRCULAR[2][0],CHANNEL_NAMES_9CH_CIRCULAR,cmode="composite")
    return


@app.cell
def _():
    mo.md(r"""
    # Spatial resolution hyperparameters
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_RESOLUTION = {

        "model":["NCA"],
        "optimizer":["nadam"],
        "block_norm":[True],
        "noise_strength":[0.005],
        "multistep":[1],
        "channels":[64],
        "ott_S":[512],
        # "ott_D":[4],
        "learn_rate":[1e-3],
        "downsample":[8,4],
        # "downsample":[8,4,2],
        "ott_sharpen":[True],
        "ott_epsilon":[0.01],
        "samples":[64],
        "stepsize_scaling":["diffusive","convective"],
        # "stepsize_scaling":["diffusive"],
        "steps_at_ds8":[32,64],

        "loss_mode":["vgg_grouped_and_l2"],  
        # "metric":["otch","otsp","l2"],
        "metric":["l2"],
        "ott_internal_loss_func":["l1"],
        "ott_K":[5],
        "loss_normalize":[False],

        "intermediate_growth":[1.0],
        "boundary_reg":[5.0],
        "contiguous_growth":[0.0],
        "TRAINING_ITERATIONS": [8000],
        "knockout":[None], # or None for baseline model training
        "finetune_lr":[1e-4],

    }
    HYPERPARAMETERS_RESOLUTION = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_RESOLUTION)

    def builld_resolution_models(HYPERPARAMETERS_RESOLUTION):
        models = []
        unique_pairs = []

        for H in HYPERPARAMETERS_RESOLUTION:
            nca,aux = load_9ch_nodal_modal(H)
            ncadict = {"model":nca,"aux":aux,"H":H}
            identifier = (
                aux["timesteps"],
                H["downsample"],
            )
            # print(identifier)
            #Only if identifier is unique do we add an NCA - there is some double counting in the hyperparameters

            if identifier in unique_pairs:
                # print("Duplicate identifier found, skipping model")

                continue
            models.append(ncadict)
            unique_pairs.append(identifier)
            print(identifier)
        return models
    NCA_MODELS_RES = builld_resolution_models(HYPERPARAMETERS_RESOLUTION)
    return (NCA_MODELS_RES,)


@app.cell
def _(NCA_MODELS_RES):
    print(len(NCA_MODELS_RES))
    return


@app.cell
def _():
    mo.md(r"""
    ## Run models on different spatial and timestep resolutions
    - takes about 6 minutes
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(NCA_MODELS_RES, RADII, run_nca):
    RESMODE = "EXTRAPOLATE"
    TS_RES = []
    _key = jr.PRNGKey(42)
    for _ncadict in tqdm(NCA_MODELS_RES):
        _key = jr.fold_in(_key,1)
        if RESMODE=="EXTRAPOLATE":
            _xs = run_nca(_ncadict,shape="circle",radius=RADII[-1],key=_key)
        else:
            _xs = run_nca(_ncadict,shape="colony",radius=1.0,key=_key)
        _timesteps = _ncadict["aux"]["timesteps"]
        _downsample = _ncadict["H"]["downsample"]
        print(_downsample)
        print(_timesteps)
        _labeled_trajectory = {
            "timesteps":_timesteps,
            "downsample":_downsample,
            "trajectory":_xs
        }
        TS_RES.append(_labeled_trajectory)
    return RESMODE, TS_RES


@app.cell
def _(TS_RES):
    for _t in TS_RES:
        print(_t["trajectory"].shape)
    return


@app.cell
def _(RADII_DATA):
    print(RADII_DATA[-1].shape)
    plt.imshow(scipy.ndimage.zoom(RADII_DATA[-1][0],zoom=(1,205/1706.0,205/1706.0)))
    return


@app.cell
def _(NCA_MODELS_RES):
    for _ncadict in tqdm(NCA_MODELS_RES):
        print(_ncadict["aux"])
    return


@app.cell
def _(RESMODE):
    cpu_device = jax.devices('cpu')[0]
    if RESMODE=="EXTRAPOLATE":
        vgg_group_loss,l2_group_loss = build_loss_functions(["vgg","l2"],loss_args=vgg_loss_dict)

    else:    
        vgg_group_loss,l2_group_loss = build_loss_functions(["vgg_grouped","l2_grouped"],loss_args=vgg_loss_dict)

    vgg_group_loss = eqx.filter_jit(vgg_group_loss,device=cpu_device)
    l2_group_loss = eqx.filter_jit(l2_group_loss,device=cpu_device)
    return l2_group_loss, vgg_group_loss


@app.cell
def _(l2_group_loss, vgg_group_loss):
    def compare_resolution_scalings_large(TS_RES,DATA_4CH):
        losses = []
        for i in range(len(TS_RES)):
            TS = TS_RES[i]["trajectory"][-1]  
            DS = TS_RES[i]["downsample"]
            timesteps = TS_RES[i]["timesteps"]

            # data = DATA_12CH_CIRCULAR[DS][0][T]
            zoomfactor = TS.shape[-1]/DATA_4CH.shape[-1]
            data = scipy.ndimage.zoom(DATA_4CH,zoom=(1,1,zoomfactor,zoomfactor)) # has shape [1,4,W,H]
            # print(data.shape)
            TS = rearrange(TS,"C X Y -> () C X Y")[:,:4]
            # print(data.shape)
            print(TS.shape)
            where = onp.ones((1,4,1,1))
            # loss_vgg = onp.array(
                # [vgg_group_loss(x[None],d[None],key=jr.PRNGKey(0),where=None)[0] for x,d in zip(TS,data)]
            # )

            # print([x.shape for x in TS])
            # print([(x[None].shape,d[None].shape) for x,d in zip(TS,data)])
            loss_vgg = onp.array(vgg_group_loss(TS,data,key=jr.PRNGKey(0),where=where))
            loss_l2 = onp.array(l2_group_loss(TS,data,key=jr.PRNGKey(0),where=where))
            loss_dict = {
                "vgg":loss_vgg,
                "l2":loss_l2,
                "total":loss_vgg+loss_l2,
                "downsample":DS,
                "timesteps":timesteps
            }
            losses.append(loss_dict)
        return losses
    return (compare_resolution_scalings_large,)


@app.cell
def _(RADII_DATA, TS_RES, compare_resolution_scalings_large):
    res_losses_large = compare_resolution_scalings_large(TS_RES,RADII_DATA[-1])
    return (res_losses_large,)


@app.cell
def _(res_losses_large):
    pprint(res_losses_large)

    def plot_resolution_loss_large(losses):
        def select_loss(losses,downsample):
            # selects losses based on downsample, and sorts them by timestep
            totals = []
            vggs = []
            l2s = []
            # for loss in losses:
            loss = [l for l in losses if l["downsample"]==downsample] # select by downsample
            loss = sorted(loss,key=lambda x:x["timesteps"]) # sort by timesteps
            totals.append(onp.array([l["total"] for l in loss]))
            vggs.append(onp.array([l["vgg"] for l in loss]))
            l2s.append(onp.array([l["l2"] for l in loss]))
            totals = onp.array(totals)
            vggs = onp.array(vggs)
            l2s = onp.array(l2s)
            totals = reduce(totals,"ts T () -> T","mean")
            # print(totals.shape)
            vggs = reduce(vggs,"ts T () -> T","mean")
            l2s = reduce(l2s,"ts T () -> T","mean")
            return totals,vggs,l2s
        losses_d8,losses_vgg_d8,losses_l2_d8 = select_loss(losses,8)
        losses_d4,losses_vgg_d4,losses_l2_d4 = select_loss(losses,4)
        colors = ["black","gray","green"]

        fig,ax = plt.subplots(figsize=(2,6),dpi=300)

        plt.bar([2,3],losses_d8,color=colors[0])
        plt.bar([2,3],losses_vgg_d8,color =colors[1])
        plt.bar([0,1],losses_d4[:2],color=colors[0])
        plt.bar([0,1],losses_vgg_d4[:2],color =colors[1])
        plt.yticks([])
        plt.legend(["Total Loss","VGG Loss"])
        ax.set_xticks([2,3,0,1],[r"$\Delta=8,t=32$",r"$\Delta=8,t=64$",r"$\Delta=4,t=64$",r"$\Delta=4,t=128$"], rotation='vertical',fontsize=13)
        # plt.bar([1,2],losses_l2_d4,color=colors[2])
        # plt.bar([3,4],losses_l2_d8,color=colors[2])
        return plt.gca()
    plot_resolution_loss_large(res_losses_large)
    return


@app.cell
def _():
    return


@app.cell
def _(l2_group_loss, vgg_group_loss):
    def compare_resolution_scalings(TS_RES,DATA_12CH_CIRCULAR,T=-1):

        losses = []
        for i in range(len(TS_RES)):
            TS = TS_RES[i]["trajectory"][T]  
            DS = TS_RES[i]["downsample"]
            timesteps = TS_RES[i]["timesteps"]

            data = DATA_12CH_CIRCULAR[DS][0][T]
            data = rearrange(data,"C X Y -> () C X Y")
            TS = rearrange(TS,"C X Y -> () C X Y")
            print(data.shape)
            print(TS.shape)
            where = onp.ones((1,9,1,1))
            # loss_vgg = onp.array(
                # [vgg_group_loss(x[None],d[None],key=jr.PRNGKey(0),where=None)[0] for x,d in zip(TS,data)]
            # )

            # print([x.shape for x in TS])
            # print([(x[None].shape,d[None].shape) for x,d in zip(TS,data)])
            loss_vgg = onp.array(vgg_group_loss(TS,data,key=jr.PRNGKey(0),where=where))
            loss_l2 = onp.array(l2_group_loss(TS,data,key=jr.PRNGKey(0),where=where))
            loss_dict = {
                "vgg":loss_vgg,
                "l2":loss_l2,
                "total":loss_vgg+loss_l2,
                "downsample":DS,
                "timesteps":timesteps
            }
            losses.append(loss_dict)

            # losses.append(onp.mean((TS-data)**2))
        return losses
    return (compare_resolution_scalings,)


@app.cell
def _(DATA_12CH_CIRCULAR, TS_RES, compare_resolution_scalings):
    res_losses = []
    for _T in [0,1,2,3,4]:
        res_losses.append(compare_resolution_scalings(TS_RES,DATA_12CH_CIRCULAR,T=_T))
    return (res_losses,)


@app.cell
def _(DATA_12CH_CIRCULAR, TS_RES, res_losses):
    def plot_resolution_scaling_loss(TS_RES,DATA_12CH_CIRCULAR,losses):

        def select_loss(losses,downsample):
            # selects losses based on downsample, and sorts them by timestep
            totals = []
            vggs = []
            l2s = []
            for loss in losses:
                loss = [l for l in loss if l["downsample"]==downsample] # select by downsample
                loss = sorted(loss,key=lambda x:x["timesteps"]) # sort by timesteps
                totals.append(onp.array([l["total"] for l in loss]))
                vggs.append(onp.array([l["vgg"] for l in loss]))
                l2s.append(onp.array([l["l2"] for l in loss]))
            totals = onp.array(totals)
            vggs = onp.array(vggs)
            l2s = onp.array(l2s)
            totals = reduce(totals,"ts T () -> T","mean")
            # print(totals.shape)
            vggs = reduce(vggs,"ts T () -> T","mean")
            l2s = reduce(l2s,"ts T () -> T","mean")
            return totals,vggs,l2s
            # return onp.array([l for l,d in zip(losses,[NCA_MODELS_RES[i]["H"]["downsample"] for i in range(len(TS_RES))]) if d==downsample])
        # losses_d8 = select_loss(losses,8)
        # losses_d4 = select_loss(losses,4)
        # losses_d2 = select_loss(losses,2)
        # losses_vgg_d8 = select_loss(losses_vgg,8)
        # losses_vgg_d4 = select_loss(losses_vgg,4)
        # losses_vgg_d2 = select_loss(losses_vgg,2)
        # losses_l2_d8 = select_loss(losses_l2,8)
        # losses_l2_d4 = select_loss(losses_l2,4)
        # losses_l2_d2 = select_loss(losses_l2,2)
        losses_d8,losses_vgg_d8,losses_l2_d8 = select_loss(losses,8)
        losses_d4,losses_vgg_d4,losses_l2_d4 = select_loss(losses,4)
        losses_d2,losses_vgg_d2,losses_l2_d2 = select_loss(losses,2)
        print(losses_d8)
        print(losses_d4)
        print(losses_d2)
        ts_d8 = [32,64]
        ts_d4 = [64,128,256]
        ts_d2 = [128,256,512,1024]
        # Plot losses as a function of time sampling. There will be multliple lines of different length for different downsampling
        plt.figure(figsize=(8,6),dpi=300)
        colors = ["red","orange","blue"]
        plt.plot(ts_d8,losses_d8,label=r"$\Delta = 8$",marker="o",color=colors[0])
        plt.plot(ts_d4,losses_d4,label=r"$\Delta = 4$",marker="o",color=colors[1])
        plt.plot(ts_d2,losses_d2,label=r"$\Delta = 2$",marker="o",color=colors[2])
        plt.plot(ts_d8,losses_vgg_d8,marker="o",color=colors[0],linestyle="dashed")
        plt.plot(ts_d4,losses_vgg_d4,marker="o",color=colors[1],linestyle="dashed")
        plt.plot(ts_d2,losses_vgg_d2,marker="o",color=colors[2],linestyle="dashed")
        plt.plot(ts_d8,losses_l2_d8,marker="o",color=colors[0],linestyle="dotted")
        plt.plot(ts_d4,losses_l2_d4,marker="o",color=colors[1],linestyle="dotted")
        plt.plot(ts_d2,losses_l2_d2,marker="o",color=colors[2],linestyle="dotted")
        plt.plot([],color="black",label="Total Loss")
        plt.plot([],linestyle="dashed",color="black",label="VGG Loss")
        plt.plot([],linestyle="dotted",color="black",label="L2 Loss")

        plt.xlabel("t")
        plt.ylabel("Loss")
        # plt.xticks([32,64,128,256,512,1024])


        plt.legend()
        # plt.loglog()
        plt.semilogx()

        return plt.gca()
    plot_resolution_scaling_loss(TS_RES,DATA_12CH_CIRCULAR,res_losses)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR, NCA_MODELS_RES, TS_RES):
    def plot_resolution_scalings(TS_RES,DATA_9CH_CIRCULAR,CHANNEL_NAMES_9CH_CIRCULAR):

        for i in range(len(TS_RES)):
            TS = TS_RES[i]["trajectory"]
            DS = NCA_MODELS_RES[i]["H"]["downsample"]
            data = DATA_9CH_CIRCULAR[DS][0][0]
            title = f"Downsample {DS}x {TS_RES[i]['timesteps']}"
            yield visualise_cell_fate(TS,CHANNEL_NAMES_9CH_CIRCULAR,tres=1,cmode="composite",title="")
    # list(compare_resolution_scalings(TS_RES,DATA_9CH_CIRCULAR,CHANNEL_NAMES_9CH_CIRCULAR))
    for _plt in plot_resolution_scalings(TS_RES,DATA_9CH_CIRCULAR,CHANNEL_NAMES_9CH_CIRCULAR):
        plt.show()
    # print(len(TS_RES))
    # print(TS_RES[0].shape)
    return


@app.cell
def _():
    mo.md(r"""
    ## Testing different resolutions on extrapolation to large MP
    """)
    return


@app.cell
def _():
    # def test_resolutions_large_mp():
    #     xs = run_nca(,shape="circle",radius=_r,key=jr.PRNGKey(42))
    #     XS_RADII.append(_xs)
    return


@app.cell
def _():
    mo.md(r"""
    # Nodal Knockout
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Run nodal knockout models
    """)
    return


@app.cell(hide_code=True)
def _(DOWNSAMPLES):
    HYPERPARAMETERS_9CH_CIRCULAR_KO = {
        # "loss_mode":["ott_grouped_and_l2","vgg_grouped_and_l2"],

        "model":["NCA"],
        "optimizer":["nadam"],
        "block_norm":[True],
        "noise_strength":[0.005],
        "multistep":[1,2,4,8],
        "channels":[64],
        "ott_S":[512],
        # "ott_D":[4],
        "learn_rate":[1e-3],
        "downsample":DOWNSAMPLES,
        "ott_sharpen":[True],
        "ott_epsilon":[0.01],
        "samples":[64],
        # "stepsize_scaling":["diffusive"],
        "stepsize_scaling":["diffusive"],
        "steps_at_ds8":[32],

        # "loss_mode":["vgg_grouped_and_l2","vgg_grouped"],
        "loss_mode":["vgg_grouped_and_l2"],  
        # "metric":["otch","otsp","l2"],
        "metric":["l2"],
        "ott_internal_loss_func":["l1"],
        "ott_K":[5],
        "loss_normalize":[False],

        # "loss_mode":["ott_grouped_and_l2","ott_grouped"],
        # "ott_internal_loss_func":["l1","l2"],
        # "loss_mode":["ott_grouped"],
        # "ott_internal_loss_func":["l2","l1"],
        # "ott_K":[7,5,3,2],
        # "metric":["l2"],
        # # "ott_K":[3],
        # "loss_normalize":[False],

        # "loss_mode":["clip_grouped_and_l2","clip_grouped"],
        # "ott_internal_loss_func":["l2"],
        # "metric":["l2","l1"],
        # "ott_K":[1],
        # "loss_normalize":[False,True],

        # "loss_mode":["l2_grouped"],
        # "metric":["l2"],
        # "ott_internal_loss_func":["l2"],
        # "ott_K":[5],
        # "loss_normalize":[False],

        "intermediate_growth":[1.0],
        "boundary_reg":[5.0],
        "contiguous_growth":[0.0],
        # "TRAINING_ITERATIONS": [8000],
        "knockout":[0,24], # or None for baseline model training
        "knockout_mode":["both"],
        # "finetune_lr":[1e-4],

        # "TRAINING_ITERATIONS": [10,100,1000,5000],
        "TRAINING_ITERATIONS": [1000,5000],
        # "knockout":[0,24], # or None for baseline model training
        "finetune_lr":[1e-4,1e-5,1e-6],
    }

    HYPERPARAMETERS_9CH_CIRCULAR_KO = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_9CH_CIRCULAR_KO)
    return (HYPERPARAMETERS_9CH_CIRCULAR_KO,)


@app.cell
def _(HYPERPARAMETERS_9CH_CIRCULAR_KO, load_and_run_nca):
    NCA_MODELS_KO = []
    TS_KO = []
    for _i,_H in enumerate(HYPERPARAMETERS_9CH_CIRCULAR_KO):
        # try:
        # NCA_MODELS_KO.append(load_9ch_nodal_modal(H=_H))
        if _i in [9]:
        # pprint(_H)
            _xs,_ncadict, = load_and_run_nca(_H,jr.PRNGKey(42))
            TS_KO.append(_xs)
            NCA_MODELS_KO.append(_ncadict)
        # except:
            # print(f"Failed to load model for hyperparameters {_H}")
    return NCA_MODELS_KO, TS_KO


@app.cell
def _(NCA_MODELS):
    _input_weights = NCA_MODELS[0]['model'].layers[0].weight.flatten()
    _hidden_weights= NCA_MODELS[0]['model'].layers[2].weight.flatten()
    _hidden_bias = NCA_MODELS[0]['model'].layers[2].bias.flatten()
    # plt.hist(onp.abs(_input_weights.flatten()),bins=100,color="blue",alpha=0.7,density=True,label="Input Weights",range=(0,0.5))
    # plt.hist(onp.abs(_hidden_weights.flatten()),bins=100,color="orange",alpha=0.7,density=True,label="Hidden Weights",range=(0,0.5))
    # plt.hist(_hidden_bias.flatten(),bins=100,color="red",alpha=0.7,density=True,label="Hidden Bias",range=(-0.5,0.5))

    print(onp.count_nonzero(_input_weights))
    print(_input_weights.shape)
    print(onp.count_nonzero(_hidden_weights))
    print(_hidden_weights.shape)
    # plt.legend()
    # plt.show()
    return


@app.cell
def _():
    _NCA_hyperparameters = {
        "N_CHANNELS":64, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":jr.PRNGKey(1234)
    }
    # STEPS_BETWEEN_IMAGES = int(256 / np.sqrt(H["downsample"]))
    # STEPS_BETWEEN_IMAGES = int(512 / H["downsample"])
    # if H["model"]=="gNCA":
        # nca = gNCA(**NCA_hyperparameters)
    # elif H["model"]=="NCA":
        # nca = NCA(**NCA_hyperparameters)

    _rawnca = NCA(**_NCA_hyperparameters)
    _input_weights = _rawnca.layers[0].weight
    _hidden_weights= _rawnca.layers[2].weight
    _hidden_bias = _rawnca.layers[2].bias
    plt.hist(_input_weights.flatten(),bins=100,color="blue",alpha=0.7,density=True,label="Input Weights",range=(-0.5,0.5))
    plt.hist(_hidden_weights.flatten(),bins=100,color="orange",alpha=0.7,density=True,label="Hidden Weights",range=(-0.5,0.5))
    # plt.hist(_hidden_bias.flatten(),bins=100,color="red",alpha=0.7,density=True,label="Hidden Bias",range=(-0.5,0.5))
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    DATA_9CH_CIRCULAR,
    DATA_9CH_CIRCULAR_KO,
    HYPERPARAMETERS_9CH_CIRCULAR_KO,
    TS_KO,
):
    print(TS_KO[0]["base"].shape)
    ## Potentially good values
    # 1,2, 5,6,12,14,15,18,21,26,29

    # 9, 12, 13, 38

    # _i=9 # for 24h KO with baseline
    _i=0

    KO_TIME_PLOT = HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]["knockout"]
    print(f"Knockout time: {KO_TIME_PLOT} hours")
    print(f"Multistep value: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['multistep']}")
    print(f"Learn rate: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['finetune_lr']}")
    print(f"Training iterations: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['TRAINING_ITERATIONS']}")
    # # pprint(HYPERPARAMETERS_9CH_CIRCULAR_KO[_i])
    # ax = visualise_single_from_tensor(TS_KO[_i]["base"],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    # plt.show()
    # ax = visualise_single_from_tensor(TS_KO[_i]["ko_24"],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    # plt.show()


    # _i=32
    compare_ko_visualise(
        ko_0_t=TS_KO[_i]["ko_0"],
        ko_24_t=TS_KO[_i]["ko_24"],
        base_t=TS_KO[_i]["base"],
        ko_0_data=DATA_9CH_CIRCULAR_KO[4][0][0],
        ko_24_data=DATA_9CH_CIRCULAR_KO[4][24][0],
        base_data=DATA_9CH_CIRCULAR[4][0])
    return (KO_TIME_PLOT,)


@app.cell
def _():
    mo.md(r"""
    ## Test baseline model on KO
    """)
    return


@app.cell
def _(run_nca, run_nca_nodal_block):
    def ko_baseline_test(ncadict,key=jr.PRNGKey(42)):
        XS = run_nca(ncadict,"colony",radius=1.0,key=key)
        XS_KO_0 = run_nca_nodal_block(ncadict,0,shape="colony",radius=1.0,key=key)
        XS_KO_24= run_nca_nodal_block(ncadict,24,shape="colony",radius=1.0,key=key)
        XS = {
            "base":XS,
            "ko_0":XS_KO_0,
            "ko_24":XS_KO_24
        }
        return XS
    return (ko_baseline_test,)


@app.cell
def _(NCA_MODELS, ko_baseline_test):
    KO_BASELINE_TEST = ko_baseline_test(NCA_MODELS[0])
    return (KO_BASELINE_TEST,)


@app.cell
def _(DATA_9CH_CIRCULAR, DATA_9CH_CIRCULAR_KO, KO_BASELINE_TEST):
    compare_ko_visualise(
        ko_0_t=KO_BASELINE_TEST["ko_0"],
        ko_24_t=KO_BASELINE_TEST["ko_24"],
        base_t=KO_BASELINE_TEST["base"],
        ko_0_data=DATA_9CH_CIRCULAR_KO[4][0][0],
        ko_24_data=DATA_9CH_CIRCULAR_KO[4][24][0],
        base_data=DATA_9CH_CIRCULAR[4][0])
    return


@app.cell
def _():
    mo.md(r"""
    ## Compare Baseline and Finetuned NCA on Nodal KO
    """)
    return


@app.cell
def _(DATA_9CH_CIRCULAR, DATA_9CH_CIRCULAR_KO, KO_BASELINE_TEST, TS_KO):
    _i=0
    ko_base_vs_finetine(
        ko_data=DATA_9CH_CIRCULAR_KO[4],
        ko_nca=TS_KO[_i],
        base_nca=KO_BASELINE_TEST,
        base_data=DATA_9CH_CIRCULAR[4],
        mode="full",
        time="0h")
    return


@app.cell
def _():
    mo.md(r"""
    ## KO time interpolation
    """)
    return


@app.cell
def _(
    NCA_MODELS,
    NCA_MODELS_KO,
    run_full_trajectory,
    run_nca,
    run_nca_nodal_block,
    run_nca_nodal_block_full,
):
    USE_KO_MODEL = True
    KO_TIMES = [0,4,8,12,16,20,24,28,32,36,40,44]
    # KO_TIMES = range(0,48,1)

    TS_KO_INTERP = []
    TS_KO_INTERP_FULL = [] # Full time resolution for rendering to video
    _key = jr.PRNGKey(42)
    if USE_KO_MODEL:
        for _kot in tqdm(KO_TIMES):
            _key = jr.fold_in(_key,_kot)
            TS_KO_INTERP.append(run_nca_nodal_block(NCA_MODELS_KO[0],_kot,shape="colony",radius=1.0,key=_key))
            TS_KO_INTERP_FULL.append(run_nca_nodal_block_full(NCA_MODELS_KO[0],_kot,shape="colony",radius=1.0,key=_key))
        TS_KO_INTERP.append(run_nca(NCA_MODELS_KO[0],"colony",radius=1.0,key=jr.PRNGKey(42)))
        TS_KO_INTERP_FULL.append(run_full_trajectory(NCA_MODELS_KO[0],"colony",radius=1.0,key=jr.PRNGKey(42)))
    else:
        for _kot in tqdm(KO_TIMES):
            TS_KO_INTERP.append(run_nca_nodal_block(NCA_MODELS[0],_kot,shape="colony",radius=1.0,key=jr.PRNGKey(42)))
        TS_KO_INTERP.append(run_nca(NCA_MODELS[0],"colony",radius=1.0,key=jr.PRNGKey(42)))
    return KO_TIMES, TS_KO_INTERP, TS_KO_INTERP_FULL


@app.cell
def _(KO_TIMES, TS_KO_INTERP_FULL):
    from Common.save_to_video import save_to_video_mono
    def render_ko_video(TS_KO_INTERP_FULL):
        cl = lambda x:onp.clip(x,a_max=1.0,a_min=0.0)
        m1p1 = lambda x:2*x - 1
        cs = lambda x:onp.concatenate([x[:,1:5],x[:,8:9]],axis=1)
        rs = lambda x:rearrange(x,"T C X Y -> T (C X) Y")
        ko_times = list(map(cs,TS_KO_INTERP_FULL))
        ko_times = list(map(cl,ko_times))
        ko_times = list(map(m1p1,ko_times))
        ko_times = list(map(rs,ko_times))
        imdata = onp.concatenate(ko_times,axis=2)
        xw,yw,zw = imdata.shape
        # print(f"Range of image data before clipping: {imdata.max()} {imdata.min()}")
        # imdata = onp.clip(imdata,0.0,1.0)
        # imdata = (imdata-imdata.min())/(imdata.max()-imdata.min())
        print(f"Range of image data after clipping: {imdata.max()} {imdata.min()}")
        ko_labels = [f"KO {t}h" for t in KO_TIMES]
        ko_labels.append("Baseline")

        print(imdata.shape)
        save_to_video_mono(imdata,f"Videos/ThesisMicropatterns/KO_interpolation.mp4",fps=30,duration=20,SCALE_UP=2,cmap="gray")    
        # plt.figure(figsize=(20,8),dpi=400)
        # plt.imshow(imdata[:,:,0],cmap="gray")
        # plt.yticks(np.linspace(xw/(5*2),xw*(5*2 - 1)/(5*2),5),["TBXT","SOX17","SOX2","FOXA2","LEF1"])
        # plt.xticks(np.linspace(yw/(len(ko_times)*2),yw*(len(ko_times)*2-1)/(len(ko_times)*2),len(ko_times)),ko_labels)
        # return plt.gca()
    render_ko_video(TS_KO_INTERP_FULL)
    return


@app.cell
def _(DATA_9CH_CIRCULAR, DATA_9CH_CIRCULAR_KO, KO_TIMES, TS_KO_INTERP):
    # print(TS_KO_INTERP[0].shape)
    print(list(KO_TIMES))
    def plot_average_intensity_ko_time(TS_KOS,KO_DATA,DATA):
        """
            Calculates the proportions of pixels with various marker co-expressions corresponding to cell types of interest.
            Does this at 48h for each KO time trajectory, and plots them as function of KO
        """
        # Want to measure co-expression of TBXT and FOXA2
        # LMBR, TBXT, SOX17, SOX2, FOXA2
        TS_KOS = onp.array(TS_KOS)
        KO_0h = KO_DATA[4][0][0]
        KO_24h = KO_DATA[4][24][0]
        NO_KO = DATA[4][0]
        # plt.imshow(KO_0h[-1,1], cmap="gray")
        # plt.show()
        print(f"Data shape: {TS_KOS.shape}")
        print(f"Average intensities at 48h: {onp.mean(KO_24h[-1],axis=(1,2))}")
        print(f"SOX2 at 0h: {onp.mean(KO_0h[-1,3])}, SOX2 at 24h: {onp.mean(KO_24h[-1,3])}, SOX2 at no KO: {onp.mean(NO_KO[-1,3])}")
        print(f"SOX17 at 0h: {onp.mean(KO_0h[-1,2])}, SOX17 at 24h: {onp.mean(KO_24h[-1,2])}, SOX17 at no KO: {onp.mean(NO_KO[-1,2])}")
        # intensity_averages = onp.mean(TS_KOS[:,-1],axis=(0,2,3)) # Average intensity of each channel over all time, space and KO time
        intensity_averages = onp.max(NO_KO[-1],axis=(-1,-2))*0.3
        # intensity_averages = onp.max(TS_KOS[:,-1],axis=(0,-1,-2))*0.3
        # intensity_averages = onp.array([0.1,0.3,0.3,0.3,0.3])
        print(f"Average intensity of each channel over all time, space and KO time: {intensity_averages}")
        notochord_nca = []
        notochord_data = []
        endoderm_nca = []
        endoderm_data = []
        mesoderm_nca = []
        mesoderm_data = []
        ratio = []

        def return_celltype_proportions(T,threholds):
            tbxt = T[-1,1]
            sox17 = T[-1,2]
            sox2 = T[-1,3]
            foxa2 = T[-1,4]
            size = T.shape[-1]**2
            tbxt_high = tbxt>threholds[1]
            sox17_high = sox17>threholds[2]
            sox2_high = sox2>threholds[3]
            foxa2_high = foxa2>threholds[4]
            notochord = onp.count_nonzero((tbxt_high & foxa2_high)&(~sox17_high))/float(size)
            endoderm = onp.count_nonzero(sox17_high & foxa2_high)/float(size)
            # mesoderm = onp.count_nonzero(sox2_high & tbxt_high)/float(size)
            mesoderm = onp.count_nonzero(tbxt_high & sox2_high)/float(size)
            return notochord,endoderm,mesoderm


        for ts in TS_KOS:
            # print(ts.shape)
            nca_cells = return_celltype_proportions(ts,intensity_averages)
            notochord_nca.append(nca_cells[0])
            endoderm_nca.append(nca_cells[1])
            mesoderm_nca.append(nca_cells[2])
        ko_0h_cell_type = return_celltype_proportions(KO_0h,intensity_averages)
        ko_24h_cell_type = return_celltype_proportions(KO_24h,intensity_averages)
        no_ko_cell_type = return_celltype_proportions(NO_KO,intensity_averages)
        print(f"KO 0h cell type proportions (notochord, endoderm, mesoderm): {ko_0h_cell_type}")
        # baseline_data_cell_type
            # tbxt = ts[:,1]
            # sox17 = ts[:,2]
            # sox2 = ts[:,3]
            # foxa2 = ts[:,4]
            # # tbxt_foxa2.append((tbxt*foxa2).sum())
            # # sox17_foxa2.append((sox17*foxa2).sum())
            # tbxt_foxa2.append(onp.count_nonzero((tbxt*foxa2)>0.1)/(125*125))
            # sox17_foxa2.append(onp.count_nonzero((sox17*foxa2)>0.1)/(125*125))
            # sox2_tbxt.append(onp.count_nonzero((sox2*tbxt)>0.1)/(125*125))
            # ratio.append(((sox17*foxa2).mean())/((tbxt*foxa2).mean()))
        cols = ["magenta","cyan","green"]
        plt.figure(figsize=(12,4),dpi=400)
        plt.plot(range(24,49),notochord_nca[24:49],label="Notochord (TBXT + FOXA2 - SOX17)",color=cols[0])
        plt.plot(range(24,49),endoderm_nca[24:49],label="Endoderm (SOX17 + FOXA2)",color=cols[1])
        plt.plot(range(24,49),mesoderm_nca[24:49],label="Mesoderm (TBXT)",color=cols[2])
        plt.plot(range(0,25),notochord_nca[:25],color=cols[0],linestyle="dashed")
        plt.plot(range(0,25),endoderm_nca[:25],color=cols[1],linestyle="dashed")
        plt.plot(range(0,25),mesoderm_nca[:25],color=cols[2],linestyle="dashed")
        plt.scatter(0,ko_0h_cell_type[0],color=cols[0],marker="X",s=100)
        plt.scatter(0,ko_0h_cell_type[1],color=cols[1],marker="X",s=100)
        plt.scatter(0,ko_0h_cell_type[2],color=cols[2],marker="X",s=100)
        plt.scatter(24,ko_24h_cell_type[0],color=cols[0],marker="D",s=100)
        plt.scatter(24,ko_24h_cell_type[1],color=cols[1],marker="D",s=100)
        plt.scatter(24,ko_24h_cell_type[2],color=cols[2],marker="D",s=100)
        plt.scatter(48,no_ko_cell_type[0],color=cols[0],marker="o",s=100)
        plt.scatter(48,no_ko_cell_type[1],color=cols[1],marker="o",s=100)
        plt.scatter(48,no_ko_cell_type[2],color=cols[2],marker="o",s=100)
        plt.scatter([],[],color="gray",marker="X",s=100,label="KO at 0h")
        plt.scatter([],[],color="gray",marker="D",s=100,label="KO at 24h")
        plt.scatter([],[],color="gray",marker="o",s=100,label="No KO")
        # plt.ylim(0,0.6)

        # plt.plot(ratio,label="Endoderm/Notochord ratio")
        plt.xlabel("Knockout time (hours)")
        plt.xticks(onp.linspace(0,48,9))
        plt.ylabel("Proportion of high co-expression pixels")
        plt.legend()
        return plt.gca()
    plot_average_intensity_ko_time(TS_KO_INTERP,DATA_9CH_CIRCULAR_KO,DATA_9CH_CIRCULAR)
    return


@app.cell
def _():
    # ko_interp_visualise(TS_KO_INTERP)
    return


@app.cell
def _(
    DATA_9CH_CIRCULAR,
    DATA_9CH_CIRCULAR_KO,
    HYPERPARAMETERS_9CH_CIRCULAR_KO,
    KO_TIME_PLOT,
    TS_KO,
):
    _i=47
    print(f"Knockout time: {KO_TIME_PLOT} hours")
    print(f"Multistep value: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['multistep']}")
    print(f"Learn rate: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['finetune_lr']}")
    print(f"Training iterations: {HYPERPARAMETERS_9CH_CIRCULAR_KO[_i]['TRAINING_ITERATIONS']}")
    compare_ko_visualise(
        ko_0_t=TS_KO[_i]["ko_0"],
        ko_24_t=TS_KO[_i]["ko_24"],
        base_t=TS_KO[_i]["base"],
        ko_0_data=DATA_9CH_CIRCULAR_KO[8][0][0],
        ko_24_data=DATA_9CH_CIRCULAR_KO[8][24][0],
        base_data=DATA_9CH_CIRCULAR[8][0])
    return


@app.cell(column=2)
def _():
    mo.md(r"""
    # Helper functions
    """)
    return


@app.function
def load_9ch_nodal_modal(H):
    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":H["channels"], # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    # STEPS_BETWEEN_IMAGES = int(256 / np.sqrt(H["downsample"]))
    # STEPS_BETWEEN_IMAGES = int(512 / H["downsample"])
    if H["model"]=="gNCA":
        nca = gNCA(**NCA_hyperparameters)
    elif H["model"]=="NCA":
        nca = NCA(**NCA_hyperparameters)
    # FILENAME_BASELINE,FILENAME_KNOCKOUT = H_to_filename_nodal(H)
    dict = H_to_filename_nodal(H)
    FILENAME_BASELINE = dict["base"]
    FILENAME_KNOCKOUT = dict["ko"]
    STEPS_BETWEEN_IMAGES = dict["timesteps"]
    if FILENAME_KNOCKOUT is not None:
        nca = nca.load(f"models/micropattern_individual_9ch/{FILENAME_KNOCKOUT}.eqx")
        print("Loaded fine tuned model for knockout ",H["knockout"])
    else:
        nca = nca.load(f"models/micropattern_individual_9ch/{FILENAME_BASELINE}.eqx")
        print("Loaded baseline model")

    return nca,{"timesteps":STEPS_BETWEEN_IMAGES}


@app.function(hide_code=True)
def get_shaped_data_synthetic(DOWNSAMPLE,shape="triangle",radius=0.28):
    def rmask(r,mask):
        W,H = mask.shape
        x,y = np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
        R = np.sqrt((x)**2+(y)**2)
        return (R>=r)
    def zoom(a, factor):
        a = np.asarray(a)
        slices = [slice(0, old, 1/factor) for old in a.shape]
        idxs = (np.mgrid[slices]).astype('i')
        return a[tuple(idxs)]

    # data_circle,aux,CHANNEL_NAMES,boundary_mask_circle = load_micropattern_circle_8ch_individual(
    #     impath="../Data/Timecourse Individual Images/*",
    #     DOWNSAMPLE = DOWNSAMPLE,
    #     BATCHES=1,
    #     PROCESSING_MODES={
    #         "map_to_0_1",
    #         "downsample"
    #         # "downsample",
    #     }
    # )

    circle_dataset = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
        impath=DATA_PATH_GROUPED,
        FILTER_KN_TIME=None,
        BATCHES=1,
        DOWNSAMPLE=DOWNSAMPLE,
        TIMESTEPS=[0,12,24,36,48],
        PROCESSING_MODES=(
            "map_to_0_1",
            "downsample"
        )
    )
    data_circle_12ch = circle_dataset.data
    aux = circle_dataset.aux
    CHANNEL_NAMES_12CH_CIRCULAR = list(circle_dataset.channel_names)
    boundary_mask_circle = circle_dataset.boundary_mask
    CHANNEL_NAMES_9CH_CIRCULAR = CHANNEL_NAMES_12CH_CIRCULAR[:4] + CHANNEL_NAMES_12CH_CIRCULAR[7:]
    # data_circle = data_circle_12ch
    _data_split = [data_circle_12ch[:,:,:4],data_circle_12ch[:,:,7:]]
    data_circle = np.concatenate(_data_split,axis=2)

    filepath = {
        "triangle":"../Data/micropattern_shapes/Max Projections */*Triangle*",
        "ellipse":"../Data/micropattern_shapes/Max Projections */*Ellipse*",
        "donut":"../Data/micropattern_shapes/Max Projections */*Donut*",  
        "circle":None,
        "full":None,
    }[shape]
    if shape=="circle":
        SHAPED_MASK = zoom(boundary_mask_circle[0][0],factor=radius)
        SHAPED_MASK = 1-rmask(0.8,SHAPED_MASK)
    elif shape=="full":
        SHAPED_MASK = onp.ones(boundary_mask_circle[0][0].shape)
    else:
        SHAPED_MASK = None        

    I = load_micropattern_shape_sequence(
        filepath,
        DOWNSAMPLE=DOWNSAMPLE*2,
        BATCH_AVERAGE=False,
        CIRCLE_DATA=data_circle,
        CIRCLE_HIST_BINS=None,
        CIRCLE_MASK=boundary_mask_circle,
        PROCESSING_MODES=(
            "map_to_0_1",
            "downsample"
        ),
        CHANNELS=CHANNEL_NAMES_9CH_CIRCULAR,
        SHAPED_MASK=SHAPED_MASK

    )
    # print(I)
    data = I.data
    mask = I.masks
    X0 = I.synthetic_initial_conditions
    SHAPE_CHANNEL_NAMES = list(I.channel_names)
    if shape=="donut":
        # hole = onp.ones_like(mask)
        hole = rmask(radius,mask)
        # plt.imshow(hole)
        mask = mask*hole
        # plt.show()
        # hole = hole.at[]

    mask = rearrange(mask,"X Y -> () X Y")
    print(f"Mask {mask.shape}")

    print(f"X0 {X0.shape}")
    X0 = X0*mask
    if data is not None:
        data = data*repeat(mask,"() W H -> () () W H ()")
    print("========================================================")
    print("Channel order",SHAPE_CHANNEL_NAMES)
    print("Desired channel order",CHANNEL_NAMES_9CH_CIRCULAR)
    return data,mask,X0,CHANNEL_NAMES_9CH_CIRCULAR


@app.cell(hide_code=True)
def _(run_nca, run_nca_nodal_block):
    def load_and_run_nca(H,key):
        nca,aux = load_9ch_nodal_modal(H)
        ncadict = {"model":nca,"aux":aux,"H":H}
        XS = run_nca(ncadict,"colony",radius=1.0,key=key)
        if H["knockout"] is not None:
            XS_KO_0 = run_nca_nodal_block(ncadict,0,shape="colony",radius=1.0,key=key)
            XS_KO_24= run_nca_nodal_block(ncadict,24,shape="colony",radius=1.0,key=key)
            XS = {
                "base":XS,
                "ko_0":XS_KO_0,
                "ko_24":XS_KO_24
            }
        return XS,ncadict    
    return (load_and_run_nca,)


@app.cell
def _(get_x0_and_bmask):
    def run_nca(ncadict,shape,radius,key):
        nca = ncadict["model"]
        H = ncadict["H"]
        t = ncadict["aux"]["timesteps"]
        x,bfunc,T = get_x0_and_bmask(H,shape=shape,radius=radius)

        XS = []
        # jnca = eqx.filter_jit(nca)
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            # x = jnca(x=x,boundary_callback=bfunc,key=key)

            # x = eqx.filter_jit(nca)(x,bfunc,key)
            if i%t==0:
                XS.append(x[:9])
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        return XS
    return (run_nca,)


@app.cell
def _(get_x0_and_bmask):
    def run_full_trajectory(ncadict,shape,radius,key):
        nca = ncadict["model"]
        H = ncadict["H"]
        t = ncadict["aux"]["timesteps"]
        x,bfunc,T = get_x0_and_bmask(H,shape=shape,radius=radius)

        XS = []
        # jnca = eqx.filter_jit(nca)
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            # x = jnca(x=x,boundary_callback=bfunc,key=key)

            # x = eqx.filter_jit(nca)(x,bfunc,key)
            # if i%t==0:
            XS.append(onp.array(x[:9]))
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        return XS
    return (run_full_trajectory,)


@app.cell(hide_code=True)
def _(get_x0_and_bmask):
    def run_nca_nodal_block(ncadict,nodal_block_time,shape="colony",radius=1.0,key=jr.PRNGKey(int(time.time()))):
        # nodal_block is in hours
        nca = ncadict["model"]
        H = ncadict["H"]
        t = ncadict["aux"]["timesteps"]
        x,bfunc,T = get_x0_and_bmask(H,shape=shape,radius=radius)

        XS = []

        realtimes = np.linspace(0,48,t*T) # time in hours
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if realtimes[i]>nodal_block_time:
                x = x.at[7].set(0)
            if i%t==0:
                XS.append(x[:9])
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        return XS
    return (run_nca_nodal_block,)


@app.cell
def _(get_x0_and_bmask):
    def run_nca_nodal_block_full(ncadict,nodal_block_time,shape="colony",radius=1.0,key=jr.PRNGKey(int(time.time()))):
        # nodal_block is in hours
        nca = ncadict["model"]
        H = ncadict["H"]
        t = ncadict["aux"]["timesteps"]
        x,bfunc,T = get_x0_and_bmask(H,shape=shape,radius=radius)

        XS = []

        realtimes = np.linspace(0,48,t*T) # time in hours
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if realtimes[i]>nodal_block_time:
                x = x.at[7].set(0)
            # if i%t==0:
            XS.append(x[:9])
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        return XS
    return (run_nca_nodal_block_full,)


@app.function
def reshape_for_videos(T,mask):

    """
    # Takes a trajectory and returns 3 versions
    # - Original (clipped to 0-1)
    # - CMY composite (Horizontal tiles of 3 channel composites)
    # - Monochrome tiled (3 by 3 tiles of each channel)
    Parameters:
    T: Array of shape (T,9,X,Y) representing the trajectory of 9 channels over time
    mask: Array of shape (X,Y) representing the boundary mask for the colony
    """

    mask = repeat(mask,"() X Y -> () () X (3 Y)")
    print(f"Mask shape: {mask.shape}")
    T = onp.clip(T,a_max=1.0,a_min=0.0)
    T_obs = T[:,:9]
    CHANNEL_ORDER = ["LMBR","TBXT","SOX17","SOX2","FOXA2","CER1","LEFTY2","NODAL","LEF1"]
    DESIRED_ORDER = ["SOX2","TBXT","SOX17","CER1","LEFTY2","NODAL","FOXA2","LEF1"]
    order_index = [3,1,2,5,6,7,4,8]
    T_for_comp = onp.zeros_like(T_obs)
    for i,ch in enumerate(order_index):
        T_for_comp[:,i] = T_obs[:,ch]
    # T_for_comp[:,-1] = 0
    T_composite = rearrange(T_for_comp,"T (cx cy cc) X Y -> T cc (cx X) (cy Y)",cx=1,cy=3,cc=3)
    print(f"Composite shape: {T_composite.shape}" )
    # T_composite = (1 - T_composite)*mask
    T_composite_cmy = onp.zeros_like(T_composite)
    T_composite_cmy[:,0] = 0.5*(T_composite[:,1]+T_composite[:,2])
    T_composite_cmy[:,1] = 0.5*(T_composite[:,0]+T_composite[:,2])
    T_composite_cmy[:,2] = 0.5*(T_composite[:,0]+T_composite[:,1])
    T_monochrome = rearrange(T[:,:9],"T (cy cx) X Y -> T () (cx X) (cy Y)",cx=3,cy=3)
    T_monochrome = repeat(T_monochrome,"T () x y -> T 3 x y")
    return T,T_composite_cmy,T_monochrome


@app.function
def save_videos(DATA,H,title_suffix,fps=5,name_func=H_to_filename_nodal):
    # DATA is of shape T C X Y
    DATA = rearrange(DATA,"T C X Y -> T X Y C")
    print(f"Data shape before video rendering: {DATA.shape}")
    FILENAME = "Videos/ThesisMicropatterns/"+name_func(H)["base"]+f"_trajectory_{title_suffix}.mp4"
    save_to_video_rgb(DATA,FILENAME,fps=30,duration=20)


@app.cell(hide_code=True)
def _(
    BOUNDARY_IND_CIRCLE_RADII,
    BOUNDARY_IND_ELLIPSE,
    BOUNDARY_IND_NO_BOUNDARY,
    BOUNDARY_IND_TRIANGLE,
    BOUNDARY_MASK_9CH_CIRCULAR,
    DATA_9CH_CIRCULAR,
    DATA_9CH_CIRCULAR_KO,
    X0_CIRCLE_RADII,
    X0_ELLIPSE,
    X0_NO_BOUNDARY,
    X0_TRIANGLE,
):
    def get_x0_and_bmask(H,shape="colony",radius=1.0):

        DS = H["downsample"]
        KO = H["knockout"]
        CH = H["channels"]
        if shape=="colony":
            if KO is not None:
                x0 = DATA_9CH_CIRCULAR_KO[DS][KO][0,0]
            else:
                x0 = DATA_9CH_CIRCULAR[DS][0,0]
            bmask = BOUNDARY_MASK_9CH_CIRCULAR[DS][0]
        elif shape=="triangle":
            x0 = X0_TRIANGLE[DS]
            bmask = BOUNDARY_IND_TRIANGLE[DS]
        elif shape=="ellipse":
            x0 = X0_ELLIPSE[DS]
            bmask = BOUNDARY_IND_ELLIPSE[DS]
        elif shape=="circle":
            x0 = X0_CIRCLE_RADII[DS][radius]
            bmask = BOUNDARY_IND_CIRCLE_RADII[DS][radius]
        elif shape=="full":
            x0 = X0_NO_BOUNDARY[DS]
            bmask = BOUNDARY_IND_NO_BOUNDARY[DS]
        bfunc = model_boundary(mask=bmask)
        x0 = np.pad(x0,((0,CH-9),(0,0),(0,0)))
        return x0,bfunc,4
    return (get_x0_and_bmask,)


@app.function(hide_code=True)
def visualise_single_from_tensor(trajectory,chnames,tres=1,title="",NCHANNELS=9):
    """
        Shows just 1 trajectory. Expects single trajectory as a tensor of T C X Y, and int tres
        describing how to downsample the trajectory in time.
    """
    trajectory = rearrange(trajectory[::tres,:NCHANNELS],"T C x y -> (C x) (T y)")
    trajectory = onp.clip(trajectory,a_max=1.0,a_min=0.0)
    xw,yw = trajectory.shape
    plt.figure(figsize=(8,12),dpi=400)
    plt.imshow(trajectory,cmap="gray")
    plt.xlabel("Time (hours)")
    plt.title(title)
    STEPS = 5

    plt.xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
    plt.yticks(np.linspace(xw/(NCHANNELS*2),xw*(NCHANNELS*2 - 1)/(NCHANNELS*2),NCHANNELS),chnames)
    return plt.gca()


@app.function(hide_code=True)
def reshape_cmy(x):
    """
        Takes x of shape [W H 3] in RGB format and returns x in CMY format
    """
    x_cmy = onp.zeros_like(x)
    x_cmy[:,:,0] = 0.5*(x[:,:,1]+x[:,:,2])
    x_cmy[:,:,1] = 0.5*(x[:,:,0]+x[:,:,2])
    x_cmy[:,:,2] = 0.5*(x[:,:,0]+x[:,:,1])
    return x_cmy


@app.function(hide_code=True)
def reshape_cmy_cells(x):
    """
        Takes x of shape [W H 3] in RGB format and returns x in CMY format
        Assumes channler order is TBXT (M) SOX17 (Y) SOX2 (C)
    """
    x_cmy = onp.zeros_like(x)
    x_cmy[:,:,1] = 0.5*(x[:,:,1]+x[:,:,2])
    x_cmy[:,:,2] = 0.5*(x[:,:,0]+x[:,:,2])
    x_cmy[:,:,0] = 0.5*(x[:,:,0]+x[:,:,1])
    return x_cmy


@app.function
def visualise_cell_fate(trajectory,chnames,tres=1,title="",cmode="cmy"):


    trajectory = onp.array(trajectory)
    # trajectory[:,1:]=0
    # trajectory[:,:1] = 0
    # trajectory[:,2:] = 0

    if cmode == "cmy":
        trajectory = trajectory[:,1:4]    
        trajectory = repeat(trajectory,"T cmy x y -> T cmy C x y",C=3)
        trajectory_cmy = onp.zeros_like(trajectory)
        trajectory[:,0,2] = 0
        trajectory[:,0,1] = 0
        trajectory[:,1,0] = 0
        trajectory[:,1,2] = 0

        trajectory[:,2,1] = 0
        trajectory[:,2,0] = 0
        trajectory_cmy[:,1] = 0.5*(trajectory[:,1]+trajectory[:,2])
        trajectory_cmy[:,2] = 0.5*(trajectory[:,0]+trajectory[:,2])
        trajectory_cmy[:,0] = 0.5*(trajectory[:,0]+trajectory[:,1])
        trajectory = trajectory_cmy
        trajectory = trajectory / onp.max(trajectory,axis=(0,1,2,3,4),keepdims=True)
        trajectory = rearrange(trajectory[::tres],"T cmy C x y -> (C x) (T y) cmy")

    elif cmode == "signal":
        trajectory = trajectory[:,5:8]    
        trajectory = repeat(trajectory,"T sig x y -> T sig C x y",C=3)
        trajectory_sig = onp.zeros_like(trajectory)
        trajectory[:,0,2] = 0
        trajectory[:,0,1] = 0
        trajectory[:,1,0] = 0
        trajectory[:,1,2] = 0

        trajectory[:,2,1] = 0
        trajectory[:,2,0] = 0
        trajectory_sig[:,1] = 0.5*(trajectory[:,1]+trajectory[:,2])
        trajectory_sig[:,2] = 0.5*(trajectory[:,0]+trajectory[:,2])
        trajectory_sig[:,0] = 0.5*(trajectory[:,0]+trajectory[:,1])
        trajectory = trajectory_sig
        trajectory = trajectory / onp.max(trajectory,axis=(0,1,2,3,4),keepdims=True)
        trajectory = rearrange(trajectory[::tres],"T sig C x y -> (C x) (T y) sig")
    elif cmode == "cmyw":
        # Want TBXT, SOX17, SOX2 and FOXA2 - all the cell fate markers
        trajectory = trajectory[:,1:5]    
        trajectory = repeat(trajectory,"T C x y -> T cmy C x y",cmy=3)
        trajectory_cmy = onp.zeros_like(trajectory)
        trajectory[:,0,2] = 0
        trajectory[:,0,1] = 0
        trajectory[:,1,0] = 0
        trajectory[:,1,2] = 0
        trajectory[:,2,1] = 0
        trajectory[:,2,0] = 0

        trajectory_cmy[:,1] = 0.5*(trajectory[:,1]+trajectory[:,2])
        trajectory_cmy[:,2] = 0.5*(trajectory[:,0]+trajectory[:,2])
        trajectory_cmy[:,0] = 0.5*(trajectory[:,0]+trajectory[:,1])
        # trajectory_cmy[:,3] = 0.5*(trajectory[:,0]+trajectory[:,1]+trajectory[:,2])/3
        trajectory = trajectory_cmy
        trajectory = trajectory / onp.max(trajectory,axis=(0,1,3,4),keepdims=True)
        trajectory = rearrange(trajectory[::tres],"T cmy C x y -> (C x) (T y) cmy")


    elif cmode=="composite":
        trajectory = trajectory[:,1:4]    
        # trajectory = repeat(trajectory,"T cmy x y -> T cmy C x y",C=3)
        trajectory_cmy = onp.zeros_like(trajectory)
        # trajectory[:,0,2] = 0
        # trajectory[:,0,1] = 0
        # trajectory[:,1,0] = 0
        # trajectory[:,1,2] = 0

        # trajectory[:,2,1] = 0
        # trajectory[:,2,0] = 0
        trajectory_cmy[:,1] = 0.5*(trajectory[:,1]+trajectory[:,2])
        trajectory_cmy[:,2] = 0.5*(trajectory[:,0]+trajectory[:,2])
        trajectory_cmy[:,0] = 0.5*(trajectory[:,0]+trajectory[:,1])
        trajectory = trajectory_cmy
        trajectory = trajectory / onp.max(trajectory,axis=(1,2,3),keepdims=True)
        trajectory = rearrange(trajectory[::tres],"T C x y -> (x) (T y) C")

    elif cmode=="shape_nca_composite":
        # Want Sox17 foxa2 and tbxt
        # trajectory = repeat(trajectory,"T cmy x y -> T cmy C x y",C=3)
        trajectory_shape = onp.zeros_like(trajectory)
        trajectory_cmy = onp.zeros_like(trajectory)[:,:3]
        trajectory_shape[:,0] = trajectory[:,2] # Sox17
        trajectory_shape[:,1] = trajectory[:,5] # sox2
        trajectory_shape[:,2] = trajectory[:,1] # Tbxt
        trajectory_cmy[:,2] = 0.5*(trajectory_shape[:,1]+trajectory_shape[:,2])
        trajectory_cmy[:,0] = 0.5*(trajectory_shape[:,0]+trajectory_shape[:,2])
        trajectory_cmy[:,1] = 0.5*(trajectory_shape[:,0]+trajectory_shape[:,1])
        # trajectory_cmy = trajectory_cmy / onp.max(trajectory_cmy,axis=(1,2,3),keepdims=True)
        trajectory = trajectory_cmy
        trajectory = rearrange(trajectory[::tres],"T C x y -> (x) (T y) C")

    elif cmode == "shape_data_composite":
        # Shape data is of form sox17 foxa2 tbxt lmbr
        trajectory = trajectory[:,:3]    
        trajectory_cmy = onp.zeros_like(trajectory)
        # trajectory[:,2] = 0
        # trajectory[:,0,1] = 0
        # trajectory[:,1,0] = 0
        # trajectory[:,1,2] = 0

        # trajectory[:,2,1] = 0
        # trajectory[:,2,0] = 0
        trajectory_cmy[:,2] = 0.5*(trajectory[:,1]+trajectory[:,2]) # Sox17 yellow
        trajectory_cmy[:,0] = 0.5*(trajectory[:,0]+trajectory[:,2]) # sox2 cyan
        trajectory_cmy[:,1] = 0.5*(trajectory[:,0]+trajectory[:,1]) # Tbxt magenta
        trajectory = trajectory_cmy
        trajectory = trajectory / onp.max(trajectory,axis=(1,2,3),keepdims=True)
        trajectory = rearrange(trajectory[::tres],"T C x y -> (x) (T y) C")
    else:
        trajectory = trajectory[:,1:4]    
        trajectory = rearrange(trajectory[::tres],"T C x y -> (C x) (T y)")


    xw,yw = trajectory.shape[:2]
    plt.figure(figsize=(8,8),dpi=300)
    # plt.imshow(trajectory,cmap="gray")
    plt.imshow(trajectory)

    plt.title(title)
    STEPS = 5

    if "composite" in cmode:
        plt.yticks([])
        plt.text(0.01,0.88,"TBXT",color="magenta",fontsize=10,transform=plt.gca().transAxes)
        plt.text(0.08,0.88,"SOX17",color="yellow",fontsize=10,transform=plt.gca().transAxes)
        plt.text(0.16,0.88,"SOX2",color="cyan",fontsize=10,transform=plt.gca().transAxes)
    elif "cmyw" in cmode:
        plt.yticks(np.linspace(xw/8,xw*7/8,4),[ch[2:] for ch in chnames[1:5]])
    elif "signal" in cmode:
        plt.yticks(np.linspace(xw/6,xw*5/6,3),[ch[2:] for ch in chnames[5:8]])
    else:
        plt.yticks(np.linspace(xw/6,xw*5/6,3),[ch[2:] for ch in chnames[1:4]])

    if "shape_data_composite" in cmode:
        plt.xticks([])
    else:
        plt.xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        plt.xlabel("Time (hours)")

    return plt.gca()


@app.function
def visualise_cellular_detail(trajectory,chnames,timestep=0,ch=0,DOWNSAMPLE=2):
    """
        Shows the cellular detail for a single timepoint. Expects trajectory as a tensor of T C X Y, and int timestep describing which timepoint to show.
    """
    traj_t = trajectory[timestep]
    W = traj_t.shape[1]
    H = traj_t.shape[2]
    print(f"Trajectory shape at timestep {timestep}: {traj_t.shape}")
    fig,axs = plt.subplots(3,3,sharex=True,sharey=True,figsize=(6,6),dpi=400)
    d=16//DOWNSAMPLE
    sw = 40//DOWNSAMPLE
    sh = 20//DOWNSAMPLE
    max_int = traj_t[:,W//2-d+sw:W//2+d+sw,H//2-d+sh:H//2+d+sh].max()
    min_int = traj_t[:,W//2-d+sw:W//2+d+sw,H//2-d+sh:H//2+d+sh].min()
    for i in range(9):
        axs[i//3][i%3].imshow(traj_t[i,W//2-d+sw:W//2+d+sw,H//2-d+sh:H//2+d+sh],cmap="gray",vmin=min_int,vmax=max_int)
        axs[i//3][i%3].set_title(chnames[i],fontsize=12)
        axs[i//3][i%3].set_xticks([])
        axs[i//3][i%3].set_yticks([])
    plt.tight_layout()
    return plt.gca()


@app.function(hide_code=True)
def compare_ko_visualise(ko_0_t,ko_24_t,base_t,ko_0_data,ko_24_data,base_data):
    # trajectory = rearrange(trajectory[::tres,:NCHANNELS],"T C x y -> (C x) (T y)")
    # chs = lambda x:opn.concatenate([x[-1,1:3],x[-1,4:5]],axis=0) # Just TBXT, SOX17 and FOXA2
    chs = lambda x:onp.concatenate([x[-1,1:5],x[-1,8:9]],axis=0) # Full KO data
    # ko_0_t = onp.concatenate([ko_0_t[-1,1:5],ko_0_t[-1,8:9]],axis=0)
    # ko_24_t = onp.concatenate([ko_24_t[-1,1:5],ko_24_t[-1,8:9]],axis=0)
    # base_t = onp.concatenate([base_t[-1,1:5],base_t[-1,8:9]],axis=0)
    # ko_0_data = onp.concatenate([ko_0_data[-1,1:5],ko_0_data[-1,8:9]],axis=0)
    # ko_24_data = onp.concatenate([ko_24_data[-1,1:5],ko_24_data[-1,8:9]],axis=0)
    # base_data = onp.concatenate([base_data[-1,1:5],base_data[-1,8:9]],axis=0)
    rs = lambda x:rearrange(x,"C X Y -> (C X) Y")
    # rs_col = lambda x:reshape_cmy_cells(rearrange(x,"C X Y -> X Y C"))
    # reshape_cmy_cells

    ko_0_t = rs(chs(ko_0_t))
    ko_24_t = rs(chs(ko_24_t))
    base_t = rs(chs(base_t))
    base_data = rs(chs(base_data))
    ko_0_data = rs(chs(ko_0_data))
    ko_24_data = rs(chs(ko_24_data))

    # print(ko_0_t.shape)
    # print(ko_24_t.shape)
    # print(base_t.shape)
    # print(base_data.shape)
    # print(ko_0_data.shape)
    # print(ko_24_data.shape)
    # plt.imshow(ko_0_t)

    imdata = onp.concatenate(
        [ko_0_t,ko_0_data,ko_24_t,ko_24_data,base_t,base_data],
        axis=1
    )
    xw,yw = imdata.shape
    imdata = onp.clip(imdata,0,1)
    plt.figure(figsize=(8,8),dpi=400)
    plt.imshow(imdata,cmap="gray")
    plt.yticks(np.linspace(xw/(5*2),xw*(5*2 - 1)/(5*2),5),["TBXT","SOX17","SOX2","FOXA2","LEF1"])
    plt.xticks(np.linspace(yw/(6*2),yw*(6*2-1)/(6*2),6),["KO 0h NCA", "KO 0h Data", "KO 24h NCA", "KO 24h Data", "Base NCA", "Base Data"])
    return plt.gca()


@app.function
def ko_base_vs_finetine(ko_data,ko_nca,base_nca,base_data,mode="full",time="0h"):
    # trajectory = rearrange(trajectory[::tres,:NCHANNELS],"T C x y -> (C x) (T y)")
    ko_0_t = ko_nca["ko_0"]
    ko_24_t = ko_nca["ko_24"]
    ko_base_t = ko_nca["base"]
    base_0_t = base_nca["ko_0"]
    base_24_t = base_nca["ko_24"]
    base_base_t = base_nca["base"]
    ko_0_data = ko_data[0][0]
    ko_24_data = ko_data[24][0]
    base_data = base_data[0]
    # chs = lambda x:onp.concatenate([x[-1,1:3],x[-1,4:5]],axis=0) # Just TBXT, SOX17 and FOXA2
    chs = lambda x:x[-1,1:5] # TBXT, SOX17, SOX2 and FOXA2
    # chs = lambda x:onp.concatenate([x[-1,1:5],x[-1,8:9]],axis=0) # Full KO data
    if mode=="composite":
        rs = lambda x:reshape_cmy_cells(rearrange(x,"C X Y -> X Y C"))
    else:
        rs = lambda x:rearrange(x,"C X Y -> (C X) Y")

    # ko_0_t = rs(onp.concatenate([ko_0_t[-1,1:5],ko_0_t[-1,8:9]],axis=0))
    # ko_24_t = rs(onp.concatenate([ko_24_t[-1,1:5],ko_24_t[-1,8:9]],axis=0))
    # base_0_t = rs(onp.concatenate([base_0_t[-1,1:5],base_0_t[-1,8:9]],axis=0))
    # base_24_t = rs(onp.concatenate([base_24_t[-1,1:5],base_24_t[-1,8:9]],axis=0))
    # ko_0_data = rs(onp.concatenate([ko_0_data[-1,1:5],ko_0_data[-1,8:9]],axis=0))
    # ko_24_data = rs(onp.concatenate([ko_24_data[-1,1:5],ko_24_data[-1,8:9]],axis=0))
    ko_0_t = rs(chs(ko_0_t))
    ko_24_t = rs(chs(ko_24_t))
    base_0_t = rs(chs(base_0_t))
    base_24_t = rs(chs(base_24_t))
    ko_0_data = rs(chs(ko_0_data))
    ko_24_data = rs(chs(ko_24_data))
    ko_base_t = rs(chs(ko_base_t))
    base_base_t = rs(chs(base_base_t))
    base_data = rs(chs(base_data))

    # ko_0_t = onp.concatenate([ko_0_t[-1,1:5],ko_0_t[-1,8:9]],axis=0)
    # ko_24_t = onp.concatenate([ko_24_t[-1,1:5],ko_24_t[-1,8:9]],axis=0)
    # base_t = onp.concatenate([base_t[-1,1:5],base_t[-1,8:9]],axis=0)
    # ko_0_data = onp.concatenate([ko_0_data[-1,1:5],ko_0_data[-1,8:9]],axis=0)
    # ko_24_data = onp.concatenate([ko_24_data[-1,1:5],ko_24_data[-1,8:9]],axis=0)
    # base_data = onp.concatenate([base_data[-1,1:5],base_data[-1,8:9]],axis=0)



    # print(ko_0_t.shape)
    # print(ko_24_t.shape)
    # print(base_t.shape)
    # print(base_data.shape)
    # print(ko_0_data.shape)
    # print(ko_24_data.shape)
    # plt.imshow(ko_0_t)
    if time=="0h":
        imdata = onp.concatenate(
            [ko_0_t,ko_0_data,base_0_t],
            axis=1
        )
        title = "Nodal knockout at 0h"
    elif time=="24h":
        imdata = onp.concatenate(
            [ko_24_t,ko_24_data,base_24_t],
            axis=1
        )
        title = "Nodal knockout at 24h"
    elif time=="base":
        imdata = onp.concatenate(
            [ko_base_t,base_data,base_base_t],
            axis=1
        )
        title = "Baseline (No knockout)"

    xw,yw = imdata.shape[:2]
    imdata = onp.clip(imdata,0,1)
    plt.figure(figsize=(8,8),dpi=400)
    # plt.title("Nodal knockout at 24h - measured at 48h",fontsize=16)
    plt.title(title,fontsize=16)
    # plt.imshow(imdata,cmap="gray")
    # legend_handles = [
    #     plt.Line2D([0], [0], color='magenta', lw=4, label='TBXT'),
    #     plt.Line2D([0], [0], color='yellow', lw=4, label='SOX17'),
    # #     plt.Line2D([0], [0], color='cyan', lw=4, label='FOXA2')]
    # plt.legend(
    #     handles=legend_handles,
    #     loc="upper left",
    #     fontsize=12
    # )
    if mode=="composite":
        plt.imshow(imdata)
        plt.text(0.01,0.9,"TBXT",color="magenta",fontsize=12,transform=plt.gca().transAxes)
        plt.text(0.1,0.9,"SOX17",color="yellow",fontsize=12,transform=plt.gca().transAxes)
        plt.text(0.21,0.9,"FOXA2",color="cyan",fontsize=12,transform=plt.gca().transAxes)
        plt.yticks([])
    # plt.xticks(np.linspace(xw/(5*2),xw*(5*2 - 1)/(5*2),5),["TBXT","SOX17","SOX2","FOXA2","LEF1"])
    else:
        plt.imshow(imdata,cmap="gray")
        plt.yticks(np.linspace(xw/(4*2),xw*(4*2 - 1)/(4*2),4),["TBXT","SOX17","SOX2","FOXA2"])
        # plt.yticks(np.linspace(xw/(3*2),xw*(3*2 - 1)/(3*2),3),["TBXT","SOX17","FOXA2"])

    plt.xticks(np.linspace(yw/(3*2),yw*(3*2-1)/(3*2),3),["24hKO NCA", "Data", "Base NCA"],rotation=0)
    return plt.gca()


@app.cell(hide_code=True)
def _(KO_TIMES):
    def ko_interp_visualise(ko_times):
        cs = lambda x:onp.concatenate([x[-1,1:5],x[-1,8:9]],axis=0)
        rs = lambda x:rearrange(x,"C X Y -> (C X) Y")
        ko_times = list(map(cs,ko_times))
        ko_times = list(map(rs,ko_times))
        # print(ko_times[0].shape)
        imdata = onp.concatenate(ko_times,axis=1)
        xw,yw = imdata.shape
        imdata = onp.clip(imdata,0,1)
        ko_labels = [f"KO {t}h" for t in KO_TIMES]
        ko_labels.append("Baseline")
        plt.figure(figsize=(20,8),dpi=400)
        plt.imshow(imdata,cmap="gray")
        plt.yticks(np.linspace(xw/(5*2),xw*(5*2 - 1)/(5*2),5),["TBXT","SOX17","SOX2","FOXA2","LEF1"])
        plt.xticks(np.linspace(yw/(len(ko_times)*2),yw*(len(ko_times)*2-1)/(len(ko_times)*2),len(ko_times)),ko_labels)
        return plt.gca()
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR_KO):
    visualise_single_from_tensor(DATA_9CH_CIRCULAR_KO[2][24][0],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


@app.cell
def _(CHANNEL_NAMES_9CH_CIRCULAR, DATA_9CH_CIRCULAR):
    visualise_single_from_tensor(DATA_9CH_CIRCULAR[4][0],CHANNEL_NAMES_9CH_CIRCULAR,tres=1,title="",NCHANNELS=9)
    return


if __name__ == "__main__":
    app.run()
