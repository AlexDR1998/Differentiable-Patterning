import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    # print(sys.path)
    import jax 
    import jax.numpy as np
    import numpy as onp
    import jax.random as jr
    import equinox as eqx
    from tqdm.notebook import tqdm
    import time
    from einops import rearrange,repeat,reduce
    from NCA.model.NCA_gated_model import gNCA
    from NCA.model.NCA_model import NCA
    from NCA.model.NCA_multi_scale import mNCA
    from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual,load_micropattern_circle_8ch_individual_explicit_colony
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations
    import matplotlib.pyplot as plt
    from pprint import pprint
    from skimage import measure
    import matplotlib.style
    matplotlib.style.use(
        "default"
    )
    from Common.dataloader.micropattern import load_micropattern_shape_sequence


@app.cell
def _():
    mo.md(r"""
    # Evaluating NCA channels and downsample for micropatterning
    - This notebook loads trained gNCA models in `models/micropattern_individual_8ch/` which have been trained on data in `Timecoarse Individual Images`.
    - These models have been trained with varying hidden channels, on data of varying spatial resolutions, to estimate what combinations have the highest accuracy
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load true data
    """)
    return


@app.cell
def _(DOWNSAMPLES):
    DATA_COLONY = {}
    BOUNDARY_COLONY = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_CHANNEL_NAMES = get_data_colony(DOWNSAMPLE=_d)
        _data_split = [_data[:,:,:4],_data[:,:,7:11]]
        _data = np.concatenate(_data_split,axis=2)
        print(_data.shape)
        DATA_COLONY[_d] = _data
        BOUNDARY_COLONY[_d] = _boundary
    CHANNEL_NAMES_COLONY = _CHANNEL_NAMES[:4] + _CHANNEL_NAMES[7:11]
    print(CHANNEL_NAMES_COLONY)
    return BOUNDARY_COLONY, CHANNEL_NAMES_COLONY, DATA_COLONY


@app.cell
def _(DOWNSAMPLES):
    DATA_COLONY_FULL = {}
    BOUNDARY_COLONY_FULL = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_CHANNEL_NAMES = get_data_colony(DOWNSAMPLE=_d)
        # _data_split = [_data[:,:,:4],_data[:,:,7:11]]
        # _data = np.concatenate(_data_split,axis=2)
        print(_data.shape)
        DATA_COLONY_FULL[_d] = _data
        BOUNDARY_COLONY_FULL[_d] = _boundary
    # CHANNEL_NAMES_COLONY = _CHANNEL_NAMES[:4] + _CHANNEL_NAMES[7:11]
    # print(CHANNEL_NAMES_COLONY)
    CHANNEL_NAMES_COLONY_FULL = _CHANNEL_NAMES
    print(CHANNEL_NAMES_COLONY_FULL)
    return CHANNEL_NAMES_COLONY_FULL, DATA_COLONY_FULL


@app.cell
def _(DOWNSAMPLES):
    DATA_IND_TRIANGLE = {}
    X0_TRIANGLE = {}
    BOUNDARY_IND_TRIANGLE = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="triangle")
        DATA_IND_TRIANGLE[_d] = _data
        BOUNDARY_IND_TRIANGLE[_d] = _boundary
        X0_TRIANGLE[_d] = _x0
    # print(X0_TRIANGLE[8].shape)
    return BOUNDARY_IND_TRIANGLE, X0_TRIANGLE


@app.cell
def _(DOWNSAMPLES):
    DATA_IND_ELLIPSE = {}
    X0_ELLIPSE = {}
    BOUNDARY_IND_ELLIPSE = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="ellipse")
        DATA_IND_ELLIPSE[_d] = _data
        BOUNDARY_IND_ELLIPSE[_d] = _boundary
        X0_ELLIPSE[_d] = _x0
    # print(X0_ELLIPSE[8].shape)
    return BOUNDARY_IND_ELLIPSE, X0_ELLIPSE


@app.cell
def _(DOWNSAMPLES):
    DATA_IND_DONUT = {}
    X0_DONUT = {}
    BOUNDARY_IND_DONUT = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="donut")
        DATA_IND_DONUT[_d] = _data
        BOUNDARY_IND_DONUT[_d] = _boundary
        X0_DONUT[_d] = _x0
    # print(X0_DONUT[8].shape)
    return BOUNDARY_IND_DONUT, X0_DONUT


@app.cell
def _(X0_CIRCLE):
    plot_matrix(X0_CIRCLE[8][1.0][0])
    return


@app.cell
def _(DOWNSAMPLES, RADII):
    DATA_IND_CIRCLE = {}
    X0_CIRCLE = {}
    BOUNDARY_IND_CIRCLE = {}
    for _d in DOWNSAMPLES:
        _DR = {}
        _XR = {}
        _BR = {}
        for _r in RADII:
            _data,_boundary,_x0,_ = get_shaped_data_synthetic(DOWNSAMPLE=_d,shape="circle",radius=_r)
            _DR[_r] = _data
            _XR[_r] = _x0
            _BR[_r] = _boundary
        DATA_IND_CIRCLE[_d] = _DR
        X0_CIRCLE[_d] = _XR
        BOUNDARY_IND_CIRCLE[_d] = _BR
    return BOUNDARY_IND_CIRCLE, X0_CIRCLE


@app.cell
def _(DOWNSAMPLES):
    DATA_IND = {}
    BOUNDARY_IND = {}
    for _d in DOWNSAMPLES:
        _data,_boundary,_CHANNEL_NAMES = get_data(DOWNSAMPLE=_d)
        # _data_split = [_data[:,:,:4],_data[:,:,7:11]]
        # _data = np.concatenate(_data_split,axis=2)
        print(_data.shape)
        DATA_IND[_d] = _data
        BOUNDARY_IND[_d] = _boundary
    CHANNEL_NAMES_IND = _CHANNEL_NAMES #= CHANNEL_NAMES[:4] + CHANNEL_NAMES[7:11]
    print(CHANNEL_NAMES_IND)
    return BOUNDARY_IND, CHANNEL_NAMES_IND, DATA_IND


@app.cell
def _(CHANNEL_NAMES_COLONY, DATA_COLONY):
    visualise_single_from_tensor(DATA_COLONY[4][0],CHANNEL_NAMES_COLONY,tres=1,title="True data (500)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Average loss between true data and each trajectory
    """)
    return


@app.cell(hide_code=True)
def _(DOWNSAMPLES):
    def plot_loss_hparams(loss):
        for D in DOWNSAMPLES:
            loss_per_sampling = list(loss[D].values())
            _CHANNELS = list(loss[D].keys())
            plt.plot(_CHANNELS,loss_per_sampling,label=f"Downsample factor {D}",marker="o")
        plt.legend()
        plt.xlabel("Total channels (Observable 8 + hidden)")
        plt.ylabel("Mean Absolute Error")
        # plt.savefig("micropattern_channel_downsample_loss")
        return plt.gca()
        # for key,val in loss.:
            # print(key)
            # print(val)
    return (plot_loss_hparams,)


@app.cell(hide_code=True)
def _(DATA, DOWNSAMPLES, hparams, output_data, plot_loss_hparams):
    _loss = {}
    for _d in DOWNSAMPLES:
        _loss[_d] = {}
    for _H in hparams:
        _d = _H["downsample"]
        _c = _H["channels"]
        _true_data = DATA[_d][0]
        # print(_true_data.shape)
        try:
            _nca_data = output_data[_d][_c]
            _loss[_d][_c] = onp.mean(np.abs((_true_data-_nca_data)))
        except:
            _nca_data = None
        # print(_nca_data.shape)
    plot_loss_hparams(_loss)
    return


@app.cell(hide_code=True)
def _(ch_8_to_11, correlation_loss, covariance_loss):
    def average_loss(nca_data,true_data,H_params,H_of_interest,LOSS_MODE="cor"):
        # pass
        downsample = H_params[0]["downsample"]
        # print(downsample)
        loss_hparam_pairs = []
        min_loss = (None,1e6)
        loss_label = {
            "cor":"Channel - Channel pearson correlation distance",
            "cov":"Channel - Channel covariance distance",
            "cos": "Cosine similarity",
            "relative": "Mean percentage error",
            "l1": "Mean absolute error",
            "l2": "Mean squared error",
            "min": "Distance between minimum pixel values",
            "hist": "Distance between histograms of pixel intensities"
        }
        for data in nca_data:
            # print(data[0])
            if LOSS_MODE=="l1":
                loss = onp.mean(np.abs(true_data[downsample]-ch_8_to_11(data[1])))
            elif LOSS_MODE=="l2":
                loss = onp.mean((true_data[downsample]-ch_8_to_11(data[1]))**2)
            elif LOSS_MODE=="relative":
                loss = onp.mean(np.abs(true_data[downsample]-ch_8_to_11(data[1]) / (np.abs(true_data[downsample]+ch_8_to_11(data[1]))+1e-9)))
            elif LOSS_MODE=="cos":
                loss = onp.mean(true_data[downsample]*ch_8_to_11(data[1]) / onp.sqrt(true_data[downsample]**2 + ch_8_to_11(data[1])**2+1e-9))
            elif LOSS_MODE=="cor":
                loss = correlation_loss(data=true_data[downsample],nca_output=ch_8_to_11(data[1]))
            elif LOSS_MODE=="cov":
                loss = covariance_loss(data=true_data[downsample],nca_output=ch_8_to_11(data[1]))
            elif LOSS_MODE=="hist":
                _x = onp.sort(true_data[downsample].ravel())
                _y = onp.sort(ch_8_to_11(data[1]).ravel())
                loss = onp.mean(onp.abs(_x-_y))
            elif LOSS_MODE=="min":
                loss = onp.abs(np.min(true_data[downsample])-np.min(ch_8_to_11(data[1])))
            else:
                raise(f"Unknown loss mode {LOSS_MODE}")


            if loss<min_loss[1]:
                min_loss = (data[0],loss)
            loss_hparam_pairs.append((data[0],loss))
        print(f"Minimum loss {min_loss[1]} with hyperparameters:")
        pprint(min_loss[0])
        hlist = list(set([H[H_of_interest] for H in H_params]))
        Losses = []
        medians = []
        mins = []
        maxs = []
        for H in hlist:
            loss_subset = select_all_with_1_match(
                loss_hparam_pairs,
                h_of_interest=H_of_interest,
                hvalue=H)
            losses = []
            for i in range(len(loss_subset)):
                losses.append(loss_subset[i][1])

            losses = np.array(losses)
            median_pos = np.argpartition(losses,len(losses)//2)[len(losses)//2]
            min_pos = np.argmin(losses)
            max_pos = np.argmax(losses)
            medians.append(loss_subset[median_pos][0])
            mins.append(loss_subset[min_pos][0])
            maxs.append(loss_subset[max_pos][0])
            Losses.append(losses)


        plt.boxplot(Losses,orientation="vertical")
        # plt.violinplot(Losses)
        plt.ylabel(loss_label[LOSS_MODE])
        plt.xticks(range(1,len(hlist)+1),hlist,rotation=90)
        plt.xlabel(f"Hyperparameter of interest: {H_of_interest}")

        # plt.xticks(hlist)
        plt.show()
        # return plt.gca()
        return min_loss[0],medians,mins,maxs
    return (average_loss,)


@app.cell
def _(
    CHANNELS,
    CHANNEL_NAMES_COLONY,
    DATA_COLONY,
    DATA_COLONY_FULL,
    DOWNSAMPLES,
    average_loss,
    nca_models,
    output_data,
    select_from_ncas,
    visualise_trajectories,
):
    _H_OF_INTEREST = "loss_mode"
    hparams_reduced = generate_hyperparameter_combinations({
        "downsample":DOWNSAMPLES,
        "channels":CHANNELS,
        "optimizer":["nadam_blocknorm","muon_blocknorm"],
        "loss_mode":["vgg_grouped","vgg_grouped_and_l2"],#"l2","l2_grad","vgg_grad","vgg_and_l2_grad"
        "model":["NCA","gNCA"],
        "noise_strength":[0.001,0.01,0.1],
        "intermediate_growth":[0.0,0.2,1.0,2.0],
        "contiguous_growth":[0.0],
    })
    best_hparams,median_hparams,min_per_hparam,max_per_hparam = average_loss(output_data,DATA_COLONY_FULL,hparams_reduced,_H_OF_INTEREST,LOSS_MODE="hist")
    # print("============ Overall best =======")
    # pprint(best_hparams)
    # print("============ Median =============")
    # pprint(median_hparams)
    # print("============ Minimum ============")
    # pprint(min_per_hparam)
    # print("============ Maximum ============")
    # pprint(max_per_hparam)
    # visualise_selected(nca_data=output_data,true_data=DATA_COLONY,hparams=best_hparams)
    # visualise_trajectories()
    _best_nca = select_from_ncas(nca_models,best_hparams)
    print(_best_nca)
    # pprint(min_per_hparam)
    print(best_hparams)
    visualise_trajectories(output_data,DATA_COLONY,min_per_hparam,_H_OF_INTEREST,chnames=CHANNEL_NAMES_COLONY)
    # print(best_nca_aux)
    return


@app.cell
def _(DATA_COLONY_FULL, average_loss, hparams, output_data):
    _H_OF_INTEREST = "contiguous_growth"
    _modes = ["cov","cor","l2","l1","relative","cos","min","hist"]
    _bests = []
    for _m in _modes:
        _best_hparams,_,_,_ = average_loss(output_data,DATA_COLONY_FULL,hparams,_H_OF_INTEREST,LOSS_MODE=_m)
        _bests.append(_best_hparams)
    for _i in range(len(_modes)):
        print(f"========== Best {_modes[_i]} ========")
        pprint(_bests[_i])
    return


@app.cell
def _(CHANNEL_NAMES_COLONY, CHANNEL_NAMES_COLONY_FULL):
    print(CHANNEL_NAMES_COLONY)
    print(CHANNEL_NAMES_COLONY_FULL)
    return


@app.cell(hide_code=True)
def _(DATA_COLONY_FULL, output_data):
    def correlation_loss(data,nca_output):
        data = rearrange(data,"() T C x y -> T C (x y)")
        nca_output = rearrange(nca_output,"T C x y -> T C (x y)")
        cor_true = onp.array([onp.corrcoef(d) for d in data])
        cor_nca = onp.array([onp.corrcoef(d) for d in nca_output])
        cor_true[:,4:,:4] = 0
        cor_true[:,:4,4:] = 0
        cor_true[:,8:,:8] = 0
        cor_true[:,:8,8:] = 0
        cor_nca[:,4:,:4] = 0
        cor_nca[:,:4,4:] = 0
        cor_nca[:,8:,:8] = 0
        cor_nca[:,:8,8:] = 0 
        # print(cor_true.shape)
        # print(cor_nca.shape)
        # plt.imshow(rearrange(cor_true,"T mx my -> mx (T my)"))
        # plt.show()
        # plt.imshow(rearrange(cor_nca,"T mx my -> mx (T my)"))
        # plt.show()
        # plt.imshow(rearrange((cor_nca-cor_true)**2,"T mx my -> mx (T my)"))
        # plt.show()
        return onp.mean((cor_true-cor_nca)**2)

    def covariance_loss(data,nca_output):
        data = rearrange(data,"() T C x y -> T C (x y)")
        nca_output = rearrange(nca_output,"T C x y -> T C (x y)")
        cor_true = onp.array([onp.cov(d) for d in data])
        cor_nca = onp.array([onp.cov(d) for d in nca_output])
        cor_true[:,4:,:4] = 0
        cor_true[:,:4,4:] = 0
        cor_true[:,8:,:8] = 0
        cor_true[:,:8,8:] = 0
        cor_nca[:,4:,:4] = 0
        cor_nca[:,:4,4:] = 0
        cor_nca[:,8:,:8] = 0
        cor_nca[:,:8,8:] = 0 
        # print(cor_true.shape)
        # print(cor_nca.shape)
        # plt.imshow(rearrange(cor_true,"T mx my -> mx (T my)"))
        # plt.show()
        # plt.imshow(rearrange(cor_nca,"T mx my -> mx (T my)"))
        # plt.show()
        # plt.imshow(rearrange((cor_nca-cor_true)**2,"T mx my -> mx (T my)"))
        # plt.show()
        return onp.mean((cor_true-cor_nca)**2)
        # plt.imshow(cor[0])
        # return plt.gca()
        # return cor
        # print(data.shape)
    # correlation_loss(DATA_COLONY[8],output_data[100][1])
    def ch_8_to_11(data):
        # Shape of data T C x y. Duplicates C from 8 to 11, for comparing with duplicate experiments
        data_split = [data[:,:4],data[:,:3],data[:,4:]]
        data = np.concatenate(data_split,axis=1)
        return data
    def ch_11_to_8(data):
        # Shape of data T C x y. Reduces C from 11 to 8, for comparing with duplicate experiments
        data_split = [data[:,:4],data[:,7:11]]
        data = np.concatenate(data_split,axis=1)
        return data

    correlation_loss(DATA_COLONY_FULL[8],ch_8_to_11(output_data[10][1]))
    return ch_8_to_11, correlation_loss, covariance_loss


@app.cell
def _(output_data):
    print(output_data[0][0])
    return


@app.cell
def _(CHANNEL_NAMES_COLONY, DATA_COLONY, output_data, visualise_trajectories):
    # Hsubset = generate_hyperparameter_combinations({
    #     "downsample":[4],
    #     "channels":[48],
    #     "optimizer":["nadam_blocknorm"],#,"muon_blocknorm"],
    #     # "loss_mode":["vgg","vgg_grouped","l2","vgg_and_l2","vgg_grouped_and_l2","l2_grad","vgg_grad","vgg_and_l2_grad"],
    #     # "loss_mode":["vgg_grouped","vgg_grouped_and_l2"],
    #     "loss_mode":["vgg_grouped_and_l2"],
    #     "model":["NCA","gNCA"],
    #     # "blocknorm":[True],
    #     # "noise_strength":[0.1,0.01,0.001],
    #     "noise_strength":[0.005],
    #     # "intermediate_growth":[0.0,0.2,1.0,2.0],
    #     "intermediate_growth":[1.0],
    #     "contiguous_growth":[1.0],
    # })


    Hsubset = generate_hyperparameter_combinations({
        "model":["gNCA"],
        "optimizer":["nadam_blocknorm"],
        "channels":[48],
        "downsample":[8],
        "K":[5],
        "D":[4],
        "S":[2048],
        "init_lr":[0.001]})

    visualise_trajectories(output_data,DATA_COLONY,Hsubset,"model",chnames=CHANNEL_NAMES_COLONY)
    return


@app.cell
def _(DATA_COLONY):
    print(DATA_COLONY[8].shape)
    return


@app.cell
def _():
    mo.md(r"""
    # Shape predictions
    """)
    return


@app.cell
def _(BOUNDARY_IND_TRIANGLE, H_best, best_nca, generate_video):
    # _XS = generate_video(best_nca[1])
    _XS = generate_video(
        best_nca[1],
        ch=best_nca[1].N_CHANNELS,
        ds=H_best["downsample"],
        t=best_nca[2]["timsteps"],
        mode="triangle",
        filepath=None)
    T_full_triangle,T_best_triangle,T_best_monocrome_triangle = reshape_for_videos(_XS,mask=BOUNDARY_IND_TRIANGLE[H_best["downsample"]])
    print(T_full_triangle.shape)
    return T_best_monocrome_triangle, T_best_triangle, T_full_triangle


@app.cell
def _(BOUNDARY_IND_ELLIPSE, H_best, best_nca, generate_video):
    # _XS = generate_video(best_nca[1])
    _XS = generate_video(
        best_nca[1],
        ch=best_nca[1].N_CHANNELS,
        ds=H_best["downsample"],
        t=best_nca[2]["timsteps"],
        mode="ellipse",
        filepath=None)
    T_full_ellipse,T_best_ellipse,T_best_monocrome_ellipse = reshape_for_videos(_XS,mask=BOUNDARY_IND_ELLIPSE[H_best["downsample"]])
    print(T_full_ellipse.shape)
    return T_best_ellipse, T_best_monocrome_ellipse, T_full_ellipse


@app.cell
def _(BOUNDARY_IND_DONUT, H_best, T_full_ellipse, best_nca, generate_video):
    _XS = generate_video(
        best_nca[1],
        ch=best_nca[1].N_CHANNELS,
        ds=H_best["downsample"],
        t=best_nca[2]["timsteps"],
        mode="donut",
        filepath=None)
    T_full_donut,T_best_donut,T_best_monocrome_donut = reshape_for_videos(_XS,mask=BOUNDARY_IND_DONUT[H_best["downsample"]])
    print(T_full_ellipse.shape)
    return T_best_donut, T_best_monocrome_donut, T_full_donut


@app.cell
def _(
    CHANNEL_NAMES_COLONY,
    T_full_triangle,
    best_nca,
    visualise_single_trajectory,
):
    # plot_matrix(rearrange(T_full_triangle[::90,:8],"T C X Y -> (C X) (T Y)"))
    visualise_single_trajectory(trajectory=T_full_triangle,chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"])
    return


@app.cell
def _(
    CHANNEL_NAMES_COLONY,
    T_full_ellipse,
    best_nca,
    visualise_single_trajectory,
):
    visualise_single_trajectory(trajectory=T_full_ellipse,chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"])
    return


@app.cell
def _(
    CHANNEL_NAMES_COLONY,
    T_full_donut,
    best_nca,
    visualise_single_trajectory,
):
    visualise_single_trajectory(trajectory=T_full_donut,chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"])
    return


@app.cell
def _():
    mo.md(r"""
    # Radius predictions
    """)
    return


@app.cell
def _(BOUNDARY_IND_CIRCLE, H_best, RADII, best_nca, generate_video):
    T_full_circle = {}
    T_comp_circle = {}
    T_monocrome_circle = {}
    for _r in tqdm(RADII):
        _XS = generate_video(
            best_nca[1],
            ch=best_nca[1].N_CHANNELS,
            ds=H_best["downsample"],
            t=best_nca[2]["timsteps"],
            mode="circle",
            r=_r,
            filepath=None)
        print(f"_XS shapes: {_XS.shape}")
        _OUTS = reshape_for_videos(_XS,mask=BOUNDARY_IND_CIRCLE[H_best["downsample"]][_r])
        T_full_circle[_r] = _OUTS[0]
        T_comp_circle[_r] = _OUTS[1]
        T_monocrome_circle[_r] = _OUTS[2]
    return T_comp_circle, T_full_circle, T_monocrome_circle


@app.cell
def _(CHANNEL_NAMES_COLONY, RADII, T_full_circle, best_nca):
    print(RADII)
    visualise_single_from_tensor(trajectory=T_full_circle[0.24],chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"],title="120")
    return


@app.cell
def _(CHANNEL_NAMES_COLONY, CHANNEL_NAMES_IND, RADII, T_full_circle, best_nca):
    def size_slices(T,chnames,tres):
        # plt.plot()
        T_slice = {}
        radii_dict = {}
        # true_radii = {
        #     0.24:220

        # }
        r_res = 200
        for R in RADII:
            slice = T[R][::tres]
            T_slice[R],_rad = calculate_radial_average(
                T=slice,
                R_res=r_res,
                padding_ratio=0.9,
                pixel_offsets=[[1,0],[1,0]],
                PLOT_DEBUG=False,
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

        linestyles = ["-","-","-","-","-","-","-","-"]
        # lines_to_plot = [1,2,3,4,5,6,7]
        lines_to_plot = [1,2,3,4]
        # cmap_cellfate = plt.cm.get_cmap("autumnn",8)
        colors1 = plt.cm.winter(onp.linspace(0,1.0,5))
        colors2 = plt.cm.autumn(onp.linspace(0,0.8,3))
        colors = list(onp.vstack((colors1, colors2)))
        # plt.plot(radii_dict[0.24],T_slice[0.24][-1].T,)
        # R = 0.64
        # plt.p
        fig,axs = plt.subplots(2,4,figsize=(16,8),sharey=True)
        for i,R in enumerate(RADII):
            for ch in lines_to_plot:
                axs[i//4,i%4].plot(
                    radii_dict[R],
                    T_slice[R][-1,ch,::-1].T,
                    color=colors[ch],
                    label=CHANNEL_NAMES_IND[ch],
                    linewidth=2,
                    linestyle=linestyles[ch])
                axs[i//4,i%4].set_title(f"{int(R*500)} ")
        axs[0,0].legend()
        plt.tight_layout()
        return plt.gca()

        # print(T[1.64][::90].shape)
    size_slices(T_full_circle,chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"])
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Define Hyperparameters
    """)
    return


@app.cell
def _():
    _colors1 = plt.cm.winter(onp.linspace(0,1.0,5))
    _colors2 = plt.cm.autumn(onp.linspace(0,0.8,3))
    CH_COLORS = list(onp.vstack((_colors1, _colors2)))
    return (CH_COLORS,)


@app.cell
def _():
    # RADII = [0.8,1.0,1.2,1.4,1.6]
    # Models trained on 500 micron diamater. Want to predict on [120,220,320,420,520,620,720,820] micron diameters
    RADII = onp.array([120.0,220.0,320.0,420.0,520.0,620.0,720.0,820.0])/500.0
    print(RADII)
    return (RADII,)


@app.cell
def _():
    # DOWNSAMPLES = [16,8,4,2]
    # CHANNELS = [8,10,12,14,16,18,24,32,64]

    # Euclidean loss
    # DOWNSAMPLES = [16,8,4,2]
    # CHANNELS = [8,16,24,32,64]

    # Textural loss
    # DOWNSAMPLES = [2]
    # CHANNELS = [32]
    # LOSS_MODES = ["l2","vgg","both_average","both_split"]
    # LOSS_MODES = ["vgg","vgg_grouped","vgg_and_l2","vgg_grouped_and_l2"]
    # INTREG = ["0.0","0.2","1.0","2.0"]
    # NOISE = ["0.001","0.01","0.1"]
    # GRADS = [True,False]

    #Textural loss with experiment groupings



    # CHANNELS = [16,24,32,48]
    CHANNELS = [32]
    # REG = ["","_int","_int_contig","_contig"]
    DOWNSAMPLES = [4,8]
    # OPTIMIZER = ["nadam_blocknorm"]

    # hparams = generate_hyperparameter_combinations({
    #     "downsample":DOWNSAMPLES,
    #     "channels":CHANNELS,
    #     "optimizer":["nadam_blocknorm","muon_blocknorm"],
    #     "loss_mode":["vgg","vgg_grouped","vgg_and_l2","vgg_grouped_and_l2"],#"l2","l2_grad","vgg_grad","vgg_and_l2_grad"
    #     "model":["NCA","gNCA"],
    #     "noise_strength":[0.001,0.01,0.1],
    #     "intermediate_growth":[0.0,0.2,1.0,2.0],
    #     "contiguous_growth":[0.0,1.0],
    # })

    #--- VGG loss hyperparameters
    CHANNELS = [48]
    DOWNSAMPLES = [4]
    hparams = generate_hyperparameter_combinations({
        "loss_mode":["vgg_grouped","vgg_grouped_and_l2"],
        "model":["NCA","gNCA"],
        "optimizer":["nadam_blocknorm","muon_blocknorm"],
        # "block_norm":[True],
        "noise_strength":[0.005],
        "channels":CHANNELS,
        "intermediate_growth":[1.0],
        "contiguous_growth":[1.0],
        "downsample":DOWNSAMPLES,
    })

    #--- OTT loss hyperparamaters
    # CHANNELS = [48]
    # DOWNSAMPLES = [2,4,8]
    # hparams = generate_hyperparameter_combinations({
    #     "model":["NCA","gNCA"],
    #     "optimizer":["nadam_blocknorm","muon_blocknorm"],
    #     "channels":CHANNELS,
    #     "downsample":DOWNSAMPLES,
    #     "K":[5],
    #     "D":[4],
    #     "S":[2048],
    #     "init_lr":[1e-3],
    # })



    print(len(hparams))
    # CHANNELS = [c+1 for c in CHANNELS] # Add 1 for boundary mask channel
    return CHANNELS, DOWNSAMPLES, hparams


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run all models on respective data
    - Takes a few minutes
    - Sweeps over Channels and Downsampling
    """)
    return


@app.cell
def _(get_x0_and_bmask, hparams):
    _key = jr.PRNGKey(int(time.time()))
    output_data = []
    nca_models = []

    def run_model_with_hparams(H,key):
        # try:
        nca,aux = load_nca_texture_grouped_noise(
            CHANNELS=H["channels"],
            DOWNSAMPLE=H["downsample"],
            MODEL=H["model"],
            LOSS_MODE = H["loss_mode"],
            INTERMEDIATE_GROWTH_COEFF = H["intermediate_growth"],
            CONTIGUOUS_GROWTH_COEFF = H["contiguous_growth"],
            NOISE_STRENGTH = H["noise_strength"],
            OPTIMIZER = H["optimizer"]
        )
        t = aux["timsteps"]
        x,bfunc,T = get_x0_and_bmask(H["channels"],H["downsample"])
        # print(nca)
        XS = []
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if i%t==0:
                XS.append(x[:8])
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        output_data.append((H,XS))
        nca_models.append((H,nca,aux))
        # except:
            # print(f"Failed with hparams {H}")


    def run_ott_model_with_hparams(H,key):
        nca,aux= load_nca_ott_grouped_and_l2(
            DOWNSAMPLE=H["downsample"],
            CHANNELS=H["channels"],
            MODEL=H["model"],
            INIT_LR=H["init_lr"],
            OPTIMIZER=H["optimizer"],
            K=H["K"],
            D=H["D"],
            S=H["S"]
        )
        t = aux["timsteps"]
        x,bfunc,T = get_x0_and_bmask(H["channels"],H["downsample"])
        # print(nca)
        XS = []
        for i in range((t*T)+1):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if i%t==0:
                XS.append(x[:8])
        XS = onp.array(XS)
        print(f"Data shape {XS.shape}")
        output_data.append((H,XS))
        nca_models.append((H,nca,aux))


    for _H in tqdm(hparams):
        _key = jr.fold_in(_key,1)
        # try:
        run_model_with_hparams(_H,key=_key)
        # run_ott_model_with_hparams(_H,key=_key)
        # except:
            # print(f"Failed with hparams {_H}")
    return nca_models, output_data


@app.cell
def _():
    mo.md(r"""
    # Analysis of single NCA model
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Run 1 model
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Good Hyperparameters:
     - Loss function is most important:
           - `vgg_grouped` or `vgg_grouped_and_l2` is best
     - Noise strength below 0.1 is best
     - Block normalisation in optimizer helps, but nadam vs muon makes no difference
     - Intermediate growth regulariser helps a bit
     - Contiguous growth or nca gating make no measureable difference at this stage
    """)
    return


@app.cell
def _(nca_models, select_from_ncas):
    # H_best = {
    #     "downsample":8,
    #     "channels":32,
    #     "optimizer":"nadam_blocknorm",
    #     "loss_mode":"vgg_grouped_and_l2",
    #     "model":"NCA",
    #     "noise_strength":0.01,
    #     "intermediate_growth":2.0,
    #     "contiguous_growth":1.0,
    # }

    #--- VGG loss best
    H_best = {
        "downsample":4,
        "channels":48,
        "optimizer":"nadam_blocknorm",
        "loss_mode":"vgg_grouped_and_l2",
        "model":"NCA",
        "noise_strength":0.005,
        "intermediate_growth":1.0,
        "contiguous_growth":1.0,
    }
    # print(nca_models[159][0])
    # print(H_best)
    # print([N for N in nca_models if N[0]==H_best])
    # for N in nca_models:
        # print(N[0])
    # print(nca_models)
    best_nca = select_from_ncas(ncas=nca_models,hparams=H_best)
    return H_best, best_nca


@app.cell
def _(BOUNDARY_IND, H_best, best_nca, generate_video):
    _XS = generate_video(best_nca[1],ch=best_nca[1].N_CHANNELS,ds=H_best["downsample"],t=best_nca[2]["timsteps"],filepath=None)
    # T_full = map_01(_XS)
    # T_best = map_01(rearrange(_XS[:,:9],"T (cx cy cc) X Y -> T cc (cx X) (cy Y)",cx=1,cy=3,cc=3))
    # T_best_monocrome = rearrange(T_best,"T c x y -> T () (c x) y")
    # T_best_monocrome = repeat(T_best_monocrome,"T () x y -> T 3 x y")
    T_full,T_best,T_best_monocrome = reshape_for_videos(_XS,mask=BOUNDARY_IND[H_best["downsample"]][0])
    print(T_full.shape)
    return T_best, T_best_monocrome, T_full


@app.cell
def _(T_full):
    # visualise_single_from_tensor(trajectory=T_full,chnames=CHANNEL_NAMES_COLONY,tres=best_nca[2]["timsteps"])
    print(T_full.shape)
    plot_matrix(T_full[-1,0])
    return


@app.cell
def _():
    mo.md(r"""
    ## Plot radial distribution of channel intensity
    - Kymograph plots show heatmap of radial average (edge -> center) of each channel.
    - Clear propagation of wavefronts here shows how NCA captures expected growth
    """)
    return


@app.cell
def _(CHANNEL_NAMES_IND):
    def radial_average(T,R_res=400,downsample=8):

        # ravs,radii = calculate_radial_average(T,R_res,padding_ratio=0.86)
        ravs,radii = calculate_radial_average(T,R_res,padding_ratio=0.85)

        t_cutoff = 4*int(256/onp.sqrt(downsample))
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
            ax.set_title(CHANNEL_NAMES_IND[ch])

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
            ax.plot(c[:,1],c[:,0],color="red")
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
                label=CHANNEL_NAMES_IND[ch],
                linewidth=2,
                linestyle=linestyles[ch])
        plt.xticks(onp.linspace(0,400,6),onp.linspace(250,0,6))
        plt.xlabel("Radial distance (microns)")
        plt.yticks(onp.linspace(0,-360,5),["0h","12h","24h","36h","48h"])
        plt.ylabel("Time")
        plt.legend()
        plt.show()
        # plt.savefig("nca_8ch_kymograph")
        # plt.legend()
        # return plt.gca()
    return (radial_average,)


@app.cell
def _(T_full, radial_average):
    radial_average(T_full)
    return


@app.cell
def _(T_full_donut, radial_average):
    radial_average(T_full_donut)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save trajectories to mp4
    """)
    return


@app.cell
def _(
    H_best,
    RADII,
    T_best,
    T_best_donut,
    T_best_ellipse,
    T_best_monocrome,
    T_best_monocrome_donut,
    T_best_monocrome_ellipse,
    T_best_monocrome_triangle,
    T_best_triangle,
    T_comp_circle,
    T_monocrome_circle,
):

    # plot_matrix(rearrange(T_best[-1],"c x y -> x (c y)"))
    save_to_video_rgb(
        data=rearrange(T_best_monocrome,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_monochrome_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_monocrome_triangle,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_monochrome_triangle_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_triangle,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_triangle_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_monocrome_ellipse,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_monochrome_ellipse_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_ellipse,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_ellipse_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_monocrome_donut,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_monochrome_donut_ds{H_best['downsample']}.mp4",
        duration=40
    )
    save_to_video_rgb(
        data=rearrange(T_best_donut,"N C x y -> N x y C"),#[:,:,:,:3],
        filename=f"micropattern_texture_donut_ds{H_best['downsample']}.mp4",
        duration=40
    )

    for _r in RADII:
        save_to_video_rgb(
            data=rearrange(T_comp_circle[_r],"N C x y -> N x y C"),#[:,:,:,:3],
            filename=f"micropattern_texture_circle_r{_r}_ds{H_best['downsample']}.mp4",
            duration=40
        )
        save_to_video_rgb(
            data=rearrange(T_monocrome_circle[_r],"N C x y -> N x y C"),#[:,:,:,:3],
            filename=f"micropattern_texture_monochrome_circle_r{_r}_ds{H_best['downsample']}.mp4",
            duration=40
        )

    # plot_matrix(rearrange(T_best[-1],"c x y -> x (c y)"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nodal knockout
    - During an NCA trajectory, simulate nodal knockout experiment by setting nodal channel to 0 after some $\tau$
    """)
    return


@app.cell
def _(H_best, best_nca, generate_video_blocked_nodal):
    _XS = generate_video_blocked_nodal(
        best_nca[1],
        ch=best_nca[1].N_CHANNELS,
        ds=H_best["downsample"],
        t=best_nca[2]["timsteps"],
        nodal_block_time=0,
        filepath=None)
    T_full_blocked = map_01(_XS)
    T_best_blocked = map_01(rearrange(_XS[:,:9],"T (cx cy cc) X Y -> T cc (cx X) (cy Y)",cx=1,cy=3,cc=3))
    T_best_monocrome_blocked = rearrange(T_best_blocked,"T c x y -> T () (c x) y")
    T_best_monocrome_blocked = repeat(T_best_monocrome_blocked,"T () x y -> T 3 x y")
    print(T_full_blocked.shape)
    return (T_full_blocked,)


@app.cell
def _(DATA_COLONY, T_full, visualise_simple):
    # plot_matrix(rearrange(T_full_blocked[::90,:8],"T C x y -> (C x) (T y) "))
    visualise_simple(trajectory=T_full,tres=90,true_data=DATA_COLONY,downsample=8)
    return


@app.cell
def _(DATA_COLONY, T_full_blocked, visualise_simple):
    visualise_simple(trajectory=T_full_blocked,tres=90,true_data=DATA_COLONY,downsample=8)
    return


@app.cell
def _():
    mo.md(r"""
    ## Channel-channel correlations
    """)
    return


@app.cell
def _(CHANNEL_NAMES_IND, CH_COLORS, T_full):
    def time_correlation(T):
        """
            Takes a trajectory T and a pair of channels and calculates how these channels correlate across timesteps
        """

        T = rearrange(T,"T C X Y -> T C (X Y)")#[:,:,2500:3500]
        # T = T-np.mean(T,axis=2,keepdims=True)

        def ch_pair(T,chs):
            T1 = T[:,chs[0]]
            T2 = T[:,chs[1]]
            # print(T1.shape)
            def correlate_pixel(T1,T2):
                c = np.correlate(T1,T2,mode="full")[len(T1)-1:]
                # c = c/np.sqrt(np.sum(T1**2)*np.sum(T2**2))
                return c
            vcor = jax.vmap(correlate_pixel,in_axes=(1,1),out_axes=1)
            # print(T1.shape)
            T1 = T1 - np.mean(T1,axis=0,keepdims=True)
            T2 = T2 - np.mean(T2,axis=0,keepdims=True)
            a = vcor(T1,T2)
            print(a.shape)
            a = onp.array(reduce(a,"Tau S -> Tau","mean"))
            return a
            # print(a.shape)
        ch_to_plot = range(1,8)
        outer_chs = range(8)
        fig,axs = plt.subplots(2,4,figsize=(16,8),sharey=True,sharex=True)
        # plt.subplots(4,2)
        for CH in tqdm(outer_chs):
            cors = []
            for c in range(8):
                cors.append(ch_pair(T,[CH,c]))
            # plt.plot(a)
            cors = onp.array(cors)
            # print(cors.shape)
            STEPS = cors.shape[1]
            # STEPS = cors[:,0,0].shape[0]
            # cors = cors/onp.max(cors,keepdims=True)
            # print(cors.shape)

            # for i,c in enumerate(cors):
            for i in ch_to_plot:
                axs[CH//4,CH%4].plot(cors[i],color=CH_COLORS[i],label=CHANNEL_NAMES_IND[i])

            # plt.plot(cors.T,color=CH_COLORS)
            axs[CH//4,CH%4].set_xlabel(r"$\tau$")
            axs[CH//4,CH%4].set_title(CHANNEL_NAMES_IND[CH])
            axs[CH//4,CH%4].set_xticks(np.linspace(0,STEPS,5),np.arange(0, 60, 12))

            # plt.legend(CHANNEL_NAMES_IND[])
        axs[0,0].legend()
        plt.tight_layout()
        return plt.gca()
        # a = correlate_pixel(T1,T2,mode="same")


    # print(T_full_circle[0.84].shape)
    time_correlation(T_full)
    return


@app.cell
def _(CHANNEL_NAMES_IND, CH_COLORS, T_full):
    def correlations_per_time(T):
        T = rearrange(T,"T C X Y -> T C (X Y)")[:,:8]
        T = T - onp.mean(T,axis=1,keepdims=True)
        cors = []
        for t in tqdm(range(len(T))):
            cors.append(onp.corrcoef(T[t]))
        cors = onp.array(cors)
        print(cors.shape)

        outer_chs = range(8)
        fig,axs = plt.subplots(2,4,figsize=(16,8),sharey=True,sharex=True)
        # plt.subplots(4,2)
        ch_to_plot = range(1,8)
        STEPS = cors[:,0,0].shape[0]
        for CH in tqdm(outer_chs):
            # cors = []
            # for c in range(8):
                # cors.append(ch_pair(T,[CH,c]))
            # plt.plot(a)
            # cors = onp.array(cors)
            # cors = cors/onp.max(cors,keepdims=True)
            # print(cors.shape)

            # for i,c in enumerate(cors):
            for i in ch_to_plot:
                axs[CH//4,CH%4].plot(cors[:,CH,i],color=CH_COLORS[i],label=CHANNEL_NAMES_IND[i])

            # plt.plot(cors.T,color=CH_COLORS)
            axs[CH//4,CH%4].set_xlabel("t")
            axs[CH//4,CH%4].set_title(CHANNEL_NAMES_IND[CH])
            axs[CH//4,CH%4].set_xticks(np.linspace(0,STEPS,5),np.arange(0, 60, 12))

            # plt.legend(CHANNEL_NAMES_IND[])
        axs[0,0].legend()
        plt.tight_layout()
        return plt.gca()
    correlations_per_time(T_full)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    # Helper functions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data loading
    """)
    return


@app.function(hide_code=True)
def get_data(DOWNSAMPLE):
    data,aux,CHANNEL_NAMES,boundary_mask = load_micropattern_circle_8ch_individual(
        impath="../Data/Timecourse Individual Images/*",
        DOWNSAMPLE = DOWNSAMPLE,
        BATCHES=1,
        PROCESSING_MODES={
            "map_to_0_1",
            "downsample"
            # "downsample",
        }
    )
    # plot_matrix(rearrange(data,"() T C x y -> (C x) (T y)"))
    return data,boundary_mask,CHANNEL_NAMES


@app.function
def get_data_colony(DOWNSAMPLE):
    """
        Helper function that wraps load_micropattern_circle_8ch_individual_explicit_colony()
        Paramaters
        ----------
        DOWNSAMPLE int
            Downsampling factor to load data at
        Returns
        -------
        data float32 [B=1 T C X Y]
            Timecourse data at specified downsampling
        boundary_mask bool [B=1 X Y]
            Boundary mask at specified downsampling
        CHANNEL_NAMES list of str
            Channel names for data

    """
    data,aux,CHANNEL_NAMES,boundary_mask = load_micropattern_circle_8ch_individual_explicit_colony(
        impath="../Data/Timecourse Seperate Colonies/",
        DOWNSAMPLE = DOWNSAMPLE,
        BATCHES=1,
        PROCESSING_MODES={
            "map_to_0_1",
            "downsample"
            # "downsample",
        }
    )
    return data,boundary_mask,CHANNEL_NAMES


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

    data_circle,aux,CHANNEL_NAMES,boundary_mask_circle = load_micropattern_circle_8ch_individual(
        impath="../Data/Timecourse Individual Images/*",
        DOWNSAMPLE = DOWNSAMPLE,
        BATCHES=1,
        PROCESSING_MODES={
            "map_to_0_1",
            "downsample"
            # "downsample",
        }
    )
    filepath = {
        "triangle":"../Data/micropattern_shapes/Max Projections */*Triangle*",
        "ellipse":"../Data/micropattern_shapes/Max Projections */*Ellipse*",
        "donut":"../Data/micropattern_shapes/Max Projections */*Donut*",  
        "circle":None
    }[shape]
    if shape=="circle":
        SHAPED_MASK = zoom(boundary_mask_circle[0][0],factor=radius)
        SHAPED_MASK = 1-rmask(0.8,SHAPED_MASK)
    else:
        SHAPED_MASK = None        

    I = load_micropattern_shape_sequence(
        filepath,
        DOWNSAMPLE=DOWNSAMPLE*2,
        BATCH_AVERAGE=False,
        CIRCLE_DATA=data_circle,
        CIRCLE_HIST_BINS=None,
        CIRCLE_MASK=boundary_mask_circle,
        PROCESSING_MODES={
            "map_to_0_1",
            "downsample"
        },SHAPED_MASK=SHAPED_MASK

    )
    # print(I)
    (data,mask,X0) = I
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
    return data,mask,X0,CHANNEL_NAMES


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model loading
    """)
    return


@app.function(hide_code=True)
def load_nca_models(DOWNSAMPLE,CHANNELS):
    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS-1}_ds{DOWNSAMPLE}_v4_long.eqx"
    print(FILENAME)
    nca = gNCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.function(hide_code=True)
def load_nca_texture_models(DOWNSAMPLE,CHANNELS):
    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    # FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS-1}_ds{DOWNSAMPLE}_texture_pretrain_v3.eqx"
    FILENAME = f"micropattern_circle_8ch_individual_NCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS-1}_ds{DOWNSAMPLE}_texture_pretrain_v3.eqx"

    print(FILENAME)
    nca = NCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.function(hide_code=True)
def load_nca_texture_grouped(DOWNSAMPLE,CHANNELS,REG,OPT):
    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    FILENAME = f"micropattern_circle_8ch_3colony_individual_vgg_{OPT}_blocknorm{REG}_NCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h.eqx"
    print(FILENAME)
    nca = NCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.function(hide_code=True)
def load_nca_texture_grouped_noise(
    MODEL,
    LOSS_MODE = "vgg_grouped",
    INTERMEDIATE_GROWTH_COEFF = "1.0",
    CONTIGUOUS_GROWTH_COEFF = "0.0",
    NOISE_STRENGTH = "0.1",
    DOWNSAMPLE = 8,
    CHANNELS = 32,
    OPTIMIZER = "nadam_blocknorm"
):

    # STEPS_BETWEEN_IMAGES = 90
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    # FILENAME = f"isambard_mp_circ_8ch_3col_ind_{LOSS_MODE}_{OPTIMIZER}_int{INTERMEDIATE_GROWTH_COEFF}_contig_{CONTIGUOUS_GROWTH_COEFF}_noise{NOISE_STRENGTH}_{MODEL}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h_stable.eqx"
    FILENAME = f"isambard_mp_circ_8ch_3col_ind_{LOSS_MODE}_{OPTIMIZER}_int{INTERMEDIATE_GROWTH_COEFF}_contig_{CONTIGUOUS_GROWTH_COEFF}_noise{NOISE_STRENGTH}_{MODEL}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h_stable_good.eqx"

    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    if MODEL=="gNCA":
        nca = gNCA(**NCA_hyperparameters)
    elif MODEL=="NCA":
        nca = NCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.function
def load_nca_ott_grouped_and_l2(
    DOWNSAMPLE,
    CHANNELS,
    MODEL,
    INIT_LR,
    OPTIMIZER,
    K,
    D,
    S
):
    key = jr.PRNGKey(0)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    LOSS_NAME = "ott_grouped_and_l2"
    INTERMEDIATE_GROWTH_COEFF="1.0"
    CONTIGUOUS_GROWTH_COEFF="1.0"
    NOISE_STRENGTH="0.005"
    EPSILON="0.01"
    OTT_METRIC="l1"

    FILENAME = f"isambard_mp_circ_8ch_ind_{LOSS_NAME}_S{S}K{K}D{D}shp{True}ep{EPSILON}{OTT_METRIC}_{OPTIMIZER}_int{INTERMEDIATE_GROWTH_COEFF}_contig_{CONTIGUOUS_GROWTH_COEFF}_noise{NOISE_STRENGTH}_{MODEL}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_lr{INIT_LR}_48h_stable.eqx"
    print(FILENAME)

    if MODEL=="gNCA":
        nca = gNCA(**NCA_hyperparameters)
    elif MODEL=="NCA":
        nca = NCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.function(hide_code=True)
def load_mnca_texture_hyperparameter_sweep(LOSS_MODE,GRADS):
    key = jr.PRNGKey(0)
    CHANNELS = 32
    DOWNSAMPLE = 2
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS, # Fix for hidden channels
        "KERNEL_STR":["ID","LAP","DIFF"],
        "SCALES":[1,2,4,8],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
    # FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS-1}_ds{DOWNSAMPLE}_texture_pretrain_v3.eqx"
    # FILENAME = f"micropattern_circle_8ch_individual_NCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_texture_pretrain_v3.eqx"
    FILENAME =f"micropattern_circle_8ch_individual_mNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_loss_{LOSS_MODE}_grad_{GRADS}_texture_pretrain_v3.eqx"

    print(FILENAME)
    nca = mNCA(**NCA_hyperparameters)
    nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
    return nca,{"timsteps":STEPS_BETWEEN_IMAGES}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data processing
    """)
    return


@app.function(hide_code=True)
def reshape_for_videos(T,mask):
    # Takes a trajectory and returns 3 versions
    # - Original (clipped to 0-1)
    # - CMY composite (Horizontal tiles of 3 channel composites)
    # - Monochrome tiled (4 by 2 tiles of each channel)
    # T shape is T C X Y

    mask = repeat(mask,"() X Y -> () () X (3 Y)")
    print(f"Mask shape: {mask.shape}")
    T = onp.clip(T,a_max=1.0,a_min=0.0)
    T_obs = T[:,:9]
    CHANNEL_ORDER = ["LMBR","TBXT","SOX17","SOX2","FOXA2","CER1","LEFTY2","NODAL"]
    DESIRED_ORDER = ["SOX2","TBXT","SOX17","CER1","LEFTY2","NODAL","FOXA2"]
    order_index = [3,1,2,5,6,7,4]
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
    T_monochrome = rearrange(T[:,:8],"T (cy cx) X Y -> T () (cx X) (cy Y)",cx=4,cy=2)
    T_monochrome = repeat(T_monochrome,"T () x y -> T 3 x y")
    return T,T_composite_cmy,T_monochrome


@app.cell
def _(
    BOUNDARY_COLONY,
    BOUNDARY_IND,
    BOUNDARY_IND_CIRCLE,
    BOUNDARY_IND_DONUT,
    BOUNDARY_IND_ELLIPSE,
    BOUNDARY_IND_TRIANGLE,
    DATA_COLONY,
    DATA_IND,
    X0_CIRCLE,
    X0_DONUT,
    X0_ELLIPSE,
    X0_TRIANGLE,
):
    def get_x0_and_bmask(ch,ds,mode="colony",r=1.0):
        if mode=="colony":
            data = DATA_COLONY[ds][0] # remove batch index
            T = data.shape[0]
            boundary = BOUNDARY_COLONY[ds][0]
            x0 = data[0]

        elif mode=="triangle":
            # data
            boundary = BOUNDARY_IND_TRIANGLE[ds]
            x0 = X0_TRIANGLE[ds]
            T = 5
        elif mode=="ellipse":
            boundary = BOUNDARY_IND_ELLIPSE[ds]
            x0 = X0_ELLIPSE[ds]
            T = 5
        elif mode=="donut":
            boundary = BOUNDARY_IND_DONUT[ds]
            x0 = X0_DONUT[ds]
            T = 5
        elif mode=="circle":
            boundary = BOUNDARY_IND_CIRCLE[ds][r]
            x0 = X0_CIRCLE[ds][r]
            T=5
        else:
            data = DATA_IND[ds][0] # remove batch index
            T = data.shape[0]
            boundary = BOUNDARY_IND[ds][0]
            x0 = data[0]

        # boundary = np.pad(boundary,((0,0),(3,3),(3,3)))
        # data = np.pad(data,((0,0),(0,ch-8),(3,3),(3,3))) 
        # data = np.pad(data,((0,0),(0,ch-8),(0,0),(0,0)))
        bfunc = model_boundary(mask=boundary)
        x0 = np.pad(x0,((0,ch-8),(0,0),(0,0)))
        # print(f"Boundary shape {boundary.shape}")
        # print(f"Data shape {data.shape}")
        return x0,bfunc,T-1
    return (get_x0_and_bmask,)


@app.cell
def _(get_x0_and_bmask):
    def generate_video(nca,ch,ds,t,filepath,mode="ind",r=1.0,key=jr.PRNGKey(int(time.time()))):
        x,bfunc,T = get_x0_and_bmask(ch,ds,mode=mode,r=r)
        XS = [x]
        for i in tqdm(range(t*T)):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            XS.append(onp.array(x))
        return onp.array(XS)
    return (generate_video,)


@app.function(hide_code=True)
def calculate_radial_average(T,R_res,padding_ratio=1.0,pixel_offsets=[[0,0],[0,0]],PLOT_DEBUG=False):
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
    T_cropped = T[:,:8,_x_low:_x_high,_y_low:_y_high]
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
def _(get_x0_and_bmask):
    def generate_video_blocked_nodal(nca,ch,ds,t,filepath,nodal_block_time,mode="ind",key=jr.PRNGKey(int(time.time()))):
        # nodal_block is in hours
        x,bfunc,T = get_x0_and_bmask(ch,ds,mode=mode)
        XS = [x]
        T = T
        realtimes = np.linspace(0,48,t*T) # time in hours
        for i in tqdm(range(t*T)):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if realtimes[i]>nodal_block_time:
                x = x.at[7].set(0)
            XS.append(onp.array(x))
        return onp.array(XS)
    return (generate_video_blocked_nodal,)


@app.function(hide_code=True)
def map_01(x):
    return (x-x.min())/(x.max()-x.min())


@app.cell(hide_code=True)
def _():
    def select_from_output_data(output_data,hparams):
        # Returns a hyperparameter-data tuple where the hyperparamer matches hparams
        subset = [D for D in output_data if D[0]==hparams]
        # print(subset)
        if len(subset)==0:
            print("No match of hparams")
            print(hparams)
            pprint(output_data)
        return subset[0]
    def select_from_ncas(ncas,hparams):
        nca_aux = [D for D in ncas if D[0]==hparams]
        if len(nca_aux)==0:
            print("No match of hparams")
            print(hparams)
        return nca_aux[0]
        # return nca_aux
    return select_from_ncas, select_from_output_data


@app.cell(hide_code=True)
def _(select_from_output_data):
    def select_many_from_output_data(output_data,hparams):
        # Returns a list of hyperparameter-data tuples where the hyperparameters are in the list hparams
        output_data_selected = []
        for H in hparams:
            output_data_selected.append(select_from_output_data(output_data,H))
        return output_data_selected
    return


@app.function(hide_code=True)
def select_all_with_1_match(output_data,h_of_interest,hvalue):
    # Returns a list of hyperparameter-data tuples where 1 of the entries in the hyperparameter dict matches a specific value
    output_data_selected = []
    for D in output_data:
        # output_data_selected
        if D[0][h_of_interest]==hvalue:
            output_data_selected.append(D)
    return output_data_selected


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plotting and visualisation
    """)
    return


@app.cell(hide_code=True)
def _(select_from_output_data):
    def plot_hparam_set(axs,H,nca_data,H_for_title,chnames,title=None):
        # Takes a matplotlib axes object, hyperparameter dict, nca trajectories/hyperparamter 
        # combination, title and channel names. Returns a matplotlib axes object with the 
        # trajectory if interest rendered and labelled. Only called as part of other plotting functions
        hparams,data = select_from_output_data(nca_data,H)
        STEPS = data.shape[0]
        data = rearrange(data,"T C x y -> (C x) (T y)")
        # print(f"Data shape {data.shape}")
        xw,yw = data.shape
        axs.imshow(data,cmap="gray")
        axs.set_xlabel("Time")
        axs.set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
        axs.set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        if H_for_title is not None:
            axs.set_title(hparams[H_for_title])
        elif title is not None:
            axs.set_title(title)
        return axs
    return (plot_hparam_set,)


@app.cell(hide_code=True)
def _(plot_hparam_set):
    def visualise_trajectories(nca_data,true_data,H_subset,H_of_interest,chnames):
        """
            Renders the true data (at correct resolution/downsampling) alongside nca trajectories corresponding to a set of 
            hyperparameters of interest. Expects nca trajectory/hparam list and hyperparamters to select from that.
        """

        # D = nca_data[1][0]["downsample"]
        D = H_subset[0]["downsample"]
        true_data = true_data[D][0]
        STEPS = true_data.shape[0]
        print(f"True data shape{true_data.shape}")
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape
        static_hparams = H_subset.copy()[0]
        static_hparams = list(static_hparams.keys())
        static_hparams = [h for h in static_hparams if h!=H_of_interest]
        static_hparams = " ".join(static_hparams)
        if H_of_interest=="_loss_mode":
            fig,axs = plt.subplots(3,(len(H_subset)+1)//3,sharex=True,sharey=True,figsize=(8,12),dpi=400)
            axs[0,0].imshow(true_data,cmap="gray")
            axs[0,0].set_xlabel("Time")
            axs[0,0].set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
            axs[0,0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
            axs[0,0].set_title("True data")
            for i,H in enumerate(H_subset):
                axs[(i+1)//3,(i+1)%3] = plot_hparam_set(
                    axs[(i+1)//3,(i+1)%3],
                    H,
                    nca_data,
                    H_for_title=H_of_interest,
                    chnames=chnames
                )

        else:
            fig,axs = plt.subplots(1,len(H_subset)+1,sharex=True,sharey=True,figsize=(14,6),dpi=400)
            axs[0].imshow(true_data,cmap="gray")
            axs[0].set_xlabel("Time")
            axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
            axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
            axs[0].set_title("True data")
            for i,H in enumerate(H_subset):
                axs[i+1] = plot_hparam_set(
                    axs[i+1],
                    H,
                    nca_data,
                    H_for_title=H_of_interest,
                    chnames=chnames
                )

        plt.suptitle(f"Varying {H_of_interest} ")# \n fixed: {static_hparams}",fontsize=15)
        plt.tight_layout()
        return plt.gca()
    return (visualise_trajectories,)


@app.cell(hide_code=True)
def _(CHANNEL_NAMES_COLONY, plot_hparam_set):
    def visualise_selected(nca_data,true_data,hparams,chnames=CHANNEL_NAMES_COLONY):
        """
            Renders just 1 trajectory (alongside true data). Expects nca trajectory/hparam list and single hyperparamter dict
        """
        D = nca_data[1][0]["downsample"]
        true_data = true_data[D][0]
        fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6),dpi=400)
        STEPS = true_data.shape[0]
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape

        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs[0].set_title("True data")
        axs[1]= plot_hparam_set(axs[1],hparams,nca_data,None,chnames=chnames,title="NCA prediction")
        axs[1].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        plt.tight_layout()
        return plt.gca()
    return


@app.cell(hide_code=True)
def _(CHANNEL_NAMES_COLONY):
    def visualise_single_from_tensor_with_data(true_data,trajectory,tres,chnames=CHANNEL_NAMES_COLONY,downsample=8):
        """
            Shows a trajectory alongside true data. Expects single trajectory as a tensor of T C X Y, and int tres
            describing how to downsample the trajectory in time.
        """
        # D = nca_data[1][0]["downsample"]
        true_data = true_data[downsample][0]
        fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6),dpi=400)
        STEPS = true_data.shape[0]
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        trajectory = rearrange(trajectory[::tres,:8],"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape

        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs[0].set_title("True data")
        # axs[1]= plot_hparam_set(axs[1],hparams,nca_data,None,chnames=chnames,title="NCA prediction")
        axs[1].imshow(trajectory,cmap="gray")
        axs[1].set_title("NCA prediction")
        axs[1].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        plt.tight_layout()
        return plt.gca()
    return


@app.function(hide_code=True)
def visualise_single_from_tensor(trajectory,chnames,tres,title=""):
    """
        Shows just 1 trajectory. Expects single trajectory as a tensor of T C X Y, and int tres
        describing how to downsample the trajectory in time.
    """
    trajectory = rearrange(trajectory[::tres,:8],"T C x y -> (C x) (T y)")
    xw,yw = trajectory.shape
    plt.figure(figsize=(8,8))
    plt.imshow(trajectory,cmap="gray")
    plt.xlabel("time")
    plt.title(title)
    STEPS = 5

    plt.xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
    plt.yticks(np.linspace(xw/16,xw*15/16,8),chnames)
    return plt.gca()


@app.cell(hide_code=True)
def _(chnames, trajectory):
    def visualise_pair_of_from_tensor(T1,T2,chanmes,TITLES):
        """
            Shows a pair of trajectories. Expects each trajectory as a tensor of T C X Y
        """

        fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6),dpi=400)
        STEPS = T1.shape[0]
        T1 = rearrange(T1,"T C x y -> (C x) (T y)")
        T2 = rearrange(T2,"T C x y -> (C x) (T y)")
        xw,yw = T1.shape

        # true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        # trajectory = rearrange(trajectory[::tres,:8],"T C x y -> (C x) (T y)")

        axs[0].imshow(T1,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),chnames)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs[0].set_title(TITLES[0])
        # axs[1]= plot_hparam_set(axs[1],hparams,nca_data,None,chnames=chnames,title="NCA prediction")
        axs[1].imshow(trajectory,cmap="gray")
        axs[1].set_title(TITLES[1])
        axs[1].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        plt.tight_layout()
        return plt.gca()
    return


@app.cell
def _():
    mo.md(r"""
    # Old / Broken / Not needed
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise multi-scale NCA on fixed channels and downsampling
    """)
    return


@app.cell(hide_code=True)
def _(CHANNEL_NAMES, DATA, multi_output_data):
    def _visualise_trajectories_mnca(nca_data,true_data,grads):
        true_data = true_data[2][0]
        STEPS = true_data.shape[0]
        true_data = np.pad(true_data,((0,0),(0,0),(3,3),(3,3)))
        print(true_data.shape)
        modes = nca_data.keys()
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape
        fig,axs = plt.subplots(1,len(modes)+1,sharex=True,sharey=True,figsize=(14,6),dpi=400)
        plt.suptitle(f" ",fontsize=16)
        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),CHANNEL_NAMES)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 72, 12))
        axs[0].set_title("True data")
        for _i,m in enumerate(modes):
            i = _i+1
            axs[i].imshow(rearrange(nca_data[m][False][:,:8],"T C x y -> (C x) (T y)"),cmap="gray")
            axs[i].set_xlabel("Time")
            axs[i].set_title(f"Loss mode: {m}")
        plt.tight_layout()
        return plt.gca()
        # print(modes)
    _visualise_trajectories_mnca(multi_output_data,DATA,True)
    return


@app.cell
def _():
    mo.md(r"""
    ## Run multi-scale NCA on fixed channels and downsampling
    - Iterates over different training hyperparameters
        - Loss function configuration
        - Spatial gradient loss
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Testing development of grouping channels by experiment
    """)
    return


@app.cell(hide_code=True)
def _(multi_output_data):

    def first_occurrence_indices(arr: np.ndarray) -> np.ndarray:
        """
        Given a sorted JAX array of integers, returns an array of indices corresponding
        to the first occurrence of each unique integer in the array.

        Parameters:
            arr (jnp.ndarray): A sorted 1D JAX array of integers.

        Returns:
            jnp.ndarray: A 1D JAX array containing the indices of the first occurrence
                         of each unique integer.
        """
        # The first element is always the first occurrence.
        first_indices = [0]

        # Compute the difference between consecutive elements.
        # Because the array is sorted, a non-zero difference means that we have encountered a new value.
        diff = np.diff(arr)

        # jnp.where returns indices where the difference is not zero, these indicate that
        # the next element is the start of a new group.
        new_value_indices = np.where(diff != 0)[0] + 1  # add 1 due to the diff operation

        # Concatenate the index 0 with the new indices.
        # first_indices = np.concatenate([np.array([0]), new_value_indices])

        return new_value_indices

    def pad_by_experiment():
        data = multi_output_data["vgg"][True][:,:8]
        data = np.array(data)
        exps = np.array([0,0,0,0,1,2,2,2])
        inds = np.argsort(exps,axis=0)
        print(inds)
        exps = exps[inds]
        data = data[:,inds]
        print(exps)
        print(first_occurrence_indices(exps))
        data_split = np.split(ary=data,indices_or_sections=first_occurrence_indices(exps),axis=1)
        data_split = [np.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0))) for x in data_split]
        for data in data_split:

            print(data.shape)
        data = np.concatenate(data_split,axis=1)
        print(data.shape)
        # data = data.at[exps.argsort()]
        # print(np.unique(exps))
        # print(np.where(exps==np.unique(exps)[np.newaxis]))

    pad_by_experiment()
    return


@app.cell(hide_code=True)
def _(GRADS, LOSS_MODES, get_x0_and_bmask):
    _key = jr.PRNGKey(int(time.time()))
    _mhparams = generate_hyperparameter_combinations(({"loss_mode":LOSS_MODES,"grads":GRADS}))
    multi_output_data = {}
    for _d in LOSS_MODES:
        multi_output_data[_d] = {}
    for _H in tqdm(_mhparams):
        try:
            # _nca,_aux = load_nca_models(CHANNELS=H["channels"],DOWNSAMPLE=H["downsample"])
            _nca,_aux = load_mnca_texture_hyperparameter_sweep(LOSS_MODE=_H["loss_mode"],GRADS=_H["grads"])

            _t = _aux["timsteps"]
        except:
            print(f"Failed to load with {_H}")
            continue

        # _data = DATA[H["downsample"]][0] # remove batch index
        # _boundary = BOUNDARY[H["downsample"]][0]

        # _bfunc = model_boundary(mask=_boundary)
        # # _bfunc = lambda x:x
        # _data = np.pad(_data,((0,0),(0,H["channels"]-8),(0,0),(0,0)))

        # _x = _data[0]
        _x,_bfunc,_T = get_x0_and_bmask(32,2)
        # _T = _data.shape[0] # Outer timesteps corresponding to data
        _XS = []
        for _i in range(_t*_T):
            _key = jr.fold_in(_key,_i)
            _x = _nca(X=_x,boundary_callback=_bfunc,key=_key)
            if _i%_t==0:
                _XS.append(_x)
        _XS = onp.array(_XS)
        multi_output_data[_H["loss_mode"]][_H["grads"]] = _XS
    return (multi_output_data,)


if __name__ == "__main__":
    app.run()
