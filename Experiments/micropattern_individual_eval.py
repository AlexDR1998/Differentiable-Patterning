import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
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
    from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations
    import matplotlib.pyplot as plt
    return (
        gNCA,
        generate_hyperparameter_combinations,
        jr,
        load_micropattern_circle_8ch_individual,
        mo,
        model_boundary,
        np,
        onp,
        plot_matrix,
        plt,
        rearrange,
        repeat,
        save_to_video_rgb,
        time,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Evaluating NCA channels and downsample for micropatterning
    - This notebook loads trained gNCA models in `models/micropattern_individual_8ch/` which have been trained on data in `Timecoarse Individual Images`.
    - These models have been trained with varying hidden channels, on data of varying spatial resolutions, to estimate what combinations have the highest accuracy
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load true data""")
    return


@app.cell
def _(CHANNELS, DOWNSAMPLES, generate_hyperparameter_combinations, get_data):
    hparams = generate_hyperparameter_combinations({"downsample":DOWNSAMPLES,"channels":CHANNELS})

    DATA = {}
    BOUNDARY = {}
    for d in DOWNSAMPLES:
        _data,_boundary,CHANNEL_NAMES = get_data(DOWNSAMPLE=d)
        DATA[d] = _data
        BOUNDARY[d] = _boundary
    return BOUNDARY, CHANNEL_NAMES, DATA, hparams


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Average loss between true data and each trajectory""")
    return


@app.cell
def _(DOWNSAMPLES, plt):
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


@app.cell
def _(DATA, DOWNSAMPLES, hparams, np, onp, output_data, plot_loss_hparams):
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
def _(mo):
    mo.md(r"""## Visualise snapshots of trajectories for different combinations of channels or downsampling""")
    return


@app.cell
def _(CHANNEL_NAMES, np, plt, repeat):
    def visualise_trajectories_per_downsample(nca_data,true_data,C):
        TD = 2
        true_data = true_data[TD][0] # use highest resolution for true data
        downsamples = list(nca_data.keys())
        print(downsamples)
        fig,axs = plt.subplots(1,len(downsamples)+1,sharex=True,sharey=True,figsize=(14,6),dpi=400)
        STEPS = true_data.shape[0]
        true_data = repeat(true_data,"T C x y -> (C x dx) (T y dy)",dx=TD,dy=TD)
        xw,yw = true_data.shape
        plt.suptitle(f"Channels 8 + {C - 9}",fontsize=16)
        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),CHANNEL_NAMES)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 72, 12))
        axs[0].set_title("True data")
        for _i,d in enumerate(downsamples):
            i = _i+1
            axs[i].imshow(repeat(nca_data[d][C],"T C x y -> (C x wx) (T y wy)",wx=d,wy=d),cmap="gray")
            axs[i].set_xlabel("Time")
            if d==1:
                axs[i].set_title("No downsampling")
            else:
                axs[i].set_title(f"Downsampling {d}")
        plt.tight_layout()
        # plt.savefig(f"micropattern_individual_8ch_snapshots_{C}ch.png")
        return plt.gca()
    return


@app.cell
def _(CHANNEL_NAMES, np, plt, rearrange):
    def visualise_trajectories_per_channels(nca_data,true_data,D):
        true_data = true_data[D][0]
        nca_data = nca_data[D]
        channels = list(nca_data.keys())
        fig,axs = plt.subplots(1,len(channels)+1,sharex=True,sharey=True,figsize=(14,6),dpi=400)
        STEPS = true_data.shape[0]
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape
        plt.suptitle(f"Downsampling {D}",fontsize=16)
        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),CHANNEL_NAMES)
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 72, 12))
        axs[0].set_title("True data")
        for _i,c in enumerate(channels):
            i = _i+1
            axs[i].imshow(rearrange(nca_data[c],"T C x y -> (C x) (T y)"),cmap="gray")
            axs[i].set_xlabel("Time")
            axs[i].set_title(f"Hidden channels {c-8}")
        plt.tight_layout()
        # plt.savefig(f"micropattern_individual_8ch_snapshots_{D}ds.png")
        return plt.gca()
    return (visualise_trajectories_per_channels,)


@app.cell
def _(DATA, output_data, visualise_trajectories_per_channels):
    visualise_trajectories_per_channels(output_data,DATA,D=4)
    # visualise_trajectories_per_downsample(output_data,DATA,C=33)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Helper functions to load models and data given hyperparameters""")
    return


@app.cell
def _(load_micropattern_circle_8ch_individual):
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
    return (get_data,)


@app.cell
def _(gNCA, jr, np):
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
    return


@app.cell
def _(gNCA, jr, np):
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
        FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS-1}_ds{DOWNSAMPLE}_texture_pretrain.eqx"
        print(FILENAME)
        nca = gNCA(**NCA_hyperparameters)
        nca = nca.load(f"models/micropattern_individual_8ch/{FILENAME}")
        return nca,{"timsteps":STEPS_BETWEEN_IMAGES}
    return (load_nca_texture_models,)


@app.cell
def _(BOUNDARY, DATA, model_boundary, np):
    def get_x0_and_bmask(ch,ds):
        data = DATA[ds][0] # remove batch index
        boundary = BOUNDARY[ds][0]
        bfunc = model_boundary(mask=boundary)
        data = np.pad(data,((0,0),(0,ch-8),(0,0),(0,0)))
        T = data.shape[0]
        x0 = data[0]
        return x0,bfunc,T
    return (get_x0_and_bmask,)


@app.cell
def _(get_x0_and_bmask, jr, onp, time, tqdm):
    def generate_video(nca,ch,ds,t,filepath,key=jr.PRNGKey(int(time.time()))):
        x,bfunc,T = get_x0_and_bmask(ch,ds)
        XS = [x]
        for i in tqdm(range(t*T)):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            XS.append(onp.array(x))
        return onp.array(XS)
    return (generate_video,)


@app.function
def map_01(x):
    return (x-x.min())/(x.max()-x.min())


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define Hyperparameters""")
    return


@app.cell
def _():
    # DOWNSAMPLES = [16,8,4,2]
    # CHANNELS = [8,10,12,14,16,18,24,32,64]

    DOWNSAMPLES = [16,8,4,2]
    CHANNELS = [8,16,24,32,64]
    CHANNELS = [c+1 for c in CHANNELS] # Add 1 for boundary mask channel
    return CHANNELS, DOWNSAMPLES


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run all models on respective data
    - Takes a few minutes
    """
    )
    return


@app.cell
def _(
    DOWNSAMPLES,
    get_x0_and_bmask,
    hparams,
    jr,
    load_nca_texture_models,
    onp,
    time,
    tqdm,
):
    _key = jr.PRNGKey(int(time.time()))
    output_data = {}
    for _d in DOWNSAMPLES:
        output_data[_d] = {}
    for H in tqdm(hparams):
        try:
            # _nca,_aux = load_nca_models(CHANNELS=H["channels"],DOWNSAMPLE=H["downsample"])
            _nca,_aux = load_nca_texture_models(CHANNELS=H["channels"],DOWNSAMPLE=H["downsample"])

            _t = _aux["timsteps"]
        except:
            print(f"Failed to load with {H}")
            continue

        # _data = DATA[H["downsample"]][0] # remove batch index
        # _boundary = BOUNDARY[H["downsample"]][0]

        # _bfunc = model_boundary(mask=_boundary)
        # # _bfunc = lambda x:x
        # _data = np.pad(_data,((0,0),(0,H["channels"]-8),(0,0),(0,0)))

        # _x = _data[0]
        _x,_bfunc,_T = get_x0_and_bmask(H["channels"],H["downsample"])
        # _T = _data.shape[0] # Outer timesteps corresponding to data
        _XS = []
        for _i in range(_t*_T):
            _key = jr.fold_in(_key,_i)
            _x = _nca(x=_x,boundary_callback=_bfunc,key=_key)
            if _i%_t==0:
                _XS.append(_x[:8])
        _XS = onp.array(_XS)
        output_data[H["downsample"]][H["channels"]] = _XS
        # print(f"Output data shape: {_XS.shape}")
    return (output_data,)


@app.cell
def _(mo):
    mo.md(r"""## Run 1 model and save video of the trajectory""")
    return


@app.cell
def _(generate_video, load_nca_texture_models, rearrange):
    _nca,_aux = load_nca_texture_models(CHANNELS=33,DOWNSAMPLE=4)
    _t = _aux["timsteps"]
    _XS = generate_video(_nca,ch=33,ds=4,t=_t,filepath=None)
    T_texture = map_01(rearrange(_XS[:,:12],"T (cx cy cc) X Y -> T cc (cx X) (cy Y)",cx=2,cy=2,cc=3))
    print(T_texture.shape)


    return (T_texture,)


@app.cell
def _(T_texture, plot_matrix, rearrange):
    plot_matrix(rearrange(T_texture[-1],"c x y -> x (c y)"))
    return


@app.cell
def _(T_texture, rearrange, save_to_video_rgb):
    save_to_video_rgb(
        data=rearrange(T_texture,"N C x y -> N x y C"),#[:,:,:,:3],
        filename="test_micropattern_texture.mp4",
        duration=40
    )
    return


if __name__ == "__main__":
    app.run()
