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
    from NCA.model.NCA_model import NCA
    from NCA.model.NCA_multi_scale import mNCA
    from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual,load_micropattern_circle_8ch_individual_explicit_colony
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations
    import matplotlib.pyplot as plt
    return (
        NCA,
        gNCA,
        generate_hyperparameter_combinations,
        jr,
        load_micropattern_circle_8ch_individual,
        load_micropattern_circle_8ch_individual_explicit_colony,
        mNCA,
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
def _(DOWNSAMPLES, get_data_colony, np):
    DATA = {}
    BOUNDARY = {}
    for d in DOWNSAMPLES:
        _data,_boundary,CHANNEL_NAMES = get_data_colony(DOWNSAMPLE=d)
        _data_split = [_data[:,:,:4],_data[:,:,7:11]]
        _data = np.concatenate(_data_split,axis=2)
        print(_data.shape)
        DATA[d] = _data
        BOUNDARY[d] = _boundary
    CHANNEL_NAMES = CHANNEL_NAMES[:4] + CHANNEL_NAMES[7:11]
    return BOUNDARY, CHANNEL_NAMES, DATA


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


@app.cell(hide_code=True)
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
        # plt.savefig(f"micropattern_individual_8ch_snapshots_{C}ch_long_textural.png")
        return plt.gca()
    return


@app.cell(hide_code=True)
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
        axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs[0].set_title("True data")
        for _i,c in enumerate(channels):
            i = _i+1
            axs[i].imshow(rearrange(nca_data[c],"T C x y -> (C x) (T y)"),cmap="gray")
            axs[i].set_xlabel("Time")
            axs[i].set_title(f"Hidden channels {c-8}")
        plt.tight_layout()
        plt.savefig(f"micropattern_individual_8ch_snapshots_{D}ds_long.png")
        return plt.gca()
    return (visualise_trajectories_per_channels,)


@app.cell
def _(
    CHANNELS,
    CHANNEL_NAMES,
    DATA,
    DOWNSAMPLES,
    generate_hyperparameter_combinations,
    np,
    output_data,
    plt,
    rearrange,
):
    def plot_hparam_set(axs,H,nca_data,H_for_title):
        hparams,data = select_from_output_data(nca_data,H)
        STEPS = data.shape[0]
        data = rearrange(data,"T C x y -> (C x) (T y)")
        xw,yw = data.shape
        axs.imshow(data,cmap="gray")
        axs.set_xlabel("Time")
        axs.set_yticks(np.linspace(xw/16,xw*15/16,8),CHANNEL_NAMES)
        axs.set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs.set_title(hparams[H_for_title])
        return axs



    def visualise_trajectories(nca_data,true_data,H_subset,H_of_interest):
        D = nca_data[1][0]["downsample"]

        true_data = true_data[D][0]
        # nca_data = nca_data[D]
        # channels = list(nca_data.keys())
        fig,axs = plt.subplots(1,len(H_subset)+1,sharex=True,sharey=True,figsize=(14,6),dpi=400)
        STEPS = true_data.shape[0]
        true_data = rearrange(true_data,"T C x y -> (C x) (T y)")
        xw,yw = true_data.shape
        plt.suptitle(f"{H_of_interest}",fontsize=16)
        axs[0].imshow(true_data,cmap="gray")
        axs[0].set_xlabel("Time")
        # axs[0].set_yticks(np.linspace(xw/16,xw*15/16,8),CHANNEL_NAMES)
        # axs[0].set_xticks(np.linspace(yw/(STEPS*2),yw*(STEPS*2-1)/(STEPS*2),STEPS),np.arange(0, 60, 12))
        axs[0].set_title("True data")
        # for _i,c in enumerate(channels):
        #     i = _i+1
        #     axs[i].imshow(rearrange(nca_data[c],"T C x y -> (C x) (T y)"),cmap="gray")
        #     axs[i].set_xlabel("Time")
        #     axs[i].set_title(f"Hidden channels {c-8}")
        for i,H in enumerate(H_subset):
            axs[i+1] = plot_hparam_set(axs[i+1],H,nca_data,H_for_title=H_of_interest)

        plt.tight_layout()
        # plt.savefig(f"micropattern_individual_8ch_snapshots_{D}ds_long.png")
        return plt.gca()

    Hsubset = generate_hyperparameter_combinations({
        "downsample":DOWNSAMPLES,
        "channels":CHANNELS,
        "optimizer":"nadam_blocknorm",
        "loss_mode":["vgg","vgg_grouped","vgg_and_l2","vgg_grouped_and_l2"],
        "model":["NCA"],
        "noise_strength":[0.1],
        "intermediate_growth":[2.0],
        "contiguous_growth":[0.0],
    })
    visualise_trajectories(output_data,DATA,Hsubset,"loss_mode")
    return


@app.cell
def _():
    return


@app.cell
def _(DATA, output_data, visualise_trajectories_per_channels):
    visualise_trajectories_per_channels(output_data,DATA,D=4)
    # visualise_trajectories_per_downsample(output_data,DATA,C=33)
    return


@app.cell
def _(mo):
    mo.md(r"""## Visualise multi-scale NCA on fixed channels and downsampling""")
    return


@app.cell
def _(CHANNEL_NAMES, DATA, multi_output_data, np, plt, rearrange):
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
def _(mo):
    mo.md(r"""## Testing development of grouping channels by experiment""")
    return


@app.cell
def _(multi_output_data, np):

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
    return


@app.cell
def _(load_micropattern_circle_8ch_individual_explicit_colony):
    def get_data_colony(DOWNSAMPLE):
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
    return (get_data_colony,)


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
def _(NCA, jr, np):
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
    return (load_nca_texture_models,)


@app.cell
def _(NCA, jr, np):
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
    return


@app.cell
def _(NCA, gNCA, jr, np):
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
        FILENAME = f"micropattern_circle_8ch_3colony_individual_{LOSS_MODE}_{OPTIMIZER}_int{INTERMEDIATE_GROWTH_COEFF}_contig_{CONTIGUOUS_GROWTH_COEFF}_noise{NOISE_STRENGTH}_{MODEL}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h_stable.eqx"

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

    return (load_nca_texture_grouped_noise,)


@app.cell
def _(jr, mNCA, np):
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
    return (load_mnca_texture_hyperparameter_sweep,)


@app.cell
def _(BOUNDARY, DATA, model_boundary, np):
    def get_x0_and_bmask(ch,ds):
        data = DATA[ds][0] # remove batch index
        boundary = BOUNDARY[ds][0]
        boundary = np.pad(boundary,((0,0),(3,3),(3,3)))
        bfunc = model_boundary(mask=boundary)
        # data = np.pad(data,((0,0),(0,ch-8),(0,0),(0,0)))
        data = np.pad(data,((0,0),(0,ch-8),(3,3),(3,3))) 
        print(f"Boundary shape {boundary.shape}")
        print(f"Data shape {data.shape}")
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


@app.function
def select_from_output_data(output_data,hparams):
    return [D for D in output_data if D[0]==hparams][0]


@app.function
def select_many_from_output_data(output_data,hparams):
    output_data_selected = []
    for H in hparams:
        output_data_selected.append(select_from_output_data(output_data,H))
    return output_data_selected


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define Hyperparameters""")
    return


@app.cell
def _(generate_hyperparameter_combinations):
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

    CHANNELS = [32]
    # REG = ["","_int","_int_contig","_contig"]
    DOWNSAMPLES = [8]
    # OPTIMIZER = ["nadam_blocknorm"]

    hparams = generate_hyperparameter_combinations({
        "downsample":DOWNSAMPLES,
        "channels":CHANNELS,
        "optimizer":"nadam_blocknorm",
        "loss_mode":["vgg","vgg_grouped","vgg_and_l2","vgg_grouped_and_l2"],
        "model":["NCA","gNCA"],
        "noise_strength":[0.001,0.01,0.1],
        "intermediate_growth":[0.0,0.2,1.0,2.0],
        "contiguous_growth":[0.0],
    })

    # CHANNELS = [c+1 for c in CHANNELS] # Add 1 for boundary mask channel
    return CHANNELS, DOWNSAMPLES, hparams


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run all models on respective data
    - Takes a few minutes
    - Sweeps over Channels and Downsampling
    """
    )
    return


@app.cell
def _(
    get_x0_and_bmask,
    hparams,
    jr,
    load_nca_texture_grouped_noise,
    onp,
    time,
    tqdm,
):
    _key = jr.PRNGKey(int(time.time()))
    output_data = []


    def run_model_with_hparams(H,key):
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
        for i in range(t*T):
            key = jr.fold_in(key,i)
            x = nca(x=x,boundary_callback=bfunc,key=key)
            if i%t==0:
                XS.append(x[:8])
        XS = onp.array(XS)
        output_data.append((H,XS))


    for _H in tqdm(hparams):
        _key = jr.fold_in(_key,1)
        run_model_with_hparams(_H,key=_key)

    # for _d in DOWNSAMPLES:
    #     output_data[_d] = {}
    # for H in tqdm(hparams):
    #     try:
    #         _nca,_aux = load_nca_models(CHANNELS=H["channels"],DOWNSAMPLE=H["downsample"])
    #         _nca,_aux = load_nca_texture_models(CHANNELS=H["channels"],DOWNSAMPLE=H["downsample"])
    #     _nca,_aux = load_nca_texture_grouped(
    #         CHANNELS=H["channels"],
    #         DOWNSAMPLE=H["downsample"],
    #         OPT="nadam",
    #         REG="_int"
    #     )

    #     _t = _aux["timsteps"]
    #     # except:
    #         # print(f"Failed to load with {H}")
    #         # continue

    #     # _data = DATA[H["downsample"]][0] # remove batch index
    #     # _boundary = BOUNDARY[H["downsample"]][0]

    #     # _bfunc = model_boundary(mask=_boundary)
    #     # # _bfunc = lambda x:x
    #     # _data = np.pad(_data,((0,0),(0,H["channels"]-8),(0,0),(0,0)))

    #     # _x = _data[0]
    #     _x,_bfunc,_T = get_x0_and_bmask(H["channels"],H["downsample"])
    #     # _T = _data.shape[0] # Outer timesteps corresponding to data
    #     _XS = []
    #     for _i in range(_t*_T):
    #         _key = jr.fold_in(_key,_i)
    #         _x = _nca(x=_x,boundary_callback=_bfunc,key=_key)
    #         if _i%_t==0:
    #             _XS.append(_x[:8])
    #     _XS = onp.array(_XS)
    #     output_data[H["downsample"]][H["channels"]] = _XS
    #     print(f"Output data shape: {_XS.shape}")
    return (output_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Run multi-scale NCA on fixed channels and downsampling
    - Iterates over different training hyperparameters
        - Loss function configuration
        - Spatial gradient loss
    """
    )
    return


@app.cell
def _(
    GRADS,
    LOSS_MODES,
    generate_hyperparameter_combinations,
    get_x0_and_bmask,
    jr,
    load_mnca_texture_hyperparameter_sweep,
    onp,
    time,
    tqdm,
):
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


@app.cell
def _(mo):
    mo.md(r"""## Run 1 model and save video of the trajectory""")
    return


@app.cell
def _(generate_video, load_nca_texture_models, rearrange):
    _nca,_aux = load_nca_texture_models(CHANNELS=33,DOWNSAMPLE=2)
    _t = _aux["timsteps"]
    _XS = generate_video(_nca,ch=33,ds=2,t=_t,filepath=None)
    T_texture = map_01(rearrange(_XS[:,:9],"T (cx cy cc) X Y -> T cc (cx X) (cy Y)",cx=3,cy=1,cc=3))
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
        filename="micropattern_texture_ds2.mp4",
        duration=40
    )
    return


if __name__ == "__main__":
    app.run()
