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
    from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations
    import matplotlib.pyplot as plt
    from pprint import pprint
    return (
        NCA,
        generate_hyperparameter_combinations,
        jr,
        load_micropattern_circle_8ch_individual,
        mo,
        model_boundary,
        np,
        onp,
        plt,
        rearrange,
        time,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load true data""")
    return


@app.cell
def _(DOWNSAMPLES, get_data):
    # hparams = generate_hyperparameter_combinations({"downsample":DOWNSAMPLES,"channels":CHANNELS})

    DATA = {}
    BOUNDARY = {}
    for d in DOWNSAMPLES:
        _data,_boundary,CHANNEL_NAMES = get_data(DOWNSAMPLE=d)
        DATA[d] = _data
        BOUNDARY[d] = _boundary
    return BOUNDARY, CHANNEL_NAMES, DATA


@app.cell
def _(CHANNEL_NAMES, DATA, nca_data, np, plt, rearrange):
    def visualise_trajectories_opt(nca_data,true_data):
        true_data = true_data[4][0]
        STEPS = true_data.shape[0]
        # true_data = np.pad(true_data,((0,0),(0,0),(3,3),(3,3)))
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
            try:
                _xs = nca_data[m]["_blocknorm"]
                # print(_xs)
                axs[i].imshow(rearrange(_xs,"T C x y -> (C x) (T y)"),cmap="gray")
            except:
                pass
            axs[i].set_xlabel("Time")
            axs[i].set_title(f"Loss mode: {m}")
        plt.tight_layout()
        return plt.gca()
        # print(modes)
    visualise_trajectories_opt(nca_data,DATA)
    return


@app.cell(column=1)
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
def _(NCA, jr, np):
    def load_nca_model(CHANNELS,DOWNSAMPLE,OPT,BLOCK,MS,TASK):
        key = jr.PRNGKey(0)
        NCA_hyperparameters = {
            "N_CHANNELS":CHANNELS, # Fix for hidden channels
            "KERNEL_STR":["ID","LAP","DIFF"],
            "FIRE_RATE":0.5,
            "PADDING":"circular",
            "key":key
        }
        STEPS_BETWEEN_IMAGES = int(256/np.sqrt(DOWNSAMPLE))
        FILENAME = f"optimizer_test_{OPT}{MS}{BLOCK}_{TASK}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}.eqx"
        print(FILENAME)
        nca = NCA(**NCA_hyperparameters)
        nca = nca.load(f"models/optimizer_test/{FILENAME}")
        return nca,{"timsteps":STEPS_BETWEEN_IMAGES}
    # load_nca_model(32,4,"muon","_blocknorm","","micropattern")
    return (load_nca_model,)


@app.cell
def _(BOUNDARY, DATA, model_boundary, np):
    def get_x0_and_bmask(ch,ds):
        data = DATA[ds][0] # remove batch index
        boundary = BOUNDARY[ds][0]
        # boundary = np.pad(boundary,((0,0),(3,3),(3,3)))
        bfunc = model_boundary(mask=boundary)
        data = np.pad(data,((0,0),(0,ch-8),(0,0),(0,0)))
        # data = np.pad(data,((0,0),(0,ch-8),(3,3),(3,3))) 
        print(f"Boundary shape {boundary.shape}")
        print(f"Data shape {data.shape}")
        T = data.shape[0]
        x0 = data[0]
        return x0,bfunc,T
    return (get_x0_and_bmask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define hyperparameters of interest""")
    return


@app.cell
def _():
    # DOWNSAMPLES = [16,8,4,2]
    # CHANNELS = [8,10,12,14,16,18,24,32,64]

    # Euclidean loss
    # DOWNSAMPLES = [16,8,4,2]
    # CHANNELS = [8,16,24,32,64]

    # Textural loss
    DOWNSAMPLES = [4]
    CHANNELS = [32]

    FULL_HYPERPARAMETERS = {
        "optimizer":["optimistic_adam","nadam","muon","sam"],
        "blocknorm":["","_blocknorm"],
        "multistep":[""],
        "task": ["micropattern"]
        # "multistep":["","_multistep2","_multistep4"],
        # "task":["micropattern","emoji"]        
    }

    # CHANNELS = [c+1 for c in CHANNELS] # Add 1 for boundary mask channel
    return DOWNSAMPLES, FULL_HYPERPARAMETERS


@app.cell
def _(nca_data):
    print(nca_data["muon"]["_blocknorm"].shape)
    return


@app.cell
def _(
    FULL_HYPERPARAMETERS,
    generate_hyperparameter_combinations,
    get_x0_and_bmask,
    jr,
    load_nca_model,
    onp,
    time,
    tqdm,
):
    def run_all_models():
        key = jr.PRNGKey(int(time.time()))
        hparam_list = generate_hyperparameter_combinations(FULL_HYPERPARAMETERS)
        # pprint(hparam_list)
        output_data = {}
        for opt in FULL_HYPERPARAMETERS["optimizer"]:
            output_data[opt] = {}
            for bn in FULL_HYPERPARAMETERS["blocknorm"]:
                output_data[opt][bn] = {}
        for H in tqdm(hparam_list):
            # print(H)
            try:
                nca,aux = load_nca_model(
                    CHANNELS=32,
                    DOWNSAMPLE=4,
                    OPT=H["optimizer"],
                    BLOCK=H["blocknorm"],
                    MS=H["multistep"],
                    TASK=H["task"]
                )
                t = aux["timsteps"]
            except:
                # print()
                print(f"failed to load with {H}")
                continue
            x,bfunc,T = get_x0_and_bmask(32,4)


            XS = []
            for i in range(t*T):
                key = jr.fold_in(key,i)
                x = nca(x=x,boundary_callback=bfunc,key=key)
                if i%t==0:
                    XS.append(x[:8])
            XS = onp.array(XS)
            output_data[H["optimizer"]][H["blocknorm"]] = XS
        return output_data
    nca_data = run_all_models()
    return (nca_data,)


if __name__ == "__main__":
    app.run()
