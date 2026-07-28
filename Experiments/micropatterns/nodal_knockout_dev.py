import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    import jax 
    import jax.numpy as np
    import numpy as onp
    # from Experiments.micropatterns.micropattern_individual_eval import get_data_colony
    from Common.dataloader.micropattern import load_micropattern_circle_nodal_knockout_9ch_explicit_colony,load_micropattern_circle_4ch_individual
    from Common.dataloader.micropattern import load_micropattern_260726
    import matplotlib.pyplot as plt
    from einops import rearrange
    from pprint import pprint

    return (
        load_micropattern_260726,
        load_micropattern_circle_4ch_individual,
        load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
        onp,
        plt,
        pprint,
        rearrange,
    )


@app.cell
def _(data_14ch, plt, rearrange):
    print(data_14ch[0].shape)
    C = 0
    plt.figure(figsize=(12,12))
    plt.imshow(rearrange(data_14ch[0][:,-1],"B C x y -> (C x) (B y)"))
    # plt.imshow(data_14ch[0][1,-1,9])
    return


@app.cell
def _(data_14ch):
    print(data_14ch[2])
    return


@app.cell
def _(load_micropattern_260726):
    data_14ch = load_micropattern_260726(
        root="../Data/260726_nca_dataset",
        downsample=4,
        experiment_groups=["cell_fate_s1","protein_response"],
        align=True,
        boundary_radius_quantile=0.98)
    return (data_14ch,)


@app.cell
def _(data_14ch, plt):
    print(data_14ch[3].shape)
    plt.imshow(data_14ch[3][2,0])
    return


@app.cell
def _(load_micropattern_circle_4ch_individual):
    data_4ch = load_micropattern_circle_4ch_individual(
            impath="../Data/Timecourse Seperate Colonies/A/*",)
    return (data_4ch,)


@app.cell
def _(load_micropattern_circle_nodal_knockout_9ch_explicit_colony):
    data = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath="../Data/Timecourse Seperate Colonies/",FILTER_KN_TIME=None)
    # print(data.shape)
    return (data,)


@app.cell
def _(data, data_4ch, plt, rearrange):
    print(data[0].shape)
    print(data_4ch[0].shape)

    plt.imshow(rearrange(data[0][0,:,0],"T X Y -> X (T Y)"))
    plt.show()
    # plt.imshow(data_4ch[0][0,-1,0])
    plt.imshow(rearrange(data_4ch[0][0,:,0],"T X Y -> X (T Y)"))
    plt.show()
    return


@app.cell
def _(load_micropattern_circle_nodal_knockout_9ch_explicit_colony):
    def get_data_colony_knockout(DOWNSAMPLE):
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
        data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath="../Data/Timecourse Seperate Colonies/",
            FILTER_KN_TIME=24,
            DOWNSAMPLE = DOWNSAMPLE,
            BATCHES=1,
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
                # "downsample",
            }
        )
        return data,boundary_mask,CHANNEL_NAMES,CHANNEL_TIMESTEP_MASK

    return (get_data_colony_knockout,)


@app.cell
def _(get_data_colony_knockout, plt):
    data_2,_mask,chnames,CHANNEL_TIMESTEP_MASK = get_data_colony_knockout(4)
    # print(_data.shape)
    plt.imshow(_mask[0,0])
    # print(_data[1].shape)
    # print(len(_data))
    return CHANNEL_TIMESTEP_MASK, chnames, data_2


@app.cell
def _(onp):
    def duplicate_x_channels_9ch(x):
        """
            Duplicates channels in x to match the individual colony experiment groups.
            X [N C H W] with C=9 -> [N 12 H W] with channels duplicated as needed.
            data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal" - "lef1" ]
            input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal","lef1"]


        """
        x_dup = [x[:,0:4],x[:,0:3],x[:,4:8],x[:,8:9]]
        return onp.concatenate(x_dup,axis=1)

    return (duplicate_x_channels_9ch,)


@app.cell
def _(
    CHANNEL_TIMESTEP_MASK,
    chnames,
    data_2,
    duplicate_x_channels_9ch,
    plt,
    pprint,
    rearrange,
):
    # _b = 5
    print(type(data_2))
    # print(len(data_2))
    # for l in data_2:
        # print(l.shape)
    print(data_2.shape)
    mask_full = duplicate_x_channels_9ch(CHANNEL_TIMESTEP_MASK)
    print(mask_full.shape)
    # print(data_2[_b].shape)
    pprint(chnames)

    plt.imshow(rearrange(data_2,"() T C X Y -> (C X) (T Y)"))
    plt.show()
    data_masked = data_2[:,1:]*mask_full[None,:,:,None,None]
    # plt.imshow(rearrange(data_masked,"() T C X Y -> (C X) (T Y)"))
    plt.imshow(mask_full.T)
    plt.colorbar()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
