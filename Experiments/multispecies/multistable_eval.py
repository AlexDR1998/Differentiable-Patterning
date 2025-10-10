import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    # print(sys.path)
    import marimo as mo
    import jax 
    import jax.numpy as np
    import jax.random as jr
    import equinox as eqx
    from tqdm import tqdm
    from einops import rearrange,repeat,reduce
    from NCA.model.NCA_gated_model import gNCA
    from Common.dataloader.emoji import load_emoji_sequence
    import matplotlib.pyplot as plt
    return (
        eqx,
        gNCA,
        jax,
        jr,
        load_emoji_sequence,
        mo,
        np,
        plt,
        rearrange,
        reduce,
        repeat,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run batch parallel multi-species growth
    - here we run the trained NCA model (vmapped across batches) from multiple initial conditions at once
    """
    )
    return


@app.cell
def _(data, jr, key, np, plot_matrix, rearrange, tqdm, vnca):
    _key = key
    _X = data[:2,0]
    # X = rearrange(data[:4,0],"B C x y -> C (B x)")
    _save_every = 32
    _T = []
    for _t in tqdm(range(512)):
        _key = jr.fold_in(_key,_t)
        # _X = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)(_X,lambda x:x,_key)
        _X = vnca(_X,lambda x:x,_key)
        # X = nca(X,lambda x:x,_key)
        if (_t%_save_every)==0:
            _T.append(_X)
    T_batch = np.array(_T)
    plot_matrix(rearrange(T_batch[:,:,:4],"T B c x y ->(B x) (T y) c"))
    # T = nca.run(iters=190,x=data[2,0])
    # print(T)
    return (T_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## One big batch including multiple initial conditions
    - We can reshape the initial conditions to have 1 large lattice with multiple initial conditions. Do the different growing structures interact then?
    """
    )
    return


@app.cell
def _(data, jr, key, nca, np, plot_matrix, rearrange, tqdm):
    _key = key
    # _X = data[4:,0]
    _shrink = 10
    _X = rearrange(data[:2,0,:,_shrink:-_shrink,_shrink:-_shrink],"(bx by) C x y -> C (bx x) (by y)",bx=2,by=1)
    print(_X.shape)
    _save_every = 32
    _T = []
    for _t in tqdm(range(256)):
        _key = jr.fold_in(_key,_t)
        # _X = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)(_X,lambda x:x,_key)
        _X = nca(_X,lambda x:x,_key)
        if (_t%_save_every)==0:
            _T.append(_X)
    T_wide = np.array(_T)
    plot_matrix(rearrange(T_wide[:,:4],"T c x y -> x (T y) c"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Linear interpolation of 2 initial conditions""")
    return


@app.cell
def _(mo):
    ic_blend_slider = mo.ui.slider(0,1,0.1)
    ic_blend_slider
    return (ic_blend_slider,)


@app.cell
def _(data, ic_blend_slider, jr, key, nca, np, plot_matrix, rearrange, tqdm):
    _ic = data[0,0]*(ic_blend_slider.value) + data[6,0]*(1-ic_blend_slider.value)
    # print(_ic.shape)
    # plot_matrix(rearrange(_ic[:4],"c x y -> x y c"))
    _X = _ic
    _save_every = 32
    _T = []
    _key = key
    for _t in tqdm(range(640)):
        _key = jr.fold_in(_key,_t)
        # _X = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)(_X,lambda x:x,_key)
        _X = nca(_X,lambda x:x,_key)
        if (_t%_save_every)==0:
            _T.append(_X)
    T_blend = np.array(_T)
    plot_matrix(rearrange(T_blend[:,:4],"T c x y -> x (T y) c"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Linear splice of 2 initial conditions""")
    return


@app.cell
def _(W, mo):
    ic_splice_slider = mo.ui.slider(W//2-6,W//2+5,1)
    # ic_splice_angle = mo.ui.slider(-np.pi,np.pi,0.1)
    ic_splice_slider

    return (ic_splice_slider,)


@app.cell
def _(data, ic_splice_slider, jr, key, nca, np, plot_matrix, rearrange, tqdm):
    # _angle = ic_splice_angle.value
    # _slider_pos = ic_splice_slider.value*
    _mask = np.zeros(data[:,0].shape)
    _mask = _mask.at[0,:,:,:ic_splice_slider.value].set(1.0)
    _mask = _mask.at[1,:,:,ic_splice_slider.value:].set(1.0)
    _ic = data[0,0]*_mask[0] + data[1,0]*_mask[1]
    _X = _ic
    _save_every = 32
    _T = []
    _key = key
    for _t in tqdm(range(256)):
        _key = jr.fold_in(_key,_t)
        # _X = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)(_X,lambda x:x,_key)
        _X = nca(_X,lambda x:x,_key)
        if (_t%_save_every)==0:
            _T.append(_X)
    T_splice = np.array(_T)
    plot_matrix(rearrange(T_splice[:,:4],"T c x y -> x (T y) c"))
    # print(_ic.shape)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Notes
    - Given the trained NCA, can we find minimal perturbations from one stable state to another?
        - If not, can we augment the training process to fascilitate it?

    - Global perturbations are probably easy to find, local ones harder. Which do we want?

    - Can we enforce contigous growth with an auxiliary loss? i.e. cells don't come alive (certain channels values get large) unless a neighbour is alive
    """
    )
    return


@app.cell(column=1)
def _():
    return


@app.cell
def _(plt):
    def plot_matrix(data):
        plt.figure(figsize=(12,8))
        plt.imshow(data)
        return plt.gca()
    return (plot_matrix,)


@app.cell
def _(eqx, gNCA, jax, jr):
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
    # nca = nca.load("models/eidf_runs/multi_species_stable_gNCA_grad_64ch_v1_cr_mi_av_al_bt_li_mu_ds_1_long.eqx")
    nca = nca.load("models/eidf_runs/multi_species_stable_gNCA_grad_64ch_contiguous_1.0_wide_perturbation_0.1_cr_mi_ds_1_long.eqx")
    # nca = lambda x,key:nca(x,lambda x:x,key)
    # nca = eqx.filter_jit(nca)
    vnca = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)
    # nca = eqx.filter_jit(nca)
    # vnca = eqx.filter_jit(vnca)
    # models/eidf_runs/multi_species_stable_gNCA_grad_64ch_v1_cr_mi_av_al_bt_li_mu_ds_1_long.eqx
    return CHANNELS, OBS_CHANNELS, key, nca, vnca


@app.cell
def _(CHANNELS, load_emoji_sequence, np, rearrange, repeat):
    DOWNSAMPLE = 1
    BATCH = 1
    data = load_emoji_sequence(
        [
            "crab.png",
            "microbe.png",
            "avocado.png",
            "alien_monster.png",
            "butterfly.png",
            "lizard.png",
            "mushroom.png",
        ],
        downsample=DOWNSAMPLE,
        impath_emojis="../Data/Emojis/",
    )
    data_filename = "cr_mi_av_al_bt_li_mu"

    data = rearrange(data, "B T C W H -> T B C W H")
    data = np.pad(data,pad_width=((0,0),(0,0),(0,0),(20,20),(20,20)))
    data = repeat(data, "B T C W H -> (B b) T C W H", b=BATCH)

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
    data = np.concatenate([data,np.zeros((data.shape[0],data.shape[1],CHANNELS-data.shape[2],data.shape[3],data.shape[4]))],axis=2)
    print("(Batch, Time, Channels, Width, Height): " + str(data.shape))
    # plt.imshow()
    # plot_matrix(rearrange(data,"B T C x y -> (B x) (T y) C"))
    return W, data


@app.cell
def _(OBS_CHANNELS, jax, np, reduce, repeat):
    def cont_reg(x,x_previous):
        x = x[:,:OBS_CHANNELS]
        x_previous = x_previous[:,:OBS_CHANNELS]
        dx = x - x_previous # How much obs growth
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.float32)
        kernel = repeat(kernel,"w h -> O I w h",O=1,I=OBS_CHANNELS)

        dilation = jax.lax.conv_general_dilated(
            lhs=x_previous,
            rhs=kernel,
            window_strides=(1, 1),
            padding="SAME",

        )
        dilation = jax.nn.sigmoid((dilation-5.0)*10)
        dilation = repeat(dilation,"B () x y -> B C x y",C=OBS_CHANNELS)
        err = dilation*dx
        print(np.max(dilation))
        print(f"Dilation shape: {dilation.shape}")
        print(f"dx shape: {dx.shape}")
        err = reduce(err,"B C x y -> B","mean")


        # return dilation*dx
        # return err
        return dilation
    return (cont_reg,)


@app.cell
def _(T_batch, cont_reg, plot_matrix, rearrange):
    print(T_batch.shape)
    _d = cont_reg(T_batch[:,0],T_batch[:,4])
    print(_d.shape)
    plot_matrix(rearrange(_d,"T C x y -> (C x) (T y)"))
    return


if __name__ == "__main__":
    app.run()
