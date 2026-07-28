import marimo

__generated_with = "0.18.4"
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
    import numpy as onp
    from einops import rearrange,repeat,reduce
    from NCA.model.NCA_gated_model import gNCA
    from NCA.model.NCA_model import NCA
    from Common.dataloader.emoji import load_emoji_sequence
    from Common.save_to_video import save_to_video_rgb
    import matplotlib.pyplot as plt
    from Experiments.multispecies.train_good import prepare_data
    return (
        eqx,
        gNCA,
        jax,
        jr,
        load_emoji_sequence,
        mo,
        np,
        onp,
        plt,
        prepare_data,
        rearrange,
        reduce,
        repeat,
        save_to_video_rgb,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run batch parallel multi-species growth
    - here we run the trained NCA model (vmapped across batches) from multiple initial conditions at once
    """)
    return


@app.cell
def _(
    CHANNELS,
    jr,
    load_new_model,
    onp,
    pad_with_hidden_channels,
    plot_matrix,
    plt,
    prepare_data,
    rearrange,
    tqdm,
):
    vnca = load_new_model()
    _key = jr.PRNGKey(123)
    # data = load_pixel_data()
    data,_ = prepare_data(BATCHES=1,mode="pixel",)
    data = pad_with_hidden_channels(data,CHANNELS)
    print(data.shape)
    _X = data[:2,0]
    print(_X.shape)
    plt.imshow(_X[0,0])
    plt.show()
    # X = rearrange(data[:4,0],"B C x y -> C (B x)")
    _save_every = 32
    _T = []
    for _t in tqdm(range(1024)):
        _key = jr.fold_in(_key,_t)
        # _X = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)(_X,lambda x:x,_key)
        _X = vnca(_X,lambda x:x,_key)
        # X = nca(X,lambda x:x,_key)
        # if (_t%_save_every)==0:
        _T.append(onp.array(_X[:,:]))
    T_batch = onp.array(_T)
    plot_matrix(rearrange(T_batch[::32,:,:3],"T B c x y ->(B x) (T y) c"))

    # T = nca.run(iters=190,x=data[2,0])
    # print(T)
    return T_batch, data


@app.cell
def _():
    return


@app.cell
def _(T_batch, onp, rearrange, save_to_video_rgb):
    print(T_batch.shape)
    T_video = T_batch[:,1,3:27]
    T_video = onp.clip(T_video,0.0,1.0)
    print(T_video.shape)
    # T_video = rearrange(T_batch[:,1,:3],"T c x y -> T x y c")
    T_video = rearrange(T_video,"Time (C Cx Cy) x y -> Time (Cx x) (Cy y) C",Cx=2,C=3)
    print(T_video.shape)

    # T_video = onp.clip(T_video,0.0,1.0)
    save_to_video_rgb(T_video,"Videos/ThesisEmojis/pixel_to_emoji_2_hidden.mp4",fps=30,duration=20,SCALE_UP=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One big batch including multiple initial conditions
    - We can reshape the initial conditions to have 1 large lattice with multiple initial conditions. Do the different growing structures interact then?
    """)
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
    mo.md(r"""
    ## Linear interpolation of 2 initial conditions
    """)
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
    mo.md(r"""
    ## Linear splice of 2 initial conditions
    """)
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
    mo.md(r"""
    # Notes
    - Given the trained NCA, can we find minimal perturbations from one stable state to another?
        - If not, can we augment the training process to fascilitate it?

    - Global perturbations are probably easy to find, local ones harder. Which do we want?

    - Can we enforce contigous growth with an auxiliary loss? i.e. cells don't come alive (certain channels values get large) unless a neighbour is alive
    """)
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    # Helper functions
    """)
    return


@app.cell
def _():
    CHANNELS = 32
    return (CHANNELS,)


@app.cell
def _(plt):
    def plot_matrix(data):
        plt.figure(figsize=(12,8))
        plt.imshow(data)
        return plt.gca()
    return (plot_matrix,)


@app.cell
def _(eqx, gNCA, jax, jr):
    def load_old_models():
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
        return vnca
    # nca = eqx.filter_jit(nca)
    # vnca = eqx.filter_jit(vnca)
    # models/eidf_runs/multi_species_stable_gNCA_grad_64ch_v1_cr_mi_av_al_bt_li_mu_ds_1_long.eqx
    return


@app.cell
def _(eqx, gNCA, jax, jr):
    def load_new_model():
        CHANNELS = 32
        OBS_CHANNELS = 4
        key = jr.PRNGKey(0)
        nca = gNCA(
            N_CHANNELS=CHANNELS,
            KERNEL_STR=["ID", "GRAD", "LAP"],
            ACTIVATION=jax.nn.relu,
            PADDING="CIRCULAR",
            FIRE_RATE=0.5,
            key=key,)
        nca = nca.load("models/signal_stability/good_emoji_multi_species_cr_mi_pixel_gNCA_intermediate_contiguous_reg_32ch_t128_standard.eqx")
        vnca = eqx.filter_vmap(nca,in_axes=(0,None,None),out_axes=0)
        return vnca
    return (load_new_model,)


@app.cell
def _(CHANNELS, load_emoji_sequence, np, rearrange, repeat):
    def load_patch_data():
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
        return data
        # plt.imshow()
        # plot_matrix(rearrange(data,"B T C x y -> (B x) (T y) C"))
    return


@app.cell
def _(np):
    def pad_with_hidden_channels(data,channels):
        data = np.concatenate(
            [
                data,
                np.zeros((data.shape[0],data.shape[1],channels-data.shape[2],data.shape[3],data.shape[4]))
            ],
            axis=2
        )
        data = np.pad(data,pad_width=((0,0),(0,0),(0,0),(20,20),(20,20)))
        return data
    return (pad_with_hidden_channels,)


@app.cell
def _(CHANNELS, load_emoji_sequence, np, rearrange, repeat):
    def load_pixel_data():
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

        initial_condition = initial_condition.at[:, :, :, : W // 2 - 1].set(0)
        initial_condition = initial_condition.at[:, :, :, W // 2:].set(0)
        initial_condition = initial_condition.at[:, :, :, :, :H // 2 - 1].set(0)
        initial_condition = initial_condition.at[:, :, :, :, H // 2:].set(0)
        data = np.concatenate(
            [initial_condition, data, data], axis=1
        )  # Join initial condition and data along the time axis
        data = np.concatenate([data,np.zeros((data.shape[0],data.shape[1],CHANNELS-data.shape[2],data.shape[3],data.shape[4]))],axis=2)
        print("(Batch, Time, Channels, Width, Height): " + str(data.shape))
        return data
    return


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
