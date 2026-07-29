import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")

with app.setup:

    import marimo as mo
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    # print(sys.path)
    import jax 
    # jax.config.update('jax_platform_name', 'cpu')
    import jax.numpy as np
    import numpy as onp
    import jax.random as jr
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
    try:
        from Common.save_to_video import save_to_video_rgb
    except ModuleNotFoundError as e:
        _save_video_import_error = e
        def save_to_video_rgb(*args, **kwargs):
            raise ModuleNotFoundError(
                "save_to_video_rgb requires optional video dependencies such as cv2."
            ) from _save_video_import_error
    def _missing_optional_helper(name, error):
        def _helper(*args, **kwargs):
            raise ModuleNotFoundError(
                f"{name} requires optional experiment dependencies that are not installed."
            ) from error
        return _helper

    try:
        from Experiments.emoji.old_scripts.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
    except ModuleNotFoundError as e:
        H_to_filename_gate = _missing_optional_helper("H_to_filename_gate", e)
    try:
        from Experiments.emoji.old_scripts.parameter_noise_sweep import H_to_filename as H_to_filename_noise
    except ModuleNotFoundError as e:
        H_to_filename_noise = _missing_optional_helper("H_to_filename_noise", e)
    try:
        from Experiments.emoji.old_scripts.fire_rate_sweep import H_to_filename as H_to_filename_fr
    except ModuleNotFoundError as e:
        H_to_filename_fr = _missing_optional_helper("H_to_filename_fr", e)
    try:
        from Experiments.emoji.old_scripts.local_perturbation import run as run_local_perturbations
    except ModuleNotFoundError as e:
        run_local_perturbations = _missing_optional_helper("run_local_perturbations", e)
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations,generate_hyperparameter_combinations_indexed
    import matplotlib.pyplot as plt
    from pprint import pprint
    from skimage import measure
    import matplotlib.style
    matplotlib.style.use(
        "default"
    )
    # from Common.dataloader.micropattern import load_micropattern_shape_sequence
    from Common.dataloader.emoji import load_emoji_sequence


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load models and data
    """)
    return


@app.cell
def _(HYPERPARAMETERS_LOSS):
    models_loss = []
    for _H in tqdm(HYPERPARAMETERS_LOSS):

        try:
            _nca = load_emoji_models_loss(_H)
        except:
            print(f"Failed to load model with hyperparameters: {_H}")
            continue
        models_loss.append((_nca,_H))
    print(len(models_loss))
    return (models_loss,)


@app.cell
def _(HYPERPARAMETERS_W_LOSS):
    models_w_loss = []
    for _H in tqdm(HYPERPARAMETERS_W_LOSS):

        try:
            _nca = load_emoji_models_w_loss(_H)
        except:
            print(f"Failed to load model with hyperparameters: {_H}")
            continue
        models_w_loss.append((_nca,_H))
    print(len(models_w_loss))
    return (models_w_loss,)


@app.cell
def _(HYPERPARAMETERS_REG):
    models_reg = []
    for _H in tqdm(HYPERPARAMETERS_REG):

        # try:
        _nca = load_emoji_models_regulariser(_H)
        # _nca = load_emoji_models_contig(_H)
        # except:
            # print(f"Failed to load model with hyperparameters: {_H}")
            # continue
        models_reg.append((_nca,_H))
    print(len(models_reg))
    return (models_reg,)


@app.cell
def _(HYPERPARAMETERS_CONTIG):
    models_contig = []
    for _H in tqdm(HYPERPARAMETERS_CONTIG):

        # try:
        _nca = load_emoji_models_contig(_H)
        # except:
            # print(f"Failed to load model with hyperparameters: {_H}")
            # continue
        models_contig.append((_nca,_H))
    return (models_contig,)


@app.cell
def _(HYPERPARAMETERS_NOISE):
    models_noise = []
    for _H in tqdm(HYPERPARAMETERS_NOISE):
        try:
            _nca = load_emoji_models_noise(_H)
            models_noise.append((_nca,_H))
        except:
            pass
    print(len(models_noise))
    return (models_noise,)


@app.cell
def _(HYPERPARAMETERS_FR):
    models_fr = []
    for _H in tqdm(HYPERPARAMETERS_FR):
        _nca = load_emoji_models_fire_rate(_H)
        models_fr.append((_nca,_H))
    return (models_fr,)


@app.cell
def _():
    data = load_emoji_sequence(
            ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
            downsample=1
        ).data
    print(data.shape) # Batch T C x y
    data = data[0] # discard batch dimenension
    plt.imshow(rearrange(data[:,:3],"T C x y -> x (T y) C"))
    return (data,)


@app.cell
def _():
    return


@app.cell
def _():
    mo.md(r"""
    # Local perturbation visualisation
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Effects of gating and regularizers
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_PERTURBATION = {
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        "model":["gNCA"],
        # "channels":[16,32],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[64],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0,0.01,0.1,1.0],
        "update_sensitivity_coeff":[0.0],
        # "update_sensitivity_coeff":[0.0], 
        "timesteps":[3],
        "fire_rate":[0.5],
        # "regen_str":["regenerate_",""],
        # "regen_str":[""],
        "regenerate":[False],
        "perturbation_mode":["large"]
    }
    HYPERPARAMETERS_PERTURBATION = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_PERTURBATION)
    pprint(HYPERPARAMETERS_PERTURBATION)
    return (HYPERPARAMETERS_PERTURBATION,)


@app.cell
def _(HYPERPARAMETERS_PERTURBATION):
    _perturbation_channel = "all"
    _coordinate_list = [(10*4,28*4),(30*4,14*4),(18*4,18*4)]
    _H = HYPERPARAMETERS_PERTURBATION[3]

    pprint(_H)
    visualise_local_perturbation(_H,_coordinate_list,ch=_perturbation_channel,name_func=H_to_filename_gate)
    # pprint(_H)
    # visualise_full_local_perturbation_map(_H,coordinate_list,ch=_perturbation_channel)
    # for _coord in coordinate_list:
        # visualise_local_perturbation_coords((_perturbation_channel,*_coord),_H)
    return


@app.cell
def _():
    return


@app.cell
def _(HYPERPARAMETERS_PERTURBATION):
    _key = jr.PRNGKey(int(time.time()))
    for _H in HYPERPARAMETERS_PERTURBATION:
        run_local_perturbations(
            _H,
            FULL_RUN=False,
            key=_key,
            NAME_FUNC=H_to_filename_gate,
            video_aux={
                "save":True,
                "output_path":"Videos/ThesisEmojis/Figure 4.10/",
                "duration":20,
                "scale_up":4,
            }
        )
    return


@app.cell
def _():
    mo.md(r"""
    ## Effect of fire rate
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_LOCAL_FIRE_RATE = {
        "loss_mode":["l2"],
        "model":["gNCA"],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        "regenerate":[False],
        # "fire_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        "fire_rate":[0.1,0.4,0.7,1.0],
        "perturbation_mode":["large"],
        "timesteps":[3],
        "parameter_noise_level":[0.0],
        "data_noise_level":[0.0],
        # "fire_rate":[0.8],
    }
    HYPERPARAMETERS_LOCAL_FIRE_RATE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_LOCAL_FIRE_RATE)
    return (HYPERPARAMETERS_LOCAL_FIRE_RATE,)


@app.cell
def _(HYPERPARAMETERS_LOCAL_FIRE_RATE):
    _perturbation_channel = "all"
    _coordinate_list = [(10*4,28*4),(30*4,14*4),(18*4,18*4)]
    _H = HYPERPARAMETERS_LOCAL_FIRE_RATE[0]
    pprint(_H)
    visualise_local_perturbation(_H,_coordinate_list,ch=_perturbation_channel,name_func=H_to_filename_fr)
    return


@app.cell
def _():
    mo.md(r"""
    ### Render videos
    """)
    return


@app.cell
def _(HYPERPARAMETERS_LOCAL_FIRE_RATE):
    _key = jr.PRNGKey(int(time.time()))
    for _H in HYPERPARAMETERS_LOCAL_FIRE_RATE:
        run_local_perturbations(
            _H,
            FULL_RUN=False,
            key=_key,
            NAME_FUNC=H_to_filename_fr,
            video_aux={"save":True,"filename":H_to_filename_fr(_H)})
    return


@app.cell
def _():
    mo.md(r"""
    ## Effect of noise on parameters and data
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_LOCAL_NOISE = {
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        # "channels":[32,16],
        "model":["gnNCA"],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],#,0.01,0.1,1.0],
        "timesteps":[3],
        "perturbation_mode":["large"],
        "regenerate":[False],
        # "fire_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        # "fire_rate":[0.5],
        "parameter_noise_level":[0.0,0.001,0.005,0.01,0.05,0.1],
        "data_noise_level":[0.005],
        # "parameter_noise_level":[0.0],
        # "data_noise_level":[0.001,0.005,0.05,0.1],
        # "data_noise_level":[0.001,0.005,0.01,0.05,0.1,0.5],

    }
    HYPERPARAMETERS_LOCAL_NOISE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_LOCAL_NOISE)
    return (HYPERPARAMETERS_LOCAL_NOISE,)


@app.cell
def _(HYPERPARAMETERS_LOCAL_NOISE):
    _perturbation_channel = "all"
    _coordinate_list = [(10*4,28*4),(30*4,14*4),(18*4,18*4)]
    _H = HYPERPARAMETERS_LOCAL_NOISE[4]
    pprint(_H)
    visualise_local_perturbation(_H,_coordinate_list,ch=_perturbation_channel,name_func=H_to_filename_noise)
    return


@app.cell
def _():
    mo.md(r"""
    # Global preserving or destructive perturbation visualisation
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Gating and regularizers
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_GLOBAL_PERTURBATION = {
        # These specify the NCA model to load
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
            # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],#,0.01,0.1,1.0],
        "update_sensitivity_coeff":[0.0,0.01,0.1,1.0],
        "regenerate":[False],
        "regen_str":[""],
        # From here on are specific to perturbation optimiser
        "timesteps":[3],
        "perturbation_mode":["global"],
        # "optimisation_mode":["maximal_preservative","minimal_destructive"],
        "optimisation_mode":["maximal_preservative"],
        "perturb_iters":[500],
        # "optimisation_mode":["minimal_destructive"],
        # "perturb_iters":[100]
    }
    HYPERPARAMETERS_GLOBAL_PERTURBATION = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_GLOBAL_PERTURBATION)
    # pprint(HYPERPARAMETERS_GLOBAL_PERTURBATION)
    return (HYPERPARAMETERS_GLOBAL_PERTURBATION,)


@app.cell
def _(HYPERPARAMETERS_GLOBAL_PERTURBATION):
    visualise_global_perturbation(HYPERPARAMETERS_GLOBAL_PERTURBATION,name_func=H_to_filename_gate)
    return


@app.cell
def _():
    mo.md(r"""
    ## Effect of fire rate
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE = {
        "loss_mode":["l2"],
        "model":["gNCA"],
        "channels":[16],
        "downsample":[1],
            # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        # "regenerate":[False,True],
        "fire_rate":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        "regenerate":[False],
        # "parameter_noise_level":[0.0,0.001,0.005,0.01,0.05,0.1],
        # "data_noise_level":[0.001,0.005,0.01,0.05,0.1,0.5],
        # "data_noise_level":[0.0]
        # From here on are specific to perturbation optimiser
        "timesteps":[3],
        "perturbation_mode":[{"channel":"obs","spatial":"global"},],
        # "optimisation_mode":["maximal_preservative","minimal_destructive"],
        # "optimisation_mode":["maximal_preservative"],
        # "perturb_iters":[500]
        "optimisation_mode":["minimal_destructive"],
        "perturb_iters":[100],
    }
    HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE)
    return (HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE,)


@app.cell
def _(HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE):
    visualise_global_perturbation(HYPERPARAMETERS_GLOBAL_PERT_FIRE_RATE,name_func=H_to_filename_fr)
    return


@app.cell
def _():
    mo.md(r"""
    ## Effect of noise
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_GLOBAL_PERT_NOISE = {
        "loss_mode":["l2"],
        "model":["nNCA"],
        "channels":[16],
        "downsample":[1],
            # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        # "regenerate":[False,True],
        # "fire_rate":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        "regenerate":[True],
        "parameter_noise_level":[0.0,0.001,0.005,0.01],
        "data_noise_level":[0.001],
        # "data_noise_level":[0.001,0.005,0.01,0.05,0.1],
        # "parameter_noise_level":[0.005],
        # From here on are specific to perturbation optimiser
        "timesteps":[3],
        "perturbation_mode":[{"channel":"obs","spatial":"global"},],
        # "optimisation_mode":["maximal_preservative","minimal_destructive"],
        "optimisation_mode":["maximal_preservative"],
        "perturb_iters":[500],
        # "optimisation_mode":["minimal_destructive"],
        # "perturb_iters":[100],
    }
    HYPERPARAMETERS_GLOBAL_PERT_NOISE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_GLOBAL_PERT_NOISE)
    return


@app.cell
def _():
    HYPERPARAMETERS_GLOBAL_FREEZE_NOISE = {
        "loss_mode":["l2"],
        "model":["gnNCA","nNCA"],
        # "regenerate":[False],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        # "regenerate":[False,True],
        # "fire_rate":[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        # "parameter_noise_level":[0.0,0.001,0.005,0.01],
        # "data_noise_level":[0.005],
        "data_noise_level":[0.005],
        "parameter_noise_level":[0.0],
        # From here on are specific to perturbation optimiser
        "optimisation_mode":["freeze"],
        "timesteps_to_target":[1],
        "timesteps_to_run":[3],
        # "optimisation_mode":["maximal_preservative","minimal_destructive"],
        # "optimisation_mode":["data_target"],
        # "timesteps_to_target":[2],

        "perturbation_mode":[{"channel":"obs","spatial":"global"},],
        "perturb_iters":[1000],
    }
    HYPERPARAMETERS_GLOBAL_FREEZE_NOISE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_GLOBAL_FREEZE_NOISE)
    for _H in HYPERPARAMETERS_GLOBAL_FREEZE_NOISE:
        _H["regenerate"] = _H["model"]=="nNCA"
        # if _H["model"]=="nNCA":
            # _H["regenerate"] = "regenerate_" if _H["regenerate"] else ""
    print(HYPERPARAMETERS_GLOBAL_FREEZE_NOISE)
    return (HYPERPARAMETERS_GLOBAL_FREEZE_NOISE,)


@app.cell
def _():
    mo.md(r"""
    ### Save videos of adversarially repgrogrammed NCA
    """)
    return


@app.cell
def _(HYPERPARAMETERS_GLOBAL_FREEZE_NOISE):
    visualise_global_perturbation(
        HYPERPARAMETERS_GLOBAL_FREEZE_NOISE,
        name_func=H_to_filename_noise,
        video_aux={
            "save":True,
            "filename":"emoji_global_perturbation_freeze_transient"})
    return


@app.cell
def _():
    mo.md(r"""
    # Regrowing damaged regions
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Contiguous regulariser and timesteps - when are models local?
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_CONTIG = {
        "loss_mode":["l2"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        # "steps_between_images":[16,32,64,96,128,192,256],
        # "steps_between_images":[16,32,64,128],#,128,256],
        # "regenerate":[True],
        "contiguous_growth_coeff":[10.0],
        "steps_between_images":[64],
        "contiguous_growth_coeff":[0.0],
        "regenerate":[True,False],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        # "contiguous_growth_coeff":[0.0,0.001,0.01,0.1,0.5,1.0,2.0,10.0],
        # "regenerate":[True],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        # "regenerate":[False,True],
    }
    HYPERPARAMETERS_CONTIG = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_CONTIG)
    pprint(HYPERPARAMETERS_CONTIG)
    return (HYPERPARAMETERS_CONTIG,)


@app.cell
def _(data, models_contig):
    x0_web = make_emoji_web_initial_state(data, channels=32, frame=0, pad=10)
    export_nca_web_assets(
        nca=models_contig[0][0],
        model_id="contiguous_regrowth",
        x0=x0_web,
    )
    return


@app.cell
def _(data, models_contig):
    Trs_contig = run_all_tr(
        models_contig,
        data,
        jr.PRNGKey(int(time.time())),
        save_every=8,
        regrowth=True,
        regrowth_aux={
            "pos":[0.5,0.5],
            "size":0.25
        },video_aux={"save":True,"filename":"emoji_regrow_hidden","channels":"hidden"}
    )
    return (Trs_contig,)


@app.cell
def _():
    def plot_contig_tr(ax,Tr,H):
        T,C,X,Y = Tr.shape
        # n_snapshots = 3
        # snapshot_indices = onp.linspace(1,T-1,n_snapshots).astype(int)
        ax.imshow(
            rearrange(
                # Tr[:10,:3],"(Tx Ty) C x y -> (Tx x) (Ty y) C",Tx=2,Ty=5
                Tr[:20,:3],"(Tx Ty) C x y -> (Tx x) (Ty y) C",Tx=2,Ty=10
            )
        )
        ax.set_yticks([])
        ax.set_xticks([])
        return ax
    def plot_contig(Trs,H,mode="contiguous"):
        fig,axs = plt.subplots(len(Trs),1,figsize=(10,2*len(Trs)),dpi=100,sharex=True,sharey=True)
        key = jr.PRNGKey(int(time.time()))
        for i,Tr in enumerate(Trs):
            axs[i] = plot_contig_tr(axs[i],Tr,H)
            # if mode=="contiguous":
            #     axs[i].set_title(f"Contiguous Growth Coeff: {HYPERPARAMETERS_CONTIG[i]['contiguous_growth_coeff']}",fontsize=8)
            # elif mode=="timesteps":
            #     axs[i].set_title(f"Steps Between Images: {HYPERPARAMETERS_CONTIG[i]['steps_between_images']}",fontsize=8)
        return plt.gca()
    # print(Trs_contig[0].shape)
    # plt.imshow(rearrange(Trs_contig[6][:10,:3],"T C x y -> x (T y) C"))
    return (plot_contig,)


@app.cell
def _(HYPERPARAMETERS_CONTIG, Trs_contig, plot_contig):
    plot_contig(Trs_contig,HYPERPARAMETERS_CONTIG,"timesteps")
    return


@app.cell
def _():
    mo.md(r"""
    ## Fire rate
    """)
    return


@app.cell
def _(data, models_fr):
    Trs_fr_contig = run_all_tr(
        models_fr,
        data,
        jr.PRNGKey(int(time.time())),
        save_every=8,
        regrowth=True,
        regrowth_aux={
            "pos":[0.5,0.5],
            "size":0.25
    })
    return (Trs_fr_contig,)


@app.cell
def _(HYPERPARAMETERS_FR, Trs_fr_contig, plot_contig):
    plot_contig(Trs_fr_contig,HYPERPARAMETERS_FR,"fire_rate")
    return


@app.cell(column=1)
def _():
    mo.md(r"""
    # Run Trajectories
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualise Loss function effect
    """)
    return


@app.cell
def _():
    NICE_LOSS_STRINGS = [
        "L2 squared",
        "L1",
        "L2",
        "Spectral",
        "Spectral (no phase)",
        "Spectral + L2",
        "Sliced Wasserstein (spatial)",
        "Sliced Wasserstein (channel)",
        "Sliced Wasserstein (full)",
        "Spectral Wasserstein (full)",
        "Sliced Wasserstein (rotational)",
        "Cosine",
    ]
    return (NICE_LOSS_STRINGS,)


@app.cell
def _():
    _w_samples = 64
    HYPERPARAMETERS_LOSS = {
        "loss_mode":["l2","l1","euclidean","spectral","spectral_no_phase","spectral_euclidean",
                     f"sliced_wasserstein_spatial_s{_w_samples}",f"sliced_wasserstein_channel_s{_w_samples}",
                     f"sliced_wasserstein_full_s{_w_samples}",f"spectral_wasserstein_full_s{_w_samples}",
                     f"sliced_wasserstein_rotational_s{_w_samples}",
                     # "bhattacharyya","kl_divergence","hellinger",
                     # "bhattacharyya_modified","hellinger_modified","kl_divergence_modified",
                     "cosine"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        "steps_between_images":[64],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0],
        "contiguous_growth_coeff":[0.0],
    }
    HYPERPARAMETERS_LOSS = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_LOSS)
    pprint(HYPERPARAMETERS_LOSS)
    return (HYPERPARAMETERS_LOSS,)


@app.cell
def _(data, models_loss):
    _key = jr.PRNGKey(int(time.time()))
    Trs_loss = run_all_tr(models_loss,data,_key,regrowth=False,video_aux={"save":True,"filename":"emoji_losses_fig_4.5"})
    return (Trs_loss,)


@app.cell
def _(NICE_LOSS_STRINGS, Trs_loss, models_loss, plot_all_models_losses):
    plot_all_models_losses(Trs=Trs_loss,models=models_loss,title_strings=NICE_LOSS_STRINGS)
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise wasserstein loss sampling parameter
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_W_LOSS = {
        "loss_mode":[
            # "sliced_wasserstein_spatial",
            # "sliced_wasserstein_channel",
            # "sliced_wasserstein_full",
            "sliced_wasserstein_rotational"
            # "spectral_wasserstein_full"
        ],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        "steps_between_images":[64],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0],
        "contiguous_growth_coeff":[0.0],
        "wasserstein_samples":[1,4,16,32,64,128,256,512]
    }
    HYPERPARAMETERS_W_LOSS = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_W_LOSS)
    # pprint(HYPERPARAMETERS_W_LOSS)
    return (HYPERPARAMETERS_W_LOSS,)


@app.cell
def _(data, models_w_loss):
    _key = jr.PRNGKey(int(time.time()))
    Trs_w_loss = run_all_tr(models_w_loss,data,_key)
    return (Trs_w_loss,)


@app.cell
def _(Trs_w_loss, models_w_loss, plot_all_models):
    _titles = [1,4,16,32,64,128,256,512]
    plot_all_models(Trs=Trs_w_loss,models=models_w_loss,title_strings=_titles)
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise Regulariser effect
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_REG = {
        "loss_mode":["l2"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        "steps_between_images":[64],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0,0.01,0.1,1.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        "regenerate":[False]
    }
    HYPERPARAMETERS_REG = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_REG)
    pprint(HYPERPARAMETERS_REG)
    return (HYPERPARAMETERS_REG,)


@app.cell
def _(data, models_reg):
    _key = jr.PRNGKey(int(time.time()))
    # plt.imshow(data[0,0])
    # plt.show()
    Trs_reg = run_all_tr(
        models_reg,
        data,
        _key,
        regrowth=False,
        regrowth_aux={"pos":[0.5,0.5],"size":0.25},
        video_aux={"save":True,"filename":"emoji_reg_video_test"}
    )
    return (Trs_reg,)


@app.cell
def _(Trs_reg, models_reg, plot_all_models):
    plot_all_models(Trs=Trs_reg,models=models_reg,title_strings=[None,None,None,None])
    return


@app.cell
def _():
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise noise effect
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_NOISE = {
        "loss_mode":["l2"],
        # "model":["NCA","gNCA"],
        # "channels":[32,16],
        "model":["nNCA"],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[64,128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],#,0.01,0.1,1.0],
        "regenerate":[True],
        "parameter_noise_level":[0.0,0.01],#,0.001,0.005,0.01,0.05,0.1],
        "data_noise_level":[0.01],
        # "parameter_noise_level":[0.0],
        # "data_noise_level":[0.001,0.005,0.01,0.05,0.1,0.5],
    }
    HYPERPARAMETERS_NOISE = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_NOISE)
    return (HYPERPARAMETERS_NOISE,)


@app.cell
def _(data, models_noise):
    _key = jr.PRNGKey(int(time.time()))
    Trs_noise = run_all_tr(
        models_noise,
        data,
        _key,
        regrowth=False,
        regrowth_aux={"pos":[0.5,0.5],"size":0.25},
        video_aux={"save":True,"filename":"emoji_noise_video_test"})
    return (Trs_noise,)


@app.cell
def _(Trs_noise, models_noise, plot_all_models):
    plot_all_models(Trs=Trs_noise,models=models_noise,title_strings=[None,None,None,None,None])
    return


@app.cell
def _():
    mo.md(r"""
    ## Visualise fire rate effect
    """)
    return


@app.cell
def _():
    HYPERPARAMETERS_FR = {
        "loss_mode":["l2"],
        "model":["NCA"],
        "channels":[16],
        "downsample":[1],
        # "steps_between_images":[128],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        "regenerate":[True],
        # "fire_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        "fire_rate":[0.1,0.4,0.7,1.0],
        # "perturbation_mode":["large"],
        # "timesteps":[3],
        # "fire_rate":[0.8],
    }
    HYPERPARAMETERS_FR = generate_hyperparameter_combinations_indexed(HYPERPARAMETERS_FR)
    return (HYPERPARAMETERS_FR,)


@app.cell
def _(data, models_fr):
    _key = jr.PRNGKey(int(time.time()))
    Trs_fr = run_all_tr(models_fr,data,_key,regrowth=True,regrowth_aux={
        "pos":[0.5,0.5],
        "size":0.25
    })
    return (Trs_fr,)


@app.cell
def _(Trs_fr, models_fr, plot_all_models):
    plot_all_models(Trs=Trs_fr,models=models_fr,title_strings=[None,None,None,None,None])
    return


@app.cell(column=2)
def _():
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.function(hide_code=True)
def load_emoji_models_loss(H):
    nca = NCA(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=jr.PRNGKey(0),
    )
    FILENAME = f"emoji_al_mi_ro_loss_{H['loss_mode']}_ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}.eqx"
    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}")
    return nca


@app.function(hide_code=True)
def load_emoji_models_w_loss(H):
    nca = NCA(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=jr.PRNGKey(0),
    )
    FILENAME = f"emoji_al_mi_ro_loss_{H['loss_mode']}_s{H['wasserstein_samples']}_ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}.eqx"
    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}")
    return nca


@app.function(hide_code=True)
def load_emoji_models_regulariser(H):
    nca = NCA(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=jr.PRNGKey(0),
    )
    FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}.eqx"
    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}")
    return nca


@app.function(hide_code=True)
def load_emoji_models_noise(H):
    FILENAME = H_to_filename_noise(H)
    if H["model"]=="nNCA":
        nca = nNCA(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=0.5,
            PADDING="REPLICATE",
            PARAMETER_NOISE_LEVEL=H["parameter_noise_level"],
            key=jr.PRNGKey(0),
        )
    elif H["model"]=="gnNCA":
        nca = gnNCA(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=0.5,
            PADDING="REPLICATE",
            PARAMETER_NOISE_LEVEL=H["parameter_noise_level"],
            key=jr.PRNGKey(0),
        )

    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}.eqx")
    return nca


@app.function(hide_code=True)
def load_emoji_models_fire_rate(H):
    FILENAME = H_to_filename_fr(H)
    if H["model"]=="NCA":
        nca = NCA(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=H["fire_rate"],
            PADDING="REPLICATE",
            # PARAMETER_NOISE_LEVEL=H["parameter_noise_level"],
            key=jr.PRNGKey(0),
        )
    elif H["model"]=="gNCA":
        nca = gNCA(
            H["channels"],
            KERNEL_STR=["ID","LAP","GRAD"],
            KERNEL_SCALE=1,
            FIRE_RATE=H["fire_rate"],
            PADDING="REPLICATE",
            # PARAMETER_NOISE_LEVEL=H["parameter_noise_level"],
            key=jr.PRNGKey(0),
        )


    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}.eqx")
    return nca


@app.function
def load_emoji_models_contig(H):
    nca = NCA(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=jr.PRNGKey(0),
    )
    if H["regenerate"]:
        regen_str = "regenerate_"
    else:
        regen_str = ""

    FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_{H['model']}_{regen_str}ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}.eqx"
    nca = nca.load(f"models/thesis_ch1/emoji/{FILENAME}")
    return nca


@app.function(hide_code=True)
def make_emoji_web_initial_state(data, channels, frame=0, pad=10):
    """Build the padded latent initial state used by the emoji NCA rollouts."""
    x0 = onp.asarray(data[frame], dtype=onp.float32)
    if x0.ndim != 3:
        raise ValueError(f"Expected data[frame] to have shape [C, H, W], got {x0.shape}.")
    if x0.shape[0] > channels:
        raise ValueError(f"Initial state has {x0.shape[0]} channels, but model has {channels}.")
    return onp.pad(
        x0,
        ((0, channels - x0.shape[0]), (pad, pad), (pad, pad)),
        mode="constant",
    ).astype(onp.float32)


@app.function(hide_code=True)
def export_nca_web_assets(
    nca,
    model_id,
    x0=None,
    grid_size=None,
    output_dir="WebDemo/public/models",
    reference_steps=8,
    display_channels=(0, 1, 2),
):
    """
    Export an already-loaded plain NCA to the WebDemo static asset format.

    Example:
        nca, H = models_reg[0]
        x0 = make_emoji_web_initial_state(data, H["channels"])
        export_nca_web_assets(nca, "emoji_good_reg_model", x0=x0)
    """
    import json
    from pathlib import Path

    def _expanded_kernel_count(kernels):
        count = 0
        for kernel in kernels:
            if kernel == "GRAD":
                count += 2
            else:
                count += 1
        return count

    def _tensor_entry(name, arr, offset):
        nbytes = int(arr.nbytes)
        return (
            {
                "name": name,
                "dtype": "float32",
                "shape": list(arr.shape),
                "byteOffset": offset,
                "byteLength": nbytes,
            },
            offset + nbytes,
        )

    def _as_float32(value):
        return onp.asarray(jax.device_get(value), dtype=onp.float32)

    def _refresh_model_index(models_dir):
        models = []
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                models.append({"id": model_dir.name, "label": model_dir.name})
        (models_dir / "index.json").write_text(json.dumps({"models": models}, indent=2) + "\n")

    source_kernels = list(nca.KERNEL_STR)
    supported_kernel_set = {"ID", "GRAD", "LAP"}
    if set(source_kernels) != supported_kernel_set:
        raise ValueError(
            "The current WebGL runtime supports plain anisotropic NCA with exactly "
            f"{sorted(supported_kernel_set)}; got {source_kernels}."
        )

    if len(nca.layers) != 3:
        raise ValueError(
            "The current WebGL asset contract supports plain NCA only. "
            "gNCA/noisy/KAN variants need a matching runtime shader first."
        )

    activation_name = getattr(nca.layers[1], "__name__", None)
    if activation_name != "relu":
        raise ValueError(f"Only relu activation is currently supported; got {activation_name}.")

    padding = nca.op.PADDING
    if padding not in ["CIRCULAR", "REPLICATE"]:
        raise ValueError(f"Only CIRCULAR and REPLICATE padding are currently supported; got {padding}.")

    channels = int(nca.N_CHANNELS)
    kernels = ["ID", "GRAD", "LAP"]
    expected_features = channels * _expanded_kernel_count(kernels)
    if int(nca.N_FEATURES) != expected_features:
        raise ValueError(
            f"Feature mismatch: model has {nca.N_FEATURES}, expected {expected_features}."
        )

    w0 = _as_float32(np.squeeze(nca.layers[0].weight))
    w1 = _as_float32(np.squeeze(nca.layers[2].weight))
    b1 = _as_float32(np.squeeze(nca.layers[2].bias))
    grad_x = _as_float32(np.squeeze(nca.op.grad_x.weight))
    grad_y = _as_float32(np.squeeze(nca.op.grad_y.weight))
    lap = _as_float32(np.squeeze(nca.op.laplacian.weight))
    average = _as_float32(np.squeeze(nca.op.average.weight))

    if w0.shape != (expected_features, expected_features):
        raise ValueError(f"Unexpected w0 shape {w0.shape}; expected {(expected_features, expected_features)}.")
    if w1.shape != (channels, expected_features):
        raise ValueError(f"Unexpected w1 shape {w1.shape}; expected {(channels, expected_features)}.")
    if b1.shape != (channels,):
        raise ValueError(f"Unexpected b1 shape {b1.shape}; expected {(channels,)}.")
    for name, kernel in {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "lap": lap,
        "average": average,
    }.items():
        if kernel.shape != (3, 3):
            raise ValueError(f"{name} has shape {kernel.shape}; current WebGL runtime supports 3x3 kernels.")

    if x0 is None:
        if grid_size is None:
            grid_size = (96, 96)
        height, width = map(int, grid_size)
        x0_array = onp.zeros((channels, height, width), dtype=onp.float32)
        x0_array[3:, height // 2, width // 2] = 1.0
    else:
        x0_array = onp.asarray(jax.device_get(x0), dtype=onp.float32)
        if x0_array.ndim != 3:
            raise ValueError(f"x0 must have shape [C, H, W], got {x0_array.shape}.")
        if x0_array.shape[0] != channels:
            raise ValueError(f"x0 has {x0_array.shape[0]} channels, but model has {channels}.")
        height, width = map(int, x0_array.shape[1:])
        if grid_size is not None and tuple(map(int, grid_size)) != (height, width):
            raise ValueError(f"grid_size {grid_size} does not match x0 spatial shape {(height, width)}.")

    out_dir = Path(output_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "w0": w0,
        "w1": w1,
        "b1": b1,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "lap": lap,
        "average": average,
    }
    tensor_entries = {}
    offset = 0
    with (out_dir / "weights.bin").open("wb") as f:
        for name, arr in arrays.items():
            arr.tofile(f)
            tensor_entries[name], offset = _tensor_entry(name, arr, offset)

    x0_array.tofile(out_dir / "x0.bin")

    deterministic = eqx.tree_at(lambda m: m.FIRE_RATE, nca, 1.0)
    reference, _ = deterministic.run(
        iters=reference_steps,
        x=np.asarray(x0_array),
        SAVE_LATENTS=False,
        key=jr.PRNGKey(0),
    )
    reference = onp.asarray(reference[-1], dtype=onp.float32)
    reference.tofile(out_dir / "reference.bin")

    manifest = {
        "modelId": model_id,
        "family": "NCA",
        "channels": channels,
        "kernels": kernels,
        "sourceKernels": source_kernels,
        "activation": "relu",
        "padding": padding,
        "fireRate": float(nca.FIRE_RATE),
        "gridSize": [width, height],
        "featureChannels": expected_features,
        "hiddenChannels": expected_features,
        "weights": {
            "path": "weights.bin",
            "tensors": tensor_entries,
        },
        "initialState": {
            "path": "x0.bin",
            "dtype": "float32",
            "shape": [channels, height, width],
        },
        "display": {
            "channels": list(display_channels),
            "range": [0.0, 1.0],
        },
        "validation": {
            "referenceSteps": int(reference_steps),
            "referenceFireRate": 1.0,
            "reference": {
                "path": "reference.bin",
                "dtype": "float32",
                "shape": [channels, height, width],
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _refresh_model_index(Path(output_dir))
    print(f"Exported WebGL assets to {out_dir}")
    return out_dir


@app.cell
def _(plot_emoji_snapshots):
    def plot_all_models(Trs,models,title_strings):
        fig,axs = plt.subplots(len(models),1,figsize=(6,1.5*len(models)),dpi=400,sharex=True,sharey=True)
        key = jr.PRNGKey(int(time.time()))
        for i,nca_hparam in enumerate(models):
            # Tr = Trs[i]
            axs[i] = plot_emoji_snapshots(axs[i],Trs[i],nca_hparam[1])
            axs[i].set_title(title_strings[i],fontsize=8)
        # plt.tight_layout()
        # axs[-1].set_xlabel("T")

        return plt.gca()

        # print(nca_hparam[1])
    return (plot_all_models,)


@app.cell
def _(plot_emoji_snapshots):
    def plot_all_models_losses(Trs,models,title_strings):
        fig,axs = plt.subplots(6,2,figsize=(6,8),dpi=400,sharex=True,sharey=True)
        key = jr.PRNGKey(int(time.time()))
        for i,nca_hparam in enumerate(models):
            # Tr = Trs[i]
            axs[i%6,i//6] = plot_emoji_snapshots(axs[i%6,i//6],Trs[i],nca_hparam[1])
            axs[i%6,i//6].set_title(title_strings[i],fontsize=8)
        plt.tight_layout()
        # axs[-1].set_xlabel("T")

        return plt.gca()

    return (plot_all_models_losses,)


@app.function
def run_emoji_models(nca_hparams,data,key,save_every=None,regrowth=False,regrowth_aux={"pos":[0.5,0.5],"size":0.5}):
    """
        Runs given nca model on initial condition, returns a trajectory
    """
    nca,H = nca_hparams # unpack nca and hyperparamter dict
    x = np.array(data[0]) # C x y
    if regrowth:
        print(f"{regrowth} selected, zeroing half the image for regrowth")
        _width = x.shape[1]
        _pos = [regrowth_aux["pos"][0]*x.shape[1],regrowth_aux["pos"][1]*x.shape[2]]
        _radius = regrowth_aux["size"]*min(x.shape[1],x.shape[2])
        x = zero_random_circle(x,_pos,_radius)
        # x = x.at[:,2*_width//6:4*_width//6,2*_width//6:4*_width//6].set(0.0)
        # x[:,:,:_width//2] = 0.0
    x = np.pad(x,((0,H["channels"]-4),(10,10),(10,10)),mode='constant') # pad to nca channels]))
    T = data.shape[0]
    t = H["steps_between_images"]
    if save_every is None:
        save_every = t
    else:
        save_every = t//save_every
    # if save_every is None:
        # save_every = t
    # print(f"Saving every {save_every} steps")
    steps = T * t
    trajectory = []
    for i in range(steps):
        key = jr.fold_in(key,i)
        x = nca(x,lambda x:x,key)
        if i % save_every == 0:
            trajectory.append(x)
    trajectory = onp.array(trajectory)
    return trajectory


@app.function
def run_full_trajectory(nca_hparams,data,key,regrowth=False,regrowth_aux={"pos":[0.5,0.5],"size":0.5}):
    """
        Runs given nca model on initial condition, returns a trajectory
    """
    nca,H = nca_hparams # unpack nca and hyperparamter dict
    x = np.array(data[0]) # C x y
    if regrowth:
        print(f"{regrowth} selected, zeroing half the image for regrowth")
        _width = x.shape[1]
        _pos = [regrowth_aux["pos"][0]*x.shape[1],regrowth_aux["pos"][1]*x.shape[2]]
        _radius = regrowth_aux["size"]*min(x.shape[1],x.shape[2])
        x = zero_random_circle(x,_pos,_radius)

    x = np.pad(x,((0,H["channels"]-4),(10,10),(10,10)),mode='constant') # pad to nca channels]))
    T = data.shape[0]
    t = H["steps_between_images"]
    steps = T * t
    trajectory = []
    for i in range(steps):
        key = jr.fold_in(key,i)
        x = nca(x,lambda x:x,key)
        trajectory.append(x)
    trajectory = onp.array(trajectory)
    return trajectory


@app.function
def zero_random_circle(image,pos,size):
    height = image.shape[-2]
    width = image.shape[-1]
    center_x,center_y = pos
    radius = size


    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2

    # Assuming image shape is [C, H, W]
    mask = rearrange(mask, 'h w -> () h w')
    image = np.where(mask, 0, image)
    return image


@app.function
def visualise_local_perturbation(H,coord,ch,name_func=H_to_filename_gate):
    fig,axs = plt.subplots(2,2,figsize=(4,4),dpi=400)
    axs[0,0] = visualise_full_local_perturbation_map(H,coord,ch=ch,ax=axs[0,0],name_func=name_func)
    axs[0,1] = visualise_local_perturbation_coords((ch,*coord[0]),H,ax=axs[0,1],name_func=name_func)
    axs[1,0] = visualise_local_perturbation_coords((ch,*coord[1]),H,ax=axs[1,0],name_func=name_func)
    axs[1,1] = visualise_local_perturbation_coords((ch,*coord[2]),H,ax=axs[1,1],name_func=name_func)
    plt.tight_layout()
    return plt.gca()


@app.function
def visualise_local_perturbation_coords(coords,H,ax,name_func=H_to_filename_gate):

    FILENAME = name_func(H)
    c,x,y = coords

    P = onp.load(f"perturbations/emoji_local/{FILENAME}_coords_{c}_{x}_{y}_{H['perturbation_mode']}_T{H['timesteps']}.npy")
    # P = numpy_image_float_to_int(P)
    print(P.shape)

    # plt.hist(P.flatten(),bins=50)
    # pprint(H)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(rearrange(P[:3],"C x y -> x y C"))
    return ax


@app.function
def visualise_full_local_perturbation_map(H,coords,ch,ax,name_func=H_to_filename_gate):

    FILENAME = name_func(H)
    P = onp.load(f"perturbations/emoji_local/{FILENAME}_counts_{H['perturbation_mode']}_T{H['timesteps']}.npy")
    print(P.shape)
    # pprint(H)
    # plt.imshow(rearrange(P[:],"C x y -> x y C"))
    colors = plt.cm.autumn(onp.linspace(0,1.0,3))
    # ax.imshow(onp.log(reduce(P,"C x y -> x y", "sum")),cmap='gray')
    ax.imshow(P,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar()
    for i,c in enumerate(coords):
        ax.scatter(c[1],c[0],s=15,color=colors[i])
    # plt.show()
    return ax


@app.function
def visualise_global_perturbation_ind(H,ax,name_func=H_to_filename_gate,video_aux={"save":False,"filename":"emoji_global_perturbation"}):
    # if "steps_between_images" not in H:
    #     if H["channels"]==32:
    #         H["steps_between_images"]=64
    #     elif H["channels"]==16:
    #         H["steps_between_images"]=128
    # if H["regenerate"]:
    #     regen_str = "regenerate_"
    # else:
    #     regen_str = ""
    # FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_{H['model']}_{regen_str}ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}"
    FILENAME = name_func(H)
    if H["optimisation_mode"] in ["freeze"]:
        FULL_FILENAME = f"perturbations/emoji_global/{FILENAME}_impulse_global_{H['optimisation_mode']}_iters{H['perturb_iters']}_T{H['timesteps_to_target']}_R{H['timesteps_to_run']}_trajectory.npy"
    elif H["optimisation_mode"] in ["data_target"]:
        FULL_FILENAME = f"perturbations/emoji_global/{FILENAME}_impulse_{H['optimisation_mode']}_iters{H['perturb_iters']}_T{H['timesteps_to_target']}_trajectory.npy"
    else:
        FULL_FILENAME = f"perturbations/emoji_global/{FILENAME}_impulse_global_{H['optimisation_mode']}_iters{H['perturb_iters']}_T{H['timesteps']}_trajectory.npy"

    # print(FULL_FILENAME)
    P = onp.load(FULL_FILENAME)
    # P = numpy_image_float_to_int(P)
    print(f"Trajectory shape: {P.shape}")
    pprint(H)
    # ax.figure(figsize=(12,6),dpi=400)
    ax.imshow(rearrange(P[::H["steps_between_images"],:3][:4],"T C x y -> x (T y) C"))
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.
    if video_aux["save"]:
        print("Saving video...")
        P = P/255.0
        P = P[:,:3]
        print("Max pixel value in trajectory:",P.max())
        print("Min pixel value in trajectory:",P.min())

        save_to_video_rgb(
            rearrange(P,"T C x y -> T x y C"),
            f"Videos/ThesisEmojis/{video_aux['filename']}_{H['LIST_INDEX']}.mp4",
            duration=20,
        )
    # plt.show()
    return ax


@app.function
def visualise_global_perturbation(Hs,name_func=H_to_filename_gate,video_aux={"save":False,"filename":"emoji_global_perturbation"}):
    fig,axs = plt.subplots(len(Hs),1,figsize=(12,2*len(Hs)),dpi=400)
    for i,H in enumerate(Hs):
        # ax = axs[i//2,i%2]
        axs[i] = visualise_global_perturbation_ind(H,axs[i],name_func=name_func,video_aux=video_aux)
        # axs.set_title(f"{H['optimisation_mode'].replace('_',' ').title()} Perturbation",fontsize=16)
    plt.tight_layout()
    plt.show()


@app.cell
def _():
    def plot_emoji_snapshots(ax,Tr,H):
        """
            Takes a trajectory (shape T C x y) and hyperparameter dict H
            Plots snapshots at T=1,2,3 * steps_between_images (i.e. ignoring initial condition)
        """
        T,C,X,Y = Tr.shape
        # n_snapshots = 3
        # snapshot_indices = onp.linspace(1,T-1,n_snapshots).astype(int)
        ax.imshow(
            rearrange(
                Tr[1:,:3],"T C x y -> x (T y) C"
            )
        )
        ax.set_yticks([])
        ax.set_xticks([])

        # fig,axs = plt.subplots(1,n_snapshots,figsize=(n_snapshots*3,3),dpi=400)
        # for i,idx in enumerate(snapshot_indices):
        #     img = Tr[idx,:3]
        #     img = rearrange(img,"C x y -> x y C")
        #     axs[i].imshow(img)
        #     axs[i].set_xticks([])
        #     axs[i].set_yticks([])
        #     axs[i].set_title(f"T={idx*H['steps_between_images']}")
        # plt.tight_layout()
        return ax
    # plot_emoji_snapshots(Tr,models[8][1])
    return (plot_emoji_snapshots,)


@app.function
def run_all_tr(
    models,
    data,
    key,
    save_every=None,
    obs_only=True,
    regrowth=False,
    regrowth_aux={"pos":[0.5,0.5],"size":0.5},
    video_aux={"save":False,"filename":"emoji_trajectory","channels":"obs"}
):
    Trs = []
    for i,nca_hparam in enumerate(tqdm(models)):
        key = jr.fold_in(key,i)
        Tr = run_emoji_models(nca_hparam,data,key,save_every=save_every,regrowth=regrowth,regrowth_aux=regrowth_aux)
        if obs_only:
            Tr = Tr[:,:4] # only keep observable channels
        Trs.append(Tr)
        if video_aux["save"]:
            Tr = run_full_trajectory(nca_hparam,data,key,regrowth=regrowth,regrowth_aux=regrowth_aux)
            if video_aux["channels"]=="obs":
                Tr = Tr[:,:3]
            elif video_aux["channels"]=="hidden":
                Tr = Tr[:,4:]
                Tr = Tr[:,:24]
                Tr = rearrange(Tr,"T (C Cx Cy) x y -> T C (Cx x) (Cy y)",Cx=2,C=3)
                Tr = onp.tanh(Tr)*0.5 + 0.5
            Tr = onp.clip(Tr,0.0,1.0)
            print("Saving video...")
            print("Max pixel value in trajectory:",Tr.max())
            print("Min pixel value in trajectory:",Tr.min())
            save_to_video_rgb(
                rearrange(Tr,"T C x y -> T x y C"),
                f"Videos/ThesisEmojis/{video_aux['filename']}_{i}.mp4",
                duration=20,
            )

    return Trs


@app.cell
def _():
    from jax.scipy.ndimage import map_coordinates
    def get_rotation_grid(shape, angle_deg):
        # Create a coordinate grid. Here we assume y,x ordering.
        ny, nx = shape
        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        # Center coordinates for rotation.
        y_center = (ny - 1) / 2.
        x_center = (nx - 1) / 2.
        y = y - y_center
        x = x - x_center

        # Convert angle to radians.
        theta = np.deg2rad(angle_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Compute inverse rotation (to sample from the input image).
        x_rot = cos_theta * x + sin_theta * y
        y_rot = -sin_theta * x + cos_theta * y

        # Shift back.
        x_rot = x_rot + x_center
        y_rot = y_rot + y_center

        return y_rot, x_rot

    def rotate_array(arr, angle_deg):
        # Compute the sampling grid for rotation.
        coords = get_rotation_grid(arr.shape, angle_deg)
        # Stack coordinates in the order expected by map_coordinates.
        coords = np.stack(coords, axis=0)
        # Use linear interpolation order (order=1). 
        # Note: You may need to experiment with mode, cval, etc.
        rotated = map_coordinates(arr, coords, order=1, mode='constant', cval=0.0)
        return rotated

    return (rotate_array,)


@app.cell
def _(data, rotate_array):
    print(data.shape)
    for _i in range(0,12):
        _rotated_data = rotate_array(data[0,1],_i*30)
        plt.plot(np.mean(_rotated_data,axis=0))

    plt.show()
    # plt.imshow(_rotated_data)
    # plt.imshow(data[0,0])
    return


@app.function
def H_to_filename_gated(H):
    if "steps_between_images" not in H:
        if H["channels"]==32:
            H["steps_between_images"]=64
        elif H["channels"]==16:
            H["steps_between_images"]=128
    FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_{H['model']}_{H['regen_str']}ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}"
    return FILENAME


if __name__ == "__main__":
    app.run()
