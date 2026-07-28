import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import jax
    import jax.random as jr
    import jax.numpy as np
    import time
    import optax
    import equinox as eqx
    import wandb
    # import sys
    import sys
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    from einops import rearrange,repeat
    from Common.model.spatial_operators import Ops
    from PDE.model.fixed_models.update_chhabra import F as F_chhabra
    from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
    from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
    from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
    from PDE.model.fixed_models.update_keller_segel import F as F_keller_segel
    from PDE.model.fixed_models.update_heat_equation import F as F_heat_equation
    from Common.save_to_video import save_to_video_mono
    from PDE.model.solver.semidiscrete_solver import PDE_solver
    from NCA.trainer.NCA_trainer import NCA_Trainer
    from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
    from NCA.model.NCA_model import NCA
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use(
        "default"
    )


@app.cell
def _():
    ITERS = 4000        # Training iterations
    CHANNELS = 16        # NCA channels
    SIZE = 64           # Grid size
    BATCHES = 2         # Data batches
    TIME_SAMPLING = 32  # Timesteps between data snapshots
    LEARN_RATE = 1e-3   # Learn rate for gradient optimiser
    return BATCHES, CHANNELS, ITERS, LEARN_RATE, SIZE, TIME_SAMPLING


@app.cell
def _(STEPS_PER_PDE, gray_scott_ic):
    def generate_pde_training_data(mode,SIZE,BATCHES,T,STEPS=None,key=jr.PRNGKey(int(time.time()))):
        if STEPS==None:
            STEPS = STEPS_PER_PDE[mode]
        if mode=="gray-scott":
            TS,X = solve_gray_scott(*gray_scott_ic(SIZE,BATCHES,T,STEPS,key))
        elif mode=="chhabra":
            TS,X = solve_chhabra(*chhabra_ic(SIZE,BATCHES,T,STEPS,key))
        elif mode=="cahn-hilliard":
            TS,X = solve_cahn_hilliard(*cahn_hilliard_ic(SIZE,BATCHES,2*T,STEPS,key))
            TS = TS[::2] # Resample time steps to match NCA training time resolution - only keep every 2nd step
            X = X[:,::2] # Resample data to match NCA training time resolution - only keep every 2nd step
        elif mode=="hillen-painter":
            TS,X = solve_hillen_painter(*hillen_painter_ic(SIZE,BATCHES,T,STEPS,key))
        elif mode=="keller-segel":
            TS,X = solve_keller_segel(*keller_segel_ic(SIZE,BATCHES,T*2,STEPS,key))
            TS = TS[::2] # Resample time steps to match NCA training time resolution - only keep every 2nd step
            X = X[:,::2] # Resample data to match NCA training time resolution - only keep every 2nd step
        elif mode=="heat-equation":
            TS,X = solve_heat_equation(*heat_equation_ic(SIZE,BATCHES,T,STEPS,key))
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # X = X[:,::t]             # Resample data to reduce number of time steps - only keep every t'th step
        print(f"Generated training data for {mode} PDE with shape [B T C x y] {X.shape}")
        print(f"Time steps: {TS.shape}, Time step size: {TS[1]-TS[0]}")
        return TS,X

    return (generate_pde_training_data,)


@app.cell
def _(BATCHES, SIZE, TIME_SAMPLING, generate_pde_training_data, gray_scott_ic):
    # ts,x0 = cahn_hilliard_ic(SIZE,BATCHES,TIME_SAMPLING*8)
    # # func = F_gray_scott(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
    # func = F_cahn_hilliard(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1)
    # v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    # solver = PDE_solver(v_func,dt=0.5)
    # T,Y = solver(ts=ts,y0=x0)
    # Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    # Y = Y[:,:,:]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    # Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    # Y = Y[:,::TIME_SAMPLING]   
    # T,Y = solve_gray_scott(*gray_scott_ic(SIZE,BATCHES,TIME_SAMPLING*8)) 
    # T,Y = solve_cahn_hilliard(*cahn_hilliard_ic(SIZE,BATCHES,TIME_SAMPLING*8))
    # T,Y = solve_chhabra(*chhabra_ic(SIZE,BATCHES,TIME_SAMPLING*8))
    # T,Y = solve_hillen_painter(*hillen_painter_ic(SIZE,BATCHES,TIME_SAMPLING*8))
    # T,Y = solve_keller_segel(*keller_segel_ic(SIZE,BATCHES,TIME_SAMPLING*8))
    _,X0 = gray_scott_ic(SIZE,BATCHES,TIME_SAMPLING*8)
    print(X0.shape)
    plt.imshow(X0[0,0])
    T,Y = generate_pde_training_data("cahn-hilliard",SIZE,2,12,STEPS=1000)
    return (Y,)


@app.cell
def _(Y):
    print(Y.shape)
    # plt.imshow(Y[0,,0])
    plt.figure(figsize=(22,12))
    plt.imshow(rearrange(Y[:,:,0],"B T X Y -> (B X) (T Y)"))
    return


@app.cell
def _():
    mo.md(r"""
    ## Define NCA for training to PDEs
    """)
    return


@app.function
def build_nca(key,CHANNELS,mode):
    if mode=="keller-segel":
        kernel_str = ["ID","LAP","DIFF"] # Include diffusion kernel for keller-segel as it has a strong diffusion term which is important to capture for stability"
    else:
        kernel_str = ["ID","LAP"] # For other PDEs, the diffusion term is less dominant, so we can try training without the diffusion kernel to see if the NCA can learn to approximate it with just the ID and LAP kernels

    nca = NCA(N_CHANNELS=CHANNELS,          
              KERNEL_STR=["ID","LAP"],      
              ACTIVATION=jax.nn.relu,       
              FIRE_RATE=1.0,                
              key=key)
    return nca


@app.function
def build_nca_diff(key,CHANNELS):
    nca = NCA(N_CHANNELS=CHANNELS,          
              KERNEL_STR=["ID","LAP","DIFF"],      
              ACTIVATION=jax.nn.relu,       
              FIRE_RATE=1.0,                
              key=key)
    return nca


@app.cell
def _():
    return


@app.cell
def _(ITERS, LEARN_RATE):
    def build_trainer(nca,XS,filename):
        opt = NCA_Trainer(nca,
                      XS,                                                
                      model_filename=filename,
                      DATA_AUGMENTER=DataAugmenter,                     
                      GRAD_LOSS=True)

        schedule = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
        optimiser = optax.chain(optax.scale_by_param_block_norm(),
                            optax.nadam(schedule))
        return opt,optimiser

    return (build_trainer,)


@app.cell
def _():
    FILENAMES = {
        "gray-scott":"thesis_ch1/pde/nca_pde_gray_scott_T12_t32_v2",
        "chhabra":"thesis_ch1/pde/nca_pde_chhabra_T12_t32_v2",
        "cahn-hilliard":"thesis_ch1/pde/nca_pde_cahn_hilliard_T12_t32_v2",
        "hillen-painter":"thesis_ch1/pde/nca_pde_hillen_painter_T12_t32_v2",
        "keller-segel":"thesis_ch1/pde/nca_pde_keller_segel_T12_t32_v2",
        "heat-equation":"thesis_ch1/pde/nca_pde_heat_equation_T12_t32_v2"
    }
    return (FILENAMES,)


@app.cell
def _(
    BATCHES,
    CHANNELS,
    FILENAMES,
    ITERS,
    SIZE,
    STEPS_PER_PDE,
    TIME_SAMPLING,
    build_trainer,
    generate_pde_training_data,
):
    def train_nca(mode,key):
        TS,Y = generate_pde_training_data(mode,SIZE,BATCHES,12,STEPS=STEPS_PER_PDE[mode])
        print(Y.shape)
        opt,optimiser = build_trainer(build_nca(key,CHANNELS,mode),Y,FILENAMES[mode])
        print(opt.OBS_CHANNELS)
        opt.train(TIME_SAMPLING,                # How many NCA timesteps between each data step?
                ITERS,                        
                WARMUP=50,                    
                optimiser=optimiser,          
                LOSS_FUNC_STR="euclidean",    
                LOOP_AUTODIFF="lax",          
                LOG_EVERY=100,
                REGULARISER_COEFFS={
                    "intermediate_state":1.0,
                    "boundary": 0.0,
                    "contiguous_growth":0.0,
                    "update_sensitivity":0.0,
                    "perturbation_conservation":0.0
                },
                wandb_args={
                    "project":"NCA-PDEs",
                    "group":"basic-nca-to-pde",
                    "tags":["training"],
                    "name":mode+"_T12_t32"
                },
                key=key)                # Random key for data augmentation and model stochasticity

    return (train_nca,)


@app.cell
def _():
    modes = [
        # "gray-scott",
        # "heat-equation",
        # "chhabra",
        "cahn-hilliard",
        # "hillen-painter",
        # "keller-segel",
    ]
    return (modes,)


@app.cell
def _():
    mo.md(r"""
    ## Train NCA models to various PDE trajectories
    """)
    return


@app.cell(disabled=True)
def _(modes, train_nca):
    _key = jr.PRNGKey(int(time.time()))
    wandb.finish()
    for _i,_mode in enumerate(modes):

        _key = jr.fold_in(_key,_i)
        print(f"Training NCA for {_mode} PDE")
        try:
            train_nca(_mode,_key)
        except Exception as e:
            print(f"Error training NCA for {_mode} PDE: {e}")
    # train_nca("heat-equation",jr.PRNGKey(1234))
    return


@app.cell
def _():
    # Random key for data augmentation and model stochasticity                      
    return


@app.cell
def _():
    mo.md(r"""
    ## Load and run trained NCA
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    from tqdm import tqdm
    import numpy as onp

    return onp, tqdm


@app.cell
def _():
    MODE_TO_EVAL = ""
    return


@app.cell
def _():
    STEPS_PER_PDE = {
        "gray-scott":5000,
        "chhabra":5000,
        "cahn-hilliard":1000,
        "hillen-painter":100,
        "keller-segel":400,
        "heat-equation":1000
    }
    return (STEPS_PER_PDE,)


@app.cell
def _(
    CHANNELS,
    SIZE,
    STEPS_PER_PDE,
    generate_pde_training_data,
    normalise,
    onp,
    tqdm,
):
    def eval_keller_segel_nca(MODE,key=jr.PRNGKey(int(time.time()))):
        timesteps = MODE["time_steps"]
        kernel = MODE["kernels"]
        kernel_to_kernels = {
            "Lap":["ID","LAP"],
            "Diff":["ID","LAP","DIFF"],
            "Grad":["ID","LAP","GRAD"]
        }
        EXTRA_STEPS_FACTOR = 1
        NCA_STEPS_TRAINING = 12*timesteps
        PDE_STEPS = STEPS_PER_PDE["keller-segel"] # For gray scott
        # TS,Y = generate_pde_training_data("keller-segel",SIZE,BATCHES,12,STEPS=400)
        # print(Y.shape)
        nca = NCA(N_CHANNELS=CHANNELS,          
                  KERNEL_STR=kernel_to_kernels[kernel],      
                  ACTIVATION=jax.nn.relu,       
                  FIRE_RATE=1.0,                
                  key=key)
        nca = nca.load(f"models/thesis_ch1/pde/nca_pde_keller_segel_T12_t{timesteps}_{kernel}.eqx")
        # ic = build_ic_for_nca(MODE_TO_EVAL)
        _,TrPDE = generate_pde_training_data(
            "keller-segel",
            SIZE*2,
            1,
            int(NCA_STEPS_TRAINING*EXTRA_STEPS_FACTOR),
            STEPS=int(PDE_STEPS*EXTRA_STEPS_FACTOR),
            key=key)
        ic = TrPDE[0,0] # Get initial condition for first batch and first time step
        ic = np.pad(ic,((0,CHANNELS-ic.shape[0]),(0,0),(0,0)))
        # Tr = [ic]
        Tr = [ic]
        for _i in tqdm(range(int(NCA_STEPS_TRAINING*EXTRA_STEPS_FACTOR))):
            ic = nca(ic)
            Tr.append(ic)
        Tr = onp.array(Tr)
        Tr = onp.clip(Tr,0.0,1.0)
        # Tr = 2*T
        # Tr = normalise(Tr)
        TrPDE = normalise(TrPDE)
        Tr = 2*Tr-1
        TrPDE = 2*TrPDE-1
        print(Tr.shape)
        print(TrPDE.shape)
        return Tr,TrPDE

    return (eval_keller_segel_nca,)


@app.cell
def _(eval_keller_segel_nca, keller_segel_video_comparison):
    _tr_diff,_trPDE = eval_keller_segel_nca({"kernels":"Diff","time_steps":16},key=jr.PRNGKey(1234))
    _tr_lap,_ = eval_keller_segel_nca({"kernels":"Lap","time_steps":16},key=jr.PRNGKey(1234))
    # print(_tr.shape)
    print("_tr_diff shape:",_tr_diff.shape)
    print("_tr_lap shape:",_tr_lap.shape)
    print("_trPDE shape:",_trPDE.shape)
    plt.figure(dpi=300)
    plt.imshow(rearrange(_tr_diff[:-1][::32,0],"T X Y -> X (T Y)"))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.figure(dpi=300)
    plt.imshow(rearrange(_tr_lap[:-1][::32,0],"T X Y -> X (T Y)"))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.figure(dpi=300)
    plt.imshow(rearrange(_trPDE[0,::32,0],"T X Y -> X (T Y)"))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    keller_segel_video_comparison(_tr_lap,_tr_diff,_trPDE,"keller-segel-kernels-comparison")
    # side_by_side_video_comparison(_tr,_trPDE,"keller-segel-Lap-16")
    return


@app.cell
def _(
    CHANNELS,
    FILENAMES,
    SIZE,
    STEPS_PER_PDE,
    generate_pde_training_data,
    normalise,
    onp,
    tqdm,
):
    def load_and_evaluate_nca(mode):
        EXTRA_STEPS_FACTOR = 1
        NCA_STEPS_TRAINING = 12*32
        PDE_STEPS = STEPS_PER_PDE[mode] # For gray scott

        nca = build_nca(jr.PRNGKey(0),CHANNELS,mode)

        nca = nca.load(f"models/{FILENAMES[mode]}.eqx")
        # ic = hillen_painter_ic(SIZE,1,TIME_SAMPLING*8)[1]
        # ic = np.pad(ic,((0,0),(0,CHANNELS-ic.shape[1]),(0,0),(0,0))) # Pad initial condition with zeros to match number of NCA channels)
        # ic = build_ic_for_nca(MODE_TO_EVAL)
        # ic = ic[0]
        _,TrPDE = generate_pde_training_data(mode,SIZE*2,1,NCA_STEPS_TRAINING*EXTRA_STEPS_FACTOR,STEPS=PDE_STEPS*EXTRA_STEPS_FACTOR)
        ic = TrPDE[0,0] # Get initial condition for first batch and first time step
        ic = np.pad(ic,((0,CHANNELS-ic.shape[0]),(0,0),(0,0)))
        Tr = [ic]
        for _i in tqdm(range(NCA_STEPS_TRAINING*EXTRA_STEPS_FACTOR)):
            ic = nca(ic)
            Tr.append(ic)
        print(ic.shape)
        Tr = onp.array(Tr)
        Tr = onp.clip(Tr,0.0,1.0)
        Tr = normalise(Tr)

        TrPDE = normalise(TrPDE)
        return Tr,TrPDE
    # _,TrPDE = solve_gray_scott(*gray_scott_ic(SIZE*2,1,STEPS=PDE_STEPS*EXTRA_STEPS_FACTOR,TIME_RESOLUTION=NCA_STEPS_TRAINING*EXTRA_STEPS_FACTOR))
    # print(Tr.shape)
    # plt.imshow(rearrange(Tr[::100,0],"T X Y -> X (T Y)"))
    # print(nca)
    return (load_and_evaluate_nca,)


@app.cell
def _(onp):
    def normalise(x):
        return (x+onp.min(x))/(onp.max(x)-onp.min(x))

    return (normalise,)


@app.cell
def _():
    # _,TrPDE = solve_gray_scott(*gray_scott_ic(SIZE*2,1,STEPS=100*12,TIME_RESOLUTION=1000))
    return


@app.cell
def _(load_and_evaluate_nca, side_by_side_video_comparison):
    _mode = "gray-scott"
    Tr,TrPDE = load_and_evaluate_nca(_mode)
    plt.imshow(rearrange(TrPDE[0,::32,0],"T X Y -> X (T Y)"))
    plt.show()
    plt.imshow(rearrange(Tr[::32,0],"T X Y -> X (T Y)"))
    plt.show()
    side_by_side_video_comparison(Tr,TrPDE,_mode)
    # print(TrPDE.shape)
    # save_to_video()
    return


@app.cell
def _(onp):
    def keller_segel_video_comparison(TR_NCA_LAP,TR_NCA_DIFF,TR_PDE,mode):
        TR_PDE = TR_PDE[0,:,0]
        TR_NCA_DIFF = TR_NCA_DIFF[1:,0]
        TR_NCA_LAP = TR_NCA_LAP[1:,0]
        # TR_NCA = onp.clip(TR_NCA,0.0,1.0)
        # TR_NCA_LAP = onp.clip(TR_NCA_LAP,0.0,1.0)
        # TR_NCA_DIFF = onp.clip(TR_NCA_DIFF,0.0,1.0)
        # TR_PDE = onp.clip(TR_PDE,0.0,1.0)
        print(TR_NCA_DIFF.shape)
        print(TR_PDE.shape)
        # print(TrPDE[0,:,0].shape)
        # print(Tr[1:,0].shape)
        # ZERO_PAD = onp.zeros((TrPDE.shape[1],TrPDE.shape[2]-Tr[1:,0].shape[1])) # Pad NCA output with zeros to match PDE trajectory shape))
        ZERO_PAD = onp.ones((TR_PDE.shape[0],TR_PDE.shape[1],32))*-1
        # TR_COMB = onp.concatenate([TrPDE[0,:,0],ZERO_PAD,Tr[1:,0]],axis=-1)
        TR_COMB = onp.concatenate([
            TR_PDE,
            ZERO_PAD,
            TR_NCA_LAP,
            ZERO_PAD,
            TR_NCA_DIFF
        ],axis=-1)

        print(TR_COMB.shape)
        save_to_video_mono(
            TR_COMB,
            f"/home/alex/PhD/Differentiable-Patterning/Videos/ThesisPDEs/{mode}_comparison.mp4",
            fps=30,
            duration=20,
            cmap="inferno",
        )
    # side_by_side_video_comparison(Tr,TrPDE)
    return (keller_segel_video_comparison,)


@app.cell
def _(onp):
    def side_by_side_video_comparison(TR_NCA,TR_PDE,mode):
        TR_PDE = TR_PDE[0,:,0]
        TR_NCA = TR_NCA[1:,0]
        TR_NCA = onp.clip(TR_NCA,0.0,1.0)
        TR_PDE = onp.clip(TR_PDE,0.0,1.0)
        TR_NCA = 2*TR_NCA - 1
        TR_PDE = 2*TR_PDE - 1
        print(TR_NCA.shape)
        print(TR_PDE.shape)
        # print(TrPDE[0,:,0].shape)
        # print(Tr[1:,0].shape)
        # ZERO_PAD = onp.zeros((TrPDE.shape[1],TrPDE.shape[2]-Tr[1:,0].shape[1])) # Pad NCA output with zeros to match PDE trajectory shape))
        ZERO_PAD = onp.ones((TR_PDE.shape[0],TR_PDE.shape[1],32))*-1
        # TR_COMB = onp.concatenate([TrPDE[0,:,0],ZERO_PAD,Tr[1:,0]],axis=-1)
        TR_COMB = onp.concatenate([TR_PDE,ZERO_PAD,TR_NCA],axis=-1)

        print(TR_COMB.shape)
        save_to_video_mono(TR_COMB,f"/home/alex/PhD/Differentiable-Patterning/Videos/ThesisPDEs/{mode}_comparison.mp4",fps=30,duration=20,cmap="inferno")
    # side_by_side_video_comparison(Tr,TrPDE)
    return (side_by_side_video_comparison,)


@app.cell(column=1)
def _():
    mo.md(r"""
    ## Generate valid initial conditions for each PDE
    """)
    return


@app.cell
def _():
    def gray_scott_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=5000,key=jr.PRNGKey(int(time.time()))):
        # key = jr.PRNGKey(int(time.time()))
        # x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
        x0 = np.zeros((BATCHES,2,SIZE,SIZE))
        x0 = x0.at[0,1,SIZE//6:5*SIZE//6,SIZE//6:5*SIZE//6].set(1.0)
        x0 = x0.at[0,1,SIZE//4:3*SIZE//4,SIZE//4:3*SIZE//4].set(0.0)
        x0 = x0.at[1,1,SIZE//4:SIZE//4+10,SIZE//4:SIZE//4+10].set(1.0)
        x0 = x0.at[1,1,:,3*SIZE//4:3*SIZE//4+5].set(1.0)
        x0 = x0.at[1,1,3*SIZE//4:3*SIZE//4+3,:].set(1.0)
        op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=2)
        v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
        # for i in range(2):
            # x0 = v_av(x0)
        # x0 = x0.at[:,1].set(np.where(x0[:,1]>0.55,1.0,0.0))

        # x0 = x0.at[:,1,:SIZE//4].set(0)
        # x0 = x0.at[:,1,:,:SIZE//4].set(0)
        # x0 = x0.at[:,1,-SIZE//4:].set(0)
        # x0 = x0.at[:,1,:,-SIZE//4:].set(0)
        for i in range(1):
            x0 = v_av(x0)
        x0 = x0.at[:,0].set(1-x0[:,1])

        ts = np.linspace(0,STEPS,TIME_RESOLUTION)
        # ts = repeat(ts,"T -> B T",B=BATCHES)
        return ts,x0
    # plt.imshow(gray_scott_ic(SIZE,BATCHES,100)[0][0,0])
    return (gray_scott_ic,)


@app.function
def chhabra_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=5000,key=jr.PRNGKey(int(time.time()))):
    scale=0.5
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
    ts = np.linspace(0,STEPS,TIME_RESOLUTION) 
    return ts,x0


@app.function
def cahn_hilliard_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=1000,key=jr.PRNGKey(int(time.time()))):
    # key = jax.random.PRNGKey(int(time.time()))
    x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))*2 - 1
    ts = np.linspace(0,STEPS,TIME_RESOLUTION)
    return ts,x0


@app.function
def hillen_painter_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=100,key=jr.PRNGKey(int(time.time()))):
    # key = jax.random.PRNGKey(int(time.time()))
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=2)
    v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(1):
        x0 = v_av(x0)
    ts = np.linspace(0,STEPS,TIME_RESOLUTION)
    return ts,x0


@app.function
def keller_segel_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=400,key=jr.PRNGKey(int(time.time()))):
    # key = jax.random.PRNGKey(int(time.time()))
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*0.5
    op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=2)
    v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(1):
        x0 = v_av(x0)
    ts = np.linspace(0,STEPS,TIME_RESOLUTION)
    return ts,x0


@app.function
def heat_equation_ic(SIZE,BATCHES,TIME_RESOLUTION,STEPS=1000,key=jr.PRNGKey(int(time.time()))):
    # key = jax.random.PRNGKey(int(time.time()))

    # x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))
    x0 = np.zeros((BATCHES,1,SIZE,SIZE))
    x0 = x0.at[0,0,SIZE//2-10:SIZE//2+10,SIZE//2-10:SIZE//2+10].set(1.0)
    x0 = x0.at[1,0].set(jr.uniform(key,shape=(SIZE,SIZE)))
    ts = np.linspace(0,STEPS,TIME_RESOLUTION)
    return ts,x0


@app.cell
def _():
    mo.md(r"""
    ## Set up solver wrappers for each PDE
    - Each PDE has different stability criteria and timescales
    """)
    return


@app.function(hide_code=True)
def solve_cahn_hilliard(ts,x0):
    func = F_cahn_hilliard(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    # Y = Y[:,:,:]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    return T,Y


@app.function
def solve_gray_scott(ts,x0):
    func = F_gray_scott(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.5)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    Y = Y[:,:,:1]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    return T,Y


@app.function(hide_code=True)
def solve_chhabra(ts,x0):

    func = F_chhabra(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    # Y = Y[:,:,:1]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    # Y = Y[:,::TIME_SAMPLING]             # Rescale data between 0 and 1
    return T,Y


@app.function
def solve_hillen_painter(ts,x0):
    func = F_hillen_painter(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.01)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    Y = Y[:,:,:]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    return T,Y


@app.function
def solve_keller_segel(ts,x0):
    func = F_keller_segel(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1,D=1.0,epsilon=0.2,alpha=0.01,c=3.0)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.02)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    Y = Y[:,:,:1]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    return T,Y


@app.function
def solve_heat_equation(ts,x0):
    func = F_heat_equation(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=1)
    v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
    solver = PDE_solver(v_func,dt=0.5)
    T,Y = solver(ts=ts,y0=x0)
    Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
    Y = Y[:,:,:]                                                   # Only include main channel, not inhibitor/other chemical - see if the NCA can learn from only 1 channel
    Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))                      # Rescale data between 0 and 1
    return T,Y


@app.cell
def _(CHANNELS, SIZE, generate_pde_training_data):
    def build_ic_for_nca(mode):
        T,Y = generate_pde_training_data(mode,SIZE*2,1,12)
        ic = Y[0,0] # Get initial condition for first batch and first time step
        ic = np.pad(ic,((0,CHANNELS-ic.shape[0]),(0,0),(0,0))) # Pad initial condition with zeros to match number of NCA channels)
        return ic

    return


@app.cell
def _():
    mo.md(r"""
    ## Try various things for Keller-Segel training
    """)
    return


@app.cell
def _(
    BATCHES,
    CHANNELS,
    ITERS,
    SIZE,
    build_trainer,
    generate_pde_training_data,
):
    # NCA kernels - just lap, lap and diff or lap and grad
    # NCA time steps between PDE samples
    from marimo_utils import generate_hyperparameter_combinations_indexed
    MODE = {
        "kernels":["Lap","Diff","Grad"],
        "time_steps":[16,32,64]
    }


    MODE = generate_hyperparameter_combinations_indexed(MODE)
    def train_keller_segel(MODE,key):
        timesteps = MODE["time_steps"]
        kernel = MODE["kernels"]
        kernel_to_kernels = {
            "Lap":["ID","LAP"],
            "Diff":["ID","LAP","DIFF"],
            "Grad":["ID","LAP","GRAD"]
        }
        TS,Y = generate_pde_training_data("keller-segel",SIZE,BATCHES,12,STEPS=400)
        print(Y.shape)
        nca = NCA(N_CHANNELS=CHANNELS,          
                  KERNEL_STR=kernel_to_kernels[kernel],      
                  ACTIVATION=jax.nn.relu,       
                  FIRE_RATE=1.0,                
                  key=key)
        opt,optimiser = build_trainer(nca,Y,f"thesis_ch1/pde/nca_pde_keller_segel_T12_t{timesteps}_{kernel}")
        print(opt.OBS_CHANNELS)
        opt.train(timesteps,                # How many NCA timesteps between each data step?
                ITERS,                        
                WARMUP=50,                    
                optimiser=optimiser,          
                LOSS_FUNC_STR="euclidean",    
                LOOP_AUTODIFF="lax",          
                LOG_EVERY=100,
                REGULARISER_COEFFS={
                    "intermediate_state":1.0,
                    "boundary": 0.0,
                    "contiguous_growth":0.0,
                    "update_sensitivity":0.0,
                    "perturbation_conservation":0.0
                },
                wandb_args={
                    "project":"NCA-PDEs",
                    "group":"keller-segel-hyperparameter-search",
                    "tags":["training"],
                    "name":f"keller-segel_{kernel}_t{timesteps}"
                },
                key=key)                # Random key for data augmentation and model stochasticity

    return MODE, train_keller_segel


@app.cell
def _(MODE, train_keller_segel):
    wandb.finish()
    _key = jr.PRNGKey(int(time.time()))
    for _mode in MODE:
        _key = jr.fold_in(_key,1)
        try:
            train_keller_segel(_mode,_key)
        except Exception as e:
            print(f"Error training NCA for keller-segel with kernels {_mode['kernels']} and time steps {_mode['time_steps']}: {e}")
    return


if __name__ == "__main__":
    app.run()
