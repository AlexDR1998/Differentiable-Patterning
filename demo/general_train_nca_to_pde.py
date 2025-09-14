#!/usr/bin/env python
# general_train_nca_to_pde.py
# ================================================================
import time, sys, os, argparse
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schnakenberg
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.fixed_models.update_keller_segel import F as F_ks
from PDE.model.fixed_models.update_fhn import F as F_fhn
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA
from NCA.model.DINCA_model import DINCA
import jax.lax as lax
import numpy as np

# ------------------------------------------------------------------
# 0 · Command-line interface
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pde",            default="sch",
                    choices=["sch", "fhn", "g1", "g2", "g3", "ks"])
parser.add_argument("--batches",  type=int,   default=6)                    
parser.add_argument("--time_sampling",  type=int,   default=32)
parser.add_argument("--learn_rate",     type=float, default=5e-4)
parser.add_argument("--channels",       type=int,   default=16)
parser.add_argument("--loss",           default="euclidean")
parser.add_argument("--model_filename", default="demo/train_nca_to_pde")
parser.add_argument("--fire_rate",      type=float, default=1.0)
parser.add_argument("--state_reg",      type=float, default=1.0)
parser.add_argument("--target_sparsity",   type=float, default=0.5)
parser.add_argument("--sparse_pruning",   type=bool, default=False)
parser.add_argument("--kernel_scale",   type=int, default=1.0)
parser.add_argument("--model", choices=["nca", "dinca"], default="nca")
args = parser.parse_args()

# ------------------------------------------------------------------
# 1 · Global training constants
# ------------------------------------------------------------------
ITERS         = 40000
SIZE          = 64
TIME_SAMPLING = args.time_sampling
CHANNELS      = args.channels
LEARN_RATE    = args.learn_rate
LOSS_FUNC_STR = args.loss
FIRE_RATE     = args.fire_rate
STATE_REGULARISER = args.state_reg
TARGET_SPARSITY = args.target_sparsity
KERNEL_SCALE = args.kernel_scale
SPARSE_PRUNING = args.sparse_pruning
MODEL_DIR     = f"{args.model_filename}_{args.pde}"
RADIUS = 6   # 3 → 7×7 square
CELL_CHANNELS   = 1 #chemotaxis
SIGNAL_CHANNELS = 1 #chemotaxis
BATCHES = args.batches
# ------------------------------------------------------------------
# Static 3×3 circular averaging kernel for Gray–Scott noisy seeds
# ------------------------------------------------------------------
AVG_OP_GS = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)



# ------------------------------------------------------------------
# 2 · PDE-specific configuration
# ------------------------------------------------------------------
PDE_CONFIGS = {  # all floats (jnp)
    "sch": dict(a=0.01,  b=2.0,   D=80.0,
                         steady=lambda p: (p["a"]+p["b"],
                                            p["b"]/(p["a"]+p["b"])**2)),
    "fhn":          dict(D=20.0, eps_v=0.5, a_v=1.0, a_z=-0.1,
                         steady=lambda p: (0.0, 0.0)),
    "g1":       dict(DA=0.1,DB=0.05,alpha=0.06230,gamma=0.06268, #labyrinth
                         steady=lambda p: (0.0, 0.0)),
    "g2":       dict(DA=0.1,DB=0.05,alpha=0.046,gamma=0.065, #worms
                         steady=lambda p: (0.0, 0.0)),
    "g3":       dict(DA=0.1,DB=0.05,alpha=0.018,gamma=0.055, #spots
                         steady=lambda p: (0.0, 0.0)),
    "ks":  dict(alpha=0.01, c=3.8, D=0.8, epsilon=0.1,
                         steady=lambda p: (0.0, 0.0)),
}

cfg = PDE_CONFIGS[args.pde]

# factory for RHS
def make_rhs(pde_name, **pars):
    if pde_name == "sch":
        return F_schnakenberg(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=KERNEL_SCALE, **pars)
    if pde_name == "fhn":
        return F_fhn(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=KERNEL_SCALE, **pars)
    if pde_name in ["g1", "g2", "g3"]:
        return F_gray_scott(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=KERNEL_SCALE, **pars)
    if pde_name == "ks":
        return F_ks(PADDING="CIRCULAR", dx=0.5, KERNEL_SCALE=KERNEL_SCALE, **pars)
    raise ValueError("unknown PDE")

rhs   = make_rhs(args.pde, **{k: v for k, v in cfg.items() if k != "steady"})
v_rhs = eqx.filter_vmap(rhs, in_axes=(None, 0, None), out_axes=0)
if args.pde.startswith("g"):
    dt = 0.2
else:
    dt = 5e-3
solver= PDE_solver(v_rhs, dt=dt)

# ------------------------------------------------------------------
# 3 · Initial-condition generator (unchanged except steady state)
# ------------------------------------------------------------------
U_eq, V_eq = cfg["steady"](cfg)
sigma      = 0.03

def make_ic(key, choice: jnp.ndarray):
    k1, k2 = jr.split(key)

    def _base():
        U = jnp.full((SIZE, SIZE), U_eq)
        V = jnp.full_like(U, V_eq)
        return U, V

    def noise(_):
        if args.pde.startswith("g"):
            """Smoothed random mask for Gray–Scott, or Gaussian noise otherwise."""
            # 1) uniform noise in both channels ...............................
            X = jr.uniform(k2, (2, SIZE, SIZE), dtype=jnp.float32)

            # 2) blur helper (expects (H,W) , returns (H,W))
            blur = lambda z: AVG_OP_GS.Average(z[None, ...])[0]

            # 3) 5× blur on each channel (vmap over channel axis) ...........
            for _ in range(5):
                X = jax.vmap(blur)(X)

            # 4) threshold first channel to binary mask ......................
            U, V = X
            mask = (U > 0.51).astype(U.dtype)
            U    = 1.0 - mask            # white background, dark blobs

            # 5) soften edges + enforce U+V = 1 ..............................
            U    = blur(U)
            V    = 1.0 - U
            return jnp.stack([U, V])     # (2,H,W)
        elif args.pde == "ks":
            # --- Uniform[0,0.1] in cell channel, zero chemo -------------
            U = jr.uniform(
                k2,
                shape=(SIZE, SIZE),
                minval=0.0,
                maxval=0.1,
                dtype=jnp.float32
            )
            V = jnp.zeros_like(U)          # chemoattractant initially 0
            X = jnp.stack([U, V], axis=0)  # shape (2, H, W)
            return X
        else:
            U, V = _base()
            r1, r2 = jr.split(k1)
            U += sigma * jr.normal(r1, (SIZE, SIZE))
            V += sigma * jr.normal(r2, (SIZE, SIZE))
            return jnp.stack([U, V])


    def gaussian_patch(r, delta=0.2, sigma=1.5):
        ax = jnp.arange(-r, r+1)
        g  = jnp.exp(-(ax**2)/(2*sigma**2))
        kern = jnp.outer(g, g)
        return kern / kern.max() * delta

    # ------------------------------------------------------------------
    # Gray–Scott “inverted” circular blobs
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Gray–Scott OLD : overlap-safe inverted blobs  ---------------------
    def gs_inverted_blob(rng, n=1, radius=SIZE // 8):
        """
        White background (U=1, V=0) with `n` circular blobs where
        U=0, V=1.  Overlapping blobs merge without square artefacts.
        """
        H = SIZE
        U = jnp.ones((H, H), dtype=jnp.float32)     # background U=1
        V = jnp.zeros_like(U)                       # background V=0

        # pre-compute coordinate grids once (static w.r.t. loop)
        xs = jnp.arange(H)[:, None]                 # shape (H,1)
        ys = jnp.arange(H)[None, :]                 # shape (1,H)

        centres = jr.randint(rng, (n, 2), radius, H - radius)  # (n,2)

        def body(carry, centre):
            A, B = carry
            cx, cy = centre
            mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius ** 2  # (H,H) bool
            A = jnp.where(mask, 0.0, A)      # set U=0 inside circle
            B = jnp.where(mask, 1.0, B)      # set V=1 inside circle
            return (A, B), None

        (U, V), _ = lax.scan(body, (U, V), centres)
        return U, V



    def _scatter(U, rng, n, delta=0.2, radius=3):
        """
        Add `n` square patches of side (2*radius+1) to U.
        delta  : peak intensity to add
        radius : 0 → 1-pixel, 1 → 3×3, 2 → 5×5, …

        Returns updated U (V untouched).
        """
        patch_size = 2 * radius + 1
        patch = gaussian_patch(radius, delta)

        # choose centres at least `radius` pixels from the border
        xy = jr.randint(rng, (n, 2), radius, SIZE - radius)

        def body(carry, coords):
            A = carry
            x, y = coords
            A = lax.dynamic_update_slice(A, patch, (x - radius, y - radius))
            return A, None

        U, _ = lax.scan(body, U, xy)
        return U

    def central(_):
        if args.pde.startswith("g") or args.pde == "ks":
            U, V = gs_inverted_blob(k2, n=1)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            U = _scatter(U, k2, n=1, delta=0.2, radius=RADIUS)   # use same helper
            return jnp.stack([U, V])

    def two(_):
        if args.pde.startswith("g") or args.pde == "ks":
            U, V = gs_inverted_blob(k2, n=2)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            U = _scatter(U, k2, n=2, delta=0.2, radius=RADIUS)
            return jnp.stack([U, V])
    def three(k): 
        if args.pde.startswith("g") or args.pde == "ks":
            U, V = gs_inverted_blob(k2, n=3)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            return jnp.stack([_scatter(U, k, n=3, delta=0.2, radius=RADIUS), V])
    def four(k):  
        if args.pde.startswith("g") or args.pde == "ks":
            U, V = gs_inverted_blob(k2, n=4)
            return jnp.stack([U, V])
        else:
            U, V = _base()
            return jnp.stack([_scatter(U, k, n=4, delta=0.2, radius=RADIUS), V])

    return lax.switch(choice,
                      (noise, central, two, three, four),
                      k2)

if BATCHES == 4:
    num_0 = 1
    num_1 = 0
    num_2 = 1
    num_3 = 1
    num_4 = 1
if BATCHES == 6:
    num_0 = 2
    num_1 = 1
    num_2 = 1
    num_3 = 1
    num_4 = 1
if BATCHES == 8:
    num_0 = 2
    num_1 = 1
    num_2 = 2
    num_3 = 2
    num_4 = 1
if BATCHES == 10:
    num_0 = 3
    num_1 = 0
    num_2 = 2
    num_3 = 2
    num_4 = 3
mix = {0: num_0, 1: num_1, 2: num_2, 3: num_3, 4: num_4}
key, *sub = jr.split(jr.PRNGKey(0), BATCHES + 1)
sub = jnp.array(sub)
choices = jnp.concatenate([jnp.full(n, c, jnp.int32)
                           for c, n in mix.items()])
x0 = jax.vmap(make_ic)(sub, choices)         # (B,2,H,W)

# ------------------------------------------------------------------
# 4 · Ground-truth trajectories
# ------------------------------------------------------------------
sampling_constant = 32
if args.pde.startswith("g"):
    if args.pde.startswith("g3"):
        ts = jnp.linspace(0, 5000, sampling_constant * 10)
    else:
        ts = jnp.linspace(0, 10000, sampling_constant * 10)
else:
    ts = jnp.linspace(0, sampling_constant * 3, sampling_constant * 10)
T, Y = solver(ts, x0)
# reshape and keep only U-channel for NCA training
Y = rearrange(Y, "T B C X Y -> B T C X Y")

if args.pde.startswith("ks"):
    # --- 5) Normalize both channels to [–1, +1] separately over the entire timeline ---
    for ch in range(CELL_CHANNELS + SIGNAL_CHANNELS):
        ch_min = Y[:, :, ch].min()
        ch_max = Y[:, :, ch].max()
        Y = Y.at[:, :, ch].set((Y[:, :, ch] - ch_min) / (ch_max - ch_min))
    # --- 6) Downsample in time by TIME_SAMPLING = 32 ---
    Y = Y[:, ::sampling_constant, :, :, :]
else:
    if args.model == "nca":
        Y = Y[:, :, :1]                             # keep only U for others
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        Y = Y[:, ::sampling_constant]                   # downsample in time
    elif args.model == "dinca":
        mins = Y.min(axis=(0, 1, 3, 4), keepdims=True)  # shape (1, 1, C, 1, 1)
        ptps = Y.ptp(axis=(0, 1, 3, 4), keepdims=True)  # shape (1, 1, C, 1, 1)
        Y = (Y - mins) / ptps
        Y = Y[:, ::sampling_constant]                       # downsample in time
    else:
        raise ValueError("unknown model flag")

# ------------------------------------------------------------------
# 5 · Build NCA + trainer
# ------------------------------------------------------------------
if args.model == "nca":
    nca = NCA(N_CHANNELS=CHANNELS,
              KERNEL_STR=["ID", "LAP", "GRAD"],
              ACTIVATION=jax.nn.relu,
              FIRE_RATE=FIRE_RATE,
              key=jr.PRNGKey(1))
elif args.model == "dinca":
    nca = DINCA(N_CHANNELS=2,           # u and v only
                #KERNEL_STR=["ID", "LAP", "GRAD"],
                FIRE_RATE=FIRE_RATE,
                key=jr.PRNGKey(1))
else:
    raise ValueError("unknown model flag")

trainer = NCA_Trainer(
    nca, Y,
    model_filename=MODEL_DIR,
    DATA_AUGMENTER=DataAugmenter,
    GRAD_LOSS=True
)

schedule = optax.exponential_decay(
    LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(
    optax.scale_by_param_block_norm(),
    optax.nadam(schedule)
)

trainer.train(
    TIME_SAMPLING,
    ITERS,
    WARMUP=50,
    optimiser=optimiser,
    LOSS_FUNC_STR=LOSS_FUNC_STR,
    LOOP_AUTODIFF="checkpointed",
    LOG_EVERY=50,
    key=jr.PRNGKey(2),
    STATE_REGULARISER=STATE_REGULARISER,
    TARGET_SPARSITY=TARGET_SPARSITY,
    SPARSE_PRUNING=SPARSE_PRUNING,
    wandb_args={"project":"NCA",
                "group":"group_1",
                "name":args.model_filename,
                "tags":["training"]}
)

if args.model == "dinca":
    print("\\nLearned PDE:")
    print(nca.pretty_print_pde())