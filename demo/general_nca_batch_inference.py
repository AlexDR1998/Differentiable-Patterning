#!/usr/bin/env python
# ------------------------------------------------------------
#   general_nca_batch_inference.py
#   Evaluate a trained NCA checkpoint against its reference PDE.
#
#   Supported PDE keys   --pde {sch | fhn | g1 | g2 | g3 | ks}
#   The script
#     • builds the same initial-condition mixture that was used
#       during training (see general_train_nca_to_pde.py),
#     • rolls out the fixed-parameter PDE,
#     • rolls out the trained NCA for an identical number of
#       micro-steps ( --time_sampling ),
#     • reports the batch-mean MSE   and optionally stores
#       grid visualisations   GT / NCA / error   per trajectory.
# ------------------------------------------------------------
import argparse, os, re, sys, pathlib, itertools

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",  "0.4")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax, jax.random as jr, jax.numpy as jnp, jax.lax as lax, equinox as eqx
import matplotlib.pyplot as plt
from einops import rearrange
sys.path.append("..")          # repo root

# ── project imports ────────────────────────────────────────────
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_schnakenberg import F as F_schn
from PDE.model.fixed_models.update_fhn           import F as F_fhn
from PDE.model.fixed_models.update_gray_scott    import F as F_gs
from PDE.model.fixed_models.update_keller_segel  import F as F_ks
from PDE.model.solver.semidiscrete_solver        import PDE_solver
from NCA.model.NCA_model                         import NCA
# ───────────────────────────────────────────────────────────────

# ------------------------------------------------------------------
# 0 · CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True, help="*.eqx checkpoint")
parser.add_argument("--pde",    required=True, choices=["sch","fhn","g1","g2","g3","ks"])
parser.add_argument("--mix",    default="0:4,1:1,2:3,3:3,4:3",
                    help="IC mixture  (e.g. '0:2,1:1')")
parser.add_argument("--time_sampling", type=int, required=True,
                    help="NCA micro-steps between successive comparisons")
parser.add_argument("--kernel_scale", type=int, required=True,
                    help="kernel scale")
parser.add_argument("--outdir", default="batch_inference")
parser.add_argument("--seed",   type=int, default=123)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# ------------------------------------------------------------------
# 1 · constants shared with the training script
# ------------------------------------------------------------------
SIZE        = 64
CHANNELS    = 16          # ← must match checkpoint
KERNEL_STR  = ["ID","LAP","GRAD"]
FIRE_RATE   = 1.0         # irrelevant during inference but required by ctor
PADDING     = "CIRCULAR"
NUM_INTERVALS = 8         # number of snapshot pairs per trajectory

# ------------------------------------------------------------------
# 2 · PDE-specific parameters and helpers
# ------------------------------------------------------------------
PDE_PAR = {
    "sch": dict(a=0.01,  b=2.0,   D=80.0),
    "fhn": dict(D=20.0,  eps_v=0.5, a_v=1.0, a_z=-0.1),
    "g1":  dict(DA=0.1, DB=0.05, alpha=0.06230, gamma=0.06268),
    "g2":  dict(DA=0.1, DB=0.05, alpha=0.046,   gamma=0.065 ),
    "g3":  dict(DA=0.1, DB=0.05, alpha=0.018,   gamma=0.055 ),
    "ks":  dict(alpha=0.01, c=3.8, D=0.8, epsilon=0.1),
}

def rhs_factory(tag:str):
    p = PDE_PAR[tag]
    if tag=="sch":  return F_schn(PADDING=PADDING, dx=1.0, **p)
    if tag=="fhn":  return F_fhn (PADDING=PADDING, dx=1.0, **p)
    if tag.startswith("g"): return F_gs(PADDING=PADDING, dx=1.0, **p)
    if tag=="ks":   return F_ks  (PADDING=PADDING, dx=0.5, **p)
    raise ValueError(tag)

# steady states (only for IC helper)
def steady(tag):
    if tag=="sch":
        a,b=PDE_PAR[tag]["a"],PDE_PAR[tag]["b"]
        u, v = a+b,  b/(a+b)**2
        return u, v
    return 0.0, 0.0          # others use zero background

U_eq, V_eq = steady(args.pde)
sigma = 0.03
AVG_OP_GS  = Ops(PADDING="CIRCULAR", dx=1.0, KERNEL_SCALE=3)

# ------------------------------------------------------------------
# 3 · initial-condition generator  (lifted from training script)
# ------------------------------------------------------------------
RADIUS = 6                                       # square scatter
def gaussian_patch(r, delta=0.2, sigma=1.5):
    ax = jnp.arange(-r, r+1); g=jnp.exp(-(ax**2)/(2*sigma**2))
    k  = jnp.outer(g,g); return k/k.max()*delta

def gs_inverted_blob(rng, n=1, radius=SIZE//8):
    H   = SIZE
    U   = jnp.ones((H,H), jnp.float32)
    V   = jnp.zeros_like(U)
    xs  = jnp.arange(H)[:,None]; ys=jnp.arange(H)[None,:]
    ctr = jr.randint(rng,(n,2),radius,H-radius)
    def body(carry,c):
        A,B=carry; cx,cy=c
        mask=((xs-cx)**2+(ys-cy)**2)<=radius**2
        A = jnp.where(mask,0.0,A);  B=jnp.where(mask,1.0,B)
        return (A,B),None
    (U,V),_=lax.scan(body,(U,V),ctr); return U,V

def make_ic(key, choice:jnp.ndarray):
    k1,k2 = jr.split(key)
    def _base():
        U=jnp.full((SIZE,SIZE),U_eq); V=jnp.full_like(U,V_eq); return U,V
    # --- choice-specific branches --------------------------------
    def branch_noise(_):
        if args.pde.startswith("g"):    # special GS noise
            X  = jr.uniform(k2,(2,SIZE,SIZE))
            blur=lambda z:AVG_OP_GS.Average(z[None])[0]
            for _ in range(5): X=jax.vmap(blur)(X)
            U,V=X; mask=(U>0.51).astype(U.dtype); U=1.-mask; U=blur(U); V=1.-U
            return jnp.stack([U,V])
        if args.pde=="ks":
            U=jr.uniform(k2,(SIZE,SIZE),minval=0.,maxval=0.1)
            V=jnp.zeros_like(U); return jnp.stack([U,V])
        U,V=_base(); r1,r2=jr.split(k1); U+=sigma*jr.normal(r1,U.shape)
        V+=sigma*jr.normal(r2,V.shape); return jnp.stack([U,V])
    def scatter(U,rng,n,delta=0.2,radius=RADIUS):
        patch=gaussian_patch(radius,delta)
        xy=jr.randint(rng,(n,2),radius,SIZE-radius)
        def body(a,xy):x,y=xy; a=lax.dynamic_update_slice(a,patch,(x-radius,y-radius));return a,None
        U,_=lax.scan(body,U,xy); return U
    def branch_central(_):
        if args.pde.startswith("g") or args.pde=="ks":
            U,V=gs_inverted_blob(k2,1); return jnp.stack([U,V])
        U,V=_base(); U=scatter(U,k2,1); return jnp.stack([U,V])
    def branch_two(_):
        if args.pde.startswith("g") or args.pde=="ks":
            U,V=gs_inverted_blob(k2,2); return jnp.stack([U,V])
        U,V=_base(); U=scatter(U,k2,2); return jnp.stack([U,V])
    def branch_three(_):
        if args.pde.startswith("g") or args.pde=="ks":
            U,V=gs_inverted_blob(k2,3); return jnp.stack([U,V])
        U,V=_base(); U=scatter(U,k2,3); return jnp.stack([U,V])
    def branch_four(_):
        if args.pde.startswith("g") or args.pde=="ks":
            U,V=gs_inverted_blob(k2,4); return jnp.stack([U,V])
        U,V=_base(); U=scatter(U,k2,4); return jnp.stack([U,V])
    return lax.switch(choice,
        (branch_noise, branch_central, branch_two,
         branch_three, branch_four), k2)

# mixture parsed from CLI
mix = {int(k):int(v) for k,v in
       (p.split(":") for p in re.split(r"[ ,]+", args.mix) if p)}
BATCHES=sum(mix.values())
key,*sub=jr.split(jr.PRNGKey(args.seed),BATCHES+1)
sub=jnp.array(sub)
choices=jnp.concatenate([jnp.full(n,c,jnp.int32) for c,n in mix.items()])
x0 = jax.vmap(make_ic)(sub, choices)        # (B,2,H,W)

# ------------------------------------------------------------------
# 4 · ground-truth PDE trajectory
# ------------------------------------------------------------------
rhs  = rhs_factory(args.pde)
vrhs = eqx.filter_vmap(rhs,in_axes=(None,0,None),out_axes=0)
dt   = 0.2 if args.pde.startswith("g") else (5e-3 if args.pde!="ks" else 5e-3)
solver=PDE_solver(vrhs,dt)
sampling_constant=32
if args.pde.startswith("g"):
    if args.pde.startswith("g3"):
        ts = jnp.linspace(0, 5000, sampling_constant*NUM_INTERVALS)
    else:
        ts = jnp.linspace(0, 10000, sampling_constant*NUM_INTERVALS)
else: 
    ts = jnp.linspace(0, sampling_constant*3, sampling_constant*NUM_INTERVALS)
T,Y = solver(ts,x0)                           # (T,B,2,H,W)
Y = rearrange(Y,"T B C H W -> B T C H W")

# channel selection & normalisation identical to training
if args.pde=="ks":
    for ch in range(Y.shape[2]):
        ch_min, ch_max = Y[:,:,ch].min(), Y[:,:,ch].max()
        Y = Y.at[:,:,ch].set((Y[:,:,ch]-ch_min)/(ch_max-ch_min))
else:
    Y = Y[:,:,:1]; Y=(Y-Y.min())/(Y.max()-Y.min())
Y = Y[:,::sampling_constant]                  # (B, NUM_INTERVALS, C, H, W)

# ------------------------------------------------------------------
# 5 · load NCA  and rollout
# ------------------------------------------------------------------
dummy = NCA(CHANNELS,KERNEL_STR,jax.nn.relu,PADDING,FIRE_RATE,
            args.kernel_scale,jr.PRNGKey(0))
nca   = eqx.tree_deserialise_leaves(args.model, dummy)

def nca_rollout(x_init,k):
    def step(carry,_):
        x,key=carry; x=nca(x,lambda z:z,key)
        return (x,jr.fold_in(key,0)),x
    (_, _), xs = lax.scan(step,(x_init,k),None,length=args.time_sampling)
    return xs[-1]
v_roll=jax.vmap(nca_rollout,in_axes=(0,0),out_axes=0)

hidden=jnp.zeros((BATCHES, CHANNELS-1, SIZE, SIZE))
x_cur=jnp.concatenate([Y[:,0],hidden],axis=1)
keys=jr.split(key,BATCHES)

pred=[]
for _ in range(NUM_INTERVALS):
    pred.append(x_cur[:,:Y.shape[2]])   # observable slice
    x_cur = v_roll(x_cur, keys)
pred=jnp.stack(pred,axis=1)             # (B,T,C,H,W)

mse = jnp.mean((pred - Y)**2, axis=(2,3,4))
mean_mse=float(jnp.mean(mse))
print(f" Batch-mean squared error: {mean_mse:.4e}")

# ------------------------------------------------------------------
# 6 · save grids
# ------------------------------------------------------------------
# for i in range(BATCHES):
#     gt,pr=Y[i],pred[i]
#     fig,ax=plt.subplots(3,NUM_INTERVALS,figsize=(3*NUM_INTERVALS,9))
#     for t in range(NUM_INTERVALS):
#         ax[0,t].imshow(gt[t,0],origin="lower"); ax[0,t].axis("off")
#         ax[1,t].imshow(pr[t,0],origin="lower"); ax[1,t].axis("off")
#         im=ax[2,t].imshow((gt[t,0]-pr[t,0])**2,origin="lower"); ax[2,t].axis("off")
#     ax[0,0].set_ylabel("GT"); ax[1,0].set_ylabel("NCA"); ax[2,0].set_ylabel("Err")
#     plt.colorbar(im,ax=ax[2,:],orientation="horizontal",shrink=0.7)
#     plt.suptitle(f"Traj {i} / N={BATCHES}  MSE={mse[i]:.2e}")
#     fig.savefig(os.path.join(args.outdir,f"grid_{i}.png"),dpi=150)
#     plt.close(fig)
for i in range(len(Y)):
    gt, pr = Y[i], pred[i]
    fig, axes = plt.subplots(3, NUM_INTERVALS, figsize=(3*NUM_INTERVALS, 9))
    
    for t in range(NUM_INTERVALS):
        axes[0, t].imshow(gt[t, 0], origin="lower")
        axes[0, t].axis("off")
        axes[1, t].imshow(pr[t, 0], origin="lower")
        axes[1, t].axis("off")
        err = (gt[t, 0] - pr[t, 0])**2
        im = axes[2, t].imshow(err, origin="lower")
        axes[2, t].axis("off")
    
    axes[0, 0].set_ylabel("GT")
    axes[1, 0].set_ylabel("NCA")
    axes[2, 0].set_ylabel("Err")
    plt.colorbar(im, ax=axes[2, :], orientation="horizontal", shrink=0.7)
    plt.suptitle(f"Traj {i} / N={BATCHES}  mean MSE={mean_mse:.3e}")
    
    filename = os.path.join(args.outdir, f"grid_{i}.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
