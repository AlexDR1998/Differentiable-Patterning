# Baseline NCA SYCL kernel

Build this shared library inside the same oneAPI environment used by the Intel
JAX plugin:

```bash
source ~/dawn-jax/envs/jaxeqx-setup.sh
NCA/model/sycl/files/build_nca_sycl.sh /tmp/libnca_sycl.so
export NCA_SYCL_LIBRARY=/tmp/libnca_sycl.so
```

`nca_sycl.cpp` currently implements the float32 NCA forward update
with ID, gradient norm, gradient, average, and Laplacian perception features;
ReLU; all existing padding modes; the fire-rate mask; and the residual update.

The implementation uses the legacy custom-call ABI required by JAX 0.5.0 and
Intel Extension for OpenXLA 0.7.0. Perception uses an 8x16 spatial tile with a
halo in shared local memory. Pointwise layers and their backward matrix
products use oneMKL GEMM. The `system.precision` values `tensorfloat32` and
`bfloat16` select oneMKL's float-to-TF32 and float-to-BF16 XMX modes,
respectively, with FP32 accumulation; `highest` selects standard FP32 GEMM.
`NCA_SYCL_XMX_MODE` can override this with `standard`, `tf32`, `bf16`,
`bf16x2`, or `bf16x3`.
`nca_sycl_backward.cpp` implements the custom VJP for the state and trainable
pointwise-layer parameters using native SYCL kernels. Circular and zero-padded
3x3 DIFF transposes use an atomic-free 8x16 tiled gather that caches the state,
gx, gy, and perception cotangents in shared local memory. Other linear
stencils use the deterministic gather; unsupported nonlinear/padding cases
retain the conservative atomic fallback.

Ordinary `vmap` execution still emits per-example parameter cotangents for JAX
to reduce correctly. `NCA_sycl.batched_call`, used by `NCA_sycl_Trainer`,
instead enters the custom VJP with each leaf's pre-batched `[N,C,H,W]` state
and accumulates shared-parameter gradients directly over its cells. Outer B
leaves remain independent calls, which avoids the slower large oneMKL shape
and leaves them available for later device sharding. The fixed perception
kernels and random update mask are non-trainable.

`nca_sycl_rollout.cpp` and `nca_sycl_rollout_backward.cpp` group sequential
updates behind one XLA custom call. Boundary constraints are applied after
every native update, and the complete trajectory is returned so existing
per-step regularisers retain their values and gradients. The reverse call
accepts cotangents for both the endpoint and every trajectory state.

`NCA_sycl_Trainer` also has an opt-in two-tile replicated-data-parallel path.
Set `trainer.sharding: 2` to place one of the two outer `B` leaves on each
visible Max 1550 tile. Model parameters are replicated and each tile evaluates
its local loss independently. Reverse-mode autodiff wraps the sharded loss, so
the transpose of its `lax.pmean` reduces parameter gradients once before the
shared optimiser update. Host-side callbacks access physical
`addressable_shards` rather than globally indexing sharded arrays. This path
requires exactly two shape-compatible outer leaves and matching boundary modes.

Set `trainer.sycl_fused_steps` to control how many sequential NCA timesteps
the SYCL trainer groups into one rollout custom call. `1` selects the original
one-step path; values greater than one must divide `run.t`. Runtime filenames
include both `_fuseK` and, when enabled, `_shard2` so sweep results remain
unambiguous.
