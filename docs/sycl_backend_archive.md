# Archived Intel SYCL NCA backend

This document records the state of the Intel SYCL neural cellular automata
(NCA) backend at archive commit `555bb8f` on branch `archive/sycl-nca`. The
backend was developed for Intel Data Center GPU Max 1550 (PVC) tiles on the
DAWN cluster. It is archived because that cluster is no longer available to
the project; CUDA/JAX is the active development target.

The archive is a complete training implementation, not merely an alternative
JAX device selection. It combines Python/Equinox model wrappers, custom JAX
primitives, a C++17 SYCL shared library, oneMKL matrix multiplication, native
reverse-mode derivatives, fused multi-step rollouts, and an optional two-tile
trainer. Its ABI and launch environment are tied to the Intel software stack
described below.

## Status at archival

The backend supports both the ordinary ReLU NCA (`NCA_sycl`) and the gated NCA
(`gNCA_sycl`). The final implementation provides:

- native single-step forward and backward custom calls;
- baseline and GLU-gated output layers;
- native fused multi-step rollout and rollout VJP calls;
- boundary application after each fused update;
- fused intermediate-state and boundary regularisers;
- direct batching of `[N, C, H, W]` leaves, accumulating shared-parameter
  gradients in the native call;
- single-tile training and replicated-data-parallel training across exactly
  two visible PVC tiles;
- FP32 accumulation with standard FP32, TF32, or BF16-family oneMKL compute
  modes; and
- targeted smoke tests, failure probes, scratch guards, and runtime/stability
  sweeps.

The final two-tile reliability mitigation is
`trainer.backend.serialize_onemkl: true`. It serializes and waits for native
oneMKL GEMMs while allowing other kernels on the two tiles to overlap. The
broader `serialize_backward_custom_calls` option is retained as a slower
reference/fallback. Strict per-stage synchronization and full custom-call
serialization are diagnostic modes rather than normal training settings.

Hardware-dependent tests cannot be reproduced without a compatible Intel GPU,
Intel Extension for OpenXLA, and oneAPI toolchain. The archive therefore
records the last working implementation and its diagnostics, but does not
claim compatibility with newer JAX or Intel plugin releases.

## Software and ABI assumptions

The custom-call bridge uses the legacy XLA custom-call ABI expected by:

- JAX/JAXlib 0.5.0;
- Intel Extension for OpenXLA 0.7.0; and
- a oneAPI installation providing `icpx`, SYCL, Level Zero, and oneMKL.

The build uses `icpx -fsycl -qmkl=sequential -std=c++17` and produces
`libnca_sycl.so`. The code registers four API-version-zero targets for the
`SYCL` platform: single-step forward and backward, and rollout forward and
backward. `NCA/model/sycl/bridge.py` loads the library with `ctypes`, registers
the symbols with JAX, defines abstract evaluation and lowering rules, and
attaches custom VJPs.

Registration is lazy: ordinary model/config inspection does not load the
library. The first SYCL-backed model call verifies that JAX's active backend is
`sycl`, locates the shared library, and registers the custom calls.

## Code map

| Area | Purpose |
| --- | --- |
| `NCA/model/NCA_sycl.py` | Equinox-compatible `NCA_sycl` and `gNCA_sycl` wrappers; validation, masks, batching, and fused-rollout entry points. |
| `NCA/model/sycl/bridge.py` | JAX primitives, SYCL lowerings, custom-call metadata, batching behavior, and custom VJPs. |
| `NCA/model/sycl/reference.py` | Portable reference operations used for comparisons and tests. |
| `NCA/model/sycl/files/nca_sycl.cpp` | Native single-step forward implementation and exported entry point. |
| `NCA/model/sycl/files/nca_sycl_backward.cpp` | Native state and parameter VJP. |
| `NCA/model/sycl/files/nca_sycl_rollout.cpp` | Fused sequential updates, boundaries, trajectories, and optional regulariser sums. |
| `NCA/model/sycl/files/nca_sycl_rollout_backward.cpp` | Reverse pass through a fused rollout. |
| `NCA/model/sycl/files/nca_sycl_kernels.hpp` | Shared SYCL kernels, metadata, scratch, and execution helpers. |
| `NCA/model/sycl/files/nca_sycl_onemkl.hpp` | oneMKL GEMM helpers and synchronization/precision policy. |
| `NCA/trainer/backend/sycl/` | Native batching, fused trainer loop, scan helpers, `shard_map` compatibility, and two-tile execution. |
| `launch_slurm.sh` | DAWN environment setup, per-job library build, diagnostics, and experiment launch. |

The lower-level implementation notes and scratch-corruption procedure remain
in `NCA/model/sycl/files/README.md`.

## Model computation

`NCA_sycl.NCA` subclasses the standard fast JAX NCA so that its parameter
PyTree and saved Equinox leaves remain compatible. It replaces the main update
with a native call while retaining JAX perception methods for diagnostics and
gradient-based losses. `gNCA_sycl` changes the output width from `C` to `2C`
and fuses `value * sigmoid(gate)` into forward and reverse execution.

For each update, the native implementation:

1. computes the selected perception features;
2. applies the hidden pointwise layer and ReLU;
3. evaluates the output pointwise layer with oneMKL;
4. applies GLU gating for `gNCA_sycl`;
5. applies the stochastic fire-rate mask; and
6. adds the residual update to the state.

Supported perception feature names are `ID`, `DIFF`, `GRAD`, `AV`, and `LAP`.
Supported padding modes are `ZEROS`, `REFLECT`, `REPLICATE`, and `CIRCULAR`.
State, masks, weights, and scratch buffers are float32. The baseline wrapper
requires ReLU and limits the constructed perception feature count to 256.

Perception uses an 8-by-16 spatial tile with a halo in shared local memory.
Pointwise forward matrices and backward parameter/state products use oneMKL
GEMM. Common circular and zero-padded 3-by-3 DIFF transposes use an
atomic-free tiled gather; other supported linear stencils use a deterministic
gather, with a conservative atomic fallback for unsupported nonlinear or
padding cases.

## Batching and fused rollouts

An ordinary `vmap(model)` produces per-example parameter cotangents for JAX to
reduce. `model.batched_call()` instead passes an already batched
`[N, C, H, W]` state into the custom VJP, allowing the native backward call to
accumulate shared-parameter gradients across cells and examples directly.

The trainer retains an outer PyTree of independently shaped batch leaves. It
uses the native batched path for compatible rank-four leaves and falls back to
the ordinary trainer path for incompatible shapes. NODAL read-block
interventions have a dedicated batching path so that model parameters are not
vmapped.

For `trainer.backend.fused_steps: K`, `batched_rollout()` groups `K` sequential
updates behind one custom call. It returns the final state and all intermediate
states, preserving the semantics and gradients of per-step losses. Soft model
boundaries, hard masks, or no boundary are encoded in static metadata and
applied after every update. `K` must divide `run.t`; `K: 1` selects the
single-step path.

Intermediate-state and boundary regularisers can be accumulated inside the
rollout. Static flags omit their work when their configured coefficients are
zero. The forward call returns two differentiable FP32 sums, and the rollout
VJP injects their analytic derivatives while traversing the timesteps in
reverse. Reduction policy is either the historical atomic implementation or a
deterministic two-stage reduction using existing rollout scratch space.

## Two-tile execution

`trainer.sharding: 2` selects `SyclTwoTileExecution`. It requires exactly two
visible local SYCL devices and an even number of outer batch leaves. The first
and second contiguous halves are paired, and each PVC tile receives one half.
Parameters and optimiser state are replicated; state, targets, and PRNG keys
are sharded.

The loss is wrapped in `shard_map` before reverse-mode differentiation.
`lax.pmean` produces a replicated scalar loss and causes parameter gradients
to be reduced once before the shared optimiser update. Host callbacks operate
on `addressable_shards`. Corresponding leaves in the two halves must have
compatible shapes and boundary modes. Multi-target assignment gathers compact
cost rows rather than complete prediction images.

`pmean_loss` and `pmean_regularisers` may be disabled to localize failures, but
those settings are diagnostic and are not numerically equivalent to normal
two-tile training.

## Configuration and dispatch

SYCL training requires the model and trainer selections to agree:

```yaml
model:
  family: NCA_sycl       # or gNCA_sycl

trainer:
  sharding: 2            # null/1 for one tile, 2 for two tiles
  backend:
    type: sycl
    fused_steps: 4
    synchronize_custom_calls: false
    strict_stage_synchronization: false
    regulariser_reduction: atomic
    pmean_loss: true
    pmean_regularisers: true
    serialize_custom_calls: false
    serialize_onemkl: true
    serialize_backward_custom_calls: false
```

`Experiments/config.py` parses `backend.type: sycl` into
`SyclTrainerBackendConfig`. `Experiments/config_helpers.py` constructs the
SYCL model. `NCA/trainer/trainer.py` then dispatches to `SyclNcaTrainer` and
rejects a SYCL model paired with a non-SYCL trainer, or vice versa.

The archived end-to-end stability configuration is
`Experiments/micropatterns/conf/experiments/nca_intel_onemkl_training_stability.yaml`.
It compares single-tile execution, two-tile oneMKL serialization, and
two-tile full-backward serialization. The associated runtime benchmark uses
`nca_intel_onemkl_runtime_benchmark.yaml`.

## Building and launching

Inside the matching Intel environment:

```bash
source ~/dawn-jax/envs/jaxeqx-setup.sh
NCA/model/sycl/files/build_nca_sycl.sh /tmp/libnca_sycl.so
export NCA_SYCL_LIBRARY=/tmp/libnca_sycl.so
```

`NCA_SYCL_LIBRARY` may be omitted only when `libnca_sycl.so` has been built in
`NCA/model/sycl/files/`. The Slurm launcher instead builds into
`${SLURM_TMPDIR}/nca-sycl-${SLURM_JOB_ID}` for every job and exports the exact
path before importing JAX or the model.

The archived `launch_slurm.sh` is DAWN-specific. It sources
`~/dawn-jax/envs/jaxeqx-setup.sh`, assumes the `pvc9` partition and project
account encoded in its SBATCH directives, sanitizes inherited Python paths,
optionally prints the Intel runtime stack, builds the library, establishes
data/model/W&B paths, and executes one generated-manifest index. These paths
and scheduler values are historical and must be adapted before reuse.

## Precision and runtime controls

`system.precision` is mapped through JAX's default matmul precision:

| Setting | Native compute |
| --- | --- |
| `highest`, `float32`, `standard` | standard FP32 GEMM |
| `tensorfloat32`, `tf32` | float-to-TF32 XMX with FP32 accumulation |
| `bfloat16`, `bf16` | float-to-BF16 XMX with FP32 accumulation |
| `bf16x2`, `bf16x3` | extended BF16-family oneMKL modes |

`NCA_SYCL_XMX_MODE` overrides this mapping. The trainer configuration exposes
the principal synchronization and reduction controls; internally these map to
the following environment variables:

- `NCA_SYCL_SYNCHRONIZE_CUSTOM_CALLS`;
- `NCA_SYCL_STRICT_STAGE_SYNCHRONIZATION`;
- `NCA_SYCL_REGULARISER_REDUCTION` (`atomic` or `two_stage`);
- `NCA_SYCL_SERIALIZE_CUSTOM_CALLS`;
- `NCA_SYCL_SERIALIZE_ONEMKL`; and
- `NCA_SYCL_SERIALIZE_BACKWARD_CUSTOM_CALLS`.

`NCA_SYCL_DIAGNOSTICS=1` enables launcher-level package, device, driver, and
shared-library diagnostics. `NCA_SYCL_TRACE=1` enables verbose OpenCL/Level
Zero tracing and should be used only for short jobs. `PROFILE_GPU=1` enables
the tracing layer needed by the profiling path.

## Tests and failure-localization tools

Portable/reference coverage is in:

- `tests/integration/test_nca_sycl_reference.py`;
- `tests/integration/test_nca_sycl_trainer.py`; and
- the SYCL-related cases in `tests/unit/`.

Hardware coverage is in `tests/hardware/`:

- `intel_sycl_smoke.*` checks the compiler/runtime and a basic SYCL kernel;
- `jax_sycl_bridge_smoke.*` checks the Intel JAX custom-call bridge;
- `nca_sycl_smoke.*` compares native forward and gradients with the fast JAX
  model, including rollouts, boundaries, and regularisers;
- `nca_sycl_two_tile_smoke.*` exercises sharded training behavior; and
- `nca_sycl_rollout_scratch_guard.py` compares guarded scratch reuse with
  independent per-step allocations.

The repeated failure matrix and its interpretation are documented in
`tests/tools/NCA_SYCL_FAILURE_PROBES.md`. It compares asynchronous baseline,
strict stage waits, oneMKL-only serialization, complete backward
serialization, and BF16 compute. Submission and summary helpers in
`tests/tools/` record job IDs and hostnames and report timing and crash counts.
The scratch-guard submitter can additionally build with Intel device
AddressSanitizer by setting `NCA_SYCL_DEVICE_ASAN=1`.

These are cluster/hardware tests. They should not be invoked as ordinary local
unit tests or on machines lacking the archived Intel environment.

## Saved-model portability

The registry records the implementation family used for training, but the
parameter PyTrees were deliberately kept compatible with standard JAX models.
`NCA/registry.py` maps:

```text
NCA_sycl  -> NCA_fast
gNCA_sycl -> gNCA
```

Loading a bundle with `implementation="portable"` reconstructs the mapped JAX
model before deserializing its Equinox leaves. This is the intended way to use
models trained by this backend on CPU or CUDA after the SYCL training code is
retired. `implementation="recorded"` reconstructs the original SYCL family and
therefore still requires this archive and its matching Intel runtime.

## Known constraints

- The backend is bound to the legacy JAX/Intel custom-call ABI and has no
  guarantee of working with current JAX releases.
- Native inputs and parameters are float32; reduced precision affects oneMKL
  compute modes while accumulation remains FP32.
- The model supports only the enumerated perception features and padding
  modes, with a 256-feature cap in the baseline wrapper.
- Two-tile mode requires exactly two visible local SYCL devices, an even outer
  batch count, and compatible paired leaves.
- Fused step count must divide the configured rollout length.
- Some diagnostic synchronization policies substantially reduce throughput.
- The DAWN setup script, account, partition, data roots, and W&B scratch paths
  are site- and user-specific historical configuration.
- Intermittent two-tile backward failures motivated the serialization and
  scratch-guard work. The archived preferred mitigation is oneMKL-only
  serialization; the probe suite should be rerun before trusting a different
  driver, plugin, or hardware stack.

This branch should be treated as a reproducibility archive. Future active
development should preserve portable loading of its saved model bundles rather
than attempting to evolve the SYCL training implementation in place.
