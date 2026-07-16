# Baseline NCA SYCL kernel

Build this shared library inside the same oneAPI environment used by the Intel
JAX plugin:

```bash
source ~/dawn-jax/envs/jaxeqx-setup.sh
NCA/model/sycl/files/build_nca_sycl.sh /tmp/libnca_sycl.so
export NCA_SYCL_LIBRARY=/tmp/libnca_sycl.so
```

`nca_sycl.cpp` currently implements the float32 baseline NCA forward update
with ID, gradient norm, gradient, average, and Laplacian perception features;
ReLU; all existing padding modes; the fire-rate mask; and the residual update.

The implementation uses the legacy custom-call ABI required by JAX 0.5.0 and
Intel Extension for OpenXLA 0.7.0. It is a correctness-oriented implementation,
not yet an XMX-tiled one. `nca_sycl_backward.cpp` implements the custom VJP for
the state and trainable pointwise-layer parameters using native SYCL kernels.
Vmapped execution uses one batched backward custom call and emits per-example
parameter cotangents for JAX to reduce correctly. The fixed perception kernels
and random update mask are non-trainable.
