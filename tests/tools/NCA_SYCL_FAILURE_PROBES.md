# NCA SYCL failure probes

Submit 20 independently scheduled repetitions of each minimal probe:

```bash
tests/tools/submit_nca_sycl_failure_probes.sh
```

The active backward-localization matrix has five fused-rollout reverse-mode
variants:

1. `baseline`: standard compute with asynchronous stages.
2. `strict_stages`: wait after every native backward stage.
3. `serialize_onemkl`: serialize and wait for only oneMKL GEMMs.
4. `serialize_backward`: serialize complete backward custom calls across tiles.
5. `bf16_compute`: accelerated BF16 compute with otherwise baseline behavior.

All variants use the same two-step rollout loss over final state and trajectory,
without regularisers or collectives. Every task reports whether the PJRT queue
is in order. Probes are interleaved across Slurm array indices, and allocated
hostnames are recorded without pinning scheduling. The default is 20 repeats
per variant (100 tasks); use `NCA_SYCL_PROBE_REPEATS=100` for a decisive final
comparison. After completion, summarize only the newly submitted job with:

```bash
python tests/tools/summarize_nca_sycl_failure_probes.py \
    nca-sycl-probe-logs --job-id JOB_ID
```

Set `NCA_SYCL_TRACE=1` only for a short diagnostic submission because Intel
runtime tracing is verbose. Normal training launchers accept
`NCA_SYCL_DIAGNOSTICS=1` and `NCA_SYCL_TRACE=1` as independent opt-in flags.

Training diagnostics can independently disable `trainer.backend.pmean_loss` and
`trainer.backend.pmean_regularisers`. Disabling the loss reduction also removes
its reverse-mode gradient collective, so the result is intentionally not a
numerically equivalent training run. `trainer.backend.serialize_custom_calls`
keeps two-tile sharding but serializes native calls across tile threads.

## Reliability and speed validation

Run 100 repetitions each of the failing baseline and the narrow oneMKL
mitigation with:

```bash
tests/tools/submit_nca_sycl_onemkl_benchmark.sh
```

The summarizer reports median/minimum/maximum successful probe time as well as
crash counts. For end-to-end measurements, generate manifests from:

- `Experiments/micropatterns/conf/experiments/nca_intel_onemkl_runtime_benchmark.yaml`
- `Experiments/micropatterns/conf/experiments/nca_intel_onemkl_training_stability.yaml`

The runtime sweep contains 18 jobs: three repeats of three execution policies
at standard and BF16 precision. The training sweep contains 15 BF16 jobs: five
repeats of the same policies for 1,000 real optimiser iterations. The policies
are single-tile baseline, two-tile oneMKL-only serialization, and two-tile
backward-wide serialization.
