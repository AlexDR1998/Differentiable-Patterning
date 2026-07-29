# NCA SYCL failure probes

Submit 20 independently scheduled repetitions of each minimal probe:

```bash
tests/submit_nca_sycl_failure_probes.sh
```

The active isolation matrix submits four probes: the `shard_map + pmean`
reverse-mode control, fused-rollout forward only, fused-rollout reverse mode,
and fused-rollout reverse mode with process-wide custom-call serialization.
The rollout loss uses its final state and trajectory without regularisers or
collectives. Differentiated probes cover every array leaf in the model/input
pair, and each task blocks on both the result and gradient tree. Probe types
are interleaved across Slurm array indices, and each task records its allocated
hostname; scheduling is deliberately not pinned. After completion, summarize
crash rates and recurring nodes with:

```bash
python tests/summarize_nca_sycl_failure_probes.py nca-sycl-probe-logs
```

Set `NCA_SYCL_TRACE=1` only for a short diagnostic submission because Intel
runtime tracing is verbose. Normal training launchers accept
`NCA_SYCL_DIAGNOSTICS=1` and `NCA_SYCL_TRACE=1` as independent opt-in flags.

Training diagnostics can independently disable `trainer.sycl_pmean_loss` and
`trainer.sycl_pmean_regularisers`. Disabling the loss reduction also removes
its reverse-mode gradient collective, so the result is intentionally not a
numerically equivalent training run. `trainer.sycl_serialize_custom_calls`
keeps two-tile sharding but serializes native calls across tile threads.
