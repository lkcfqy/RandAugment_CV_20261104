# Ctrl-A mechanism diagnostic — local archive

Outcome: **NO_GO_FOR_ONLINE_FORK**. Two seeds and all pre-registered frozen measurements completed. No online branches or Colab usage.

- [Chinese results and limits](RESULTS_ZH.md)
- [Machine-readable outcome, budget and remaining ceiling](execution-summary.json)
- [Protocol](protocol.json) · [pre-outcome lock](protocol-lock.json) · [sampling manifest](sampling-manifest.json)
- [Source provenance](source-manifest.json) · [CIFAR asset hashes](asset-manifest.json) · [interface patch manifest](patch-manifest.json)
- [Compatibility/layout recovery](compatibility-amendment-layout.json) · [spawn recovery](recovery-spawn.json) · [reader equivalence](reader-benchmark.json)
- [Checkpoint continuation semantics](CHECKPOINT_HANDOFF.md)
- [Aggregate metrics](aggregate-metrics.csv) · [all operation strata and CIs](operation-metrics.csv) · [official ROR and CIs](official-ror-metrics.csv)
- [Seed0 full results](seed0-frozen-results.json) · [seed1 full results](seed1-frozen-results.json)
- [19 pre-training tests](tests/all-pretrain-tests-recovery.txt) · [result invariance](result-integrity.json) · [historical preservation](preservation-after.json)
- [Paired rendering effects](figures/paired-rendering-gain.png) · [training health](figures/training-health.png) · [official ROR](figures/official-ror.png)

All raw predictions, per-image CE, model/BN and optimizer/RNG states, and full timing logs remain in local ignored raw/, weights/ and logs/. No test batch was extracted or loaded. No commit or upload was performed. Final SHA256 values are in deliverable-manifest.json.
