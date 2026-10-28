# MPS agent-follows-doc witness receipt

Scope: environment contract check only. The documented command was executed exactly once, without modification, retry, installation, or repair.

Working directory: `/Users/lkc/Documents/Codex/2026-10-24/task-9`

Execution environment: default `exec_command` sandbox. No elevated execution requested.

Command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/encoder-readiness-20261021/runtime/.venv/bin/python' -B '/Users/lkc/Documents/Codex/2026-10-24/task-9/bzr_mps_prepared/kernel_witness.py'
```

Exit code: **1**. Tool-reported command duration: **0.95116159 seconds**.

Output:

```text
Traceback (most recent call last):
  File "/Users/lkc/Documents/Codex/2026-10-24/task-9/bzr_mps_prepared/kernel_witness.py", line 4, in <module>
    assert torch.backends.mps.is_available()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
AssertionError
```

Expected sentinel: `BZR_MPS_KERNEL_WITNESS`. It was not emitted.

Doc/reality divergence: the spec selects `mps`, but `torch.backends.mps.is_available()` returned false in this invocation. The readiness witness therefore failed before a kernel-dispatch witness could be produced. This does not establish the cause or whether MPS would be available in a different execution context. The ledger itself correctly leaves readiness conditional on validation; readiness must remain unconfirmed.

Status: completed invocation, failed readiness check. This was a process failure, not an approval denial or an uncertain execution outcome. No training or further experiment was started. No bypass or retry was attempted. The tool receipt did not verify a model identity or service tier.

## Second witness after execution-context documentation was revised

The parent supplied material new evidence of an identical successful command under the approved native execution context and requested an independent check of the revised documents. The revised spec and ledger were read before execution. They explicitly require `exec_command` with `sandbox_permissions=require_escalated` for MPS visibility. The original default-sandbox failure remains recorded above.

The exact same documented command was executed once in that revised context, without code edits, installation, or training.

Exit code: **0**. Tool-reported command duration: **0.464612773 seconds**.

Output:

```text
BZR_MPS_KERNEL_WITNESS {"device": "mps", "shape": [16, 16], "torch": "2.14.1", "finite": true}
```

Expected sentinel was emitted. The revised documented invocation passes this independent agent-follows-doc witness: MPS output was a finite 16 by 16 tensor under Torch 2.14.1. No doc/reality divergence was observed for this bounded invocation in the specified context. This validates the readiness witness only, not a worker or training run.

Status: completed and successful. Elevated execution was not denied; there is no uncertain execution outcome. No further invocation was attempted. The tool receipt did not verify a model identity or service tier.
