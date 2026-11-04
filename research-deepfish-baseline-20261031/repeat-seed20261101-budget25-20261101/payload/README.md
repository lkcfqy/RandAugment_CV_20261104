Authorized post-result exploratory repeat: training seed20261101 only; one additional A/B pair; formal U200 fixed. Original seed20261031 results already seen. Data-selection seed and all support/dev records remain unchanged. No seed20261102. Full final checkpoints persist before dev evaluation.

Authorized resumption20261101: cumulative cap raised from2CU to2.5CU only. Carry1.2577442410187722CU/3718.752113674164GPU seconds; cumulative7200GPU seconds remains fixed. Earlier seed20261101 attempt had formal0/dev0. Resume from original weights and empty optimizers, never preflight state. Full checkpoints retained; no tensor-delta encoding. Current-allocation8MiB return bandwidth and live rate feed the same fixed-U component estimator before payload upload or dependency/model preparation. Preparation allowance554.525631070137seconds includes25percent over the previous measured cold preparation; elapsed preparation is subtracted once. Four initializations,1.5p90 update/inference factors,120seconds overhead and180seconds release reserve remain; checkpoint floor692175962bytes covers full files.

The historical r5 preparation notes below remain provenance; this repeat cannot select a smaller U.

以下是 r5 的历史说明，仅作溯源；其中 seed20261031 和 U 候选选择不适用于本次 seed20261101、固定 U200 的探索性重复。

# DeepFish fixed 5-shot exploratory A/B package

Status: explicitly authorized execution; infrastructure-only recovery v1. The scientific data/configuration files remain byte-identical to the reviewed initial package. Prior consumption is carried forward; model efficacy remains unseen before the timing-only U freeze.

This package contains 69 public JPEGs, the original five-image support records and a fixed 64-image / 187-box dev subset, two complete configurations and a small official-MMDetection runner driver. No official test images/labels, model weights, full repositories or previous experiment data are included. The dev set is prefix-group disjoint, not a new-scene benchmark: habitat code 7463 is shared with support. Its 187 boxes are additional target-label budget.

The one episode uses seed 20261031. Both arms call the official Color(prob=1, level=0): A min_mag=max_mag=1.0, B=0.8. Input is explicitly cv2 uint8 BGR. The same official flip/resize/crop geometry follows. JSON tuple fields required by the official loaders/transforms are restored before Runner construction. Recording wrappers retain the original transform objects and only record source IDs, step outputs, homography/scale/flip, GT, and RNG-state hashes before transforms and after every update, including the final update. Any paired drift stops the comparison. An actual installed-Color synthetic gate runs before model construction.

Each preflight has 5 warmup + 10 timed optimizer updates (15 total, three exposures per support image); it also times prediction on the five support images without metrics. Its entire process/model/optimizer state is discarded. Formal runs start fresh from the original checkpoint and empty optimizers; U=5E gives exactly E exposures of each support image in each arm. Tensor initialization, trainable names, exposure counts and paired traces are checked.

U is chosen once from {25,50,100,200} using timing only, before any dev metrics. Remaining cost is conservatively estimated as:

`4 * max_init_seconds + 1.5 * U * (p90_step_A + p90_step_B) + 1.5 * 128 * p90_support_prediction_seconds + 4 * state_bytes / measured_write_bytes_per_second + 120`.

The largest candidate fitting the work deadline is frozen. The four remaining initializations cover two formal workers and two separate fixed-final evaluation workers. If none fits, the status is RESOURCE_UNDETERMINED; the dataset is not reduced. Loss values cannot enter the selection function. General checkpoint/save_best hooks are disabled. Only the final tensor state dict at U is saved. Both formal traces must agree before either dev evaluator can run.

Dev annotations 4195 and 4660 retain their original xywh and area, including the 0.5-pixel overhang. Evaluation uses the official order LoadImage → FixScaleResize → LoadAnnotations → PackDetInputs, so the original labels are loaded after image resize. No dev-label clipping or new boxes are introduced.

The candidate is Linux x86_64 / Python 3.10, torch 2.1.0+cu118, torchvision 0.16.0+cu118, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0. Core packages are obtained only through PyPI, official PyTorch and OpenMMLab distribution sources. The CPU preparation validates dependency resolution, pip check, actual official Color behavior and compiled CPU operators; CUDA training remains gated by the authorized preflight. The isolated CPython 3.10.19 interpreter is installed from the SHA-pinned Astral python-build-standalone 20251014 release using a pinned uv 0.9.5 binary from PyPI. Sources and hashes are in python_bootstrap_pins.json. The notebook interpreter is not modified. Headless workers explicitly use MPLBACKEND=Agg. The runtime needs 12 GiB free local disk.

Weights are downloaded only in the future authorized Colab runtime from asset_pins.json and verified by size/SHA256. BERT uses safetensors and fixed local files. All PyTorch loads use weights_only=True directly or TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 through MMEngine. No unsafe fallback, dynamic pickle allowlist or trust_remote_code is used. Installed key MMDetection source files are checked against implementation_pins.json.

Before any future run, an external controller must obtain this experiment's explicit approval, verify the actual CU rate upper bound, record allocation start, arm an independent owned-session release watchdog and verify a public marker round trip through the official Colab CLI into atomically written, hash-checked local storage. It must reconcile actual usage and endpoint release afterward. The independent in-VM watchdog validates the same explicit authorization context and bounded deadline before importing Colab management. It and native unassign are additional fallbacks; their return never claims confirmed release. Content-addressed JSON stdout receipts and atomic fsync/readback preserve progress evidence. Remote /content is ephemeral. Every completed phase emits a complete hash/chunk index and waits for a local acknowledgment after all files are downloaded, atomically persisted and rehashed. Final checkpoints, logs and raw dev predictions are collected before release. Timeout or a missing receipt is reported as incomplete; it never makes ephemeral storage durable.

Both ceilings are charged from allocation start, including preparation, preflight, training, inference and finalization: at most 2.5 CU and 7200 allocated-GPU seconds. The hard deadline is allocation_start + min(7200 - prior_GPU_seconds, (2.5 - prior_CU) * 3600 / verified_CU_per_hour); work stops 180 seconds earlier. Unknown rate, persistence, release coverage or compatible environment blocks execution. These are the explicitly approved cumulative ceilings. Prior failed checks and CPU setup are included in the CU ledger; the earlier GPU allocation is included in the GPU-time ledger.

The external runtime context is NOT shipped with approval. `run_context.example.json` is disabled and cannot run. After approval, resource verification and package SHA verification, the unique pending command in the extracted package directory is:

```sh
python -B driver.py --run-context /content/deepfish-run-context.json
```

Only the authorized controller may execute it after the real current-allocation rate, watchdog and local persistence probe have been recorded. The live context must include this reviewed MANIFEST.json digest and refer only to the owned current allocation. No credentials belong in that context or this package. An existing artifact directory is not overwritten, and this package is not automatically resumed after failure.

MANIFEST.json lists every payload file except itself and SHA256SUMS. SHA256SUMS covers those files plus MANIFEST.json; the delivery receipt separately hashes the archive and SHA256SUMS. This avoids a self-referential checksum. See ATTRIBUTION.md for data provenance limits.

Infrastructure revision v4: PyTorch 2.0 deterministic CUDA scalar indexed assignment failed before the first optimizer update. Official 2.1.0 fixes the relevant slice-size expansion and has a matching OpenMMLab MMCV 2.1.0 cu118 binary. Deterministic algorithms remain enabled. A synthetic scalar-index/NMS gate runs before weight downloads; scientific configurations and timing-only U rule remain unchanged.
