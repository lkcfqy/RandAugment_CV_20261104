# DeepFish fixed 5-shot exploratory A/B package

Status: LOCAL PREPARATION ONLY. Upload, GPU assignment, dependency installation and model execution have not been authorized or performed for this package.

This package contains 69 public JPEGs, the original five-image support records and a fixed 64-image / 187-box dev subset, two complete configurations and a small official-MMDetection runner driver. No official test images/labels, model weights, full repositories or previous experiment data are included. The dev set is prefix-group disjoint, not a new-scene benchmark: habitat code 7463 is shared with support. Its 187 boxes are additional target-label budget.

The one episode uses seed 20261031. Both arms call the official Color(prob=1, level=0): A min_mag=max_mag=1.0, B=0.8. Input is explicitly cv2 uint8 BGR. The same official flip/resize/crop geometry follows. JSON tuple fields required by the official loaders/transforms are restored before Runner construction. Recording wrappers retain the original transform objects and only record source IDs, step outputs, homography/scale/flip, GT, and RNG-state hashes before transforms and after every update, including the final update. Any paired drift stops the comparison. An actual installed-Color synthetic gate runs before model construction.

Each preflight has 5 warmup + 10 timed optimizer updates (15 total, three exposures per support image); it also times prediction on the five support images without metrics. Its entire process/model/optimizer state is discarded. Formal runs start fresh from the original checkpoint and empty optimizers; U=5E gives exactly E exposures of each support image in each arm. Tensor initialization, trainable names, exposure counts and paired traces are checked.

U is chosen once from {25,50,100,200} using timing only, before any dev metrics. Remaining cost is conservatively estimated as:

`4 * max_init_seconds + 1.5 * U * (p90_step_A + p90_step_B) + 1.5 * 128 * p90_support_prediction_seconds + 4 * state_bytes / measured_write_bytes_per_second + 120`.

The largest candidate fitting the work deadline is frozen. The four remaining initializations cover two formal workers and two separate fixed-final evaluation workers. If none fits, the status is RESOURCE_UNDETERMINED; the dataset is not reduced. Loss values cannot enter the selection function. General checkpoint/save_best hooks are disabled. Only the final tensor state dict at U is saved. Both formal traces must agree before either dev evaluator can run.

Dev annotations 4195 and 4660 retain their original xywh and area, including the 0.5-pixel overhang. Evaluation uses the official order LoadImage → FixScaleResize → LoadAnnotations → PackDetInputs, so the original labels are loaded after image resize. No dev-label clipping or new boxes are introduced.

The candidate is Linux x86_64 / Python 3.10, torch 2.0.0+cu117, torchvision 0.15.1+cu117, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0. Core packages are obtained only through PyPI, official PyTorch and OpenMMLab distribution sources. The complete resolver/transitive environment and CUDA ABI have NOT been tested. A Python 3.10 interpreter must already exist; there is no interpreter installation or alternate-environment fallback. The runtime needs 12 GiB free local disk.

Weights are downloaded only in the future authorized Colab runtime from asset_pins.json and verified by size/SHA256. BERT uses safetensors and fixed local files. All PyTorch loads use weights_only=True directly or TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 through MMEngine. No unsafe fallback, dynamic pickle allowlist or trust_remote_code is used. Installed key MMDetection source files are checked against implementation_pins.json.

Before any future run, an external controller must obtain this experiment's explicit approval, verify the actual CU rate upper bound, record allocation start, arm an independent owned-session release watchdog and confirm an already-mounted persistent destination. It must reconcile actual usage and endpoint release afterward. The independent in-VM watchdog validates the same explicit authorization context and bounded deadline before importing Colab management. It and native unassign are additional fallbacks; their return never claims confirmed release. Content-addressed JSON stdout receipts and atomic fsync/readback preserve progress evidence. Checkpoints remain in the verified persistent destination.

Both ceilings are charged from allocation start, including preparation, preflight, training, inference and finalization: at most 2 CU and 7200 allocated-GPU seconds. The hard deadline is allocation_start + min(7200, 7200 / verified_CU_per_hour); work stops 180 seconds earlier. Unknown rate, persistence, release coverage or compatible environment blocks execution. These ceilings are a proposed future scope, not an existing permission or runtime estimate.

The external runtime context is NOT shipped with approval. `run_context.example.json` is disabled and cannot run. After approval, resource verification and package SHA verification, the unique pending command in the extracted package directory is:

```sh
python -B driver.py --run-context /content/deepfish-run-context.json
```

Do not execute it now. The live context must include this reviewed MANIFEST.json digest and refer only to the owned current allocation. No credentials belong in that context or this package. An existing artifact directory is not overwritten, and this package is not automatically resumed after failure.

MANIFEST.json lists every payload file except itself and SHA256SUMS. SHA256SUMS covers those files plus MANIFEST.json; the delivery receipt separately hashes the archive and SHA256SUMS. This avoids a self-referential checksum. See ATTRIBUTION.md for data provenance limits.
