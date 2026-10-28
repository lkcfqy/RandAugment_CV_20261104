# Experiment integrity audit — fresh MPS B/Z/R

Date: 2026-10-24. Completed: 2026-10-24T13:10:33.576583+00:00.

**Overall verdict: WARN for claim scope and accounting labels; all deterministic integrity checks PASS. No blocking execution or metric defect found.** This supports the bounded, one-seed development result below. It does not establish equivalence, statistical significance, independent-seed replication, or a general Posterize effect.

Auditor: fresh native Codex / GPT-6 reviewer `/root/mps_run_integrity_review`. Exact model backend/effort are not independently observable. Review independence is **same-family** and semantic acceptance is **provisional**. No GPT6 Pro or additional reviewer was consulted. The protocol records requested configuration `gpt-6-astra / max / Standard` with `backend_configuration_verified=false`; this is requested-only metadata, not an attestation of the reviewer runtime. Deterministic checks are independently reproduced as described below.

Experiment: `/Users/lkc/Downloads/RandAugment_CV_20261104/augmentation-research-20261022/posterize-three-arm-mps-20261024`.

## Verified outcome

| Arm | Correct / development examples | Accuracy | NLL recomputed from saved logits |
|---|---:|---:|---:|
| B | 552 / 589 | 93.71816638% | 0.232625735584511 |
| Z | 552 / 589 | 93.71816638% | 0.232593774347967 |
| R | 550 / 589 | 93.37860781% | 0.233622471235803 |

The frozen primary contrasts are dZ=0 and dR=−2 correct examples (0 and −0.33955857 percentage points). Classification is **unresolved_below_direction_threshold**: neither the reversal rule nor the same-direction rule is met. The ±3-example gate is descriptive, not a significance, equivalence, or power test. NLL does not change the classification.

B and Z have identical per-example correctness vectors. Relative to B or Z, R corrects 0 errors and introduces 2. Joint B/Z/R correctness patterns are 000:37, 110:2, 111:550; all other patterns are zero. Identical correctness does not assert identical logits or identical predicted labels on the 37 shared errors. The three worker-reported final trainable-parameter digests are distinct.

## A. Ground-truth provenance — PASS within the inspected pipeline

Labels are loaded from the hash-locked `pilot-20261022/fit-manifest.json`, not generated from model outputs. The pipeline asserts 591 training and 589 development examples, all 37 classes in each partition, unique source IDs, and disjoint duplicate clusters (`worker-v2.py:84–98`). Independent metadata verification confirmed 1,180 unique source IDs and duplicate clusters, exact partition counts, and no source overlap. Saved prediction labels were checked against that manifest for all 589 records per arm.

The reviewer did not reopen original annotation archives or inspect raw images. Thus this is verification of the frozen manifest's role and consistency, not a fresh audit of the entire upstream annotation lineage. Evaluation type: **real_gt**.

## B. Loss and score normalization — PASS

Formal training has 36 full batches of 16 plus one tail of 15 per epoch. Cross entropy is reduced by sum then divided by the actual batch size, with gradients cleared once and one optimizer step per batch (`worker-v2.py:197–224`). The tail is divided by 15. The discarded readiness benchmark explicitly compares this expression to mean reduction (`worker-v2.py:155–168`). All saved batch sizes, weighted epoch losses, and 1,110-step learning-rate schedules per arm matched independent checks.

Accuracy is raw correct count divided by the fixed 589 examples. NLL is conventional log-softmax from the stored 37-class logits (`worker-v2.py:235–241`). The reviewer independently recomputed argmax, correctness, log-sum-exp NLL, mean NLL, and accuracy from all 1,767 saved records. All matched. No metric is normalized by a model's own maximum performance.

## C. Artifact existence, integrity, and budget — PASS with an accounting qualifier

All three arm summaries, all 90 epoch receipts, all 3,330 formal step records, all 12 scheduled checkpoints (epochs 5, 10, 20, 30), and final joint/result/exit files exist and agree. Actual checkpoint bytes matched their recorded SHA-256 hashes, ZIP CRC checks passed, and no tensor was deserialized. Each epoch's saved sample-order hash matched its frozen order. The final verifier recorded 141 input/artifact hashes, including source locks, base checkpoint, prediction files, and final outputs.

The original phase began **12:43:56.683066 UTC**, with unchanged deadline **14:13:56.683066 UTC**. Readiness passed at **12:50:51.819661 UTC** and the formal-start receipt records **12:50:51.922713 UTC**. The process exited normally at **2026-10-24T13:07:15.948930+00:00**, after **1399.265873 seconds** cumulatively (23m19.266s), leaving 4000.734127 seconds before the cap. No deadline termination or kill was sent.

The v1 startup failure remains preserved. It failed at the first budget guard before importing torch/model creation or image loading; every counter is zero and its checkpoint is null (`attempt-001/failure.json:6–26`). Its incorrect stage elapsed reflects the cross-interpreter monotonic clock origin. The v2 diff only adds conversion from the immutable UTC start into the current interpreter's monotonic origin and changes the code-lock filename (`worker-v2.py:9–11,67`). The budget file is unchanged. Supervisor v2 retains the original UTC deadline and has no retry loop (`supervisor-v2.py:14–29`). This was one formal training run, preceded by a zero-training startup failure and a discarded benchmark.

**Accounting qualifier:** the protocol's 3,330-update/54,957-forward maxima and final formal counters exclude the separately recorded benchmark. Report both:

| Work | Optimizer updates | Model-forward image presentations |
|---|---:|---:|
| Formal training + final evaluation | 3,330 | 54,957 (53,190 train + 1,767 dev) |
| Discarded readiness benchmark | 9 | 141 training |
| Combined | 3,339 | 55,098 |

There are 3,339 backward calls including the benchmark, plus 48 image reads for pixel parity with no model forward. Total image reads are 55,146. The final `train_image_reads` counter includes the 141 benchmark reads, while `training_forward_images` and `optimizer_updates` are formal-only. These counter scopes must not be presented as interchangeable. The small synthetic kernel witness uses a matrix multiply, with no images or optimizer update.

## D. Live computation, pairing, and evaluation separation — PASS

The relevant functions are invoked: formal model construction and optimizer at `worker-v2.py:183`, schedule/loss at `:204–209`, clean evaluation at `:227–241`, and final contrasts/gate at `:248–253`. There is no unused primary metric standing in for an executed evaluation. Runtime events contain exactly 3,330 update completions, 90 epoch completions, 111 development batches, three arm completions, and one final completion. Every arm begins development evaluation only after its 1,110 updates.

Initialization uses separate model, head, and adapter-slot seeds (`common.py:64–81`); each formal arm is rebuilt with an empty optimizer, and initial/frozen digests match across arms and the discarded benchmark (`worker-v2.py:183–191`). The benchmark optimizer/model are deleted and never reused for formal training (`:175,183`). All frozen-before/after digest receipts agree.

The reviewer independently regenerated every CPU RNG draw from manifest dimensions, using local generators rather than executing the worker: all 30 orders, 17,730 weak crops/flips, 35,460 B operator slots, and 2,562 R replacement slots matched exactly. All 17,730 trace rows preserve non-Posterize slots across B/Z/R; Z replaces each target slot with Identity and R replaces it from the 13 survivors. Trace SHA-256 is `858035231e62a4ebd85af78581fae0fbb7e8a8f671bffad46d0e014c2c3b4bed`. Code references: `worker-v2.py:106–128`; `common.py:29–46`.

Development evaluation is clean and runs only at the frozen epoch 30 under `eval()` and inference mode. No earlier checkpoint, NLL, or interim ranking is used to choose an endpoint. Incomplete execution is explicitly labeled incomplete rather than a scientific negative (`worker-v2.py:254–259`). This reviewer did not read development predictions or interim summaries before `result.json` existed.

## E. Scope and claim impact — WARN

The supported claim is: **one local MPS development experiment completed the frozen 30-epoch B/Z/R comparison, and its registered directional result is unresolved (dZ=0, dR=−2).** The NLL values remain secondary.

Seed 20261024 is the same as the historical CPU screen; this is not independent-seed replication. Device, microbatching, and pacing differ, so old/new differences cannot be attributed solely to epoch count. Software-input pairing is verified, but numerical repeatability across additional MPS executions was not tested. Repeated use of the development set and one seed limit generalization. R and Z differ in Identity probability, so their difference is not an unrestricted intrinsic-effect estimate for Posterize.

The stored flag `original_30_epoch_claim_tested=true` is supportable only as **the planned local 30-epoch endpoint was executed**. It must not be expanded into a replication of the original paper's operator-contribution finding or a causal isolation of training duration.

No corrective experiment is needed or authorized by this audit. Preserve the fixed classification and the accounting/scope qualifiers in delivery. No blocking action remains.

## Reproducibility and audit limits

- `verify_results.py`: standard-library verifier of saved logits, step/epoch records, counter events, deadlines, source hashes, and checkpoint bytes. It refuses to read predictions without all-arm completion and normal-exit receipts.
- `verification-evidence.json`: detailed successful checks and the 141 hashes.
- `verify_random_streams.py` and `random-stream-evidence.json`: exact CPU RNG reconstruction from metadata, with zero image/model reads.
- `IMMEDIATE_CODE_REVIEW.md`: findings recorded before all arms completed.

No project files were modified; no training, model inference, tensor deserialization, image access, package installation, cloud action, extra seed, or external reviewer call was performed. Parameter-content and rendered-pixel hashes remain worker-produced receipts: their consistency was checked, but their underlying tensors/pixels were not independently reconstructed. The 48 saved baseline pixel-parity receipts were read but not rerendered.
