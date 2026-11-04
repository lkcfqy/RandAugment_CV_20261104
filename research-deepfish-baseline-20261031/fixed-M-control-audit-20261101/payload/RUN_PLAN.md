Fixed M candidate: PREPARED, NO EXECUTION GO

This package contains 69 already approved public DeepFish images, support/dev annotations, one fixed candidate configuration, historical A control hashes, official dependency/asset pins and phase scripts. It does not contain a runtime context granting execution, any model weight, official test data, a cloud allocator or an automatic training driver. The host must reuse the previously verified Colab CLI persistence and external release watchdog workflow. Preparation does not authorize allocation or upload.

Scientific scope: one M model, seed20261031, original GroundingDINO Swin-T, same official support5 and dev64, U200=40 epochs, batch1/workers0, fixed final checkpoint. A is the already completed historical control. M retains Color(prob=1,level=0,factor=1) and adds the predeclared Mosaic(.6)/HSV(5,30,30)/MixUp(.3) pipeline. Cache state starts empty for every independently initialized phase and persists across epochs. This is an exploratory whole-strategy comparison at fixed anchor/update budget, not source-exposure matching or an ETS paper reproduction.

Before execution, the parent must record the final resource/margin policy and explicitly grant this candidate GO. User total CU cap remains2.5; cumulative history1.8088330815795528CU and5443.899788473129GPU seconds must carry forward. The current scripts retain the parent's existing7200-second GPU ceiling. Disk6GiB and GPU2h are parent limits, not user-set limits; this package does not revise either. The example context is deliberately unauthorized.

Execution sequence after GO:

1. Verify package manifest and the retained local historical A checkpoint hash. Before expensive environment setup, use the existing host CLI to verify current allocation/rate, external watchdog, a small remote-to-local atomic download/hash receipt, bandwidth and the FULL remaining cost. Do not rely only on true-valued context flags. Stop and release if the complete run cannot fit.
2. Reuse `bootstrap_environment.bootstrap(...)` with requirements_runtime_lock.txt (the complete successful r5 distribution versions) and its pinned official Python3.10.19/core wheels. The phase worker verifies every locked installed version before model imports; historical_install_artifacts.json records54 original public downloads and checks the53 archived SHA256 values. The historical Jinja2 record lacks an archive digest; its installed version is locked, but historical byte identity is unknown. Fetch weights directly from the official URLs in asset_pins.json, verify every hash, then enable the existing offline and weights-only environment. No new OAuth, paid resource purchase, third-party fork, unsafe pickle fallback, or alternate model weight is permitted.
3. Use the returned verified interpreter path; do not assume /usr/bin/python3.10 exists. With a newly verified runtime context, run:

   `<verified_python> candidate_worker.py --context runtime_context.json --phase preflight_M`

   This phase performs15 updates (5 warmup,10 timed) and5 support-image forwards without saving predictions or computing dev metrics. Before any update it must match historical A's complete initial-state hash, trainable-name hash, empty optimizer and effective optimizer values, then first-update Torch/CUDA RNG. A mismatch is a STOP, not permission to enlarge a key allowlist or alter initialization.
4. Discard the complete preflight model/optimizer/RNG/cache state. Recompute full remaining cost from current elapsed usage, bandwidth and measured M throughput under the already chosen margin definition. Freeze U200 or stop; no shorter-U fallback and no dev/loss-driven selection. A matching `U_FREEZE.json` is required. Run:

   `<verified_python> candidate_worker.py --context runtime_context.json --phase formal_M`

   Independent initialization and the same control gates apply again. Save complete tensor-only final checkpoint atomically. Source/GT/NumPy trajectories may differ from A because that is part of the declared strategy; the anchor sampler schedule and all other controls remain fixed.
5. Return the full final checkpoint and logs to the Mac, verify file hash and safe tensor-only state, and recheck the local historical A checkpoint identity. Only then create CANDIDATE_CHECK_OK.json with `candidate_checkpoint_sha256`, `reference_A_final_checkpoint_sha256`, `freeze_sha256`, and `updates_per_arm=200`. Run:

   `<verified_python> candidate_worker.py --context runtime_context.json --phase evaluate_M`

   Evaluate all64 dev images once with the original metric mapping. No test data or save_best. Reuse historical A as an explicitly marked historical control only if pinned training/evaluation controls remain intact; any material runtime/control change requires reconsidering whether A must be rerun, with separate complete cost accounting.
6. Persist complete predictions, metrics, initialization/optimizer/source traces and phase receipts locally with hashes. Release the owned allocation and independently confirm assignments/sessions/rate are zero. No additional seed, arm, color-strength search, altered labels or annotation filtering is included.

Pending: actual candidate initialization/control gates and CUDA throughput have NOT been run; live allocation/rate/bandwidth/watchdog checks and the final25% margin definition remain unresolved. This is a concrete phase plan and prepared package, not a completed or authorized GPU experiment.
