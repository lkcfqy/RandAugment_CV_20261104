# Common checkpoint handoff

This task authorizes common training and frozen diagnostics only. No online fork is started by any script here. Passing the resource gate is not authorization for a new stage.

Final checkpoints are `weights/seed0-epoch10.pt` and `weights/seed1-epoch10.pt`. Their checksums are in `seed0-latest-checkpoint.json`, `seed1-latest-checkpoint.json` and the final deliverable manifest. All earlier completed epoch checkpoints are retained. A failed pre-training spawn attempt's epoch0 checkpoint is separately named.

The state includes the exact official airbench94 model/BatchNorm tensors, optimizer slots, explicit scheduler state and horizon, Python/NumPy/Torch/MPS RNG states, phase histories, normalization moments, ξ/Γ/α, seed, exposure count, source commit and protocol hashes. Each final checkpoint has `actual_epoch=10`, `train_images=920000`, and `phase_update_done=True`.

The policy in the final checkpoint is the post-epoch10 controller update, ready for epoch11 on the frozen 20-actual-epoch horizon. Do not confuse it with `Gamma_used`/`alpha_used` in the epoch10 history row. Diagnostics use the post-update frozen policy. All raw augmented losses from training were observed under changing model weights; frozen CE matching is a separate measurement.

The pinned source recreates SGD every actual epoch. For any separately authorized continuation, recreate SGD at epoch11; do not carry epoch10 momentum into epoch11. The saved optimizer state is retained for audit, not a license to change this reset rule. The explicit learning-rate function is `0.5*0.05*(1+cos(pi*(epoch-1)/21))`. Future training loader seeds follow `seed*1000+epoch`, with worker Python/NumPy seeds from `torch.initial_seed() mod 2**32`.

The official channels-last network layout was converted to contiguous NCHW to resolve the installed MPS backward stride error. Structure and parameter values are unchanged; CPU/MPS train-gradient agreement is recorded in `benchmark.json`. Keep `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0`. CPU Pillow preprocessing is deliberate, with model operations on MPS.

Never regenerate the data split or challenge manifests when continuing this experiment. `data/*_indices.npy` and `data/*_challenges.npz` are fixed and hashed. The original official test member is never extracted or loaded. Raw prediction files are image-major, operation-minor, with 15 terminal operations in the official `OPS` order; original-image indices and all prefix/sign/quantile recipes are recoverable from those manifests.

For E/F, a prefix image is the 32x32 readout of its rendering state. A full sequence may retain the latent expanded canvas or fused matrix when the terminal operation extends a continuous affine block. Reported conditional ratios summarize the fixed, uniformly sampled prefix-context distribution for each terminal operation; they are not exhaustive estimates for every fixed prefix sequence.

The original overall planning ceiling was 200 original-46k exposures and 8,000,000 inference forwards. This common stage uses exactly40 exposures; any remaining allowance must be read from the final actual ledger and reviewed by the parent thread before further execution. The present eight-hour wall deadline is never reset by the recovery scripts.
