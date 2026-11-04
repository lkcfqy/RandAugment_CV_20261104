# Related second-encoder control: DeiT-Tiny

Authorized once on 2026-11-02. This reuses every identity, episode, nested label budget, query image, four-view RNG key, policy, ridge lambda/objective and tie rule of augmentation-selection-pilot-20261102. It is correlated with the DINOv2 run and is not fresh-data confirmation. Do not add a third run or modify presentation material.

Clock: 04:58:47 UTC start; stop by 05:58:47 UTC (one hour), also before the user's 10:00 UTC / 19:00 Korean time cutoff. CPU4/interop1, max RSS12 GiB, new artifacts <=100 MiB, no install/data or weight download/upload/Colab/CU. Preserve prior results.

Model: existing Meta DeiT-Tiny original .pth SHA256 a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97, loaded weights_only=True with strict complete state into existing official timm1.0.22 deit_tiny_patch16_224.fb_in1k (1000-class pretrained head retained but unused). Frozen final normalized CLS192 is unit-L2 normalized per view. No unsafe fallback.

Preprocessing declared before utility: pinned timm model config crop_pct=.9, bicubic center crop224, hence floor(224/.9)=248 shorter-edge resize, versus DINOv2's256. Means/stds are identical. Weak RRC and RA operate identically on original64px images and are unchanged; all872 support originals x8 weak/RA tensor hashes will be compared against the exact frozen first-run transformation code. Standard validation/query inputs therefore differ by the documented model recipe; this is not a pure-weight-only causal comparison.

Reuse the small runner with only model loading/features, dimension, clean resize and budgets changed. No re-selection of data. Verify original manifest/core hashes, safe weight hash, native eval transform equivalence, view equivalence, ridge tests. Freeze protocol/code. Time support-only forwards and synthetic ridge solves without query utility; the only resource option is the full1000-query design (no shrink).

Run all360 heads/24 selections; seal before query feature extraction/evaluation. Save all360 heads, OOF and query predictions and all policies. Recompute from persisted results and verify score reprojection/normal equations. Compare both encoders' complete paired distributions, frequencies, regret and ties, with no arbitrary pp-success threshold. Archive new results and stop; assess whether a cross-dataset check is worthwhile without starting it.
