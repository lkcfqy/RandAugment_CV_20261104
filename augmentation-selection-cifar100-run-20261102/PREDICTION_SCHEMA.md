# Saved prediction and identity schema

The immutable `../augmentation-selection-cifar100-preflight-20261102/IDENTITIES.json` supplies ordered support_union (8,751) and query (9,990) records with original IDs, source indices, RGB SHA256 and fine labels. The original images and model weights are referenced, not copied. Protocol SHA256: `25c62a9ac48f42b619e85158d377d33da1e263f083b732bc370c1cd17298631d`; identity SHA256: `a79508bafe2ddeedccf07096cc977f8a15aa8940304f7b6f7ef2fbf7010339ae`.

For each encoder:
- `HEAD_IDENTITIES.json`: all 360 rows in canonical episode/B/policy/fold order; fold=-1 is the full-label refit, folds0..3 are CV training heads. Columns0..99 in W/b and classIDs refer to original CIFAR fine labels.
- `support_features.npy`: float32 [8751,9,D], view0 standard;1..4 weak;5..8 weak+RA. Standard policy uses view0 repeated4 times, with the same per-original objective weight.
- `query_features.npy`: float32 [9990,1,D], standard input only.
- `W.npy`, `b.npy`: float64 [360,D,100] /[360,100], D=384 (DINO) or192 (DeiT).
- `QUERY_PREDICTIONS.npy`: uint8 [360,9990], every head/query argmax. Full float64 scores reconstruct as features.astype(float64) @ W[h] + b[h] in the original 256-query blocks; argmax chooses the smallest class index on score ties. All reconstructed argmax values have been verified.
- `CV_PREDICTIONS.npz`:86,400 rows; head_index uint16, support_union_index uint32, fine label uint8, predicted class uint8. Every held-out original is predicted once per fold head/policy/episode/B. CV uses the original standard view, not augmented validation views.
- `FIT_SEAL.json`: hash-pins all heads/support features/CV decisions and24 original selections. Both such seals must precede `JOINT_QUERY_GATE.json` (720 heads/48choices), which precedes both query evaluations.
- `QUERY_SEAL.json`: all exact query feature/prediction hashes and the common gate hash.

`ALL_HEAD_METRICS.csv` has720 complete query metrics; `ALL_FOLD_COUNTS.csv` has576 rows of actual validation/query counts; `RESULTS.json` contains all48 settings, full candidate/tie sets and four complete summaries. `STAGE_RANKINGS.md` exposes complete stage orderings, including equalities. No missing setting is imputed and no query model was refit after selection.
