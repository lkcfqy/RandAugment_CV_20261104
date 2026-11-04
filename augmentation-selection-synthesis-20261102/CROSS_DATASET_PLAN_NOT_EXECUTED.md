# Minimal falsifiable follow-up — NOT EXECUTED

This is a prospective proposal motivated by post-result EuroSAT analysis. It is not approval to acquire assets, run a third round, or change the current selector. No reliable paper contribution has yet been established.

## Question

Does the observed opportunity/cost pattern extend beyond EuroSAT: with a fixed three-policy pool and frozen encoders, is the episode-wise oracle gain over a strong constant policy smaller than actual CV selection regret, even after granting all CV ties their most favorable query-labelled assignment? Are most full-head mistakes already detectable as strict inferiority when the same fold heads are evaluated on queries, rather than arising only after full-label refitting?

These are empirical, conditional questions, not a claim that CV is universally harmful or that validation noise has been newly discovered. The oracle and best-constant comparisons remain diagnostics; the two prespecified fixed baselines W and R stay the primary deployable comparisons.

## One data source; no adaptive expansion

Candidate: CIFAR-10 as a non-remote-sensing image source. Its local availability, lawful provenance, identity split and previous experimental exposure have NOT been inspected in this follow-up. Do not download it or substitute another dataset automatically. Before any new query utility, verify this asset; if it is unavailable under the separately authorized scope, stop. If prior use prevents a fresh-confirmation claim, label the run exploratory. Fix the data choice before utility and do not search datasets until one gives the desired pattern.

Subject to that asset check, use all ten classes, a fixed 100-per-class training candidate pool and a disjoint fixed 100-per-class query set from documented allowable splits. Select identities by one registered hash salt, not GT difficulty or model scores. Reuse 12 nested episodes with B=8/16 per class, four folds grouped by original, four views, C/W/R, RRC range/flip/RA N2 M9, feature normalization, lambda=.001 ridge, per-original weights, and C > W > R tie order exactly as before. Do not tune the regularizer to the new dataset. Preserve each encoder's already documented native clean input recipe.

Both existing DINOv2-Small and DeiT-Tiny are required for this specific two-encoder question. This is one new dataset, not two independent confirmations. All 720 heads and both models' 48 choices must be sealed before ANY query evaluation, then all fixed-policy and selected results are reported. No encoder training or view/strategy search.

## Frozen outputs and falsification

Report the paired CV−W and CV−R distributions, all 12 episode outcomes for each encoder/budget, choices, regret conditional on mistakes, oracle space versus EVERY constant policy, query-best sets, and all CV-tie assignments. Also report full-wrong/fold-wrong and full-wrong/fold-correct cases, with integer per-fold correct counts; separately compare the actual CV choice to each eventual full-query optimum in the fold-query scores. A third candidate being better at the fold stage does not prove the eventual full-stage mistake was already present for the same pair. No confidence interval treating overlapping supports or the shared query as independent; no post-result deletion or extra seed.

The current qualitative pattern would fail to extend if the new data show that the CV cost is no longer larger than the oracle opportunity above the strongest constant strategy, especially after complete favorable tie bounds, or if full-refit changes rather than fold-validation/query discrepancies dominate the observed errors. Evaluate this separately for each encoder and budget; do not pool away a contradiction. Report small or ambiguous differences as ambiguous. A reversed direction is a limitation to the claim, not a reason to strengthen augmentation, alter the tie rule, choose another dataset or add repetitions.

The secondary budget comparison is whether B16 leaves less oracle space than B8 in both encoders. This is NOT a general monotonicity theorem: EuroSAT already shows opposing directions for the CV regret change. If the opportunity shrinkage does not repeat, limit that description to the original data.

Even agreement would justify only a narrow empirical boundary under these heads, policies and budgets. Before proposing a paper, identify what this adds beyond Evaluating the Evaluators / CLAP and existing training-scale model-selection work; repetition alone is insufficient. No new-method claim follows from the accounting identity.

## Cost and stopping

The fixed 100-per-class candidate pool caps the support union at 1,000 images. The workload is therefore at most 1,000 support originals x 9 views plus 1,000 queries per encoder: 20,000 formal image-forwards across both encoders, plus bounded timing checks. The previous realized union was only 872; it MUST NOT be assumed for a new hash sample. Exact counts must be measured from the identity plan before encoding.

Observed numeric pipelines on this Mac were 146.69 seconds (DINO) and 39.77 seconds (DeiT), using 8,848 formal image-forwards each. A count-scaled estimate for the 1,000-support union bound is about 211 seconds combined, before contingency, preparation, validation and archival. This is an estimate, not a measured third-run cost. Reserve at least 1.5x numeric time plus 10 minutes for local persistence/recomputation; a separate one-hour task cap would be sufficient only if the asset checks and actual no-utility timing pass. No new weights, installation, cloud or CU. The user's 10:00 UTC research cutoff overrides any task window.

No third-run permission, asset check, identity choice, encoding, fitting or query evaluation has occurred here. The parent must decide whether this limited external check is worth performing.
