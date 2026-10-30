# Draft for parent / Pro review — DO NOT EXECUTE

Date: 2026-10-29 UTC. Scope if separately approved: one CPU-only exploratory audit of public cached detections. This draft is not a frozen experiment authorization. No new method claim. No model execution, training, weight download, grid search, or old sealed data.

## Question and estimand

H1: after geometry-only coverage-preserving view thinning, CAR incurs a larger increase in false positives at a frozen source-policy threshold than raw-score NMS, even if its AP advantage remains. This is a directed operational hypothesis about this declared reference, not a universal claim about TTA.

Define D=E[(FP_CAR,sparse-FP_CAR,dense) -(FP_raw,sparse-FP_raw,dense)] per image on the evaluation partition. Each method's threshold is fitted **once** on dense-policy calibration images at beta=1 FP/image. Source and target outcomes use the same held-out images. Report all four FP rates, recall, precision, AP, view counts and their paired differences. Primary practical gate: D>=0.25 FP/image and a paired interval excluding zero. Also report beta=5 as a predeclared sensitivity, never select whichever looks favorable.

AP persistence is a secondary condition for the “ranking gain but operational failure” interpretation, not required to force a generic negative result. If CAR has no dense-policy operational benefit, record that baseline limitation and do not silently choose a different checkpoint or threshold budget. A positive D without useful recall is not a useful contribution.

## Fixed inputs and integrity gates

- One detector: published **yolov8s-visdrone**, one dataset: **VisDrone2019-DET val**, one dense view set **S512/r0.60 + full view**. It is not a reproduction of all CAR experiments.
- Prediction index: author Zenodo v1.2.2, record23644217. Required parquet `visdrone-val__yolov8s-visdrone__baseline_S512_r060.parquet`, SHA256 `519d667e2be057d40115e93a560184dabb9385d6f29533b82d248eee2340b441`; also its sidecar. They are in `detections_visdrone-val_part09of10.zip` (98,941,990 bytes). Current work read only the index, not these data.
- Obtain official val labels, author image-size manifest and full view geometry. Verify data licenses, digest, file count, class mapping, ignored regions, detector provenance and whether arrays are pre-merge. No test-dev or other test data in this stage. An author benchmark checkpoint may have used val for development; therefore this is exploratory, not independent confirmation of a paper claim.
- Do not infer absent/zero-detection views from rows. Reconstruct all views from sizes and configuration and verify populated rows agree exactly. A metadata failure stops execution as **data/implementation unresolved**.
- Common candidate score floor s>=0.02 (paper methods); verify the cache contains all required rows below/above this floor. If it does not, stop rather than fabricate missing candidates. Use all ten published classes; no post-result class selection.
- Fixed class-wise NMS IoU0.6, max500/image, deterministic ties. Keep floor and cap constant. Report cap saturation; if it affects >1% of images, qualify as a cap-dependent result and seek a revised protocol instead of increasing the cap mid-run.

## Split and leakage control

Freeze an image-group split and hashes before looking at predictions or labels. Prefer officially verifiable source-sequence/scene groups; never treat tiles/boxes as independent samples. Group assignment must not be guessed from a filename prefix. If no trustworthy grouping exists, use a deterministic image-ID SHA256 split for an explicitly exploratory descriptive result and withhold claims of independent scene generalization; any image-bootstrap interval must be labeled potentially optimistic. Do not quietly promote the authors' image halves to sequence-independent validation.

The concrete group IDs, count and split cannot be filled in until public metadata is inspected. That is a pre-execution gate, not permission to adjust the split after seeing outcomes. Calibration/evaluation images never overlap; no group can be split across them. No target-policy labels used for the primary threshold.

## Two policies, two scoring rules

Dense: published S512/r0.60 view set. Sparse: retain every second sorted tile start on each axis, also first/last, Cartesian product; retain the full image. This is a **thinned CAR reference**, not exact native SAHI r0.2. No score/content-conditioned view selection. Check complete pixel coverage and savings from metadata. If fewer than 30% of total views are removed, mark inadequate cost separation for this input; do not tune thinning to produce a desired effect.

Raw: native raw score with fixed NMS. CAR reference: native eligibility and same-class IoU0.75 per-view existence matching, m0.5, lambda5, own view eligible when geometry allows. Undefined agreement maps to0 as in source. Use score s(1+5A)/6 with a **fixed** divisor; never refit a per-partition maximum. No learned rescorer in the initial test.

For each rule, set the smallest dense calibration threshold meeting beta, with an explicit deterministic tie rule. On evaluation, freeze it across both policies. Do not call the resulting score a probability. The raw and CAR operational outputs need not have the same number of candidates or match assignments.

## Controls that distinguish explanations

1. **Normalization artifact control:** recover source CAR rankings on a fixed array; fixed divisor must preserve the same tie-broken order within numerical tolerance. Any discrepancy is an implementation issue, not H1 evidence.
2. **Fixed-candidate diagnostic:** form raw-score NMS anchors from retained-view predictions, freeze boxes/source IDs/raw scores, then compute agreement using dense versus sparse eligible views. Re-score this identical anchor set. This isolates vote-set effects from missing candidate boxes; it still changes opportunity, match count and correlation jointly. Do not claim a pure K intervention.
3. Report K, own-view eligibility, agreement, boundary distance, object/predicted box size, full-image-vs-tile source, and same-class matching jitter for all groups. Condition/standardize using frozen coarse source-defined bins. No mining of classes or bins. GT-based strata are explanatory and cannot be used by a deployment policy.
4. If operational drift disappears in the fixed-anchor diagnostic and is explained by proposal loss or context, it does not support the proposed vote-evidence explanation. If the issue only appears with per-array score normalization, reject the scientific claim.
5. Duplicate **the entire view ensemble uniformly** as a bookkeeping placebo: with deterministic duplicated predictions and deduplicated geometry it should add no evidence. Duplicating just selected views changes weights and is not an independence-preserving null. Do not bill cached duplicates as new model observations.
6. Report raw-score NMS, source-fitted per-view/scale calibration and the released 13-feature rescorer as nearest baselines in the review. Do not implement/fill their label-trained versions in this first run. If H1 survives, a second protocol must compare these strong baselines at equal labels and preprocessing before any improvement claim.

## Outcome rules

- If D fails the directed gate, report all results and close **this particular hypothesis/configuration**. Do not reverse its sign, select a favorable class or change seed/threshold to rescue it.
- If uncertainty is too wide, grouping unverified, or candidate/score metadata invalid, report unresolved execution/evidence, not proof of no mechanism.
- If D passes but ordinary source-conditioned calibration or a scalar target threshold explains everything, a “new rescoring algorithm” claim is unavailable. A documented operating boundary may still have application value; parent/Pro judge whether independent confirmation merits its cost.
- A later, separately approved labeled-target recalibration comparator would diagnose whether the effect is only threshold drift. It must not replace the frozen-source primary result or use evaluation labels. No label-budget sweep in this stage.
- Any positive result remains a mechanism/application screen. At least one independent detector or scene set, with separately frozen protocol and suitable uncertainty, is needed before broader claims.

## Resource plan

No detector forward pass is necessary for this cached-prediction test. Budget about 99 MB predictions archive + official val archive labeled0.07 GB + <1 MB image-size/reproduction metadata; arrays used are about17.3 MB. Check storage and safe extraction. No third-party mirror or access-gate bypass. Read-only module lookup in the existing isolated environment found numpy, but not pandas, pyarrow, sklearn, pycocotools or scipy. Environment setup is therefore a real future prerequisite; no package was installed in this phase.

Proposed later CPU wall-clock ceiling30 minutes, process one image/class at a time, bounded scratch memory, first5 metadata-fixed images used only for integrity and throughput. Estimate full cost before analysis; if infeasible, stop and report it. Count download, preparation and calibration time. View-count reduction is a cost proxy; actual deployment latency/energy is unmeasured. No model/weights, cloud, payment, training, parameter search or push.
