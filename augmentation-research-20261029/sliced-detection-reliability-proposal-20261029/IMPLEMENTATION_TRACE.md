# Fixed implementation traces

These are read-only source observations. They are not allegations of bugs, performance results or execution logs.

## SAHI — b5a153ee7bcb1fd84a7b868575ff14131e7423ce

- `sahi/slicing.py:64–94`: explicit height/width and integer overlap determine the grid; final starts are snapped to the image boundary. Real coverage must use returned geometry, not a uniform-count approximation.
- `sahi/predict.py:181–189`: default overlap0.2, full-image pass enabled, GREEDYNMM/IOS default.
- `sahi/predict.py:321–332`: when model confidence is low and `force_postprocess_type` is false, the implementation switches to NMS/IOU. Any reliability experiment must pin postprocessing explicitly.
- `sahi/predict.py:367–386`: batch tile inference → coordinate shifts → filtered predictions appended in full-image coordinates.
- `sahi/predict.py:390–393`: optional intermediate merging; leave `merge_buffer_length=None` when comparing full-pool postprocessing.
- `sahi/predict.py:398–418`: optional whole-image pass and final merge.
- `sahi/postprocess/utils.py:400–414,417–429,449–484`: merged score=max; bbox=union; mask=union; category from higher score. Thus maximum-score arithmetic alone does not describe the whole controller.

## CAR — Zenodo23644217/v1.2.2; scientific files identical to archived v1.2.1

All paths below are archive-relative `code/...`; extracted local prefix is `sources/car/code/`.

- `src/views.py:30–39,84–99`: step=int(S*(1-r)), final boundary start, rectangular grid + optional full view. This grid is not guaranteed byte-for-byte identical to current SAHI's different overlap-rounding expression.
- `src/features.py:21–37`: bbox coverage with margin0.5 sqrt(area); image-edge waiver controlled by `margin_at_image_border=False`.
- `src/features.py:147–192`: reconstruct views from image dimensions; compute K from eligibility; per-view same-class IoU existence; agreement M/K; NaN for K=0. Own view is not automatically excluded.
- `src/car.py:6–14`: score=s*(1+lambda*agreement); NaN agreement→0; normalization divides by the maximum score of the supplied array.
- `src/car.py:16–20`: score normalization then class-wise NMS. `:23–32` also provides a post-merge, fixed-candidate rescoring variant. Therefore “hold boxes fixed and rescore” is already an author control; our fixed-anchor diagnostic is not a new method.
- `src/calibrate.py:5–25`: 13 features include original/network size, scale, boundary proximity, normalized coverage, agreement, score dispersion, localization dispersion, density and class.
- `src/calibrate.py:81–92`: supervised/pseudo-label GBDT fitting; `:131–146` low-coverage shrinkage also exists. Neither extra coverage features nor low-coverage shrinkage may be claimed as new proposals.
- `revision/pod/rev_a11_viewcalib.py:2–14,61–112`: per-view/per-scale isotonic calibration, image-half cross-fit, same frozen policy, AP-oriented comparisons.
- `revision/pod/rev_a12_rescorer.py:2–14,59–69,84–116`: 13-feature supervision; frozen S512/r0.6 or prefreeze S640/r0.25 are separate within-policy evaluations. This does not by itself verify frozen-threshold transfer between policies. It is a mandatory strong neighbor, not a reason to claim “labels do not help reliability.”
- `configs/frozen_test.yaml:33–46,66–85`: detector floor0.001, max300/view, max500/image, matching0.75, NMS0.6, dense S512/r0.6. Main manuscript mentions analysis floor0.02, so the proposed cache audit must inspect sidecars and apply the declared analysis floor consistently.
- `src/data/visdrone.py:12–34`: ignore categories0/11 or score0. This reader still requires comparison to the official evaluator before scientific use; copying its behavior is not automatic validation.
- `src/splits.py:6–14`: random image halves, not sequence-group verification. `revision/local/rev_a14_cluster_boot.py:2–12` clusters checkpoint/dataset for an across-configuration statistic; it does **not** establish scene independence for our image sample.

## ECCV2024 calibration implementation — bf59c92f5416ac4b66a7c2c67b44487777ff2ea4

`DetectionCalibration.py` exposes fit→transform→evaluate with separate val/test annotations. `common_objects.py` documents class-specific LRP thresholds before and after calibration, and held-out domain-shift evaluation. This prevents an unfair baseline consisting only of one arbitrary temperature or fixed0.3 threshold. No dependencies were installed and no calibrator was fitted here.
