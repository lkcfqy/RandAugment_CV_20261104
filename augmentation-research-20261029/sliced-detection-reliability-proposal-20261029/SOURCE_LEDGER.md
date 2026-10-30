# Source ledger

Read 2026-10-29 UTC. A source being listed does not mean it validates H1. No search-result absence is treated as proof of novelty.

## Primary papers and scope of reading

1. Nikouei et al., [Small Object Detection: A Comprehensive Survey on Challenges, Techniques and Real-World Applications](https://arxiv.org/abs/2503.20516), v1, 2025-03-26. Abstract and §7/Table3 locator (PDF pp35–36). Used only to locate the efficiency/robustness application space; not a novelty certificate.
2. Akyon et al., [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934), ICIP2022. Abstract and official benchmark README; application/availability baseline, not a claim that this audit reproduces the paper.
3. Gilg et al., [The Box Size Confidence Bias Harms Your Object Detector](https://arxiv.org/html/2112.01901v1), §4.5/Table4; source text reports separate scale calibration before NMS. This directly defeats a generic “TTA confidence bias is new” claim.
4. Kuzucu et al., [On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines](https://arxiv.org/html/2405.20459v1), §3–4, especially §4.2–4.4. Read threshold choice, split design, Platt/isotonic and distinction between probability and localization objectives. Official implementation pinned below.
5. Gilg et al., [Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates and Implicit Duplication Modeling with IoU-Aware Calibration](https://arxiv.org/html/2309.03110v1), abstract, §3.2.3/§4 opening. WACV2024 near prior for conditional calibration and duplicate modeling. We did not reproduce its experiments or claim its full transfer coverage has been exhausted.
6. Na, Kim, Jang, [Cross-view agreement rescoring improves tiled small-object detection without retraining](https://doi.org/10.21203/rs.3.rs-10908559/v1), posted2026-09-29. Read main PDF Results, Scope/Limitations (PDF p9), Methods (PDF pp9–10), data/code availability. The author's current source record says **submitted**, so do not call it accepted Scientific Reports. Primary PDF: [covered v1](https://assets-eu.researchsquare.com/files/rs-10908559/v1_covered_205103f6-7ba6-43b0-8d62-a3890c5a45c5.pdf). Current code/data-index record [Zenodo23644217, v1.2.2](https://doi.org/10.5281/zenodo.23644217), 2026-10-26; API metadata acquired directly. Main methods audited against actual source, not just abstract. The supplemental PDF exceeded the web tool size cap; released A11/A12 code was read instead. Do not claim complete supplement coverage.
7. Tang et al., [QASHI: Quality-Aware Scale-Protected Hybrid Inference for Small-Object Detection in Optical and SAR Remote Sensing Imagery](https://www.mdpi.com/2072-4292/18/20/3587), Remote Sensing18(20),3587,2026. Official indexed snippets name reliability calibration, source scale and crop boundary. Full page HTTP429. Method and threshold-transfer coverage remain unverified; no exclusion/novelty claim rests on the missing page.
8. [ROI-Gated SAHI](https://arxiv.org/html/2608.23923v1),2026-08-25. Abstract/introduction and cost-setting locator read as a near efficiency direction. Not used as an empirical claim about this candidate; its reported COCO128/illustrative tiny sample is not our planned validation set.

## Fixed public repositories

| Repository | Commit | Retrieved evidence |
|---|---|---|
| [obss/sahi](https://github.com/obss/sahi/tree/b5a153ee7bcb1fd84a7b868575ff14131e7423ce) | b5a153ee7bcb1fd84a7b868575ff14131e7423ce (2026-10-16) | predict.py, slicing.py, combine.py, postprocess/utils.py, torchvision adapter/registry, MIT license |
| [fiveai/detection_calibration](https://github.com/fiveai/detection_calibration/tree/bf59c92f5416ac4b66a7c2c67b44487777ff2ea4) | bf59c92f5416ac4b66a7c2c67b44487777ff2ea4 (2025-03-26) | README, DetectionCalibration wrapper, common_objects tutorial, license |
| [VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset/tree/4364e8265275dfa44fd8f767b5180af58580194d) | 4364e8265275dfa44fd8f767b5180af58580194d (2023-09-24) | Official val-download and DET-toolkit links; README snapshot |
| [fcakyon/small-object-detection-benchmark](https://github.com/fcakyon/small-object-detection-benchmark/tree/d923428f380763dec2bd809cd4f7d532c6e00d03) | d923428f380763dec2bd809cd4f7d532c6e00d03 (2025-01-17) | README read; old MMD/CUDA recipe and some released weights identified, not installed/downloaded. Not the preferred CPU-cache route. |

Twelve saved GitHub text snapshots were checked against Git blob SHA1 (including the `blob <length>\0` prefix), then SHA256 recorded in `SOURCE_SNAPSHOT_VERIFICATION.json`. A serialization-added final newline was detected and removed **only after** matching the expected Git blob; no source semantics were edited.

## CAR version and archive verification

- Initial pinned record23642547/v1.2.1 and current record23644217/v1.2.2 metadata are saved. Current publication date2026-10-26 is before this audit.
- v1.2.1 `CAR_code.zip`: 852,791 bytes, SHA256 `5ad3bd3947b9fe718dde3de76978fc7fcecf21729d8e01be3be24bcd31009383`.
- v1.2.2 `CAR_code.zip`: 852,873 bytes, SHA256 `9cad38bc8b1f0fc876bffad794425495ede309f669b4e239b08b31fcf9000633`.
- Both official MD5 checksums passed. Zip member comparison found exactly one changed member: `code/revision/figures/make_fig1_photo.py`. Scientific source files cited in this audit are byte-identical between these versions. Extracted source under `sources/car/code/` is v1.2.1; do not mislabel it as a distinct Git commit. v1.2.2 archive and `VERSION_DIFF.json` are retained.
- Code AGPL-3.0-or-later, data CC BY4.0 per author README. Public benchmark images retain their original licenses and are not redistributed by the authors. No copied code was installed or executed.
- Prediction **index only**: 532 rows, SHA256 `d71de831c8cdadfb6660a627a79df6c176e999f831dc51cafe5e5343bfcc2af2`. It lists individual SHA256, sizes and containing zip; no predictions/results were computed here.

## Availability limits

Public source and metadata are verified available; usability of the proposed parquet, annotations, sidecar and scene groups is not yet verified. Availability of a download link is not proof of local data readiness. The old MPS acceptance report supports only an existing working torch/torchvision environment, not detector throughput. A [Torchvision SSDLite model](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) was checked as a possible public-weight fallback, but it is **not selected** for this initial cached-prediction protocol and was not downloaded.
