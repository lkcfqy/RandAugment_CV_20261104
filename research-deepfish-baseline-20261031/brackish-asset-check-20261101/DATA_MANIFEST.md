# Brackish v4 local asset manifest

Status: official archive acquired and metadata qualification completed. No model or image decoding.

| Asset | Bytes | Integrity / scope |
|---|---:|---|
| `assets/brackish-dataset-v4.zip` | 181,102,697 | SHA256 `95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da` |
| `original_v4/` | 197,144,979 | 15,193 unchanged release files; matches API expanded total |
| `sources/kaggle_metadata.json` | 16,091 | SHA256 `3f6475332c14c3b9c2b3cface5750dfae563f02f972945e2eb1bd694645bf0fc` |
| `sources/kaggle_files_v4.json` | 9,100 | SHA256 `cc180867835625b6ea4b64025fcfc19026535bfc2e4ee51ca276c3559e345bc0`; first API page only, not complete file manifest |

Download: `https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/brackish-dataset?datasetVersionNumber=4`; one successful HTTP attempt. Final official storage origin and path, headers, ETag/MD5 equality, timestamps, and version are recorded in `sources/DOWNLOAD_RECEIPT.json`. Signed redirect query strings were not retained.

`reports/ZIP_MEMBERS.json` is the complete archive member listing inspected before extraction. `reports/EXTRACTED_FILE_SHA256.json` provides all original file sizes and SHA256. `FILE_MANIFEST.json` covers the final local deliverables; its own hash and post-archive verification are recorded outside the directory in the archival receipt. No received upstream code was executed.

Important deliverables:

- `REPORT_ZH.md`: findings, qualification and limited future protocol.
- `ATTRIBUTION_AND_LICENSE.md`: publisher CC BY-SA 4.0 and unresolved COCO MIT-field discrepancy.
- `reports/VIDEO_COVERAGE.csv`: all 89 videos, date/time filename fields, declared coverage, split counts, classes, hashes. `class_*` fields count all published COCO boxes including disputed tails; `all_animal_boxes` and `fish_target_boxes` use declared-coverage-qualified frames.
- `reports/DATE_INVENTORY.csv`: all 10 dates and qualified counts.
- `reports/ALL_FRAME_QUALIFICATION.csv`: all 15,084 frame filenames represented by released YOLO metadata, with original split and coverage status; not a decoded-video frame count.
- `reports/UNCOVERED_TAIL_FRAMES.csv`: 566 filenames outside the two explicit annotation limits; includes the 156 disputed tails with AAU/COCO annotations.
- `reports/ORIGINAL_SPLIT_AUDIT.json`: shared video/date groups, adjacent-frame split crossing and timestamp-interval overlap.
- `reports/DATA_QUALITY_AND_MAPPING.json`, `YOLO_COCO_DIFFERENCE_SUMMARY.json`, `EMPTY_FRAME_CROSS_FORMAT_CAVEAT.json`: complete raw quality evidence and explicit retained coordinate-bound issues.
- `reports/PROPOSED_*_POOL_FRAMES.csv`, `DATE_BLOCK_AND_5SHOT_FEASIBILITY.json`: metadata-only date partition proposal, not an actual selected support episode or model run.
- `reports/VERIFICATION.json`: independent all-file and annotation checks.
- `download_official.py`, `audit_metadata.py`, `verify_metadata.py`: local stdlib acquisition and audit programs, separate from untouched publisher scripts.
- `sources/SOURCE_EVIDENCE.json`, `reports/ACTION_LOG.json`: scope, evidence strength, nonblocking source access failure and execution status.
- `reports/COMPLETION_AND_BUDGET.json`: actual elapsed time, local byte totals and unchanged no-model/no-upload/no-CU status.

All original split labels belong to **Brackish**. Reading Brackish `test` annotations here is metadata qualification and does not involve DeepFish test access or model evaluation. Original split files were not changed. The proposed date partition is a new possible research protocol and does not reproduce the publisher's frame split benchmark.

Limits: raw archive ≤300 MiB, this asset directory ≤1.5 GiB, full existing research directory ≤12 GiB. The final exact byte receipt checks all three limits without deleting old assets. Filesystem allocation overhead is separate from the recorded logical byte totals.
