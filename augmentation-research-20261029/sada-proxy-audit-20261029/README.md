# SADA proxy formula audit, 2026-10-29

Read `VERDICT_ZH.md` first. This is a deterministic mathematical audit of arXiv:2510.00434v1, **not a SADA implementation, benchmark or new-method experiment**.

Requirements: Python 3.9+ standard library only. No network access, project data, checkpoints or third-party packages are used by the audit script.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 audit.py --out /tmp/sada-audit-reproduction-NEW
```

The output directory must not exist; files are never overwritten. Run from this directory or use an absolute script path. The script verifies the local source hashes, executes every case in `protocol.json` at 90 and 60 decimal digits, writes the complete outputs and runs `test_audit.py`. It exits nonzero on test failure. Runtime metadata varies; mathematical output is deterministic on the same runtime.

Files:

- `protocol.json`: fixed cases and limits, frozen before the first numerical execution.
- `audit.py`, `test_audit.py`: final executable audit and validation checks.
- `results/raw-cases.jsonl`: every probability, state, finite difference, gradient projection and both KL directions.
- `results/windows.jsonl`, `results/window-summary.csv`: every window, metric, eta normalization, raw order and precision flag.
- `results/ema.jsonl`: all declared zero-initialized EMA sensitivities; no magnitude mapping.
- `results/numeric-summary.json`, `results/test-results.json`, `results/run-metadata.json`: results, checks and provenance.
- `precision_review.py`, `precision-review.json`: independent full-window recalculation at 60 digits against saved 90-digit results.
- `SOURCE_AUDIT.md`, `sources.json`, `sources/`: source boundaries and version-pinned snapshots.
- `initial-implementation/`, `results-initial/`: preserved initial implementation/output before repairing endpoint CE antisymmetry; do not use their near-zero strict rankings as scientific evidence.
- `checksums.sha256`: SHA-256 inventory of delivered files, excluding the inventory itself.

The initial code used an algebraically equivalent log-partition increment for CE. An out-and-back trajectory exposed a numerical pathology: theoretical zero window variance became tiny nonzero Decimal residuals. The final code evaluates CE directly at deterministic endpoints, preserving exact antisymmetry, keeps the original expression as an independent check, and adds a dedicated regression test. No cases, labels, rates or epsilon scales were changed. All initial numeric outputs are retained.
