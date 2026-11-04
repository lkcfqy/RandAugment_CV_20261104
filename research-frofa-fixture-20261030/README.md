# Frozen-feature clipping fixture

Completed CPU-only mathematical fixture; 919 assertions pass, one formal run, 1.20084 seconds within 600-second hard wall-clock budget. No training or new-contribution claim.

Start with [REPORT_ZH.md](REPORT_ZH.md). Exact derivation: [DERIVATION.md](DERIVATION.md). Fixed configuration: [protocol.json](protocol.json). Original records: [results/summary.json](results/summary.json), [budget.json](budget.json), [worker-attempt-01.log](worker-attempt-01.log).

This archive is a completed run. The budget supervisor refuses additional execution after success. Do not reset its state as a continuation of this authorized fixture. Any future numerical experiment requires its own explicitly agreed scope and retained protocol.

The source `.py.txt` files are evidence only and must not be imported/executed. Actual independent implementation is in `controls.py`.

Archive hash verification excludes only `SHA256SUMS` itself and `archive-verification.json` from the payload manifest to avoid self-referential hashes. The latter records the manifest's own SHA-256 and the full source/destination payload equality check.
