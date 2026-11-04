# NSegment 有限 CPU 审计

先读 [中文报告](REPORT_ZH.md)。本次实验已经完成，结论限定为固定公开实现、环境、参数、尺寸和 seeds 的审计；不是原 Table I 重现或新的训练贡献。

主要证据：

- `protocol.json`、`PROTOCOL.md`：实施前固定的协议、预算、停止规则。
- `per-seed.csv`：1024 行逐 seed 指标。
- `fixture-results.json`：12 项校准记录，0 次修正。
- `scan-summary.json`：两尺度与四参数的扫描汇总。
- `trace-summary.json`、`trace-records.json`、`trace-states.json`、`trace-outputs-{A,B,C}.npz`：合成管线结果、完整状态与输出。
- `audit_core.py`、`fixtures.py`、`scan.py`、`trace.py`：独立最小实现；不导入下载的 NSegment 源码。
- `SOURCE_COMPARISON.md`、`sources/manifest.json`、`sources/LICENSE`：逐行对照、固定源码哈希、MIT 许可。
- `environment.json`、`opencv-build.txt`、`dependency-install.log`、`requirements.lock.txt`：环境及官方 PyPI 安装记录。
- `budget.json`、`checkpoints.jsonl`、`validation.json`：实际预算、过程检查点、结果验证。
- `SHA256SUMS.json`：归档 payload 内每个文件的校验值（清单本身除外）。归档外的 `DELIVERY_VALIDATION.json` 记录 tar SHA256 和复制后复核。

实际执行解释器位于隔离 staging 的 `.venv/bin/python`，原项目 `.venv` 未改动。venv 不纳入归档。归档含锁定版本，复查现有结果只需安装的 NumPy/OpenCV 与 Python 标准库；重新运行实验必须另建目录和新协议，保留本次所有证据。脚本内固定本次绝对截止时间，不会在期限后自动开启新扫描。

本次命令（均在 staging 中，日志一并保存）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python fixtures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scan.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python trace.py
.venv/bin/python verify.py
```

`verify.py` 核对现有结果和已记录执行时间，不重跑实验，可在本轮期限后复查；`scan.py` 用独占创建方式打开 CSV，拒绝覆盖已有扫描。原始下载源码和论文仅被读取、解析为文字及哈希，从未执行。
