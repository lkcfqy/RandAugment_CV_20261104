# LA3 补边顺序：2026-10-29 CPU 审计

先读 [中文报告](REPORT_ZH.md)，再查 [来源账本](SOURCE_LEDGER.md)。结果：窄 nearest 像素机制成立；原生条件支持、事件稀少、最终缩放混合边界与严重度限制尚不足以进入训练。未声称创新、泛化损害或方法无效。

- `PROTOCOL_ZH.md`、`protocol-freeze.json`：像素运行前写定的协议与哈希。
- `audit.py`：从固定源码提取函数，原生策略清点、完整链和后处理检查。
- `summarize_and_verify.py`：运行后的描述汇总、失败归档一致性与 168 次原始采样器独立回放核对。
- `results/all-cases.jsonl`：全部 3,584 个完整链结果和操作参数。
- `results/operator-controls.json`：800 个基础操作检查。
- `results/conditional-support.json`：实际条目位置、类条件概率、支持与事件质量。
- `results/failure-cases.jsonl`、`failure-arrays.zip`：全部 974 个非严格通过例，含无变化例。
- `results/representative-*.npz`、`postprocessing.json`：固定首个种子／载体的实例与边界检查。
- `sources/`：固定源代码、受限解析的原生策略、四篇完整 PDF 与哈希。`.pkl` 不使用通用任意对象反序列化，已限制操作码和对象构造。

依赖沿用项目已有隔离环境：Python 3.14.10、NumPy 2.5.4、Pillow 13.0.0。无需 torch、GPU 或联网运行像素审计。

重现时复制本目录到一个**新目录**，保留原始记录，将副本中的 `results/` 改名为 `results-original/`，然后用项目环境运行 `python audit.py` 和 `python summarize_and_verify.py`。脚本故意拒绝覆盖已有 `results/`。这会重算合成图像检查，不会训练或下载数据。

本轮未安装依赖、训练、下载数据／权重、打开封存测试、提交或推送 Git。旧 SADA/SRA 等记录保持原状。
