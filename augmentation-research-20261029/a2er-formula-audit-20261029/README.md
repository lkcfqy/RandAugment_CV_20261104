# A2ER 公式与数据准备，2026-10-29

先看 [中文结果](REPORT_ZH.md)、[逐式实现对照](FORMULA_MAP_ZH.md) 和 [原生路径门](NATIVE_PATH_GATE_ZH.md)。公式检查通过、数据已准备；第六轮条件小试未开始，因为公开样例回放输入固定、完整图像实验的增强入口仍未验证。

- `PROTOCOL_ZH.md`、`protocol-freeze.json`：执行前写定的范围、用例与哈希。
- `formula_audit.py`：111 个确定性记录、109 项检查；已有隔离环境直接运行，CPU-only，无优化器步骤。
- `dtype_and_beta_check.py`：补充解释 input dtype 提升及 beta 实数空间导数。
- `results/`：全部用例、断言、准确核心耗时与精度记录。NaN／Inf 使用字符串保存，未删掉边界例。
- `native_path_check.py`、`NATIVE_PATH_GATE_PROTOCOL_ZH.md`、`native-path-freeze.json`：独立冻结的回放路径检查；没有训练。
- `sources/`、[来源账本](SOURCE_LEDGER.md)：论文、补充材料、两个固定源码版本及哈希。
- `prepare_cifar100.py`：从官方地址首次下载、校验、安全提取 TRAIN 和开发划分。
- `resume_cifar100.py`、`finalize_cifar100.py`：真实慢速续传及安全成员白名单停止后的恢复记录。
- `prepare_cifar100_v2.py --root /absolute/new/directory`：识别官方备份元数据、只提取 TRAIN 的修订复现入口；新目录限定。最终数据状态见 [DATA_README_ZH.md](DATA_README_ZH.md) 和 `data-preparation-receipt.json`。

复现数值检查：复制本目录到新目录，把副本 `results/` 改名为 `results-original/`；使用项目 `environment-preparation-20261029/venv/bin/python formula_audit.py`，再执行 `dtype_and_beta_check.py`。脚本拒绝覆盖原始 results。无需联网；不安装作者包、不运行 `setup.py`、不载入神经网络。

没有训练、权重下载、Colab、旧封存测试、提交或推送。该目录保留公式门的证据，不宣称获得科研贡献。
