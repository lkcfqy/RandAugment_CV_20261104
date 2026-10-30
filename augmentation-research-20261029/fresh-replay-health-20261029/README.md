# Fresh-replay 实现健康检查，2026-10-29

先读 [结果报告](REPORT_ZH.md) 和 [正常DER++基线参考](BASELINE_REFERENCE_ZH.md)。未发现已执行训练路径的确定bug；唯一20图、无回放健康检查在250步达到100%训练准确率及CE .02925。旧760步试验仍为基线不足，门A/B未执行。

- `PROTOCOL_ZH.md`、`protocol-freeze.json`：诊断前固定范围与哈希。
- `diagnose.py`：旧产物审计、确切首批损失/梯度重算、一次微型过拟合。
- `supervise.py`：独立完整子进程计时。
- `run-001/diagnostic.json`、`microfit-steps.jsonl`：完整数据、参数、梯度、曝光、计时及所有健康检查步数。
- `process-timing.json`、`process.log`：外部计时和原始输出。
- `sources/`：固定Mammoth官方基线配置和来源哈希。

不要重跑覆盖这些路径；没有新完整候选训练、test、Colab、安装、提交或推送。
