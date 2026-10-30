# Fresh-replay 残差降权参考组合，2026-10-29

结果见 [REPORT_ZH.md](REPORT_ZH.md)。这是声明的新组合，不是A2ER复现。共同前缀760步已完成；新任务基线学习不足，门A及五个短分支未执行，无机制结论。

执行前设计见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md) 与 `protocol-freeze.json`。`pilot.py` 包含固定数据、模型、学习门及条件分支；`run-001/` 保存全部真实记录。`verify_saved_run.py` 仅用保存的logits与TRAIN标签独立复核，不再运行模型。`sources/` 保存固定Mammoth增强配方和Git blob/SHA-256。

本次有绝对UTC硬截止，**不要直接重跑pilot或复用run-001路径**。任何另一次试验应先另立协议和输出路径，不能修改此冻结版本后伪称复现。原始TRAIN和旧封存测试不变；没有提交或推送。
