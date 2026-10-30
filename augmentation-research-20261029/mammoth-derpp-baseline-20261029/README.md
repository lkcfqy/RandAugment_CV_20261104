# 固定Mammoth DER++基础诊断，2026-10-29

完整结果见 [REPORT_ZH.md](REPORT_ZH.md)，执行前设计见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。v2已完成6300步、两个task各50epoch；只有一个固定seed，未执行门A或五分支。

`baseline.py`/`run-001`保留2步logger中断；`baseline_v2.py`/`run-002`是独立修复后的完整运行。`LOGGER_FIX_V2_ZH.md`说明唯一修复；两个freeze文件和原始日志均保留。`supervise_v2.py`及`process-timing-v2.json`记录完整子进程时间。

`sources/manifest.json`保存官方固定提交、Git blob、SHA-256；`verify_completed.py`独立验证步数、曝光、源映射、抽样/入库随机数、checkpoint和指标，不再次运行模型。`run-002/independent-verification.json`保存结果。

所有终点固定，不重跑当前路径、不选择中间checkpoint。旧760步基线不足结果不变；本次是新的基础学习诊断，不是科研贡献结论。
