# 实际恢复进度

更新时间：2026-10-31T14:07:24.266594+00:00。

- Astral CPython 3.10.19 已在 CPU 实际安装。归档 SHA 为 c0b09b744293f2aad85b1ef84544f8a7ba383675b29d1f7efd1e96bb9984399f；官方依赖、Color 恒等/RNG、MMCV 人工算子及源码哈希门已通过。
- 8 MiB 公开标记通过 Colab CLI 下载到本机并原子落盘、SHA 回读；CPU 端已释放。证据在 ../deepfish-recovery-v1/cpu-check3/received/import_gate.json 和 controller_report.json。
- 第一轮 T4 在 Runner 构造前因 setuptools 缺 pkg_resources 失败，优化器更新 0、dev/test 效用 0。失败日志全部取回，实例已释放。
- 已从 PyPI 核验 setuptools 69.5.1 wheel SHA c636ac361bc47580504644275c9ad802c50415c7522212252c033bd15f301f32，并实际验证 pkg_resources.packaging 接口。新一轮 bootstrap 增加 Torch cpp_extension 导入门。
- 当前进行第二轮 T4，费率 1.15 CU/小时，本轮已运行约 149.2 秒。此前保守账本 0.211608864 CU、464.288 GPU 秒全部保留。本轮实际回取速度 3597610 字节/秒。
- 当前未收到完整 CUDA 预检回执，U 未冻结；没有正式 A/B 或 dev 效用结果。数据、两份科学配置、U 函数均与原包逐字节一致。

实时记录：gpu-run/execution.stdout.log、latest_usage.json、progress.json；终态将写 controller_report.json。目标仍是完成两臂；未把环境门通过当成实验完成。
