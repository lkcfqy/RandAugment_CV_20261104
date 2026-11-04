# DeepFish 检测资产与 detector 配置核验

本次完成有限资产检查。数据、权重和固定源码已留在本机；没有运行模型、训练、CUDA、MPS、Colab、生成器或检索器，没有安装依赖、上传数据或推送 GitHub。

当前事实：官方 FISH 归档可安全读取；Swin-T 权重通过 CPU `weights_only=True` 张量检查。检测框的上游制作过程、检测版独立许可说明、视频/场景分组语义及独立 dev 策略仍未核实。本报告不对整个数据集作“可用/不可用”裁决，也不宣称复现、创新或增强效用。

交付入口：

- [DATA_MANIFEST.md](DATA_MANIFEST.md)：来源、真实数量、support 身份、划分和许可边界。
- [DATA_MANIFEST.json](DATA_MANIFEST.json)：可机器读取的核心资产事实。
- [配置审计](reports/CONFIG_AUDIT_ZH.md)：原始 GroundingDINO Swin-T 继承关系、生效增强、优化器、更新次数及另一路径隔离。
- [权重检查](reports/WEIGHT_CHECK_ZH.md) 与 [完整键/形状记录](reports/WEIGHT_CHECK.json)。
- [下一阶段单 CUDA 吞吐检查需求](reports/NEXT_CUDA_CHECK_ZH.md)：尚未执行，也未承诺两小时可行。
- [六图人工核验](reports/VISUAL_REVIEW_ZH.md)、[support 清单](reports/SUPPORT_IDENTITIES.csv)、[归档逐文件哈希](reports/FISH_archive_members.json)。
- [操作状态与异常记录](reports/OPERATION_LOG_ZH.md)、[最终核验](reports/FINAL_VERIFICATION.json)。

最需要保留的配置事实：

1. 提交中的 Swin-T 文件默认仍指向 DIOR；仅把第一项数据基类换成 FISH，才是本报告分析的 FISH 路径。该替换只在静态解析中发生，没有修改或执行上游文件。
2. 上层 pipeline 覆盖 FISH 文件中的 Mosaic/HSV/MixUp；直接解析的训练循环是 12 轮，独立变量 `max_epochs=30` 只用于 scheduler 的结束范围。
3. 自动化脚本实际临时继承 Swin-B 并启动 4 GPU，且默认是 ArTaxOr 1-shot；没有运行该脚本。
4. `mm_grounding_dino` 的 FISH 配置使用另一权重，并每轮在 test 上 `save_best`。本次没有下载该权重或混用这一路径。

来源代码、原始归档和图片仅本地保留。`sources/` 中所有 Python/脚本均作为文本读取；本次执行的是 `scripts/` 中自写的受限下载、归档检查、标注统计、预览和静态解析脚本。旧研究目录没有更改或重启。

