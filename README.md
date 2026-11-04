> **2026-11-04 开发交付更新。** 本次保留自有源码、配置、测试、说明和经筛选的实验成果，补充换机开发入口。先读 [换机开发指南](docs/PORTABLE_DEVELOPMENT_ZH.md)、[资源恢复与交付边界](docs/RESOURCE_RESTORE_ZH.md) 和 [第三方资源说明](docs/THIRD_PARTY_RESOURCES_ZH.md)。实际交付文件以 `PORTABLE_DELIVERY_MANIFEST.json` 为准。
>
> 四个大数组、依赖环境、特征/图片缓存、原始运行日志及私人认证记录没有加入本次上传。缺少这些资源时，普通源码开发与合成测试仍可单独验收，但不代表原实验可完整精确复现。历史 seal/manifest 中的 SHA、完成状态与时间约束仍指向原实验；新增开发版本的路径调整和省略项另行记录，不覆盖原本地实验。
>
> **隐私派生记录。** 61 份历史 JSON 中的 85 处账号余额已置为 `null`，254 份原始 CLI 台账未包含于本次公开副本。精确路径、字段位置及当前文件 SHA 见 [隐私派生清单](docs/PRIVACY_DERIVATIVES.json)。这些脱敏记录不再满足原文件 seal/hash，也不代表当前账号状态；原 SHA 字段、实验参数及本地原件保留历史含义。
>
> 下方的 2026-10-30 备份说明描述先前版本。此次新增研究目录与恢复步骤以以上新文档为入口。

> **2026-10-30 已停止并保存备份快照。** 研究和实验均已暂停；恢复前请读 [当前停止状态](CURRENT_STATUS_20261030_STOPPED.md) 与 [备份恢复说明](backup-20261030/RECOVERY_ZH.md)。

# RandAugment：课堂演示与数据增强研究记录

本项目收集 RandAugment 论文、交互式教学网页，以及围绕因果增强、LoRA 位置与增强收益、Posterize 操作的探索性实验。研究记录主要形成于 2026 年 10 月 20–24 日。

## 从哪里开始

| 内容 | 入口 |
| --- | --- |
| RandAugment 原论文 | [RandAugment_NeurIPS_2020.pdf](RandAugment_NeurIPS_2020.pdf) |
| RandAugment 动画讲解 | [randaugment-animation.html](randaugment-animation.html) |
| AutoAugment / RandAugment / TrivialAugment 对比 | [augmentation-comparison.html](augmentation-comparison.html) |
| 早期因果增强研究 | [causal-augmentation-research-20261020/](causal-augmentation-research-20261020/) |
| 后续增强研究 | [augmentation-research-20261022/](augmentation-research-20261022/) |
| 历史结果的解释与更正 | [CURRENT_EVIDENCE_STATUS_20261021.md](CURRENT_EVIDENCE_STATUS_20261021.md) |

下载后用浏览器打开两个 HTML 文件即可查看教学演示，无需安装 Python。GitHub 文件页面显示 HTML 源码，演示需要在本地打开。网页中的教学简化、图片署名和来源说明保留在原页面内。

根目录的 `CURRENT_CHECKPOINT_*` 是各阶段的进度与结论记录。它们带有历史时间点，阅读时应结合对应实验的最终报告和后续更正。

## 已有实验结论

这些实验有明确的数据、配方与统计范围限制，目前没有建立一项可普遍推广的新增强算法收益。

| 实验 | 当前可支持的结论 | 报告 |
| --- | --- | --- |
| 因果增强与条件分配 | 22 个 CPU 读出完成；预设的收益交互未获支持。该结果不能推出数据增强普遍无效。 | [2026-10-22 检查点](CURRENT_CHECKPOINT_20261022.md) |
| LoRA 位置 × RandAugment | 四格单 seed 开发实验完成；交互约 +0.340 个百分点，未达到预设 1 个百分点工程投入门。 | [四格实验报告](augmentation-research-20261022/pilot-20261022/finding-zh.md) |
| Posterize B/Z/R 三臂实验 | 三臂均完成 30 轮；冻结判定为 `unresolved_below_direction_threshold`，不能据此主张等效或无效。 | [2026-10-24 MPS 结果](augmentation-research-20261022/posterize-three-arm-mps-20261024/RESULT_ZH.md) |

旧报告中的 `completed`、`PASS` 或 `verification passed` 通常指执行、交付或核验完成，并不等于科学收益或原创性已经成立。具体边界以各报告为准。

## 仓库上传范围

本仓库保存代码、协议、研究报告、实验配置、数据来源与许可记录、日志、CSV/JSON 结果、小型预测数组、质量检查图、论文 PDF 和教学 HTML。

以下资源由 [`.gitignore`](.gitignore) 排除，继续保留在原本地项目中：

- Python 虚拟环境、PyTorch 下载缓存及 Python 编译缓存。
- 原始数据集照片、掩码、压缩包，以及逐图生成的视图缓存。
- `.pt` / `.pth` 等模型权重与训练检查点。
- 大型预计算特征库及下载产生的二进制中间文件。
- 本地凭据、环境设置和 macOS 元数据。

因此，克隆此仓库不能代替完整本地实验备份。历史 `delivery-manifest.json`、哈希记录和验证脚本可能引用上述未上传的资源；在这些资源恢复前，完整文件校验或重新训练可能无法执行。报告中“未上传”“不是 Git 仓库”等陈述保留为原实验时点的历史记录。

## 阅读与复现

1. 先阅读目标实验的中文报告、`protocol.json`、数据清单和环境记录。
2. 需要重训时，按相应来源记录准备数据和预训练权重，或从原本地项目恢复它们。
3. 按相应实验的环境记录准备依赖；本项目没有统一的安装包或一键训练入口。
4. 检查脚本中的原机器绝对路径、设备设置、输出目录和历史时间预算后再运行。

主要实验入口包括：

- [因果增强实验](causal-augmentation-research-20261020/causal-resume-20261022/experiment.py)
- [LoRA / RandAugment 训练](augmentation-research-20261022/pilot-20261022/train.py) 与 [结果分析](augmentation-research-20261022/pilot-20261022/analyze.py)
- [Posterize 三臂 MPS 实验](augmentation-research-20261022/posterize-three-arm-mps-20261024/worker-v2.py)

论文、第三方来源代码和图片保留原作者与原许可信息；仓库未额外授予这些材料新的许可。
