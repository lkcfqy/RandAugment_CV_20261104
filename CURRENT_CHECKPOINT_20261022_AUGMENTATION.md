# RandAugment扩围研究检查点

2026-10-22 UTC。用户明确研究不限因果。仍在原RandAugment项目，旧阴性、因果研究记录、课堂与封存资料保留，不把教师评价当作结果保证。

- [主方案与唯一备选](augmentation-research-20261022/proposal-zh.md)：主推同参数预算的前/后层Q/V-LoRA × 固定RA训练收益交互；备选是同RGB的CutMix后增强来源面积目标一致性。
- [独立提案与挑战](augmentation-research-20261022/independent-proposal-review.md) / [近邻证据与访问边界](augmentation-research-20261022/novelty-evidence.json)
- [设计草案](augmentation-research-20261022/design-draft.json) / [候选数据清单](augmentation-research-20261022/candidate-data-plan.json)：排除177已有JPEG ID；训练/开发/预留各592，仅元数据，非最终测试锁。
- [实际MPS兼容计时](augmentation-research-20261022/mps-vit-probe.json) / [实现代码](augmentation-research-20261022/mps_vit_probe.py)：4布局16步真实mps:0反向更新，非数据集效果。
- [状态](augmentation-research-20261022/outcome.json) / [保留核验](augmentation-research-20261022/preservation-final.json) / [后台](augmentation-research-20261022/background-status.json)

主方案conditional GO到数据QA、实现和正式协议锁定；尚无数据集训练、未宣布原创机制或新贡献。不是只做冻结特征：适配器/完整骨干训练路径已实际检查。近邻Surgical FT、Hint-Aug、DynamicAug和CoCa直接覆盖相关动机；精确位置×RA经验差异仍待验证。备选与OA-CutMix重叠风险高，不并行开训。

下一步只推进主线：核验新公共照片/标签、重复组、增强可辨识性及实际loader/AdamW资源，再锁定完整协议。计划4核心格10epochs开发，2小时墙钟上限；完整30runs/最多30epochs/12小时累计上限，均含明确优化与停止边界。欠训练/预算耗尽记测量不足，不能变科学阴性。8A优先，单MPS进程、受限CPU/内存和计算节奏；A100仍需另问。

旧34+12+12份交付与3旧检查点、19旧回合/3课堂/3源输入SHA保持。旧J*解释v2与代理候选否决不改，不解封旧final，不重跑旧smallNORB。0新图片/权重/依赖下载，0后台训练。sysctl权限读拒绝后停；官方Pets二进制浏览不支持与308属于技术格式/重定向，固定镜像HEAD200。历史验证/403与不可访问目标不绕过。

所有资料与代码在Mac。无A100/Colab/付费/ARiS/私有上传/发布，未操作8A。Sol/max与既有Fast请求配置不变，实际后端档位回执未暴露。
