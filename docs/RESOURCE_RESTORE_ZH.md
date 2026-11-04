# 资源恢复与交付边界

本仓库的开发副本包含源码、配置、测试、研究说明和经筛选的数值成果。它不代替原机器的完整实验备份。先按 [开发指南](PORTABLE_DEVELOPMENT_ZH.md) 在全新环境完成显式的合成测试，再选择需要恢复的真实实验资源。第三方来源和许可见 [资源说明](THIRD_PARTY_RESOURCES_ZH.md)。

## 四个明确省略的大文件

四个文件均留在原本地项目，未随本次上传，也未删除。恢复到下列项目相对路径后，应核对完整 SHA-256；仅文件名相同不代表内容正确。

| 相对路径 | 字节数 | 性质及缺失影响 |
|---|---:|---|
| `augmentation-selection-cifar100-run-20261102/dinov2_small/W.npy` | 110592128 | 原实验 360 个 ridge 头；缺失时不能按原参数重投影。 |
| `augmentation-selection-cifar100-run-20261102/dinov2_small/support_features.npy` | 120973952 | 原支持集九视图特征；原完整拟合/封存校验需要。 |
| `research-frofa-pilot-20261030/cache/features.fp32.npy` | 3010560128 | SigLIP 的 5000×196×768 FP32 特征；精确使用原特征训练/评估需要。 |
| `research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy` | 184320128 | CIFAR-100 全部 60000 张 RGB 图像；真实数据入口读取它。 |

按表顺序的 SHA-256：

```text
5e792bce7c8e3f1cf65a96fa652edb3d4ebb5473592b437eeabcc1ae705420fe
0838549168985a2a57793eeeda8519827cf431a1fa7e4a67b828bd56de8ee75b
ae3a3efd142cbff6a2806976e9e95c4ab1c5a0b0779500ff27730b27f22c6201
826e4c8a99d0344fec81a3317f3d4f41562ed3399ca995aa3b84dfc3e5da0dce
```

CIFAR 数组有确定性生成方法：固定原包中 train 后接 test，将 CHW 转为 HWC 后保存。原包 SHA-256 为 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`；来源记录见 `research-tta-conformal-20261030/PROTOCOL_ZH.md`，生成逻辑见 `research-view-route-20261030/cifar100-amendment-v1/manifest.py`。旧生成入口拒绝已有 manifest，应在独立目录按原版本重建并对照目标 SHA，不能直接在历史目录补跑。

另外三个数组可从准确的数据、模型、样本顺序、变换和环境重新计算，但没有验证跨硬件/版本逐字节一致。需要原实验精确结果时，应从自己保留的原件恢复；不要用合成 fixture 替代后宣称原实验重现。

## 环境、数据与模型

- 虚拟环境、`deps/`、`.python/`、wheel 缓存和本机二进制不随新增源码交付。依赖按固定版本和适用平台重新安装；Mac arm64 的 wheel/hash 不适用于全部操作系统。不要把历史“已安装”回执当作新机器的实际状态。
- DINOv2、DeiT、SigLIP、GroundingDINO、CountGD/BERT 等模型，从对应主题的固定模型 revision、来源清单和许可获取。校验配置、tokenizer 和权重组成，不能只下载一个权重文件就认为完整。
- EuroSAT、CIFAR、AeBAD、MILL、PairTally、DeepFish/Brackish 等数据按各自来源与条款恢复。图片许可、代码许可与派生标注许可分别核对；缺完整分发依据的材料没有因它能公开下载就被打包公开。
- 科学特征缓存与生成图片缓存也可能是精确复核的输入。它们不属于普通源码开发依赖，但省略会使相应原实验的完整校验失败。参数、预测、身份/类别映射、fold 划分和必要摘要按实际交付清单保留。
- Colab 或其他服务需要时在新机器独立认证；本仓库不提供旧 Cookie、令牌、账号额度或已登录会话。CLI 用法见 `colab-parallel-preparation-20261030/READINESS_ZH.md`，该文件是开发说明，不是原认证回执。

## 历史封存与当前开发版本

原始实验记录中的 `PASS`、`completed`、SHA、截止时间和路径属于当时的环境。某些检查还绑定旧 `.git/index`、Git HEAD、mtime，或会重写封存文件。本次文件选择及可移植路径调整不会使这些旧断言自动适用于新克隆。

先使用新开发入口验证已交付源码与合成测试。恢复真实资源时分别检查路径、大小、SHA、数组形状、样本顺序及标签映射；需要数值重投影的实验，另在隔离输出目录运行对应验证。缺文件应如实报告，不删除断言来制造通过结果。

`PORTABLE_DELIVERY_MANIFEST.json` 描述本次实际交付文件与省略边界；[路径派生记录](PORTABILITY_DERIVATIVES.json)记录路径调整的派生关系。旧 seal/manifest 保留作为历史参考，不是“所有被引用附件都已公开”的保证。

原始运行日志没有加入此次新增文件，即使有旧 seal 引用它们。来源记录中不需要的 Cookie/个人认证信息也未迁移。原件仍留在原机器。以后分享日志时，应先检查其具体字段，移除会话与身份信息，并明确它与原封存文件的差异。
