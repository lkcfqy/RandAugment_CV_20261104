# 首选问题的最后可行性检查

日期与文献截止：2026-10-20。只针对“跨编辑器的增强收益能否迁移至自然 OOD”收口；不追加泛搜、训练或推理。核验原始记录见 [focused-feasibility-gate.json](focused-feasibility-gate.json)。

**明确结论：当前数据门尚未通过，不能把跨编辑器→自然 OOD 的最小实证验证称为可直接启动。可继续做 manifest 与人工审查准备；尚没有取得样本级配对清单、可用图像／掩码与自然测试元数据，不能报告机制或收益。** 小型类别元数据确实已读取，但它不等于样本级配对。研究问题可保留为条件可行，先不扩大算力。

## 最贴近主问题的三项先例

这里按与主问题的接近程度提取已有报告中的三项，不把2026修订版误称三篇2026首次发表的新论文。

| 原始来源及本轮沿用的阅读范围 | 已有精确覆盖 | 仍待检验的差异与边界 |
|---|---|---|
| [ForAug: Mitigating Biases in Image Classification via Controlled Image Compositions，v4，2026-07-01](https://arxiv.org/html/2503.09399v4)，§3–4、附录G/L/M | 原／同类／全类背景，两种修补器，边缘平滑、掩码扰动，自然OOD和分类型过滤留存 | 已读部分未确认“同一前景的sham／去相关配对对比预先锁定后，在预留编辑器上重做完整训练对比”的联合检验。仅换编辑器、加自然测试或边界处理已不足构成贡献 |
| [Breaking Spurious Correlations via Generative Randomization and Cross-Variant Self-Supervised Learning，v1，2026-07-07](https://arxiv.org/html/2607.05850v1)，§3、附录D–F；作者自报workshop接收 | FLUX背景生成、前景贴回、检测过滤，同主体跨背景正对、SSL、ERM warmup与GroupDRO | 背景随机化＋一致性组合已经存在；已读部分未确认配对sham及预留第二编辑器的锁定训练复验。可检验的是其收益在控制流程变化后能否复现，而非重新命名组合 |
| [From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Spurious Correlations in Image Recognition，v4，2026-08-08](https://arxiv.org/html/2308.04553v4)，§3–4.4、附录B相关文字；首次提交2023 | 已提出真实／合成来源与偏置联合捷径；用合成预训练、真实微调及质量退化分析应对 | “伪影也会成为捷径”不是新发现。剩余问题是配对协议及跨编辑器的训练风险差，是否揭示分阶段训练等已有控制未解释的失败 |

这些是待检差异，不是“未被任何人做过”的结论。OntoAug正文此前403，未重新尝试；它及完整引用链仍限制查新判断。若最近先例已解释锁定后的全部效应，停止提出新增方法。

## 实际元数据、配对与许可核验

| 对象 | 本轮实际取得什么 | 能否支持样本配对与使用 |
|---|---|---|
| IN-9 [官方Release索引](https://api.github.com/repos/MadryLab/backgrounds_challenge/releases?per_page=5)、[九类表](https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/master/in9_classes.txt)、[映射](https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/master/in_to_in9.json) | 三次HTTP200。分别3,344／198／10,520字节；实际解析9类和1000项ImageNet类别索引映射。Release仅列一个279,581,133字节的tar.gz，无单独CSV/JSON/TXT清单附件；未下载该图像包 | **未取得样本级配对。** 1000项是类别映射，非1000张图；附件索引未列清单不证明压缩包内部没有清单。不能按目录顺序假定同前景，也未确认背景来源ID。图片许可链仍待核；仓库代码可读不是图片再分发授权 |
| Waterbirds [官方loader](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/waterbirds_dataset.py)、[构造代码](https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_scripts/generate_waterbirds.py) | 代码定义CSV字段和CUB `img_id`／背景来源字段；对官方bundle根路径下推导的`metadata.csv`子路径发起有上限GET，在HTTP响应前因TLS证书过期失败，没有CSV行 | **不能确认包内实际字段、组数或跨split复用。** 此失败也未验证推导子路径是否存在。`img_id`是源图像ID，非同一物理鸟的身份。标准合成图还不足以代替原始前景与掩码 |
| iWildCam [官方loader](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/iwildcam_dataset.py) | 代码定义filename/y/split/location/sequence/time；同样的有限CSV请求在TLS阶段失败，没有取得实际行 | **自然任务未就绪。** 位置不是单独背景变量，sequence不是动物身份；没有掩码或背景配对标签。182／186类别版本问题仍需实际CSV解决 |

两次CodaLab失败原因为 `CERTIFICATE_VERIFY_FAILED: certificate has expired`，均未获得HTTP响应或数据；没有关闭证书校验、换HTTP或访问其他会话路径。不是有待批准的训练，也不是结果不明的下载。

许可方面：[CUB官方](https://www.vision.caltech.edu/datasets/cub_200_2011/)明确限非商业研究教学，且不拥有全部图片版权；[Places完整条款入口](https://places2.csail.mit.edu/download-private.html)本轮仍502，Waterbirds的完整来源约束未核清。iWildCam loader明确 [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/)，条款可读，使用和发布需遵守其中归属／修改通知要求；这不解决当前CSV与配对缺失。未下载图片、权重或完整数据集。

## 不把流程差异冒充机制

E1和E2可能同时改变主体细节、背景物体、可识别性、难度与边界痕迹。相同提示、掩码和seed不保证相同图像内容；同背景类别的sham也不保证相同难度。因此，跨编辑器一致或自然OOD提升，最多是**指定训练流程收益的可迁移证据**，不能单独识别“纯背景语义机制”。

最简单的控制是：同一源前景图像配对，固定前景像素、掩码、裁剪、分辨率和目标背景池；增加直接贴图与同语义sham；在每个Y内独立均衡分配背景而非一律反转；保留全部分配、失败和过滤记录，人工分开审查标签、目标变化和处理损坏。原图难度可预先分层，但不能根据编辑后的模型置信度挑出看似难度相同的样本，再宣称排除了混淆。上述控制仍不能保证编辑内容与伪影完全分离。

固定分类器的配对损失只作诊断；增强训练的收益要用不同训练运行比较。E2输入压力测试不能代替E2完整训练复验。自然任务须在iWildCam自己的train/val/test重新建立对照，不能从Waterbirds分类器直接跨不同标签空间评分。

## 进入最小实证验证前的通过条件

首先取得并核对至少一个机制数据源和一个自然任务的实际元数据。配对manifest至少要有：源前景图像ID及哈希、掩码哈希、背景文件与来源、目标背景类别、split、编辑器版本、处理分配、输出哈希、人工实际标签／背景及失败原因。分组统计是实际结果，不能仅写字段模板。

只有确认前景变体不跨split、背景来源留出可执行、两类背景在各Y有支持、失败／过滤未被静默丢弃，并取得可用图像与掩码和适用研究使用条件后，才能称数据关通过。当前没有这些实际配对行，故结论仍是 **NO-GO：暂不启动原定最小实证验证**。这不妨碍继续准备清单和审查规则；本轮收口到此，不运行模型或GPU。
