# 原始来源与阅读覆盖

截止 2026-10-20。`正文`表示本轮阅读相关方法／讨论，不表示每一表格、代码和实验数字均已复核；`摘要/入口`不足以支持未读细节。文献检索通过公开原文，数据核查通过官方页面与仓库；没有使用综述代替关键方法证据。

## 已用作主要判断的来源

| 编号 | 原始来源、版本或状态 | 实读范围 | 支持的有限事实；不能外推的内容 |
|---|---|---|---|
| P01 | [AutoAugment](https://arxiv.org/pdf/1805.09501)，作者 v3，2019-04-11，CVPR 2019 | §3 | 子策略、顺序操作、概率／幅度及验证反馈搜索；不是自然因果识别 |
| P02 | [RandAugment](https://papers.nips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)，NeurIPS 2020 正式 PDF | §3–4，Algorithm/图3；前轮本机12页可读核验 | N/M、等概率有放回抽样、操作尺度；另研究变化强度方案，不能说所有 RA 实现固定 M |
| P03 | [TrivialAugment](https://arxiv.org/pdf/2103.10158)，作者 v2，2021-08-17，ICCV 2021 | §3 Algorithm1、§4 | 随机单操作和随机强度；定义范围不自动适合任意任务 |
| P04 | [CGN](https://www.cvlibs.net/publications/Sauer2021ICLR.pdf)，Sauer/Geiger，ICLR 2021 | §2–3 | 形状、纹理、背景机制的归纳偏置与组合；不保证自然因素唯一恢复 |
| P05 | [ReLIC](https://arxiv.org/pdf/2010.07922)，2020 作者版 | §2 内容／风格／标签假设与方法 | 内容保持及独立假设下的增强不变性；本轮不以未经核验的会录作为证据 |
| P06 | [Noise or Signal](https://arxiv.org/pdf/2006.09994)，Xiao等，2020 | §2、附录 D 控制 | BG-Gap、Mixed-Same/Rand；处理伪影控制早有先例 |
| P07 | [GroupDRO/Waterbirds](https://arxiv.org/pdf/1911.08731)，Sagawa等，ICLR 2020 | 方法与附录 C | Waterbirds 工程偏置和组信息；半合成不能直接证明自然世界背景因果效应 |
| P08 | [ALIA](https://arxiv.org/html/2305.16289v2)，v2 2023-10-29 22:52:04 UTC；v1 2023-05-25 | §2–3、§5.4、§6、附录10.1–10.2 | CLIP语义与高置信度编辑过滤、两编辑器比较；未确认配对 sham、真实组×人工有效性×留存统计。正式会录未在本轮独立核验 |
| P09 | [AGA会录](https://openaccess.thecvf.com/content/WACV2025/html/Rahat_Data_Augmentation_for_Image_Classification_using_Generative_AI_WACV_2025_paper.html)，WACV 2025，4173–4182；[作者v1](https://arxiv.org/html/2409.00547v1)，2024-08-31 | 会录索引；根线程补读作者§3–4、§5.3、§6 | 分割、生成背景与合成，作者承认上下文不兼容；SIC是使用已有PIC的softmax information curve，不是新前景保持指标。正式详情403，正式版差异未比对，未逐项复核数值 |
| P10 | [DDB作者 v2](https://arxiv.org/html/2503.17226v2)，2025-07-27 14:26:14 UTC；[ICCV 2025 会录](https://openaccess.thecvf.com/content/ICCV2025/html/Parast_DDB_Diffusion_Driven_Balancing_to_Address_Spurious_Correlations_ICCV_2025_paper.html)，17526–17535 | §2、§4、§5.3表2–5、附录9.2–9.4；会录索引可核，详情曾403 | 改前景及标签、原类概率／IG过滤、已有剪除计数；不应误称保持原标签或统一低置信度删除 |
| P11 | [From Fake to Real](https://arxiv.org/html/2308.04553v4)，作者 v4，2026-08-08；首次2023，正式会录本轮未核 | 摘要、引言、§3–4.4、附录 B–E 相关文字 | 合成来源与偏置联合捷径、分阶段训练、质量退化实验；未完整审计全部实验，不能宣布缺少任何某种控制 |
| P12 | [Generative Randomization + Cross-Variant SSL](https://arxiv.org/html/2607.05850v1)，v1 2026-07-07 05:13:26 UTC；[摘要元数据](https://arxiv.org/abs/2607.05850v1)自报CVPR2026 GCV workshop接收 | §2–3、相关工作、附录D–F | 背景生成、同主体正对、SSL、warmup、GroupDRO已组合；FLUX、贴回、检测过滤。未见本文所拟完整交叉设计；不据此保证新颖 |
| P13 | [SAGE](https://arxiv.org/html/2609.01051v1)，v1 2026-09-01 10:47:11 UTC，预印本 | Related Work、Method、消融、附录D开头 | 少数簇生成、合成验证集、DFR；簇不等于真实少数组。未逐项复核消融数值，接收状态待核 |
| P14 | [DSCM](https://arxiv.org/abs/2006.06485v2)，v2 2020-10-22，元数据为NeurIPS2020 | 摘要／模型概述 | 可在给定结构及机制假设下推断外生噪声；非无假设识别证据 |
| P15 | [CausalDiffAE](https://arxiv.org/html/2404.17735v3)，v3 2024-08-23，元数据ECAI2024 | 引言、方法结构、因素监督描述 | 图／因素辅助信息下的反事实生成；不支持纯观测图像无假设恢复 |
| P16 | [Diffusion Counterfactual Generation with Semantic Abduction](https://proceedings.mlr.press/v267/rasal25a.html)，ICML2025，PMLR267:51201–51228；[作者正文](https://arxiv.org/html/2506.07883v1) | 引言、方法框架、Limitations | 噪声反演、身份／控制权衡；Markovian等假设、可识别性及噪声限制 |
| P17 | [Causal-Adapter](https://arxiv.org/html/2509.24798v6)，v1 2025-09-29，v6 2026-05-14，作者journal-ref ICML2026 | 引言、计算说明、附录F–G | 冻结扩散加因果控制；反演近似、恢复不推出识别、图误设和不完整图讨论。未独立核正式会录，作者A100速度不移用于本机 |
| P18 | [Counterfactual (Non-)identifiability of Learned SCMs](https://arxiv.org/pdf/2301.09031)，Nasr-Esfahany/Kiciman，2023 | 摘要、引言与反例要点 | 已知图／无隐藏混杂仍不足以一般识别个体反事实；不声称所有特定任务都不可识别 |
| P19 | [Bias Challenges in Counterfactual Data Augmentation](https://arxiv.org/abs/2209.05104v2)，v2 2022-09-13，元数据UAI2022 CRL workshop | 摘要／元数据 | 生成上下文猜测可能引入偏差；NLP例子不是图像过滤问题已被证实的证据 |
| P20 | [On Counterfactual Data Augmentation Under Confounding](https://arxiv.org/html/2305.18183v2)，v2 2023-11-21，非2026新论文 | §4–6、Algorithm1、附录B相关讨论 | 条件生成与混杂设定的分析／构造测试；不支持自然混杂自动被识别。主报告不据其报告性能数字 |
| P21 | [ForAug: Mitigating Biases in Image Classification via Controlled Image Compositions](https://arxiv.org/html/2503.09399v4)，[v4元数据](https://arxiv.org/abs/2503.09399v4)，2026-07-01 | 独立审查者实读§3.1–4.3、附录A/G/L/M和相关表格文字；根线程复核正文／附录相关返回 | 背景策略、两修补器、边缘平滑、掩码扰动、自然OOD及分类型过滤留存已有。没有逐幅视觉检查／运行／复算；未读到完整锁定配对检验不证明它不存在 |

## 官方数据与模型入口

| 编号 | 来源 | 实际核验／未完成部分 |
|---|---|---|
| D01 | [Waterbirds loader](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/waterbirds_dataset.py)；[生成代码](https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_scripts/generate_waterbirds.py) | 代码已读，字段、训练常数可见；没有实际CSV、图片或完整性验证 |
| D02 | [CUB官方](https://www.vision.caltech.edu/datasets/cub_200_2011/)；[CaltechDATA](https://data.caltech.edu/records/65de6-vp158) | 官方页面、数量和使用说明已读；没有下载图像或分割包 |
| D03 | [Places条款](https://places2.csail.mit.edu/download-private.html) | 官方索引摘录可见；全文超时／502，不能宣布衍生数据发布许可已确认 |
| D04 | [backgrounds_challenge](https://github.com/MadryLab/backgrounds_challenge)；[九类](https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/master/in9_classes.txt)；[映射](https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/master/in_to_in9.json)；[ImageNet条款](https://www.image-net.org/download.php) | 小型类别／映射／loader已读；LICENSE 404、附件未载、没有配对manifest，图片许可待确认 |
| D05 | [WILDS官方](https://wilds.stanford.edu/datasets/)；[iWildCam loader](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/iwildcam_dataset.py)；[作者论文](https://arxiv.org/html/2012.07421v3) | 官网页面和loader已读，论文数量取作者正文索引；没有实际CSV，182/186类冲突待核 |
| D06 | [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/)；[WILDS MIT](https://raw.githubusercontent.com/p-lambda/wilds/main/LICENSE) | 条款已读，软件与图片许可区分；未代替法律意义上的所有权审查 |
| M01 | [SD1.5 inpainting镜像](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) | 仅模型卡，OpenRAIL-M与非原厂镜像说明可读；没取权重或推理 |
| M02 | [SD2 inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 401 Unauthorized；停止，未绕过 |

## 检索收据与待补覆盖

已使用原论文引用链、精确标题和 arXiv/CVF/PMLR/OpenReview 主域检索。2026-10-20 再针对候选差异执行的六个表述为：

1. `"background augmentation" "artifacts" diffusion`
2. `"synthetic" "spurious correlations" "editor"`
3. `"background" "sham" "augmentation"`
4. `"augmentation" "selection bias" "image"`
5. `"confidence filtering" "spurious correlations"`
6. `"synthetic data" "filtering" "minority" image diffusion`

第3项返回大量作者姓氏 Sham 等噪声，不能据“没有精确命中”下不存在结论；词面检索不构成完备查新。部分长结果被截断；未见部分不算已读。2024–2026与最近六个月版本通过标题／版本记录定向核查，未使用搜索排序当时间证据。

追加发现后的核查状态如下；不把检索片段当完整覆盖：

- ForAug 已提升为实读直接先例 P21，v4在截止日前。其覆盖使候选一差异进一步收窄。
- [OntoAug，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_OntoAug_Rethinking_Generative_Data_Augmentation_via_Ontology_Guidance_CVPR_2026_paper.pdf)：官方索引片段确认条目，涉及前景掩码、背景提示；独立审查者读PDF、HTML、会录索引均403，未取正文，具体控制和精确公开日期未核。已停止追加尝试，不能排除更近重合。
- [Uncertainty-Guided Synthetic Training Data Augmentation](https://arxiv.org/html/2606.31603v1)，2026-06-30检索结果；是否涵盖候选的有效样本误拒尚未核验。
- [Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](https://arxiv.org/html/2410.13674v4)，2025-09-26；片段中已有CLIP过滤阈值，尚未读完整方法。

因此报告使用“候选／待检差异”，不使用“首个”“无人做过”“通过完整查新”。报告没有引用这些未深入来源的实验数值。
