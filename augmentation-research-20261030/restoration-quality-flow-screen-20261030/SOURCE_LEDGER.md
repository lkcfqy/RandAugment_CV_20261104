# 来源账本

所有访问日期 2026-10-30 UTC。`raw/` 保存工具返回（包括失败和仅定位命中）；未下载完整数据集、模型或论文二进制。网页版本以以下 URL 为准。

| 来源 | 已核实范围 | 用途/边界 |
|---|---|---|
| [Self-supervised image denoising and restoration method for atomic force microscopy / SPIDER](https://www.nature.com/articles/s41467-026-76973-8) | 2026-08-20；Results/Fig.2，Methods/Data augmentation，Code availability | 方向性增强直接先例；官方代码 psichen/SPIDER 只定位，未审计 |
| [Masked and Shuffled Blind Spot Denoising for Real-World Images / MASH](https://arxiv.org/html/2404.09389v1) | Hamadi Chihaoui、Paolo Favaro；2024-04-15 v1；§3.2–3.3 | 相关噪声与内容保留的增强先例；未运行 |
| [Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review](https://arxiv.org/html/2308.00247v4) | 2024-03-25 v4；§V | 问题定位；不是“2026 最新综述”或新颖性证据 |
| [A Survey on Image Quality Assessment: Insights, Analysis, and Future Outlook](https://arxiv.org/html/2502.08540v1) | Chengqian Ma 等；2025-02-12 v1；§3–4 讨论 | 面向应用的质量标准，仅用于定位 |
| [JNDMix: JND-Based Data Augmentation for No-reference Image Quality Assessment](https://arxiv.org/html/2302.09838v1) | 摘要/引言 | 已知增强影响质量标签；不声称它做过本轮拟议的复合三元组干预 |
| [From Patches to Pictures (PaQ-2-PiQ)](https://arxiv.org/html/1912.10088v1) | §3、§4.2–4.4 | 图片/局部质量分别标注与建模已有先例；未下载数据 |
| [TRIQA: Image Quality Assessment By Contrastive Pretraining on Ordered Distortion Triplets](https://arxiv.org/html/2507.12687v1) | Rajesh Sureddi、Saman Zadtootaghaj、Nabajeet Barman、Alan C. Bovik；arXiv 2025-07-16 v1；ICIP 2025 DOI 10.1109/ICIP55913.2025.11084443；§3、Table 3 | 复合增强、距离三元组、已报告总体收益；代码固定版本见 IMPLEMENTATION_TRACE |
| [Degraded Reference Image Quality Assessment](https://arxiv.org/html/2110.14899v1) | Shahrukh Athar、Zhou Wang；2021 v1；§III-B/Fig.10、§V-A | 非单调失真先例、数据类型核对；本文分析含客观 SQB/FSIMc，不能全称 MOS |
| [dipIQ: Blind Image Quality Assessment by Learning-to-Rank Discriminable Image Pairs](https://arxiv.org/html/1904.06505v1) | Kede Ma、Wentao Liu、Tongliang Liu、Zhou Wang、Dacheng Tao；原工作 TIP 2017，所读 arXiv v1 为 2019；§III-A–B | FR 指标共识、不确定性排序权重；不是泛化“删不可靠对”新颖性空白 |
| [LIVE Wild Compressed Image Quality Database](https://live.ece.utexas.edu/research/twostep/index.html) | 官方下载说明与数据概况 | 需要表单后密码；未提交、未下载；不能写成已准备可运行数据 |
| [SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](https://arxiv.org/html/2405.14793v1) | Yihan Wang、Lahav Lipson、Jia Deng；ECCV 2024；§3.2、§4 设置 | 歧义与 MoL；官方源码固定并审计，未跑模型 |
| [Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation / ARFlow](https://arxiv.org/html/2003.13045v2) | Liang Liu 等；CVPR 2020，arXiv v2；§4.1–4.2 | 原/新遮挡区分、增强一致性；官方代码仅定位，不称源码双重验证 |
| [Reviving Unsupervised Optical Flow: Concept Reevaluation, Multi-Scale Advances and Full Open-Source Release](https://cv-stuttgart.github.io/Reviving-Unsupervised-OpticalFlow/) | Azin Jahedi、Marc Rivinius、Noah Berenguel Senn、Andrés Bruhn；WACV 2026；官方项目与代码链接 | 正文访问返回 403，方法/公式/是否覆盖干预未核实；不当精确反证 |

失败与材料缺口：官方 CVF 正文 URL `https://openaccess.thecvf.com/content/WACV2026/html/Jahedi_Reviving_Unsupervised_Optical_Flow_Concept_Reevaluation_Multi-Scale_Advances_and_Full_WACV_2026_paper.html` 返回 403；MDIVL 官方站点访问失败。没有绕过访问限制。搜索索引和第三方摘要只做定位。

近邻不等于完全相同干预。本轮的拒绝理由必须结合缺少具体新增预测/未具备可执行材料；不能仅凭标题相似或“存在前作”关闭全部家族。
