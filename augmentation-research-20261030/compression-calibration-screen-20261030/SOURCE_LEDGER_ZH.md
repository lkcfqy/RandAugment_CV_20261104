# 来源账本与读取边界
记录日期：2026-10-30 UTC。原始工具返回保存在 raw/。搜索结果仅用于定位，不作为“未有前作”的证据。未获得全文的条目明确标记。

| 来源 | 固定版本/链接 | 实际读取与本轮用途 |
|---|---|---|
| FKD — Zhiqiang Shen, Eric Xing | [arXiv 2112.01528](https://arxiv.org/abs/2112.01528)，[作者 ECCV2022 camera-ready](https://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)；本轮全文读取 [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2112.01528) | §3.1/3.3/3.5，Eqs3–7，Tables2/6/7，§4.1。缓存记录变换并重现；压缩和同图多crop已经实验。主实验通常基本crop/flip，不能泛化为所有RA条件已测。 |
| Soft Label Pruning and Quantization for Large-Scale Dataset Distillation — Lingao Xiao, Yang He | [2604.18135v1](https://arxiv.org/html/2604.18135v1)，HTML标注2026-04-20；TPAMI DOI10.1109/TPAMI.2026.3664488 | §III-C–E，Eqs8–14/Algorithm1。保存top-k pre-softmax logits、减少保存增强对、温度调度和学生温度。直接覆盖H2泛化预算权衡。没有核验作者实验代码，不评价其实现是否忠实。 |
| Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance — Zhihang Yuan et al. | [2303.13003v1](https://arxiv.org/pdf/2303.13003v1)，[版本页](https://arxiv.org/abs/2303.13003) | §3.2和§4.1，PDF pp4–6/Table1。noise、inter-class imbalance、sample-size以及多校准种子的逐类性能；不把摘要中的augmentation字样当作H1已经测过。首次误用v2 HTML为404，版本页确认仅v1，已纠正。 |
| MetaAug — Cuong Pham et al. | [2407.14726v1](https://arxiv.org/html/2407.14726v1)，ECCV2024 | §2/3和附录A.2 TablesA.5–A.7、A.3 TableA.8。**A.6/A.7是表号，不是章节号**。已查通用、光度、几何增强及校准图像规模；未找到匹配边际的配对预算实验，不作穷尽性证明。 |
| A Group-Theoretic Framework for Data Augmentation — Shuxiao Chen, Edgar Dobriban, Jane H. Lee | [1907.10905v4](https://arxiv.org/html/1907.10905v4)，2020-11-06；[JMLR原文PDF](https://jmlr.org/papers/volume21/20-163/20-163.pdf) | §3.4、4.2、4.3；轨道平均、反变量、ERM/M-estimation方差理论是强理论前作。不得把本轮等预算协方差代数当创新。 |
| SelectQ — Zhao Zhang, Yangcheng Gao, Jicong Fan, Zhongqiu Zhao, Yi Yang, Shuicheng Yan | [官方期刊页](https://www.mi-research.net/en/article/id/035c9cbd-8934-43b1-9235-6eacdb207004)，MIR2025 22(3):499–510，DOI10.1007/s11633-024-1518-0 | **官方摘要**：利用激活统计和动态聚类选择校准来源。未取得可阅读全文/官方代码，本轮不宣称精确消融排除。搜索有2022 TechRxiv版本，不把2025期刊年误当首次公开年。 |
| Enhancing Post-training Quantization Calibration through Contrastive Learning / CL-Calib — Yuzhang Shang, Gaowen Liu, Ramana Rao Kompella, Yan Yan | [CVPR2024正式页](https://openaccess.thecvf.com/content/CVPR2024/html/Shang_Enhancing_Post-training_Quantization_Calibration_through_Contrastive_Learning_CVPR_2024_paper.html)，pp15921–15930；[OpenReview](https://openreview.net/forum?id=Wxyyc2vvGd) | **全文未取得**。正式HTML/PDF请求403；OpenReview跳验证页。仅确认题名/作者/会议和定位，不能声称其具体augmentation对照不存在；没有绕过访问限制。 |
| On Calibration of Modern Quantized Efficient Neural Networks — Joey Kuang, Alexander Wong | [2309.13866v2](https://arxiv.org/html/2309.13866v2)，2023-09-26 | §2.1/3：固定1024校准样本、3次抽样、多个bit配置，研究校准质量/TS。与PTQ配对来源分配不同，但阻止把“量化后置信度变化”当新增发现。 |
| Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized Language Models — Zhen Yang et al. | [2608.21019v1](https://arxiv.org/html/2608.21019v1)，2026-08-21 | §3.1–3.3及摘要：分布/边界保持风险与目标依赖的校准组合。语言模型任务，不能直接证明图像H1；也不能把通用“score fidelity比accuracy更重要”重提。 |
| When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization — Yanlong Zhao et al. | [2610.11226v1](https://arxiv.org/html/2610.11226v1) | §1/2/3.1。已有降低重构损失不保证下游提升的实测与分析。H1需要下游门，不能只做MSE。 |
| DQAKD、SQuaT | [2509.03850v1](https://arxiv.org/html/2509.03850v1)，[SQuaT](https://arxiv.org/html/2608.10709v1) | 复用并复核先前排除：DQAKD按teacher的DEV/CMI评估DA并做QAT/KD；SQuaT teacher/student量化范围。未将这两者当H1独立来源采样实验。旧排除文件保留。 |

## 官方代码可定位事实
源码是文献/实现准备，不执行导入。
- pytorch/pytorch tag v2.13.0 解析为 **cf30153c4c131c8164ee7798e5022d810682e2cb**。
  [observer.py](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/ao/quantization/observer.py)：
  :987 HistogramObserver，:1002 histogram动态range，:1055 dtype决定dst_nbins，:1076 L2近似，:1130非线性搜索，:1289更新，:1349 qparams。
  本候选默认8-bit完整range；不擅自通过缩quant_min/max假设得到了忠实低bit搜索。
- pytorch/vision tag v0.28.0 解析为 **8fb87713a24951e639c494b0f2a8a81b5f8e33a6**。
  [functional.py](https://github.com/pytorch/vision/blob/8fb87713a24951e639c494b0f2a8a81b5f8e33a6/torchvision/transforms/functional.py):654入口；
  [_functional_tensor.py](https://github.com/pytorch/vision/blob/8fb87713a24951e639c494b0f2a8a81b5f8e33a6/torchvision/transforms/_functional_tensor.py):124镜像直接flip(-1)。
- sources/ 收录固定commit的observer、tensor helper及两仓库LICENSE；Git blob逐项校验见SOURCE_FILES.json。入口完整工具返回保留raw/。
- 现有本地训练基础只读了协议、结果报告和manifest；没有为本轮加载checkpoint。引用的model SHA来自既有存档并再次以文件hash核验，未反序列化。

## 本地旧材料
- augmentation-research-20261030/ocsr-closeout-20261030/MANIFEST.json：MolScribe否决与MolSight/Delta更正，保留。
- augmentation-research-20261030/augmentation-family-screen-20261030/MANIFEST.json：此前三家族筛查保留。
- augmentation-research-20261029/frozen-derpp-gate-a-20261029：指定fresh-replay组合已NO-GO，本轮不读取buffer预测或重开。
- 本轮另存PRIOR_RECORD_HASHES.json，核验manifest和checkpoint字节未变。此项不是重做旧科学验证。

