# 本轮排除/未决增量表

2026-10-29，只读。继承此前 `new-family-screen-20261029/EXCLUSION_LEDGER_ZH.md` 和后续独立家族报告；不覆盖旧记录。这里停止的是具体宽泛设想，不是判定整个研究家族没有机会。

| 窄设想 | 核实证据及定位 | 本轮决定 |
|---|---|---|
| “同一增强在不同量化位宽可由收益转为损害” | Data-Augmented Quantization-Aware Knowledge Distillation，arXiv 2509.03850v1，Table 6，PDF p8；CIFAR100 ResNet32 minimal/TA：W2A2 70.62/70.20，W3A3 73.70/74.09，W4A4 74.76/75.11 | **该现象已报告**，不重复包装成新边界。不是本地实验，不据此声称有显著差异或普遍反转 |
| “通用/光度/几何增强对量化有不同效果” | MetaAug，ECCV2024 Appendix Tables A.6/A.7；学生量化重构已参与增强优化；EWGS 固定源码显示量化范围本身可学习 | 简单加 RA 或只指出 clipping 不足以成为增量。SQuaT 又有 student-aware quantized features，宽泛教师—学生失配解释已存在 |
| “数据选择/主动学习应考虑增强轨道，而不是逐点代表性” | Data-Efficient Augmentation（2210.08363 摘要）；GRINCO（2607.01089v1 §1–2 定位）；When Dynamic Data Selection Meets Data Augmentation（2505.03809v1 §3） | 泛泛 selection×augmentation / orbit redundancy 已有直接问题设置；尚无区别于这些解释的具体新预测，不启动 |
| “增强让机器遗忘更好”或“用增强攻击重新测遗忘” | Data Augmentation Improves Machine Unlearning（2508.18502v1 §III–IV）已有七种增强和多种遗忘方法；Auditing Approximate Machine Unlearning for Differentially Private Models（2508.18671v1）已有 augmentation-based LiRA 与保留/遗忘隐私检查 | 原泛化主张不足。首篇 evaluation-pipeline 增强表述未由代码消歧，不称 bug；GLI 全文访问受限，不能说已排除所有相邻方法 |
| “长尾重复采样的类计数不等于有效信息，增强应按类调节” | Data-Centric Long-Tailed Image Recognition（2311.01744v1 §I–II）明确 observed/true support、信息多样性与分布偏移；DODA ICLR2024 §1–2、CUDA ICLR2023 摘要 | 具体宽泛动机已经存在。没有新定义可避免只是换重复率/增强强度网格，暂不保留 |
| “相机陷阱分类准确率与生态统计误差不一致” | Deep learning-based ecological analysis...（2408.14348 摘要）；Essential tools but overlooked bias（2025 DOI 10.1111/2041-210X.70132 摘要） | 改应用指标本身不是新增量；尚未验证增强特有、可识别的日夜/活动模式偏差，不虚构结果 |
| “细粒度标签被增强破坏而粗粒度标签保留” | Better Aggregation in Test-Time Augmentation（2011.11156v2 §3.3–3.4）已有部件/相似细粒度特征和类特定响应；DiffDA-Eval（2603.08364v1 §4.3 定位）已有层级粒度评价 | 泛泛 granularity×augmentation 不足。生成式增强的结果不能直接等同 RA；尚缺视图级标签保真证据，不能仅用本体层级保证粗标签仍真 |
| “TF/PyTorch Inception crop 分布差别、方向偏差或短预算下最优强度不同” | You May Be Running the Wrong Inception Crop（2610.04147v1 §3、算法1/2）已直接分析前两者；旧台账已禁止短预算机制同义重开 | 不立题，不改成实现 bug 项目 |
| “TTA 可改善 conformal set 或应另留 calibration split” | Test-time Augmentation Improves Efficiency in Conformal Prediction，CVPR2025 / 2505.22764v1 | 这两个一般说法已知；有限样本/动态预算方面仅做了定位，没有完整最近邻评审，不声称整个家族已关闭 |
| “增强多视图用于类别比例估计” | Unified View / DFM 已有任意映射和丰富特征；GMNet 已有相关性与图像代码 | 一般方法主张关闭。只保留报告中固定查询预算下的具体实证问题给 Pro 审核，训练准入仍为 0 |

“有前作”并不自动证明所有差分都已回答。表中没有强行把访问缺失写成无贡献，也没有把未执行的方向写成负实验。

本轮唯一数值复述是原论文表格，不是本地复现。所有搜索结果和代码快照见 `raw/` 与 `sources/`；最终采用的是原始论文/作者代码，不以检索摘要的二手解读替代正文。

