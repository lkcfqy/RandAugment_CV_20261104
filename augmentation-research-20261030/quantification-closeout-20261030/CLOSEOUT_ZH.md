# 多视图响应类别比例估计：关闭记录

记录日：2026-10-30 UTC；父任务在 2026-10-29 末转达实际 Pro 最终结论 **NO-GO**。本任务随后独立核对下列一手来源，接受关闭。旧提案和原始证据原样保留，不改写成“从未提出”或“实证失败”。

权威后继状态：原 `augmentation-quantification-screen-20261029/REPORT_ZH.md` 中的“1 条待评审候选”现已关闭；当前该路线保留候选数为 0，训练准入为 0。没有运行其草案、模型或采样实验，不换 augmentation、数据集或下游模块挽救同一命题。

## 直接重叠证据

1. **Kernel Density Estimation for Multiclass Quantification（KDEy）**，Alejandro Moreo、Pablo González、Juan José del Coz，arXiv **2401.00490v2，2024-01-02**。§3 给出各一维边际相同但联合模式不同的明确例子；§4.1 Eq(3)及后续方法使用多元 KDE 表示后验概率。保留联合关系而非各维边际并非未覆盖的量化思想。\
   https://arxiv.org/html/2401.00490v2

2. **Generalized Bayes Quantification Learning under Dataset Shift（GBQL）**，Jacob Fiksel、Abhirup Datta、Agbessi Amouzou、Scott Zeger，arXiv **2001.05360v2，2021-02-17**（PDF封面另标2021-02-18）。§2.4（印刷页9–10）定义同一实例多个分类器输出的 ensemble 损失；§3.3（印刷页18）明确这些输出不独立、共享已标注实例，理论容纳这种依赖。\
   https://arxiv.org/pdf/2001.05360v2

3. **GMNet** 的联合特征/协方差表示与固定官方 CIFAR 入口已在前报告核查。它和上述来源共同削弱了“普通特征丰富化加同预算比较”作为独立科学增量的理由。\
   https://arxiv.org/html/2501.13638v1 \
   固定代码：https://github.com/AICGijon/gmnet/tree/f7ac591dfd63c755c00d6fa440ab563268d97896

准确边界：GBQL 并不是本提案同一 TTA 联合 KDE 实现；KDEy 的直接例子是同一分类器的多类输出。不能声称这些论文逐项运行了我们的4-view配置。关闭理由是提案尚无超出已知丰富特征/相关输出解释的不同预测，不能靠换输入来源和普通成本比较自动产生贡献，不是“所有增强的量化用途永远无效”。

## 不能用打乱视图冒充机制识别

原始同图联合分布为 P(z1,z2)=sum_y pi_y P(z1,z2|y)。在大目标池中跨图片独立重配视图，接近 P(z1)P(z2)，其中含 y与y' 的双重混合 pi_y*pi_y'；有限不放回打乱还有有限样本修正。这样同时打掉共同类别和同图依赖，不是只删除某个增强机制。下降不能隔离提案所称因素；换新 shuffle 或下游任务也不解除该问题。

原报告已提醒该风险；现明确不得将其做成“对照通过”的证据。理论表述只作识别边界说明，不是新增研究结果。

## 保存及后续边界

原提案归档 manifest SHA-256：`86faa2d4f48eb75980c0af0d8c30242699b207b5559835fe4ebfa98628382794`。本关闭记录独立保存；归档时核验原 manifest 仍一致。

本轮继续其他增强家族的只读筛选；不会再调整本提案的视图、估计器、样本量、数据集或阈值。没有训练、模型下载、费用、推送或访问封存测试。

