# PTQ covariance-pilot 当前版本关闭

日期：2026-10-30 UTC。依据：父任务转达的实际 Pro 最终评审。状态为 **NO-GO：理论与决策设计不足**；不是负实验，不是已由精确先例证明完全覆盖。Stage A 的 8,880 次前向和 Stage B 的 48 个量化网络均不运行。

## 保留的准确理由

1. 方差方向不保证量化器选择后悔值。对于两个候选、真实风险差为 delta 的选择，regret = |delta| × 错选概率（需明确 tie 规则）。均值 gap、尾部形状与 ties 影响错选概率；较低方差本身不够。
2. 强统计对照应使用同一 pilot 的 loss-difference 均值、方差、尾部信息；仅与 raw activation correlation 比较会弱化基线。
3. 五个 hook、多个候选范围的聚合和最终选择规则未冻结。不能事后挑位置、候选对或权重。
4. 镜像混合目标上的局部 MSE 不保证干净部署 NLL。跨越该目标差异不能靠方差恒等式推出。
5. 预算应准确拆开：共享 grid pilot 400 次单独记录；预测 pilot 800 + 单臂 calibration 160，需对比直接 calibration 960，而非只比较各臂末尾 160。更强基线还应直接用 pilot 选范围并复用其前向。
6. 12 次校准抽样重复是研究设计，不能当成 12 次真实部署来摊销 pilot 成本。
7. 等方差、同一相关系数 rho 的理想例子中，P16 的方差为 sigma^2(1+rho)/16，I96 为 sigma^2/96，因此前者更小需要 rho < -5/6。这只是代数示例，不是实际数据阈值或科学准入门；也不推出 selection regret。

## 读取边界和保留记录

Pro 未完整读取 CL-Calib / SelectQ；本地先前也分别仅有受限全文访问和官方摘要。因此不得把本次 NO-GO 写为这些精确先例已证明覆盖本题。CL-Calib 的官方 403 / OpenReview 验证页属于材料访问缺口；没有绕过。

原提案、公式推导、58 个文件归档与所有失败访问记录完整保留在 `augmentation-research-20261030/compression-calibration-screen-20261030/`。原 manifest SHA-256：`933c09742a889f684806529dc94ebf2c45ddac4b318d3550f75a4eace3a5d52d`。该材料后续仅作采样和 observer 备忘，不作为待执行候选。

本轮未运行训练、推断、observer、量化或新数值实验；未下载模型/数据、安装、付费、提交或推送。研究总项目继续独立家族只读筛查，不通过降低位宽、换 seed、加强增强或更换小校准集救此版本。

