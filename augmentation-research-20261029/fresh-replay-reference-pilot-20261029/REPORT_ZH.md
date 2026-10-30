# Fresh-replay 参考组合：基线学习不足，未进入机制门

2026-10-29，seed=20261029。**已实际执行760个共同前缀 optimizer steps；任务2学习门未通过，因此门A和五个64步短分支均未运行。** 本结果只说明这套预定小型参考配方不足以支持本轮机制检验，不是候选机制失败，也不支持 A2ER 原论文受增强误导的归因。

本轮对象明确是新的 snapshot fresh-replay DER + residual sampling 组合：真实回放重新裁剪／翻转原图，A2ER来源的Q/eta/gamma公式仅用于采样权重更新，target correction和alpha/beta自适应关闭，当步损失不另乘gamma。所有条目在任务1第600步从同一模型快照取得历史target、同龄；这些设计差异在执行前写入协议。原生A2ER路径审计及旧负结果未改动。

## 固定设计与验收

官方CIFAR100 TRAIN前10个fine class，0..4为旧任务、5..9为新任务；每类400 fit、100 dev。10类共同分类头，没有task mask。旧类buffer512，类别数103/103/102/102/102；与开发集无交集。没有读取官方test或旧封存测试。

175,258参数的三stage小型GroupNorm ResNet，无BN/Dropout；任务1 SGD lr=.05训练600步，任务2 lr=.02训练160步。标准裁剪padding4、水平翻转p=.5，未用RA或强增强。此变换来自固定[Mammoth CIFAR100源码](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/datasets/seq_cifar100.py)。协议与脚本在18:49:19 UTC冻结，原始SHA-256复核未变。

- 裁剪／翻转实现与torchvision固定参数像素比较最大差0。
- 方差恒等式误差1.78e-15；Q及权重已知例通过。
- CPU/MPS该模型logit最大差2.38e-7，参数梯度最大差6.85e-7；均通过执行前阈值。
- GPU吞吐验收首16步属于原共同前缀，未重跑；暖态步时中位数0.007016秒。

## 预定学习门的实际结果

| 检查点 | 旧类dev NLL | 旧类准确率 | 新类dev NLL | 新类准确率 |
|---|---:|---:|---:|---:|
| 初始化 | 2.346701 | 20.0% | 未评估 | 未评估 |
| 任务1，第600步 | 1.077999 | 53.6% | 未评估 | 未评估 |
| 共同分支点，第760步 | 1.310241 | 50.0% | 3.154597 | 10.0% |

任务1达到预定的准确率≥50%、NLL≤1.5、较初始化降低≥.30 nat。分支点要求旧／新类各准确率≥35%、NLL≤1.8；新类未达到。最后32步训练均值新类CE=3.080429、旧类CE=1.226589、logit MSE=2.535429，没有NaN或Inf。

从保存的logits以独立NumPy log-sum-exp重算开发NLL，最大逐例误差<9e-16，准确率一致；训练记录恰为600+160步，fit/dev和buffer/dev均不相交。因此停止原因是观察到的基线学习不足，不是数据标签错配、未运行优化器或评分文件缺失造成的伪结论。这里没有诊断性证据将学习不足单独归因于降权、蒸馏强度或步数；需要新的预注册基线设计才能区分，不在本轮追调。

**没有门A的TV/稳定性结果，没有N/A/B/R/U性能比较，也没有uniform优越或新分配有效的证据。** 不将预定投入门偷换为成功条件，不追加步数、换学习率、checkpoint或seed来救结果。停止时仍有时间预算；它不是超时。

## 运行时间、状态和产物

- 设备：Apple M5 Pro的MPS，float32；torch2.13.0、numpy2.5.4。CPU fallback关闭。
- 实际进程18:49:50.460–18:50:00.071 UTC，执行墙钟 **9.610949秒**，包含验收、数据读取、训练和评估；任务1段4.825335秒、任务2段2.596738秒。MPS训练步同步计时。此前约8分钟用于来源、协议和脚本准备，不计为GPU训练时间；全部在19:11:01 UTC硬截止内完成。
- 进程peak RSS713,900,032字节；结束时MPS allocator22,870,528字节、driver214,908,928字节。这些不是独立可相加的物理内存峰值。
- `run-001/summary.json`：最终状态 `baseline_insufficient_branchpoint`。
- `run-001/training-losses.jsonl`、`progress.jsonl`：完整760步损失和阶段日志。
- `data-and-sequences.npz`、`buffer-at-capture.npz`、`prefix-replay-history.npz`：划分ID、两任务序列、历史target来源／增强参数、实际回放ID与最终权重。
- `task1-checkpoint.pt`、`common-checkpoint.pt`：自训练模型及优化器状态，不是下载权重。
- `initial-old-dev.npz`、`task1-old-dev.npz`、`branchpoint-dev.npz`、`verification.json`：逐例输出和独立指标复核。

原数据和旧实验保持；没有新安装、Colab、提交或推送。下一步不是把此候选记作科学NO-GO，而是由父任务决定是否值得另立一次有明确理由的基础学习配方；本轮授权的一套配方已经完整执行并如实停止。
