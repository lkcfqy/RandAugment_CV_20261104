# SRA 样本—动作配对候选：最终门槛审查，2026-10-29

**结论：冻结，NO-GO。** 当前剩余问题是一个更严格的消融设计，尚不是有区别于现有解释的预测的研究方向。实质性门槛失败后，本轮停止扩大检索、追查代码和实验设计；没有训练、安装、下载数据/模型、运行网络或改动旧记录。SADA 保持上一阶段冻结状态。

| 门槛 | 结果 | 理由 |
|---|---|---|
| 1. 实质问题 | **FAIL** | 最强现有表述仍是：给定同一完整动作多重集，正确实例配对是否优于打乱配对。它可证伪，却只是检验既有实例适应性解释的消融；没有预先说明新的机制、条件、方向性预测，能与该解释区分。 |
| 2. 可识别性 | **UNVERIFIED** | 已恢复核心静态路径，但没有完整动作记录/重放和状态控制验证。两个独立在线臂不能保证相同动作边际；仅打乱强度分数也不等于排列完整动作。 |
| 3. 证据与成本 | **FAIL（执行就绪性）** | 当前项目没有模型权重、所需训练照片/掩码；可重放 SRA donor 轨迹未验证存在。全成本及验证协议未锁定，不能把 donor 当免费已有资源。 |

这里的 FAIL 是候选成熟度判断，不是证明配对无效，也不是认定精确消融已被别人逐字做过。

## 最近先例：足以停止，未做穷尽查新

- **PBA，ICML 2019，§4.2、Table 3，PDF 第 8 页**：[官方论文](https://proceedings.mlr.press/v97/ho19b/ho19b.pdf)。比较最后固定策略、保留持续时间但打乱策略顺序、以及按持续时间混合的平稳策略。它检验时间安排，不是逐批完整动作的实例配对；表中消融单次评估，部分组合未跑。不能把它称为本方案的完全复现，但“分布与安排拆开检验”不是新问题。
- **MADAug，arXiv:2309.04747v2，§5 “Strategy of MADAug”、Table 7**：[版本固定正文](https://arxiv.org/html/2309.04747v2#S5)。已拆分 model-adaptive、data-adaptive 与二者结合；§5 Table 6 另测 monotonic curriculum。它没有在所读位置保证逐批完整动作边际完全相同，不能冒称其已回答本候选的严格因果量；但实例/模型适应性解释已有直接消融。
- **InstaAug，arXiv:2206.00051v3，§5.2、Table 1 与 Appendix E.3**：[版本固定正文](https://arxiv.org/html/2206.00051v3#S5.SS2)。对比 instance-specific、without-input 的共享分布和 class-specific；另解释全局随机裁剪基线经过较充分调参。关闭输入会改变学得的分布，仍非逐批全动作匹配。不过“局部不变性需要实例依赖”已有明确预测与控制。

因此，本候选没有通过“非平凡新预测”门槛。没有依据把更严格的 shuffle 控制本身包装为新方法/新机制，也不据有限检索宣称精确实验不存在。

## 已确认的 SRA 代码路径

2026-10-29 通过公开 GitHub API 重新核对默认分支，HEAD 仍为 **`785c3b12944251af81201083a9698a87f1d490f2`**，提交时间 2024-11-14T09:17:42Z。旧目录 `augmentation-research-20261022/sra-shape-impact-20261023/source/` 的 10 份文件重新检查 SHA-256 和 Git blob SHA 全部与旧 manifest 相符。下列均指该固定提交，不使用可变 main 链接：

| 环节 | 已读来源与结论 |
|---|---|
| 基础训练配置 | [SRA_CIFAR.py:40](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L40)：默认不启用固定 seed；44–45 行两半批 label smoothing=0；65–67 行 depth=2，应用概率上下限都为 1。通用 SRA 类默认概率不等于该入口实际配置。 |
| 评分前的更新 | [SRA_CIFAR.py:188](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L188)：train 模式、批次分半；200–213 行随机增强第一半批并执行 optimizer/scaler/scheduler 更新。 |
| 评分 | [preact_resnet.py:74](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L74)：归一化、网络前向、softmax 与标签 cosine，再取 `2/log(C)` 次幂（83–85 行）。旧 B=C 形状问题保持原有结论，本轮不重新开启。 |
| 分数进入增强 | [SRA_CIFAR.py:219](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L219)：no_grad 获取分数和 image_basic；223–229 行将分数传回模型增强分支。 |
| 类型、顺序、幅度、开关 | [transforms/SRA.py:217](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/transforms/SRA.py#L217)：同一标量分数复制到两层；逐样本逐层用 torch 抽类型、NumPy 抽概率，幅度取对应分数；195–206 行再抽应用开关。当前 CIFAR 入口概率=1，因此开关总启用，但 RNG 调用仍存在。 |
| 符号和实际参数 | [operations.py:320](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/transforms/operations.py#L320)：以 ShearX 为例，内部另用 NumPy 抽符号再缩放；其他几何/颜色操作也有自己的参数实现。类型/顺序/符号不包含在强度标量中。 |
| 后续图像与网络 | [preact_resnet.py:88](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L88)：SRA 后还调用 cutout、normalize、网络前向。基础 crop/flip、cutout、全部算子参数及 RNG/设备状态没有形成可重放账本。 |
| 损失和状态 | [SRA_CIFAR.py:226](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L226)：第二次 CE/backward/optimizer/scaler/scheduler 更新。no_grad 的评分仍处于 train 模式；仅凭 no_grad 不能认定没有 BN/Dropout 等状态影响。NormAct/全部模块状态实现未继续追查，故该项未验证。 |

完整动作至少需要保存每层的类型、层顺序、强度转实际参数、符号、启用状态及全部随机决定；图像基础预处理也需一致。只排列 `cos_sim` 会改变强度与随机类型/符号的联合配对，不保证相同完整动作多重集。相同 seed 也不能替代逐动作核对。

两个独立在线臂一旦更新不同，后续参数、分数及动作生成都会分叉，所以“各自在线算分再匹配强度”不保证边际相同。固定 donor 动作库的条件配对效应与原始在线反馈 SRA 的总效果是不同问题；当前没有可用 donor 账本或通过验证的实现。

这里不要求把干预后的损失、更新和 BN 状态强行固定相同：它们可能正是配对改变的后果。需要一致的是起始条件和非处理因素，并把作用量讲清楚；不能既改变训练数据配对，又宣称整个优化轨迹未变。

## 数据与全成本

本轮文件元数据检查：仓库 `.pt/.pth/.ckpt/.safetensors` 共 0 个；`augmentation-research-20261022/pilot-20261022/data/images`、`data/masks`、`data/vit_b_16-c867db91.pth` 均不存在。先前环境检查已确认无可用 torch 环境。本轮未再运行环境或模型。

旧 shape-impact 的分数/决策记录是固定分数诊断，不等同于逐训练步完整 donor 动作轨迹；本轮没有验证到可直接使用的轨迹。成本必须包括独立环境和移植验证、数据获取/划分、donor 生成与记录、两种 recipient 训练、重复、评估、存储/校验。形式上至少是 `C_setup + Σ_r(C_donor,r + C_matched,r + C_permuted,r + C_evaluation,r) + C_storage/validation`；不存在本轮已证明免费的 donor，也没有足够实测依据填入小时数。

**停止状态：已保存审查结论；不排队、不训练、不追加候选。** 只有实质性的新证据或新科学理由，才能重新审查。此次没有这样的依据。
