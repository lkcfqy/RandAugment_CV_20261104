# Fresh-replay DER 残差降权参考组合：单 seed 开发初筛

2026-10-29。**这是明确声明的新参考组合，不是 A2ER 作者实现或其图像实验复现。** 原生路径失败记录不变。目的仅检验：固定降权强度后，样本身份分配是否值得继续研究；方差分解本身无创新性。

硬截止：2026-10-29 19:11:01 UTC（从本次准备开始计 30 分钟）。MPS float32，禁止静默 CPU fallback，不用 Colab。只运行一套配方、seed=20261029；共同前缀 600+160=760 optimizer steps（≤1000），门通过后五支各 64 步。时间或基础学习不足记录执行未决／基线不足，不搜索另一配方。

## 数据和模型，结果前固定

- 官方 CIFAR-100 TRAIN，使用已保存 SHA256 分层划分。选 fine labels 0..9；任务1=0..4（apple, aquarium_fish, baby, bear, beaver），任务2=5..9（bed, bee, beetle, bicycle, bottle）。每类 400 fit、100 development。评估为固定 10 类 class-incremental head，不按 task mask。test 不读；旧封存测试不碰。
- 小型 CIFAR GroupNorm ResNet：3×3 stem16，三个 stage 宽度16/32/64，每stage两个两卷积 residual block，stage2/3首block stride2，GN8组、ReLU、global average、10类线性头；无 BN、Dropout 或预训练权重。
- SGD momentum=.9、weight_decay=.0005；任务1 lr=.05，任务2及短分支 lr=.02。batch32；任务2另有独立 supervised replay32 和 distillation replay32。损失 `CE_current + .5*CE_replay + .5*MSE_all_logit_elements`。不对当步损失另乘 gamma，以隔离这个参考组合的采样模块。
- 原图 fresh views：32×32、零填充4、均匀裁剪offset0..8、水平翻转p=.5、标准 CIFAR100 mean/std=(.5071,.4867,.4408)/(.2675,.2565,.2761)。与固定 Mammoth `e75a491...` 的 seq_cifar100 配方对应。不用 RA、Random Erasing、强度搜索。
- 旧类 buffer512：各类[103,103,102,102,102]，从fit按预定seed/row SHA256选取，无结果筛选。task1第600步后用各一张固定普通增强视图生成历史10维logit；同一模型快照，**所有条目同龄**。它是snapshot replay reference，而非声称逐次入库的作者轨迹。target、成员随后固定；task2不把新类加入buffer。
- N 身份权重由 task2 的160步真实回放访问形成：Q初值0，访问distillation batch残差D=.5∑(h−z)^2，Dstable=D+1e-5，按 b/512 更新median EMA Q；eta按审计公式计算，gamma=1−sqrt(ratio)，访问条目 w←.5w+.5(1−gamma)。未访问保持原值。target correction、alpha/beta自适应从头关闭；这只是公式来源明确的采样参考组合。
- 两个 replay batch 都按 w 进行无放回 Plackett–Luce 抽样（Gumbel top-k）。同一支内两批独立，允许交叠；短分支间用共同随机数耦合。报告的是归一化基础权重概率，不冒称无放回边际纳入概率。

## 执行与学习门

开始前验证裁剪/翻转像素、权重重分配保真、方差恒等式、Q更新和CPU/MPS小模型数值。首16步作为共同前缀的一部分做吞吐验收，不丢弃重跑。若预测剩余总训练加诊断预算不能在硬截止内完成，停止并记录时间预算未决。

任务1步600：旧类dev准确率≥.50、NLL≤1.5且比初始化降低≥.30 nat才继续。任务2步160分支点：旧类与新类dev准确率各≥.35、NLL各≤1.8；否则基线不足，不把机制记为失败。没有 checkpoint 选择。监测训练loss但不据其改配方。

## 门 A：固定状态，仅评分

冻结模型、512个buffer条目、历史target、alpha/beta和Q/N权重。全buffer两组独立固定视图，各8张；Davg=平均单视图到z的半平方残差；Dmean=平均logit到z的半平方残差。两个评分共享全部前向。检查其差等于半视图方差；**全buffer评分绝不更新Q**。

分层固定为 true class × 同龄snapshot cohort × branchpoint clean true-class confidence是否≥.5。年龄恒定，不伪造年龄差异。每层保留N原始权重完整多重集：A按Davg从高到低分配从低到高权重；B按Dmean同样分配；tie以原始TRAIN row打破；R按固定seed层内置换；U均匀。N/A/B/R层质量、全局概率多重集、entropy、ESS必须一致。可交换层（size≥2）须覆盖≥80%条目，单例保留如实报告。

预定投入门：两组均TV(A,B)≥.05；两组的Davg及Dmean Spearman各≥.80；两组向量(pB−pA)余弦≥.50。零差异按失败处理。未通过即停止，无短分支。第二组仅独立复核；若通过，正式分支使用第一组分配。

同时保存：去除每条logit公共平移后的D/方差与匹配TV；残差的平移分量精确分解；logit范数、clean真类概率、entropy与方差/权重差关联及预定OLS解释R²。它们是解释诊断，不能独立证明因果。分层过细或精度/来源前提异常单独记不可识别／实现不足。

## 门 B：只有门 A 通过才执行

从同一模型及优化器状态运行 N/A/B/R/U，各64步。成员、target、Q、alpha/beta、各支权重均固定。当前任务序列、增强参数种子、步数、batch与共同随机数相同；不同身份权重会改变实际source曝光，逐步保存并报告，绝不称逐样本完全相同。每支同等计算/诊断存储；不因前一支结果改后续支。

主终点是buffer外旧类dev与新类dev的NLL和错误率。探索通过需 B 相对 N/A/R 各自旧类NLL改善≥.01 nat，新类NLL恶化≤.01；并且B相对U也须旧类改善≥.01、新类恶化≤.01，否则uniform已足够有竞争力，不包装新方法。单seed开发结果只决定是否值得预注册第二seed，不自动运行第二seed。

保存所有失败门、完整loss/评分/抽样ID/指标、唯一共同checkpoint、版本、时间与内存。既有所有负结果不改动，不提交/推送。
