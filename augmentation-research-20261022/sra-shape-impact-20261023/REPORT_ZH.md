**SRA 形状缺陷：官方输入契约与实际决策影响核验**

2026-10-23 UTC。结论是：缺陷属于官方评分入口，合法的一维整数标签可以触发；它不是旧适配器制造的差异。但所核查的官方默认配置和本项目既有配置没有暴露这个触发条件，因此本分支归档为实现核查案例，不进入下游训练，不宣称新的增强算法或原论文性能失效。RandAugment 研究项目继续，本结论只关闭这一候选的训练升级。

请求档位记录为 gpt-6-astra / max / **Standard**，没有后端路由回执，不声称已经独立确认。所有新增文件只在本项目，8A 的最高资源优先级保持；未操作其他项目、Codex 应用或 PPT。本轮没有 ARiS、图像读取、模型训练、GPU、费用、上传或外发。

**源码真实性和输入契约。** [官方仓库](https://github.com/ainieli/Sample-awareRandAugment)固定在 `785c3b12944251af81201083a9698a87f1d490f2`。10份源码快照按原始字节保存，逐份 Git blob SHA 和 SHA256 检查通过，见 [source-manifest.json](source-manifest.json)。不是把网页格式化后的内容当作原文件。主要入口是 [models/cifar/preact_resnet.py 第66行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L66)，缺陷在第83–85行。

可复查调用链是：

1. [dataset/load_data.py 第15–20、48–50行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/dataset/load_data.py#L15)使用 torchvision 的 CIFAR 类别索引标签，DataLoader 没有目标 one-hot 变换。正常批标签为 int64 `(N,)`。
2. [SRA_CIFAR.py 第195–198行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L195)将图像和标签同样分成前后两半；第221行把后半批标签直接传给 `model(..., training=True, y=train_label_b2, use_basic_aug=False)`。前半批走随机增强，不计算此标签分数。
3. [ResNet.forward 第73–85行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L73)在网络产生 `(B,C)` logits 后用 `y.shape[-1] != ori.shape[-1]` 判断是否转 one-hot。作者本文件第119–121行的示例也传入一维整数标签；README 允许修改训练超参数，并没有 B≠C 的接口前置条件。
4. B=C 时末维相同，转换被跳过。整批类别编号向量 `(C,)` 经余弦操作广播，变成每个样本共享的目标向量。原目标应是每个样本自己的 one-hot。这里 B 指**后半批的评分批大小**，不是脚本名义 batch。
5. 第223–229行把评分作为 `cos` 传回模型，模型再交给 SRA。 [transforms/SRA.py 第217–229行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/transforms/SRA.py#L217)将它作为每个样本的两层增强强度。算子名称由随机数独立抽取，并不按评分排序。

本轮直接从固定文件提取并执行完整 `ResNet.forward` AST。编码器替换为返回同一缓存 logits 的桩，输入图像用合法 `(B,3,32,32)` 浮点零张量，避免解码图片或执行网络。修复组只把上述条件改为 `y.ndim == 1`，其余 AST 经恢复比较确认完全一致；该最小改动只针对本轮有效 hard/one-hot 输入，不冒充完整输入校验器。修复组同时与旧 [mis_score.py](../sra-shape-audit-20261023/mis_score.py)、显式 one-hot 原函数和独立公式比较，均一致。此前本地适配器没有引入缺陷。

[SRA 论文式(1)–(2)](https://arxiv.org/html/2508.08004v1)的正确评分为 `s_i = (p_i[y_i] / sqrt(sum_c p_i[c]^2))^(2/log C)`。本轮 NumPy 公式和第二份纯 Python exp/fsum 实现均直接据此计算，没有复用错误形状条件。[PyTorch 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cosine_similarity.html)解释了为何广播与类型提升使此路径输出数值而非立刻报错。原仓库声明 PyTorch1.13.1；本机2.14.1通过，未另装旧版，不声称原依赖栈已完整复现。

ImageNet 当前活跃模型是 VMamba；同一判断位于 [models/imagenet/vmamba.py 第1873–1875行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/imagenet/vmamba.py#L1873)。这里做了静态链路和 batch 算术核验，未运行 VMamba 网络。上一阶段其他模型文件的范围仍见旧报告，不把这些同仓库文件当作独立方法。

**真实配置、尾批与人工探针分开。** 官方训练数据未下载或运行；以下是固定源码配置、数据集规定规模以及本项目既有协议的核对。还用 PyTorch BatchSampler/DistributedSampler 对整数索引验证了边界，再以独立闭式算术复算。

| 核查对象 | 总样本数 / 类别数 | 每设备完整 loader 批 / 尾批 | 完整评分批 / 尾评分批 | 此缺陷触发批数 |
|---|---|---|---|---|
| 官方 CIFAR-10 默认 | 50,000 / 10 | 256 / 80 | 128 / 40 | 0 |
| 官方显式可选 CIFAR-100，其他参数同默认 | 50,000 / 100 | 256 / 80 | 128 / 40 | 0 |
| 官方 ImageNet 默认，4个进程 | 1,281,167 / 1,000 | 512 / 292 | 256 / 146 | 每进程0 |
| 本项目 Pets 旧 train | 591 / 37 | 16 / 15 | 没有 SRA 评分调用 | 0 |
| 本项目 Pets 旧 dev | 589 / 37 | 16 / 13 | 没有 SRA 评分调用 | 0 |

CIFAR [第48行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L48)完整 batch=256；loader 未设置 drop_last，默认保留尾批；use_proxy=False。ImageNet [第90–91、203–204行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_ImageNet.py#L90)全局 batch2048、4进程，每进程512；[分布式 loader 第201–213行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/dataset/load_data_distributed.py#L201)保留尾批。DistributedSampler 每进程 ceil(1,281,167/4)=320,292，全局补齐1个索引；625个完整批后尾批292，评分146，不等于1000。未假设整除，也未忽略 sampler 补齐。

本项目历史 [protocol.json](../pilot-20261022/protocol.json)的 batch16 与 [train.py](../pilot-20261022/train.py)循环一致。[common.py](../pilot-20261022/common.py)使用普通 torchvision RandAugment，front-weak 缓存来自弱增强组；两者均不是 SRA 训练。结果 JSON 对本地行列出的假想半批8/8或8/7，只是“若接入 SRA”的算术，不是实际日志里的评分批。

额外兼容性重放用同589条 dev 缓存，分别模拟完整批16、256的 SRA 后半批取法，各评分295条，评分批为8×36+7和128+128+39。两种都没有 B=C；原函数、修复和公式最大差2.22e−16。没有把这种缓存重放称为真实训练暴露。

预先声明的**人工**例子另列：若把 CIFAR-10 完整 batch改为20，每个评分批为10，2,500批会触发；README允许这种超参数，但没有发现用户计划采用或历史采用它。N=50,004、完整 batch64 的20条尾批则纯为余数演示，**不是真实 CIFAR 数据量或已有配置**。旧15×37分组同样是刻意边界探针。不能靠这些人工选择声称真实任务受损，也不为制造阳性而变更训练 batch。

**预定实验与实际决策结果。** [protocol.json](protocol.json)在数值执行前冻结，SHA256为 `66069c41d58cee912f6d1f917c79cf9aa3d2903b6a4a1b3739a63c4f5da5aa2e`。只读旧 front-weak epoch30 的589条已用 dev logits（37类），核对 ID、标签和 train/dev manifest；复用既有固定顺序15×37+34，不重新挑模型、epoch、分组或种子。原实现、单条件修复、独立公式使用完全相同 logits/标签。

算子追踪执行原 SRA 的构造、forward、_apply_op 和各算子 `_magnitude_to_arg`，用记录参数的空操作替换实际像素变换。采用训练脚本的 depth2、p_min_t=p_max_t=1，而不是构造器默认0.2/0.8；14个算子顺序固定。各组在每次对照前重设 torch seed=2026102311+组号、NumPy seed=2026102312+组号。几何符号随机数也匹配。零图像只服务合法张量形状，未测实际像素或准确率。

| 指标 | 人工 B=C 的555条记录 | B=34末批负对照 |
|---|---:|---:|
| 原评分与公式平均 / 最大绝对差 | 0.618495 / 0.986306 | 最大2.22e−16 |
| 算子名称变化 | 0/1,110次 | 0/68次 |
| 执行开关变化 | 0/1,110次 | 0/68次 |
| 实际传入算子的强度参数变化 | 876/1,110次 | 0/68次 |
| 至少一个参数改变的照片 | 533/555（96.04%） | 0/34 |
| 两两评分顺序翻转 | 5,723/9,990（57.29%）；无并列 | 0 |
| 每组 Spearman 的平均 | −0.189237 | 1 |
| 每组最高四分位集合重合 | 38/150个名额（每组取10） | 全部重合 |

排序是描述性检查：**SRA没有根据此排名选择算子**，不能把排序翻转写成操作选择改变。每个强度敏感算子的参数都变了；Equalize、AutoContrast、Identity共234次选择忽略强度，是负对照。555张是固定矩阵的行，不是555次独立训练，以上不是训练效应显著性。

| 算子 | 触发组选择次数 | 参数差绝对值均值 | 最大差 | 原实现参数单位 |
|---|---:|---:|---:|---|
| ShearX | 89 | 0.183201 | 0.277192 | 剪切系数 |
| ShearY | 83 | 0.185750 | 0.269482 | 剪切系数 |
| TranslateX | 83 | 0.286360 | 0.436784 | 相对宽度平移比例 |
| TranslateY | 77 | 0.279098 | 0.441447 | 相对高度平移比例 |
| Rotate | 63 | 18.483463 | 26.036698 | 角度（度） |
| Brightness | 66 | 0.552320 | 0.832962 | 乘性/混合因子 |
| Color | 81 | 0.525719 | 0.808447 | 混合因子 |
| Sharpness | 95 | 0.553681 | 0.831576 | 混合因子 |
| Contrast | 73 | 0.573130 | 0.887676 | 混合因子 |
| Solarize | 93 | 158.858000 | 235.729314 | 0–256阈值 |
| Posterize | 73 | 2.405827 | 3.607840 | 传入的连续 bits 参数；非像素结果 |
| Equalize | 71 | — | — | 忽略强度 |
| AutoContrast | 80 | — | — | 忽略强度 |
| Identity | 83 | — | — | 忽略强度 |

Posterize 源码第534–540行返回连续值并交给后续实现，本轮没有自行添加取整，更不把参数差当作有效位数或像素差。完整1178条决策在 [decisions.csv](decisions.csv)，589条评分和排名在 [scores.csv](scores.csv)，汇总在 [result.json](result.json)。修复、旧适配器、显式one-hot和公式的评分最大误差均为2.22e−16。

**已有 issue 和修复历史。** 本轮只读公开默认分支，返回2个提交、1个issue和0个PR，见 [history.json](history.json)。初始提交为2024-11-08的 `cd3f8d182d7908d8c56fe9d1a926fb113512e717`；当前2024-11-14提交只改 Solarize 阈值方向及 Posterize 注释/换行。初始和当前 preact_resnet.py 的blob均为 `b557e9fdf23deed2a1acb02568910556775201eb`，该形状分支并未修复。唯一 [issue #1](https://github.com/ainieli/Sample-awareRandAugment/issues/1)请求发布ResNet50权重，非本缺陷。没有发现这段公开历史中的报告或修复；这不证明首次发现，也不涵盖私有分支、外部讨论或作者未公开修订。未发布、联系、评论或提交任何内容。

**复核、预算和完整性。** 两次数值调用均完成：主审计0.736秒、独立复核0.416秒（均含脚本内导入、不含工具调度），CPU与interop各1线程；峰值RSS分别285.4MB、229.0MB；60秒单次上限未触发。2次预算已用完，0训练、0神经网络前向、0图像解码、0像素变换、0GPU。

[verify.py](verify.py)没有导入主审计、旧适配器或官方AST；用纯Python exp/fsum重算2,945个评分，手写随机抽样顺序核对1,178次算子/开关，并独立参数方程核对2,781个值，最大误差2.22e−16。还复算16组排名、所有参数汇总和批次算术，见 [verification.json](verification.json)。这是同一代理的另一份实现，不是外部审稿者。

源码快照、输入缓存、旧审计目录、旧训练代码、协议哈希前后相同。结果独占创建，未覆盖旧成果。之前归档脚本的Git blob头误写为字面反斜杠零，4次预写校验和随后1次清单生成失败；改用NUL字节后全部通过，发生在数值执行前，没有写入错误源码或改科学比较。完整记录保存在 [technical-events.json](technical-events.json)，不是权限拒绝。

**先例、可讲范围与下一步。** 本轮只确认 SRA 既有公式到实现的偏离。[SADA 2025](https://arxiv.org/html/2510.00434v1)已利用训练动态调整增强，[ActiveAugment 2026-08](https://arxiv.org/html/2608.28923v1)已用候选视图不确定性及特征差异选增强；这些是上一阶段读过的直接近邻，不是本次重新完成的穷尽查新。[Metamorphic Testing，ISSTA2018](https://arxiv.org/abs/1808.05353)也说明关系检验不是新范式。上述来源不支撑本轮参数数字，数字只来自本地冻结比较。

按事前停止条件，所核查的真实记录及默认配置触发数为0，**本候选不获下游训练预算**。即使人工边界上参数变化很大，也不升级18–21run或另挑batch救阳性。保留既有修复适配器作为未来SRA接入前的接口检查；只有将来有明确计划采用的真实配置满足评分B=C，才另立同batch、同初始化、同标签/算子轨迹、同BN和更新顺序的原版/修复版比较，不能用换batch替代修复对照。现在没有这样的配置证据。

11月4日演讲可使用配套 [TALK_NOTE_ZH.md](TALK_NOTE_ZH.md)，讲“公式、张量输入与训练配置需要逐层核查”。它明确同时给出缺陷、条件影响及默认不暴露的边界，没有修改PPT。研究主线下一步从候选筛选继续：先核对已有train/dev缓存可支持哪些真实反事实比较，再对照已读的LA3、AutoDO、SRA、SADA、ActiveAugment及新增直接先例；不能把此次实现修正、普通样本感知调强度或旧失败方向重新命名为贡献。本轮没有登记新的训练候选或承诺后台继续运行。

**访问状态。** 没有新的权限阻塞、待决写入或未知提交；本轮项目内写入的正式审批全部成功。历史 `ps` 进程表读取被OS/沙箱拒绝，状态“已拒绝并停止”，未重试或绕过，其他任务实时负载未验证；本次仅两个不足1秒的单CPU调用。ExploreAugment OpenReview PDF 与另一个 OpenReview 验证页面保持“验证阻断并停止”，未换入口。本轮源码、历史、缓存及参数核查已全部完成，不依赖这些受阻操作。

复查入口：[协议](protocol.json)、[固定源码清单](source-manifest.json)、[历史](history.json)、[主脚本](impact.py)、[评分CSV](scores.csv)、[决策CSV](decisions.csv)、[结果](result.json)、[验证](verification.json)。不要在原目录直接重跑主脚本：它拒绝覆盖结果；以后若需要复现，应在项目内新建同级目录并记录新预算，保留本轮全部文件。
