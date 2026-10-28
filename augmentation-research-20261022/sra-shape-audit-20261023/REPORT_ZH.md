**RandAugment 阶段结果：SRA 的标签形状判别存在可复现的边界缺陷**

2026-10-23 UTC。完成了一个有明确源码依据、实际运行并复核的最小证伪：在固定 logits 下，SRA 的特定实现会在“评分批大小 B 恰等于类别数 C”时错误解释整数标签。这会让增强评分随分批、批内顺序和类别编号改变。新增本地修复适配器及回归核查已通过；没有开展新训练，没有宣布新方法或论文贡献。

请求档位记录为 gpt-6-astra / max / **Standard**；工具未提供实际后端路由回执，不能宣称已独立验证档位。全程只在 RandAugment 项目新增本目录及新检查点。8A 保持最高资源优先级；未操作 8A、SCI-CV 或 Codex 应用。未使用 ARiS、收费算力、上传、分享或新凭据；PPT 与旧结果未修改。

**为何从这里入手。** 已先读标签双入口 AUDIT_ZH.md、PROTOCOL_DRAFT.md，以及 LoRA、温度、插值/patch 相位、操作实现、批内耦合和代理错误记录。没有重开失败训练或 18–21 次标签噪声草案。第一候选是“评分前向是否偷偷更新 BatchNorm”：SRA 的 train + no_grad 路径确实需要匹配模型状态，但增强与 BN 统计的相互作用已有 [Split BatchNorm 前作](https://arxiv.org/html/2010.07810)。宽泛机制缺乏增量，未训练。沿同一评分入口继续核对，才发现下面的具体形状歧义。

**缺陷与触发范围。** [SRA 2025 的式 (1)–(2)](https://arxiv.org/html/2508.08004v1)规定：每个样本的预测向量和它自己的 one-hot 标签求余弦，再取 2/log(C) 次幂。[官方固定 commit 的 WRN 实现第83行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L83)却以 `y.shape[-1] != ori.shape[-1]` 判断是否需要 one-hot。

整数标签形状为 `(B,)`，末维是 B；logits 为 `(B,C)`，末维是 C。B=C 时跳过转换，于是整个批的类别编号向量被当成每个样本共同的 C 维标签向量。[PyTorch 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cosine_similarity.html)允许广播与类型提升，所以本机运行没有报错，而是产生有限但错误的评分。正确目标应逐样本取 one-hot；本轮适配器按维数与完整形状判别，并验证类型、类别范围及软标签行和。

相同形状判断也见于固定 commit 的 [ShakeShake 第163行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/shakeshake.py#L163)、[ResNet-timm 第49行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/imagenet/resnet_timm.py#L49)、[DeiT 第50行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/imagenet/deit.py#L50)和 [Swin 第50行](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/imagenet/swin.py#L50)。这五个文件属于同一仓库，不是五个独立方法。后两者使用未取幂的余弦；本地适配器保留选项，不把指数差异也改掉。本轮数值审计执行的是 WRN 的三行评分表达式，其 AST 与源片段一致；未运行五种完整网络。当前 main 的 WRN 文件 blob 与固定 commit 相同。

**实际对照与结果。** 协议与代码哈希先于运行保存。仅使用已消费的 front-weak epoch30 开发缓存，589张、37类，核对 source_id 和标签与 fit-manifest 一致；未解码照片或读取确认/测试集。按 source_id 排序后，用固定 seed 20261023 打乱，得到15批×37=555条触发记录和34条末批负对照。重排及类别重编号 seed 为20261024；只用这一次预定排布，不挑模型、epoch 或种子。

| 比较（同一 logits、标签语义与照片） | 平均绝对评分差 | 最大差 | 超过预定 1e-12 的记录 |
|---|---:|---:|---:|
| B=C 原实现 vs 公式 | 0.618495 | 0.986306 | 555/555 |
| 原37条一批 vs 拆成≤18条的小批 | 0.618495 | 0.986306 | 555/555 |
| 批内联合重排后复位 vs 原排列 | 0.121317 | 0.448069 | 555/555 |
| 同时重编号标签与 logits 列 vs 原编号 | 0.133590 | 0.508697 | 555/555 |
| B=34末批原实现 vs 公式 | 3.27e-17 | 2.22e-16 | 0/34 |

显式 one-hot、修复适配器、拆小批三种控制与公式的最大误差均为 2.22e-16；修复后重排误差为0，重编号误差至多2.22e-16。全589条一次评分 B≠C 也符合公式。

预定随机顺序中的第一条是 Birman_105：公式0.954462，原B=37评分0.477865；仅重排批中记录变成0.421978，仅重编号变成0.331777，修复后为0.954462。选择该例的规则是“第一条”，不是取最大差。

这不是统计显著性研究：这里验证确定性代码关系，555条不是训练重复，不能据此推断分类损害大小。旧开发缓存来自非 SRA 训练的模型，用作已知 logits 测试载体；绝不把它当 SRA 性能复现。

**验证与资源。** 24组边界检查覆盖 C∈{2,10,37,100}、B∈{C−1,C,C+1}、FP32/FP64，核对整数标签、one-hot、软标签与不取幂分支；14个非法输入被明确拒绝。主运行使用本机 PyTorch2.14.1/NumPy2.5.4，CPU与interop均1线程，记录0.551秒（含脚本内导入），峰值RSS约237MB。没有图片前向、反向更新、GPU调用或新下载。原作者标注的 PyTorch1.13.1 环境未另装或复测。

另一份纯 Python `math.fsum` 写法逐条重算589×7=4,123个保存数值及汇总，最大差2.22e-16，通过；只用NumPy读取NPZ，不调用评分实现。复核核心耗时0.019秒，不包含独立进程启动/导入。这是同一代理的独立写法复算，不是第二位审查者。输入、协议与冻结代码哈希均未变。60秒运行上限未触发，退出码0，没有本任务遗留后台工作。

**先例与研究价值边界。**

| 直接或近期工作 | 本轮核读与实质区别 |
|---|---|
| [SRA，2025](https://arxiv.org/html/2508.08004v1) | 评分公式、训练入口和官方源码。当前结果修复公式到实现的偏离，不提出新评分。 |
| [SADA，2025-10](https://arxiv.org/html/2510.00434v1) | §3 用预测动态近似影响并作时间窗口统计。已覆盖“随训练动态调整增强”，不能换措辞当新点；未核其官方代码。 |
| [ActiveAugment，2026-08](https://arxiv.org/html/2608.28923v1) | §3.1–3.2 用候选视图的预测不确定性与表征差异选择增强。进一步否定泛称“首次按模型状态选增强”；本轮未核其代码或性能。 |
| [EntAugment，2024官方代码](https://github.com/Jackbrocp/EntAugment/blob/e6ed83ad41beddf47cff434b512abc222dc3ee8f/train_EntAugment.py#L141) | 当前核读评分表达式为预测熵，不经过本次整数标签形状分支。仅限该入口，不是整套代码无缺陷认证。 |
| [Split BatchNorm，2020](https://arxiv.org/html/2010.07810) | 已直接研究增强分布与BN统计的关系；本轮第一个候选因此未进入训练。 |
| [Metamorphic Testing，ISSTA2018](https://arxiv.org/abs/1808.05353) | 以应保持的关系检测图像分类器实现缺陷已有先例，本轮不声称发明了这类检验。 |

另有2026-09的《Test Accuracy Conflates Learned Weights with Normalization State》检索线索，但 [OpenReview 落地页](https://openreview.net/forum?id=Ff9byKDIY7)返回浏览器验证，全文未取得。它不用于判定精确重合或支撑实验数字。官方 SRA issues 此次返回1条模型发布请求，没有本缺陷报告；这只是有限检索范围，不证明首次发现。未向作者发消息、开issue或提交PR。

**不能据此说论文结果失效。** 源码默认 CIFAR 总batch为256，评分取后半批128；50,000训练样本的末批80，评分40。这两个值均不等于10或100。该静态默认路径不触发本缺陷。若将SRA迁移到小数据、不同类数或不同每设备批大小，才需检查评分处的实际B；B是评分批，不是名义全batch。完整训练还有BN与更新时序效应，不能用端到端分批差异单独归因此缺陷。没有测得准确率下降、修复后的收益或跨架构泛化。

**下一步与停止门。** 本阶段可交付为11月4日演讲的一个“公式正确也要核对标签张量”的实测案例，配上上表和默认配置不受影响的边界。保留 [mis_score.py](mis_score.py) 作为项目内以后接入SRA前的已测适配器；没有改作者远端仓库或旧实验。研究主线暂不扩跑：只有在真实计划采用的配置中先找到触发批，并明确它会影响什么实际实验结论，才值得比较原实现与修复版训练；两者必须保持同批大小、同样本轨迹和同模型状态，不能用“改变batch大小”作唯一修复对照。若实际配置不触发，就把此项关闭为复现检查，不故意选 B=C 制造论文阳性，也不为它重启18–21次噪声实验。

**文件与复现。** [protocol.json](protocol.json) 与 [protocol-lock.json](protocol-lock.json)记录预算、种子、停止条件；[diagnose.py](diagnose.py)为实际诊断；[mis_score.py](mis_score.py)为修复适配器；[result.json](result.json)、[per-photo.csv](per-photo.csv)、[group-mapping.json](group-mapping.json)保存逐条证据；[verify_results.py](verify_results.py)、[verification.json](verification.json)保存独立写法复核；[evidence-ledger.json](evidence-ledger.json)、[evidence-addendum.json](evidence-addendum.json)记录先例和源码范围。可用 `replay.py` 在项目内新建同级目录再现，脚本拒绝覆盖已有结果；本轮未启动重放。

**访问状态。** 查询本机进程表的 `ps` 被OS/沙箱拒绝，状态为“已拒绝并停止”，未提权重试或换工具绕过；因此不能确认其他项目当下负载。本轮实际计算只占单CPU线程约一秒，没有与8A并行大型训练。新增 OpenReview 落地页遇浏览器验证，状态为“验证阻断并停止”；旧 ExploreAugment PDF 的封锁保持，不访问镜像、GUI或其他入口。上述两项不阻止本次源码/缓存核查完成，没有待决或结果不明的写入。项目内新增文件的正式写权限调用均成功；没有自动审批拒绝。所有旧失败、封存资料和PPT保持原状。
