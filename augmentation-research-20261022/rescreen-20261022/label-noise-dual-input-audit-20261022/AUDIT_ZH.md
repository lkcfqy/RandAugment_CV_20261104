# 标签噪声“双入口”：限时全文与官方代码核查

2026-10-22；检索开始 18:52:33 UTC，限定 45 分钟。仅检索、静态代码阅读和协议设计；未启动训练、推理或 GPU，未读取任何封存测试。模型偏好记录为 gpt-6-astra / max / Fast，未验证后端实际路由。未使用 ARiS。

**结论：关闭“首次发现标签同时影响训练和增强”“首创置信度控制增强抗噪”的宽泛立题。狭义的 SRA 损失标签 × 控制器标签四格在此次核读的算法和消融中未找到同构实验，但这不证明新颖性，也不足以支持现在进入训练或论文主线。保留一份可实施的课堂机制审计草案；目前没有新贡献的实验证据。**

这不是把工程投入门当科学否定。上一轮温度尺度 10% 门只约束那个固定 checkpoint 的投入判断；本报告不复用该门，也不使用旧 589 个开发 logits 否定跨模型机制。旧 LoRA pilot、数值诊断、范数审计及修订记录均保留，未再修补或重跑。

## 最接近前作究竟控制什么

这里把“直接依赖标签”限定为：固定模型参数、策略参数、历史缓存、图像和随机数，只更改送入控制器的当前标签，增强决定是否会变。通过过去训练改变模型，是另一条间接路径。

| 前作及阅读范围 | 实际标签入口与官方代码证据 | 与拟议四格的关系 |
|---|---|---|
| [AugDesc，CVPR 2021](https://arxiv.org/html/2103.02130v3)，正文 §3、§4 及附录 A.1–A.3 | 弱增强供分析/伪标签，强增强供梯度更新；图像变换自身只接收图像。给定标签仍进入 CE→GMM 样本筛选及软标签修正。[变换接口](https://github.com/KentoNishi/Augmentation-for-LNL/blob/908a22b23ca4cd5c70c52bdaa6a4304a5cbb1fda/dataloader_cifar.py#L220)、[GMM](https://github.com/KentoNishi/Augmentation-for-LNL/blob/908a22b23ca4cd5c70c52bdaa6a4304a5cbb1fda/train_cifar.py#L242)、[标签修正](https://github.com/KentoNishi/Augmentation-for-LNL/blob/908a22b23ca4cd5c70c52bdaa6a4304a5cbb1fda/train_cifar.py#L100)。 | 已覆盖“分析与更新应使用不同增强”的泛称。WW/SS/WS 是增强视图配置，不是独立切换两个标签入口。不能因此称整套 AugDesc 与标签无关。 |
| [AutoDO，CVPR 2021](https://arxiv.org/html/2103.05863v2)，§3–5、联合优化与消融 | 联合优化逐样本增强、损失权重和软标签；标签经内外层目标影响策略。运行时增强接口是 index/image，损失另接 target。[调用链](https://github.com/gudovskiy/autodo/blob/ad80c5ca84d52fcc0cb4b8e5f6204b7e0eb19f7a/custom_models/automodels.py#L325)。 | “标签噪声与增强联合处理”已做。A/AW/AWS 消融并非同一 SRA 控制器的 clean/noisy 标签路由四格。 |
| [CBS，arXiv 2024 v1](https://arxiv.org/html/2402.11242v1)，§III-B–D、§IV-D | 样本池筛选依赖给定标签，包括 `p_y > tau`；固定池之后，混合系数偏向最大预测概率更高者，该比较本身不用给定标签；混合目标又含标签。[筛选与混合](https://github.com/NUST-Machine-Intelligence-Laboratory/CBS/blob/215708e383e3017b1628854f65469003158bbccb/main.py#L119)。 | 置信度控制的增强不是新概念；“最大概率与标签无关”不能推导整条训练链与标签无关。所读消融未见本四格。 |
| [CTW，IJCAI 2023](https://www.ijcai.org/proceedings/2023/0450.pdf)，§3、§4.3–4.4 | 时序领域按给定标签的损失及类别归一化选择可信样本，再做 TimeWarp，同标签也进入增强样本损失。[入口](https://github.com/qianlima-lab/CTW/blob/1a622be6cb352d9af29b90ebc4012f47c0bbc74f/src/utils/training_helper_CTW.py#L404)。 | 已直接讨论为何不能无差别增强带错标数据。CAug/AAug 和组件消融改变选择/增强方式，不能当作本四格，但足以否定泛化的“标签双用从未考虑”。 |
| [LA3，ECCV 2022 / arXiv 2023](https://arxiv.org/html/2304.10310)，§3.2–3.3、§4.4 | 当前 target 直接索引类别增强策略，随后仍返回 target 给训练损失。[CIFAR10 数据入口](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/dataset.py#L492)、[训练](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/train_label_aware.py#L47)。 | “同一个标签决定增强又监督分类”已经存在。区别仅可能在独立干预后量化额外风险；所读文中未见该错标四格。 |
| [AdaAugment，arXiv v3，2025-07-13](https://arxiv.org/html/2405.11467v3)，§III、分析与消融 | 策略 action 读特征，但 reward 由三种增强的带标签 CE 差值形成，影响后续策略及按样本缓存强度。[reward/更新](https://github.com/Jackbrocp/AdaAugment/blob/d854af15467285ad8f46500d6fc2da5225fb5bd0/train.py#L151)。所查 WRN 虽有 labels 形参，函数体未使用它。[WRN](https://github.com/Jackbrocp/AdaAugment/blob/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/Wide_Resnet.py#L76)。 | 属于标签影响策略学习信号，非 SRA 当前标签直接算强度。若对它做四格，必须拆 reward 的标签与分类 CE 的标签，不能只改 action 输入。不要与 ICLR 2022 的 AdaAug 混称。 |
| [EntAugment，ECCV 2024](https://arxiv.org/html/2409.06290v1)，§3 与算法 | 强度采用预测熵，固定 logits 时不读当前标签；标签仍通过 CE 训练改变未来模型。代码把熵强度写入逐样本缓存。[公式实现](https://github.com/Jackbrocp/EntAugment/blob/e6ed83ad41beddf47cff434b512abc222dc3ee8f/train_EntAugment.py#L141)、[缓存读取](https://github.com/Jackbrocp/EntAugment/blob/e6ed83ad41beddf47cff434b512abc222dc3ee8f/Dataset.py#L77)。 | 可借其标签无关分数作路由负对照；不能把含缓存时序/不同训练目标的整套方法当完全匹配基线，更不能称其免疫错标。 |
| [SelectMix，arXiv 2025 v1](https://arxiv.org/html/2509.11265v1)，方法、消融、附录 B Algorithm 1 | 样本给定标签与 out-of-fold 预测是否一致决定是否混合；给定标签还影响伙伴池及混合目标。未确认作者官方代码，结论仅来自全文算法。 | 对“错误标签影响增强选择与目标”非常接近；所读 α 消融不是独立 oracle 标签路由。仅换成 Fashion-MNIST 也不产生新意。 |

补充核读：SADA [2025 v1 §3](https://arxiv.org/html/2510.00434v1) 的推导先出现标签，但其高效评分明确用当前预测替代 one-hot 标签；未取得官方代码，不能把推导式误说成实际标签入口。MADAug [2023 v2 §3](https://arxiv.org/html/2309.04747v2) 是以特征为策略输入、训练/验证损失驱动双层学习，未做官方代码核查。ActiveAugment [2026 v1 §3–4](https://arxiv.org/html/2608.28923v1) 提供近期基于预测不确定性/表征差异选增强的近邻；本轮只核读正文，未核代码，未用它声称已覆盖四格。

## SRA 的两个入口已在代码中分离定位

依据 [SRA 正文 §3](https://arxiv.org/html/2508.08004v1) 和固定 commit `785c3b12944251af81201083a9698a87f1d490f2`：

- [模型评分](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/models/cifar/preact_resnet.py#L74) 将 softmax 向量与标签 one-hot 的余弦相似度取 `2/log(C)` 次幂，直接依赖当前标签；[增强器](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/transforms/SRA.py#L208) 把结果作为每样本强度。
- [训练循环](https://github.com/ainieli/Sample-awareRandAugment/blob/785c3b12944251af81201083a9698a87f1d490f2/SRA_CIFAR.py#L194) 先更新随机增强半批，再给另一半批评分，最后增强并更新；第 221 行的 `y` 是控制器入口，第 205/230 行 criterion 的标签是监督入口。改造时必须拆成两份索引一致的标签数组。
- 评分在 `no_grad` 中，但循环仍处于 train 状态。它不自动关闭 BatchNorm/Dropout；任何基线必须匹配这次前向及更新节奏。不能偷偷改成 eval 后声称只改变标签。

固定概率向量 p 时，若标签从 a 换成 b，MIS 比值为 `(p_b/p_a)^(2/log(C))`（分母概率正）。这是已知评分公式的代数结果，**不是贡献**。错标若对应低概率，会让强度变小；它可能减少破坏，也可能减弱正则化、帮助记忆错标。最终风险方向不能从这条公式推出。

## 可实施，但四格回答有边界

令 y₀ 为原标签、ỹ 为训练集固定人工错标。独立切换损失入口 L 与控制器入口 C：

| 组 | 损失标签 | 控制器标签 |
|---|---|---|
| L0C0 | y₀ | y₀ |
| L0C1 | y₀ | ỹ |
| L1C0 | ỹ | y₀ |
| L1C1 | ỹ | ỹ |

用干净开发集错误率 E 定义 `ΔC|L1 = E11 − E10`，交互 `I = (E11 − E10) − (E01 − E00)`。前者问固定监督噪声下额外污染控制器有何影响；后者问这种影响是否随损失标签污染改变。两者都不是“同一个错误被重复使用”的完整识别：还需同错标位置、同错误率及类别转移分布、但错误类别独立重抽的第五组。否则“同错标”与“两个入口各有噪声”混在一起。

改变控制器标签还会改变整体强度分布，所以四格也不能单独归因为“分配给了错误样本”。若未来要讲分配机制，需预设边际强度匹配的回放/分配对照。此项不是为了给阴性找新阈值。

详见 [未执行协议](PROTOCOL_DRAFT.md)。方案采用新数据分割与小 CNN，不围绕旧 Pets/LoRA logits 再做数值修饰。当前没有训练代码、下载、性能读数或训练耗时测量，不能把协议可实施说成实验已验证。

## 是否值得继续

**作为新论文方向：本轮不放行。** 最重要的反证是 AutoDO、LA3、SelectMix、CBS/CTW 已覆盖标签参与增强与噪声处理；仅把一个已有控制器拆成两份标签，通常是合理诊断，不自动是研究贡献。本轮尚没有反直觉风险、效应规模、跨模型复现或可部署修复的证据。

**作为课程中的原创机制核查：有条件保留。** 若后续确实要做，最有信息量的一步是新小模型上预先固定的四格加“同位置独立错误类别”对照，区分一般错标正则化效应与同一错误的耦合。不要先重复旧 589 logits、调温度门或挑 seed。若无资源或无法落实匹配控制，可直接停止；这不等于科学证明机制不存在。

淘汰规则：若后续最近前作已有同一干预与结论，关闭新意叙事；若只能看到固定 softmax 公式的必然变化，关闭贡献主张；若控制器干预与基线更新次数/数据划分不能匹配，协议无效、先停；若结果未支持有害方向或区间过宽，如实记录，不追加噪声率/模型/seed 搜索救阳性。尚未设定效应大小的科学等价界，不能将任意资源门写成“无效”。

## 已完成范围与阻塞

23 个定向检索式；8 个官方研究仓库的 25 个文件已按 commit 固定读取，核读相关入口而非声称逐行审计整个仓库；另外核对 Fashion-MNIST 官方 README 与 MIT LICENSE。全文核读范围、源码 blob SHA、检索式与未核实项见 [证据账本](evidence-ledger.json)。检索未找到同构四格只是有限的未检出，不是全网无前作证明。

AugDesc 版本边界：官方 dataloader 在 `eval_train` 使用无增强，作者在代码注释说明本应为弱增强；“影响很小”是作者注释，未由本任务复验。其 `randaugment.py` 存储 m 却用随机强度，因此不能直接把这一代码路径称为固定 M 的复现。这些实现差异只作复现记录，不当作研究贡献，也不改动它们。

**阻塞：**尝试读取 [ExploreAugment PDF](https://openreview.net/pdf/44737a43d938147e967d6e47a4996467fc4a44db.pdf) 时被重定向至 OpenReview 浏览器验证页，未取得全文，已停止该访问；没有通过 GUI、镜像或其他入口绕过。仅凭检索片段不能排除其覆盖本问题，也不能据此判定重复。状态是网页验证阻断，不是本轮文件写入被拒。其余独立检索、静态代码核查和协议已完成。

本轮没有访问 SCI-CV 项目、改 PPT、开启 GPU/MPS/A100、收费资源或后台任务。原项目追加归档是否成功，以同目录 `archive-receipt.json` 为准；旧报告及旧结果不覆盖。
