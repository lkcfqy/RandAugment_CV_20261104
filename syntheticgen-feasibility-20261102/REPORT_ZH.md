# SyntheticGen：只读可行性审查，停止当前候选

日期：2026-11-02。结论：**STOP，不下载像素或权重，不启动比较实验。** 公开数据池确实存在；当前缺口在可解释比较所需的监督谱系、论文子集身份和下游实现。缺一个常见对照，不能直接转化为我们的科研贡献。

固定依据：SyntheticGen 代码 `ca724b2693bbf07725606dbaf4024602bcc0f17a`；HF 数据版本 `225bb99a7a70588dd37e0e7e4e566e6be20ea82f`；DAFormer 代码 `e6f6e041e7477b642106c134c1715fc934ccd210`。所有代码均以 .txt 留存并校验 Git blob，未导入或执行。对应链接和 SHA256 见 [source_index.json](source_index.json)。

## 1. 生成器究竟使用什么监督

| 环节 | 发布配置及静态调用证据 | 能确认的范围 |
|---|---|---|
| Stage A 布局生成器 | `train_layout_d3pm_masked_sparse_80k.yaml` 指定 `Train`、`Urban,Rural`；脚本 318–329 行将参数传给 LoveDADataset，665 行起对语义标签和比例训练 | 两个域的 Train 分割标签、比例及域身份进入训练。公共 Dataset 也加载配对 RGB，但此阶段所读损失路径使用标签，不能说 RGB 参与该阶段优化 |
| Stage B 图像生成器 | `train_controlnet_ratio_loveda_1024.yaml` 同样指定两个域的 Train；脚本 826–841 行读取 `pixel_values`、`layout_512`、`layout_64`，RGB 经 VAE 编码 | 两个域的 RGB 和分割标签参与配对条件生成训练；同时使用已有 SD v1.5 基础模型 |
| 类比例先验 | `compute_ratio_prior_loveda_train.yaml` 指定两个域的 Train，脚本读取布局/比例统计 | 两个域的标签分布可影响先验 |
| 合成池配比和接受筛选 | `Generator/gen.py` 780–789 行扫描 Rural 与 Urban 的真实掩码、计算域配额；1031–1053 行计算实际类别比例并记录接受结果 | 不是仅用域名字的无监督生成。即使 `only_domain` 限制输出域，这段代码仍先统计两个域，不能据此证明源域独占监督 |

配置均指向 Train，**这里没有发现 Val 标签用于上述训练的证据**。不能把“目标域训练标签可见”改写为“验证集泄漏”；也不能用配置证明任何未发布的原始运行实际使用了这些文件。[生成器配置](https://github.com/Buddhi19/SyntheticGen/tree/ca724b2693bbf07725606dbaf4024602bcc0f17a/configs)、[配对数据加载](https://github.com/Buddhi19/SyntheticGen/blob/ca724b2693bbf07725606dbaf4024602bcc0f17a/src/scripts/dataset_loveda.py)、[Stage B](https://github.com/Buddhi19/SyntheticGen/blob/ca724b2693bbf07725606dbaf4024602bcc0f17a/src/scripts/train_controlnet_ratio.py)。

另核作者生成器 HF 仓库固定版本 `098fa01a35185774bc56bdd9daf091dda8d6399f`：文件索引只列 `.gitattributes`、README、`controlnet.zip`、`layout.zip`、image.png。数据卡仍指向上述两域配置，没有单列训练 ID/按迁移方向的 checkpoint 映射。ZIP 内是否包含进一步配置**未检查**，因为本任务禁止权重下载；不能声称内部一定没有。生成器卡声明 Apache-2.0，和合成数据池的 MIT 声明分开记录。[生成器数据卡](https://huggingface.co/buddhi19/SyntheticGen/blob/098fa01a35185774bc56bdd9daf091dda8d6399f/README.md)。

论文 §II-D 使用 2000 对合成样本（Urban 1106、Rural 894），加原 LoveDA Train；下游包括 U-Net、PSPNet、FactSeg、HRNet、AerialFormer，在官方 Val 评估。表 I 包含 Urban→Rural 和 Rural→Urban，并只列原数据与加入合成数据的比较。原文没有给出这两个方向各自的生成器训练样本清单、是否重新进行源域专属训练、所用合成子集，或对应的源/目标标签访问声明；不能核实其严格 source-only DG 合约。[论文 v1](https://arxiv.org/html/2602.04749v1#S2.SS4)。

**公平性含义（本次推论）：** 如果按当前发布配置使用双域标注训练的生成器，真实数据对照也必须获得相同的双域训练标签访问权，且描述为相应的监督设置。将这种合成数据与“只许看源域标签”的 RCS 对照比较，会混入额外监督。若研究严格源域泛化，则需要每个方向源域专属的生成器/先验/生成清单证据；本次没有。两种设置不能混在一条性能差中。

## 2. 3321 对能否恢复论文 2000 对

固定 HF 版本的 CSV 实读 **388,630 字节**，SHA256：
`c48d62ffc8eeb69833be2c133ba4fac590897b4c217f2f6c6b718ba2e81be30b`。

| 元数据核查 | 实际结果 |
|---|---|
| 样本数/域 | 3321；Urban 1766，Rural 1555 |
| 与论文数量之差 | Urban 多 660，Rural 多 661；这不是子集身份映射 |
| 路径与唯一性 | 3321 张图像、3321 个语义掩码、3321 个 RGB 掩码路径；全部存在于固定版本文件索引，无重复行、缺失路径或域名错配 |
| 编号 | Urban 000000–001765；Rural 000000–001554，均连续；编号连续不能证明其生成顺序或论文成员资格 |
| CSV 字段 | image_file_name、mask_file_name、mask_rgb_file_name、domain、source_dataset；最后一列全部 LoveDA |
| 缺少的身份/统计 | 无论文成员字段、生成器 checkpoint/hash、源训练 ID、种子、生成配置、请求/实际类别比例、接受日志或逐类像素数 |
| 完整树中其他文件 | README.md、metadata.csv、.gitattributes、file.py；未见单独论文清单或生成日志 |

[固定版本数据清单](https://huggingface.co/datasets/buddhi19/SyntheticGenV5/blob/225bb99a7a70588dd37e0e7e4e566e6be20ea82f/Train/metadata.csv)；可复核统计见 [metadata_audit.json](metadata_audit.json)。

作者 HF `file.py` 仅按域及文件名排序枚举匹配图像/两类掩码，写上述五列，并未选择论文子集。生成代码本身**会**写包含 ratios/focus_class/metrics 的 manifest 和包含实际比例 p 的 generation_log；这些运行产物不在本次发布池中。证据是“代码支持记录而当前池未发布”，不是声称作者没有保存过。[生成清单写入代码](https://github.com/Buddhi19/SyntheticGen/blob/ca724b2693bbf07725606dbaf4024602bcc0f17a/Generator/gen.py#L1031)。

所以不能自行选“前 1106/894 张”冒充原实验。固定使用全部 3321 对可以定义另一实验，但既不能复现 2000 对，也不能补上训练谱系缺口；本次不以此替代任务。逐类像素统计须读取语义掩码才能核实，CSV 不能提供；此次没有读掩码像素，也没有根据图片/效果选样。

数据卡声明 MIT，代码仓库亦有 MIT LICENSE；这只记录发布声明，不扩展为对全部上游数据权利的保证。此前页面约 6.78 GB 的池大小本次没有逐文件字节复核：HF API 的 usedStorage=20,292,529,573 不能当成当前待下载包大小；viewer 的 parquet 69,937 字节是五列元数据，不是图像池。没有下载任一 PNG、parquet 或权重。

## 3. 下游和强对照是否可执行

当前 GitHub 递归树完整，只有 33 个文件；发布页 releases 为空。README/项目页未给出额外五模型训练配置的入口。本地项目文件名盘点没有找到 LoveDA 像素资产及上述论文实验清单；已有通用 HRNet/timm/MMDetection 源文件不等于该分割实验已准备好。这是项目范围盘点，未搜索整台电脑。

已发布的 `eval_downstream_segmentation.py` 是真实可读的示范，但要与论文模型区分：

- 固定构造 **SimpleSegNet**，浅层卷积编码器/转置卷积解码器，没有论文五种模型的具体配置、初始化和 checkpoint。
- 默认 512×512 resize，Adam LR=1e-3，batch=8，**max_train_steps=2000**，交叉熵，最后状态评估。其固定更新预算确实公开；不能反过来称论文五模型必然也用了此预算。
- 原样本与合成样本直接 ConcatDataset，shuffle=True；没有域配额、RCS 或逐类像素曝光匹配。
- 同一个 `loveda_domains` 传给 Train 与 Val，未提供按方向分别设置训练域/验证域的参数。
- 只在 main 开始 manual_seed 一次，先训练 real，再新建 real+synth 模型；第二次初始化会接着消耗 RNG。此脚本没有实现逐臂相同初始状态；不据此断言论文原实验也如此。
- 不能拿这个小网络替代论文中的模型再叫公平复现，也不使用冻结特征替代像素级分割。

[示范训练器](https://github.com/Buddhi19/SyntheticGen/blob/ca724b2693bbf07725606dbaf4024602bcc0f17a/src/scripts/eval_downstream_segmentation.py)、[SimpleSegNet 定义](https://github.com/Buddhi19/SyntheticGen/blob/ca724b2693bbf07725606dbaf4024602bcc0f17a/src/models/segmentation.py)。

论文及已检查发布材料**未报告**域比例/类别像素曝光/更新匹配的真实 RCS 对照。这个结论限于公开证据，不是证明作者未做过任何未公开实验。论文的“相同训练设置”未细化到可核验的五模型 optimizer、scheduler、步数、crop、batch、预训练权重、种子及 checkpoint 选择；不能凭示范文件默认值补齐。

DAFormer RCS 已是既有方法：官方实现用真实源标签的类别频率 f 设 p(c)=softmax((1-f(c))/T)，先抽类别再抽含足够该类像素的图，按 min_crop_ratio 尝试重新裁剪。它会改变训练时类别曝光，**并不自动精确匹配一个合成池的域×类别像素联合计数**。需要的只是它的已知采样组件，不能把完整 UDA 算法或改名版 RCS 当本项目新方法。[DAFormer 官方实现](https://github.com/lhoyer/DAFormer/blob/e6f6e041e7477b642106c134c1715fc934ccd210/mmseg/datasets/uda_dataset.py)、[CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)。

## 4. 即使补齐资产，这个检验能回答什么

有实践意义的估计对象是：**在相同可访问标注、原分割架构与初始化、固定 U×batch、相同训练处理/选 checkpoint 规则下，合成数据是否比真实采样带来超出域比例和少数类曝光变化的收益。** 它不是“合成方法比默认均匀采样强”这一现成表格的重述。

成立所需的控制与边界：

1. 先固定监督制度和全部训练/验证 ID；按真实监督访问权给对照公平权限，不能把目标标签生成的结果算作无目标监督收益。
2. 域配比和逐类像素曝光必须在**实际训练 resize/crop/ignore 后**统计，至少按域×类别记账；固定 batch/U 并记录重复样本、有效像素及 CE 的有效像素归一化权重。只数 2000 张/平均原掩码比例不够。
3. RCS 不是七个独立旋钮。一张裁剪包含多个类别，有限真实裁剪池的联合比例不一定能达到合成目标；即便期望值可匹配，有限步整数计数也可能无法严格匹配。应事前检查可行性、报告残差，不能在失败后仍叫“已匹配”。完整曝光匹配也不等于上下文、标签质量或梯度噪声已相同。
4. 要分辨“何时真实重采样足够、何时新增有效语义上下文仍有收益”，需要事前定义的多个不平衡/监督条件及独立保留评估；单个 LoveDA 结果，无论 RCS 胜负，都最多提供该设置的对照事实。若匹配后优势消失，只能说这些因素在该设置足以解释收益；若保留优势，也不能直接证明新上下文机制。

这些是可识别性要求，不是已经满足的实验计划，更不是新算法。本次没有独特且有证据支持的可发表命题，亦没有证明“合成无用”或“收益仅是重平衡”。**当前候选因关键资产不可核验且知识增量尚不足而关闭。** 后续只有获得论文 2000 对及按方向生成谱系、配套下游配置/权重身份，并提出超出单一方法排名的可证伪问题，才值得重新立项；本次不发作者消息、不等待补件、不自动开跑。

## 5. 资源、验证与交付

执行内容：公开论文/项目页阅读、17 个 GitHub 文本源文件及固定树、HF API/README/CSV/file.py/尺寸 JSON、本地标准库元数据核验。图片/掩码像素/权重下载 0；外部仓库代码执行、安装、模型构造、训练、推理、上传、新云会话和新增 CU 均 0。演讲资料和既有研究未修改。

运行时长/产物体积及最终 SHA 清单见 [RUN_RECEIPT.json](RUN_RECEIPT.json)、[MANIFEST_SHA256.json](MANIFEST_SHA256.json)。[verification.json](verification.json) 保留 34 项检查，覆盖固定版本/CSV hash、所有路径唯一性/存在性/域与 stem 配对、树完整性及每个源码 Git blob/SHA256；这些验证元数据身份，不验证未读取的图像内容或模型效果。可在归档目录执行 `python3 audit_metadata.py` 复核（只读本地文本并更新本检查回执）。

未来实跑的完整时间、内存、下载/磁盘和训练预算**未定**：缺原模型训练合约，且当前没有 LoveDA 像素资产；不能用 DeepFish 检测耗时或 SimpleSegNet 2000 步代填原分割成本。本次没有为成本估算试跑模型。

一次默认沙箱 HTTPS 请求在 DNS 阶段失败；按标准权限机制重试同一已授权公共元数据读取后成功。浏览器工具无法抓 HF 页面是工具能力限制。没有自动审批拒绝、未决云动作或后台训练。详见 [evidence_and_decision.json](evidence_and_decision.json)。

