# CAR / QASHI 来源、定义及重叠核验

核验日期：2026-10-29 UTC。状态：只读补证；切片 exposure / 网格阈值迁移归 first-cv，Rand 不再作为独立候选推进。本报告补充上一轮材料，不修改其历史内容或视其为实验授权。

## 1. 出处与时间：两个不同发表状态

| 项目 | CAR | QASHI |
|---|---|---|
| 完整题名 | Cross-view agreement rescoring improves tiled small-object detection without retraining | QASHI: Quality-Aware Scale-Protected Hybrid Inference for Small-Object Detection in Optical and SAR Remote Sensing Imagery |
| 作者（顺序） | Hyun-Woo Na; Eunkyung Kim; Haneol Jang，均为 Hanbat National University | Pingpeng Tang; Lixing He; Mo Tao; Gangshuo Yuan; Pengyu Qian; Yinlin Li; Qiao Shi; Lei Zhang |
| DOI / 发布 | [Research Square v1](https://doi.org/10.21203/rs.3.rs-10908559/v1)，2026-09-29 | [Remote Sensing 18(20), 3587](https://doi.org/10.3390/rs18203587)，2026-10-19 在线发表 |
| 独立注册元数据 | Crossref `posted-content`；不是已核实的期刊正式发表 | Crossref `journal-article`，publisher MDPI AG；全文链接标为 version of record |
| 必须保留的限制 | 当前作者 README 写的是 **Scientific Reports, submitted 2026**；不能称已被该刊接收 | 公式、数据划分和代码可用性尚未读到正文，不能从摘要补写 |

QASHI 的作者及正式日期来自本轮直接取得的 Crossref 原始记录，不依赖第三方引用。官方索引另列 received 2026-09-01 / revised 2026-09-30 / accepted 2026-10-04 / published 2026-10-19；版本页索引显示 HTML 在 10-23 更新。后两项是索引证据，不等价于本轮已读全文。

原始元数据：`car-crossref.json`、`qashi-crossref.json`。官方登记接口：

- https://api.crossref.org/works/10.21203/rs.3.rs-10908559/v1
- https://api.crossref.org/works/10.3390/rs18203587
- https://www.mdpi.com/2072-4292/18/20/3587/notes

## 2. CAR 的 Zenodo 不是凭标题猜出的第三方包

主稿 Data availability / Code availability（[Research Square v1 PDF](https://assets-eu.researchsquare.com/files/rs-10908559/v1_covered_205103f6-7ba6-43b0-8d62-a3890c5a45c5.pdf)，正文 p11；含封面文件的第12页）直接指向初始记录 **22256020**。本轮 API 核实它与后续记录属于同一 concept DOI **10.5281/zenodo.22256019**，题名及三位作者一致。

| 固定记录 | 作者版本号 | metadata publication_date | 服务端 created（UTC） |
|---|---|---|---|
| [22256020](https://doi.org/10.5281/zenodo.22256020) | 1.0.0 | 2026-09-06 | 2026-09-06 09:51:35 |
| [23642547](https://doi.org/10.5281/zenodo.23642547) | 1.2.1 | 2026-10-25 | 2026-10-25 13:59:22 |
| [23644217](https://doi.org/10.5281/zenodo.23644217) | 1.2.2 | 2026-10-26 | 2026-10-25 15:05:38 |

三者都早于本次 10-29 核验。v1.2.2 的填报发布日期比创建时间晚一天，已保留两个字段；不能把填报日期当期刊审稿日期。证据支持“主稿链接的作者配套归档”，不证明上传者身份已被我们独立鉴定，也不验证论文数值真伪。

上一轮已取得并校验的 v1.2.2 `CAR_code.zip`：852,873 bytes，SHA-256 `9cad38bc8b1f0fc876bffad794425495ede309f669b4e239b08b31fcf9000633`。与 v1.2.1 逐成员比较，仅 `code/revision/figures/make_fig1_photo.py` 不同；下列科学源文件字节相同。本轮未再下载代码或预测包。

**没有已核实的公开 GitHub 仓库地址。** 主稿说匿名仓库提供给审稿人，公开实现以 Zenodo 归档存在。不能将“没找到 GitHub”写成“没有官方代码”。当前 README 还明确将原文 sealed/pre-registered 强措辞改为作者自行遵守的 held-out 配置冻结；`TEST_USED.json` 不是公共预注册。新版材料也增加了非航拍和 oriented-box 分析，不能把 v1 正文的未测范围直接套到后续代码版本。

## 3. CAR 正文公式与实际实现

正文 Effective view set and agreement（正文 p10，含封面文件第11页）Eq.(1)：

\[
\operatorname{agr}(d)=\frac{|M(d)|}{|C(d)|},\qquad
\widehat{s}(d)=s(d)\,[1+\lambda\operatorname{agr}(d)],\quad\lambda=5.
\]

- `C(d)`：框完整落在 crop 内，四边再留 `0.5*sqrt(w*h)`；crop 边与原图边重合时豁免该边留白。
- `M(d)`：`C(d)` 中存在至少一个同类、IoU≥0.75 框的 view。按 view 判断存在性，不是一对一配对；自己的 view 若满足资格也计入。
- 先重评分，再 class-wise NMS，IoU=0.6；因此同时影响排序及保留框，不能假设输出框固定。
- 实现对 `|C|=0` 返回 NaN；打分时 NaN agreement 置0。`normalize_scores` 再除以**传入数组的最大值**，不是固定除6。阈值迁移分析必须固定实际调用链和分数单位，不能把自定义 `/6` 冒充原始执行。

可核对的实现位置（均在已归档的 `../sliced-detection-reliability-proposal-20261029/sources/car/code/code/`）：

| 文件与行 | 已核实内容 |
|---|---|
| `src/features.py:21–37,147–192` | 有效 view、重建全部 views、同类存在性、M/K 与零覆盖 |
| `src/car.py:6–20` | 公式、数组最大值归一化、NMS |
| `src/car.py:23–32` | 作者已有 post-merge 固定框重评分对照 |
| `src/views.py:30–39` | stride=int(S*(1-r))，末端边界对齐 |
| `src/calibrate.py:5–25,81–92,131–146` | 13特征 GBDT，包含尺寸/scale/边界/coverage/agreement；已有低 coverage shrinkage |
| `revision/pod/rev_a11_viewcalib.py:2–14,61–112` | full/tile 与 magnification 条件 isotonic，图像两折 cross-fit，策略内 AP 对照 |
| `revision/pod/rev_a12_rescorer.py:2–14` | 原始1500张 VisDrone-train监督控制及公开包不含该 train dump 的声明 |

有真实函数、脚本入口、requirements 和 tests，不是伪代码；但本轮**没有导入或执行作者代码**，不能报本机可运行/复现通过。上轮环境未有 pandas、pyarrow、pycocotools 等，未为此安装。

## 4. 预测包许可、分组和训练/测试关系

以下为已取得作者 README、源代码及元数据的声明/结构核验，不是对预测数据的独立统计验证：

- 归档的结果表、预测 dumps、manifests：**CC BY 4.0**；代码：**AGPL-3.0-or-later**。原始图像和 annotations 不随包发布，必须分别遵守基准许可；不能用归档的数据许可替代原数据条款。自训 checkpoint 不在包内。
- dump 通常按 detector × benchmark split × view generator 划分，并有 `.meta.json`；列含 image_id、view_id、crop坐标、sigma、bbox、score、class_id、detector、weights_hash。索引的532行不是532个独立实验。
- 主稿使用 VisDrone val(548)/test-dev(1610)、DOTA1.5 val、AI-TODv2 val/test；修订包又加入 Argoverse-HD val每10帧及DOTA原生OBB。不能将共享图像、共享检测器或不同 view 的行当独立样本。
- `src/splits.py:6–14` 是固定seed的**图像ID随机两半**，不是已证实的 sequence/scene 独立划分。`revision/local/rev_a14_cluster_boot.py` 的checkpoint/dataset聚类，也不能替代图像序列分组。原始group映射未取得。
- CAR全局配置在 VisDrone val/YOLOv8s 上选择，再用于 held-out 分割。重切这些开发图像不构成从未使用过的新确认集。
- checkpoint来源是混合的：部分公开预训练，部分作者在对应train集自训后冻结。逐checkpoint训练来源/hash应查 `CAR_results.zip` 的 `WEIGHTS_MANIFEST.json`；该结果包本任务未下载，故逐一独立核验未完成。
- A12说明原始标签训练控制使用1500张 VisDrone-train，应用到val的 S640/r0.25；这些train dumps未公开。修订脚本改为val图像两半cross-fit，分别评估 frozen S512/r0.6 或prefreeze配置。这两个设计不能混为同一训练/测试关系。

**本轮及前轮均没有下载预测 arrays、benchmark images/annotations 或模型权重；没有打开旧封存测试。**

## 5. QASHI：已知内容与明确未核实内容

Crossref登记的作者摘要可确认：固定检测器，full view + 4个重叠tile + 最多3个额外ROI；QAD综合候选密度、弱响应、候选竞争与full/tile不一致；融合包含graph ranking、bounded BoxVote与scale protection。摘要列 VisDrone2019-DET、DIOR、SeaDronesSee v2、SRSDD-v1.0。官方索引还明确 FTCI 对source scale与crop boundary做reliability calibration。

**准确公式及式号：未核实，不给伪公式。** 官方HTML经web读取返回429；Crossref登记的标准PDF链接 `https://www.mdpi.com/2072-4292/18/20/3587/pdf` 经web不可访问，本轮一次普通直连读取于22:59:26 UTC返回 **HTTP403 Forbidden**。没有使用代理、镜像或绕过措施；不继续重试。日志见 `qashi-official-pdf-access.json`。

因此官方仓库/可执行代码、预测包存在性与许可、训练测试和场景分组、校准拟合/阈值选择规则均**未核实**，不能写“无代码”或“没有跨网格阈值试验”。Crossref中的 **CC BY4.0仅已确认是论文许可**，不是预测包许可。题名搜索未发现作者仓库只是一项有限搜索结果，不是不存在的证据。

## 6. 最强重叠与唯一可能差分

| 已有内容 | 对当前候选的约束 |
|---|---|
| CAR把重检次数除以“有资格看见”的view数；已有边界/scale/coverage特征、条件校准、固定框对照及多view配置 | “多看几次导致confidence变”“除coverage”“控制框集合”“加边缘和尺度特征”都不是可宣称的新方法 |
| QASHI明确已有预算受限的局部重推理与source-scale/crop-boundary可靠性调整 | 泛称“成本与可靠性边界”或“source-aware融合”不足以区别；正文受阻也不能据此断言新颖 |
| first-cv已在做exposure/网格阈值迁移 | Rand不能再以同一干预、同一终点另起一条贡献线 |

相对**已读到的CAR分析**，唯一尚未确认被直接回答的精确对比是：在源view策略上固定FP/image操作阈值，迁移到预先指定的几何view稀疏化策略，测其联合FP/recall变化。A11/A12是策略内AP校准，不直接等同于此跨策略操作点迁移；但本轮未读全CAR补充稿，也未取得QASHI正文，不能宣布文献缺口已经成立。

相对 **first-cv正在推进的同题**，目前**没有已证实的独立新增差分**。结论是将出处、公式和限制交回first-cv统一评审；Rand撤出平行实验。已知相关工作不自动否决整个问题，同样，“换一个可靠性终点”也不自动产生贡献。

本轮剩余阻塞仅为QASHI官方全文访问及由此无法核清的项目；不是Mac权限拒绝，也不是全项目停工理由。父任务/first-cv若取得合法官方全文，可从第5节继续，而不是重新下载CAR预测包。
