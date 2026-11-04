# DeepFish 分组与检测框来源补查

日期：2026-10-31。仅核资料、已有标签与少量图像；没有执行检测器或改官方 split。输入是上一轮已哈希的 CD-ViTO `FISH.tar.gz`。这份补查不改变上一轮资产事实。

## 分组语义与证据强度

| 对象 | 能确认的含义 | 证据强度与边界 |
| --- | --- | --- |
| 20 个文件名首段数字 | 原作者代码 `slice_df_reg(df, habitat)` 用文件名第一个下划线前字段筛选 `habitat`；分类集则用目录字段筛选 | **强：官方代码的操作定义**。不能据此把每个编号称为已验证的独立 GPS 站点或独立相机部署 |
| 46 个完整前缀（去掉 `_f数字.jpg`） | 同数字编号下面还区分 `F1/F2/...` 或鱼名文本；有连续帧号，是本归档可复算的文件分组 | **中：序列线索；弱：具体视频身份**。没有找到前缀→原视频文件、剪辑、部署、GPS 的正式映射。不能说是 46 个独立视频/新场景 |
| `_f数字` | 文件内帧序线索，用于本次确定性抽样排序 | **中**。未核到原始帧率、是否抽帧或重编号，不能把帧号差换算为秒 |

官方代码固定在 [alzayats/DeepFish@1ddd7a6，`slice_df_reg`](https://github.com/alzayats/DeepFish/blob/1ddd7a62e9a6157992e5ecf7198210933e9fc855/DeepFish/datasets/__init__.py)；[fish_loc.py](https://github.com/alzayats/DeepFish/blob/1ddd7a62e9a6157992e5ecf7198210933e9fc855/DeepFish/datasets/fish_loc.py) 实际调用此函数。完整响应保存在 `sources/official_metadata_evidence.json`，没有运行原作者代码。

[原论文](https://researchonline.jcu.edu.au/64493/1/64493.pdf)第 3–5 页说明：图像来自 20 个 habitat 的水下视频；相机部署时记录深度与 GPS；原始任务有分类、点和分割标注，按每个 habitat 内部分层随机分为 50/20/30 的 train/val/test。它没有提供这 46 个前缀的映射，也没有解释此检测归档的框。论文原始 val 不能视为 CD-ViTO 检测归档已随附的 dev。PDF 已存本地并核 SHA-256；本次依赖论文文本核查，未安装 PDF 渲染依赖。

作者项目页提供 habitat 名称/数量表，并注明源表 `SummeryOFdatset_190915.csv`，但所读页面没有数字编号映射。原始归档目录的一次有界 GET 超时；未下载 7.6 GB 原始归档，因此内部 CSV/视频映射**未核**。官方页面也没有把完整前缀定义为视频或站点。[固定项目页源码](https://github.com/alzayats/DeepFish/blob/1ddd7a62e9a6157992e5ecf7198210933e9fc855/docs/index.html)

## 排除 support 所在组后的 train 余额

下面均从官方 `train.json` 的 3,596 张 / 12,434 框计算；没有写出新 split。

| shot | support 完整前缀数 | 排除这些前缀后：组 / 图 / 框 | support habitat 编号数 | 排除这些 habitat 后：habitat / 图 / 框 |
| --- | ---: | --- | ---: | --- |
| 1 | 1 | 45 / 3,531 / 12,369 | 1 | 19 / 3,517 / 12,355 |
| 5 | 3 | 43 / 3,305 / 12,122 | 3 | 17 / 2,831 / 10,533 |
| 10 | 7 | 39 / 3,018 / 11,645 | 6 | 14 / 2,663 / 10,792 |

因此都有非空、与各自 support 分组键不重叠的剩余记录。这里只确认分组键的隔离，未证明地理、鱼个体或拍摄会话统计独立。完整 46 组库存与 1/5/10-shot 身份见 `TRAIN_GROUP_INVENTORY.csv` 和 `GROUP_AUDIT.json`。若同时排除三个 shot 集的全部 8 个 habitat 编号，还剩 12 个 habitat、2,183 张、9,302 框；此余额也未生成 dev。

5-shot 的三个前缀是 `7463_F7`、`9866_acanthopagrus_palmaris`、`9870_Gerres`。只排除完整前缀时，编号 7463 和 9866 的其他前缀仍可能留在候选池。这正是前缀隔离与 habitat 隔离的区别。

## 检测框来源：已知链与未知链

能追溯的链是：CD-ViTO 官方 README → 官方 Drive 目录 → 已核文件 ID 与 SHA-256 的 FISH 检测归档。归档只提供图像和 5 个 COCO JSON，train JSON 顶层没有 `info`、`licenses` 或制作说明，类别是 `fish`。检测框确实随该基准发布，而不是本次从点标签造出。

[CD-ViTO 论文 §9.1 / Table 11](https://arxiv.org/html/2402.03094v3#S9.T11)引用 Saleh 2020，并列出 909 图 / 3,029 框；这与前一轮实测 test 对应。所查论文文字、README、原始作者说明、针对 habitat/bounding box/annotation 的公开 issue 检索，均没有补齐框的标注者、人工/自动派生方式、点或 mask 转换规则、质量复核与原始样本映射。**这些仍未知**。数字相合或框带半像素坐标都不是派生方式的证明。

原始 DeepFish 作者卡和项目页标 CC BY 4.0；检测派生标注的单独许可/署名说明在归档中未找到。保留原作者和基准来源，不把代码许可证当作检测标签许可证。[作者数据卡](https://huggingface.co/datasets/Alzayats/DeepFish)

检索在上述有界范围停止；另一个论文网页返回 HTTP 429，原始数据门户超时，均未反复重试。没有用其他人新做的 mask→box 数据证明本归档的来源。未知项限制来源审计的完备性；相邻帧和未知派生说明没有被用来自动宣判探索任务可用或不可用。

## 允许的结论范围

官方 split 可以明确用于同场景条件下的探索；本次 5-shot dev 只能称前缀组隔离。不能据此声称跨场景、跨地点或新视频泛化，更不能声称形成科研贡献或完成论文复现。本轮起封存官方 test；上一轮资产 QA 曾检查 3 张 test 图像，但没有任何模型输出或分数，本轮不再解码或查看 test，也不运行其 evaluator。
