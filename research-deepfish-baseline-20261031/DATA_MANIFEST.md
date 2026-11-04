# DeepFish / FISH 数据事实

检查日期：2026-10-31 UTC。本次只检查 CD-ViTO 发布的检测归档，没有下载原始 7,621,627,392 字节 DeepFish.tar，没有从点标注或分割掩码生成任何框。

## 来源链与尚未证实的部分

- [CD-ViTO / CDFSOD-benchmark README 固定版本](https://github.com/lovelyqian/CDFSOD-benchmark/blob/929413d7d61caf72808a8c27cce19bf51c2973dd/README.md) 明确链接到 [Google Drive 数据目录](https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q)。本地保存的该目录 HTML 将文件 ID `1Y7NEUZ4IJjHEvbmHuj0NcfnoVhvMasaW` 对应到 `FISH.tar.gz`。
- 文件实际大小 **761,948,029 字节**（约 726.65 MiB）。本地计算 SHA-256：`714ed9b731efab05ddfb4bd3b73127b67ebcb2a3a5f7efa07f66f314b9ad6094`。未找到发布方提供的 FISH 校验和，不能把本地哈希称为发布方认证。
- [Saleh 等，Scientific Reports 2020](https://www.nature.com/articles/s41598-020-71639-x)、[作者项目页](https://alzayats.github.io/DeepFish/) 和 [作者 HF 数据卡](https://huggingface.co/datasets/Alzayats/DeepFish) 对应水下 DeepFish。作者资料描述分类、点级定位/计数与分割，数据卡标明 CC BY 4.0。没有把其他同名数据集作为来源。
- 此次可追溯的是 **CD-ViTO 发布的 COCO 检测 JSON 与 JPEG**。归档仅有 JPEG 和 5 个 JSON；没有 README、LICENSE、原始点/掩码、标注者说明、框生成脚本或映射表。固定仓库的 DATASETS.md 仅给通用 COCO 转换/划分说明，不能证明这些框究竟由人工重标、掩码转换或其他来源生成。检测框制作来源仍未解决；没有与 7.6 GB 原始归档逐图核对。
- [CD-ViTO 论文 Table 11](https://arxiv.org/html/2402.03094v3#S9.T11) 的 909 图、3029 框与此归档 **test.json** 精确一致，并非整包总量。论文提供一个 fish 类，未在所读说明中解释检测框来源。

## 归档和实际标注

归档扫描检查了所有 4514 个成员，含 4510 个普通文件和 4 个目录；没有符号链接、硬链接、绝对路径、上级路径或重复成员。展开文件总量 776,254,041 字节。4505 个 JPEG 全部做了字节 SHA-256；只复制 5 个 JSON 和 6 张样例图，没有全量解包图像。

| 文件 | 图像数 | 框数 | 类别 | 实际图像目录 |
| --- | ---: | ---: | --- | --- |
| train.json | 3596 | 12434 | id=1, fish | train/ |
| test.json | 909 | 3029 | id=1, fish | test/ |
| 1_shot.json | 1 | 1 | id=1, fish | train/ |
| 5_shot.json | 5 | 5 | id=1, fish | train/ |
| 10_shot.json | 10 | 10 | id=1, fish | train/ |

总计 4505 个独立 train/test 文件、15463 个完整 train/test 框；support 是 train 的子集，不应重复加总。所有 JSON 声明尺寸为 1920×1080；6 张实际解码图尺寸相符。类别名称与 id 一致，support 的 supercategory 是 `none`，train/test 是 `object`。

每个 JSON 内图像 ID、标注 ID、文件名都无重复，图像引用均存在；未发现空图、无效类别、非有限坐标、非正框宽高或 area 与 w×h 不符。严格图像边界检查发现 train 137 框、test 34 框略越界；最大越界分别约 1.0 和 0.5 像素。三个 support 文件无此问题。原 JSON 全部保持不变；这些小幅边界偏差没有被直接判为数据损坏。框裁剪和 evaluator 的实际处理需未来固定环境核验。

## Support 身份和重复性

[SUPPORT_IDENTITIES.csv](reports/SUPPORT_IDENTITIES.csv) 给出全部 16 条 support 记录的文件名、图像 ID、标注 ID、原始 bbox 与图像 SHA-256。由于 5-shot 与 10-shot 共享一张图，合计 15 张不同 support 图像。

- 1-shot：image ID 2416，annotation ID 9085，`9852_Acanthopagrus_palmaris_f000070.jpg`，bbox `[1139.5, 606.5, 489.0, 215.0]`。
- 5-shot 图像 ID：930、1892、1900、2244、2363。
- 10-shot 图像 ID：1419、1425、1880、1990、2217、2260、2296、2363、3218、3355。
- 每条 support 图像及标注都与 train.json 的相应记录精确一致。此次这 15 张图在完整 train.json 中也各只有一个框，未发现因 k-shot 选取而遗漏的已有 train 框。
- 这三套 support **并不嵌套**：1-shot 与另外两套都不重叠；5-shot 与 10-shot 仅共享 image ID 2363。不能将 shot 数增加解释为同一 support 集逐步加样。
- 仅有一套发布的 1/5/10-shot 文件，未发现 support 随机 seed、重复抽样副本或生成日志。通用 kshot_split.py 使用 random.sample 且无固定 seed，但不能据此断言归档就是由该脚本按某 seed 产生。

## train/test 不重叠的精确范围

- train 与 test 的文件名交集、图像字节 SHA-256 交集均为 **0**；各自内部也没有字节完全重复图像。每套 support 与 test 同样为 0。
- train/test 的 numeric image ID 各自重新编号，有 909 个数值重合；它们对应不同文件，不能仅凭数值 ID 重合判定图像泄漏。数据跨 split 关联必须用 split+文件身份。
- 去掉文件名末尾 `_f数字.jpg` 后，train 和 test 各有 **46 个前缀组且全部共享**。909 张 test 图中，843 张在同前缀 train 图中存在文件帧号差 1 的图；另外 66 张最小差为 2–7。此为文件名邻近性统计，不等于已核实视频时间戳或站点映射。
- 1-shot 支持图 `...f000070.jpg` 与 test ID 609 的 `...f000071.jpg` 帧号差 1；实际六图检查中画面背景和鱼的位置非常接近。不能将文件级不重复扩展为视频、栖息地或场景独立，亦不能把全 train 的邻近图统计等同于实际 few-shot 训练看过全 train。
- **没有独立 dev/val JSON**。Domain-RAG 配置把 test.json 用作 val。任何后续研究均需事先解决调参/选 checkpoint 的依据，不能借 test 指标反复挑增强方案。此次未划新 dev、未修改官方 support/test。

## 许可及范围

作者项目页与 HF 卡对**原始 DeepFish**标注 CC BY 4.0，HF 卡写明数据拥有者 Michael Bradley / James Cook University。CD-ViTO 仓库代码有 Apache-2.0 LICENSE；这不自动证明检测派生标注获得同样的独立许可。FISH 归档无许可文件或 attribution 文件；当前应同时保留原始作者署名、基准来源和上述未决项，不宣称完成检测派生许可核验。本次没有再分发或上传。

检测框来源和分组语义的缺失是待补充的事实；资产能读取与科学用途是否成立是不同层面的结论。此次既不据缺说明直接判不可用，也不据可读性直接放行正式效用实验。

