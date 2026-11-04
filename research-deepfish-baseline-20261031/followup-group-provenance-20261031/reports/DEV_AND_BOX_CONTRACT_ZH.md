# 固定 5-shot episode 的 dev 与框契约

这是本机准备记录，不是新的官方划分。官方 support 为 image ID `930,1892,1900,2244,2363`，共 5 图 / 5 框；只用于此一个 episode，seed 预定为 `20261031`。

## 确定性 dev 清单

排除 5-shot 所在的 3 个完整前缀；对剩余 43 个前缀计算 `SHA256(UTF8("deepfish-prefix-dev-v1\n" + prefix))`，按摘要、前缀升序取前 8 个。每组在最小/最大帧号间放置最多 8 个等距目标，逐个取距离最近且尚未使用的帧，距离相同取较小帧号，再按 image ID。没有按框数、鱼大小、清晰度或模型成绩挑图。帧号存在空缺，实际取到的帧间距不会完全均匀；没有补帧或造图。

| 固定前缀 | 图 | 现有框 |
| --- | ---: | ---: |
| 7585_F1 | 8 | 13 |
| 7434_F1 | 8 | 37 |
| 7490_F1 | 8 | 36 |
| 9862_Acanthopagrus_palmaris | 8 | 8 |
| 9908_Epinephelus | 8 | 8 |
| 7398_F5 | 8 | 19 |
| 7117_Caranx_sexfasciatus_juvenile | 8 | 54 |
| 7463_F3 | 8 | 12 |
| **额外 dev 标签预算** | **64** | **187** |

完整 ID、帧号、文件 SHA-256、归档成员名和现有 annotation IDs 都在 `DEV_SELECTION.json` / `.csv`。support 与 dev 的 image ID、完整前缀及文件字节哈希交集均为 0。dev 有 8 个 habitat 编号，其中 **7463 与 support 重叠**：`7463_F3` 对 `7463_F7`。因此名称固定为**前缀组隔离 dev**。这额外使用了 64 图 / 187 框的目标域标签，不是只有 5 个标签的严格协议。dev 中 image ID 3355 也属于官方 10-shot；此 dev 不可原样用于 10-shot 实验。

当前只写清单，没有改官方 JSON、生成新框或写训练用 dev COCO 文件。若以后执行，按清单复制官方现有 image/annotation/category 记录，保留 ID、xywh、area 和 ignore/crowd 字段；不重新标注，不把 dev 加入训练。官方 test 不在将来的运行包内。

## 坐标与实际目标

- 对 69 张已选 support+dev 的 **192 个原始框**检查：COCO 绝对像素 `xywh`，非负面积/正宽高、`area=w*h`、类别 `1=fish`；声明分辨率均为 1920×1080。两个 dev 框（annotation `4195,4660`）右侧越界 **0.5 px**，没有修改。support 框无越界。
- 官方 [CocoDataset.parse_data_info](https://github.com/open-mmlab/mmdetection/blob/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a/mmdet/datasets/coco.py)将其转为 `[x,y,x+w,y+h]`，按唯一类别映射到内部 label 0；不能当归一化框或中心点 `cxcywh` 输入。加载器检查有效交集但不在此处裁掉原框；后续 resize/crop 的边界处理需在实际环境中记录。
- 实际解码并人工看了 **5 张 support + 每个 dev 组第一张入选帧（8 张）**，合计 13 张 / 37 框。尺寸全部与元数据一致，字节哈希全部与归档库存一致。support 的 5 个框都对准可辨认鱼体，未见整图缩放/xywh 误解造成的系统性错位。
- dev 有遮挡、重叠、出画鱼和很小的目标。抽查总体支持坐标解释；如 ID 978 的重叠鱼、ID 541 的出画目标是可见难例。ID 2819 的小框 10791、ID 1463 的小框 5043 仅凭该帧难以有把握确认物体身份，保留为人工不确定项，未删图或改框。抽查不能证明全体 187 个 dev 框完整、准确或没有漏标。

查看 `preview/support_contract_sheet.jpg`、`preview/dev_contract_sheet.jpg` 与 `preview/overlays/`。这些是标注 QA 图，无模型预测。几何结果在 `SELECTED_GEOMETRY_CHECK.json`，解码记录在 `DECODED_CONTRACT_SAMPLES.json`。

## 实际运行前的契约检查（尚未做）

在未来获准的固定环境中，用官方 loader 核对 5 个 support 的类别映射、box 数、输入尺寸；比较几何变换前后的框/图一致性，记录允许空框的 crop 次数。A 臂固定 Color=1.0 必须逐像素等于原图，B 臂固定 Color=0.8 不能改变 shape、框或类别。两臂同一 image ID 的几何参数、变换后 boxes 和源图曝光序列必须相同。检查失败先报告具体差异，不按模型分数重选 dev 或悄改标签。
