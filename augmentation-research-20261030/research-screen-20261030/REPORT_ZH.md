# Rand：量化候选关闭后的独立家族筛选

记录：2026-10-30 UTC。范围：一手文献、固定官方源码、数据卡和解析推导；没有模型调用、训练、数据集/权重下载、安装、费用、推送或封存测试访问。

## 当前结论

- 多视图响应类别比例估计：按实际 Pro 最终意见关闭，独立关闭记录见同级 `quantification-closeout-20261030/CLOSEOUT_ZH.md`。旧提案未改写。
- 数据集蒸馏的“收益来自合成图还是附加增强/软标签”：**不保留**。DD-Ranking §2.2–2.3、§3.2–3.3 已直接设计随机数据同配方比较和 augmentation-robust score。仅再次做这个比较不足以构成差分；不是说整个蒸馏领域关闭。[原文](https://arxiv.org/html/2505.13300v1)
- 化学结构图识别：不保留“原生镜像破坏手性”的说法——已读实现没有启用镜像；不保留泛泛的“坐标精度影响手性”提案。MolSight 已有坐标学习改进与立体化学消融；UDP 也早已研究坐标变换和编码/解码误差。[MolSight](https://arxiv.org/html/2511.17300v1)；[UDP](https://arxiv.org/abs/1911.07524v2)
- **仅提交 1 个窄假设给父任务/Pro 评审**：原生增强之后，即使完全预测正确的离散训练标签，符号解码是否仍会改变原有立体化学？这只是训练标签的语义一致性审计问题，不是新方法、贡献结论或真实失败证据。详见 `OCSR_REVIEW_QUESTION_ZH.md`。

目前合格的实证贡献数为 0；训练准入为 0。该窄假设的语义差分可明确陈述，但真实事件频率、标注质量、数据使用条件、软件兼容性都尚未通过准入。不能把“没有检索到同一句话”当新颖性。

## 已核实的实际调用链

MolScribe 固定 commit：`7296a30413eb55436702011efdff78131f66d162`。

1. 官方训练脚本 `scripts/train_uspto_joint_chartok.sh:25–29` 启用 `chartok_coords,edges`、动态 Indigo、增强及 64 档坐标。
2. `dataset.py:36–60` 同步图像/keypoints，旋转、白边裁剪、少量 CropAndPad、白边填充、降采样/模糊/噪声、resize；没有启用水平/垂直翻转。
3. `dataset.py:365–378,405–418` 变换后将坐标归一化，再送入 tokenizer。辅助 USPTO 路径可使用 `pseudo_coords` 的 min/max 重新归一化，不能与动态图像路径混同（`dataset.py:347–360`）。
4. `tokenizer.py:164–178` 使用 round(x·(B−1)) 与除以 B−1；不是 floor，也不是 B 个像素分母。
5. `chemistry.py:84–149,497–558` 从坐标和有向 wedge/dash bonds 进入 RDKit 立体化学恢复。其异常处理、非碳手性清除、官能团展开都必须与网格问题分开，不能笼统归因增强。
6. `augment.py:49–64` 的旋转 keypoint 路径还包含提前 int 转换。未运行，不称 bug；后续若仅这一细节解释结果，不能包装成新的增强机制。

[固定源码](https://github.com/thomas0809/MolScribe/tree/7296a30413eb55436702011efdff78131f66d162)，已读小文件原样及 Git blob 校验位于 `sources/` 与 `SOURCE_FILES.json`。

## 数据与可执行性约束

MolRecBench-Wild 官方发布代码 commit `500da87d767ba8ea48b1fec5f765a01e8c9a2394` 固定 HF revision `e8999662272ee5a9fd565e5dabf28feb12962dee`。数据卡说 **2026-08-19 版为 5,024 图、818 来源文章、只有 test split**，含 float64 coords、symbols、bonds 和 source DOI；这不同于论文 v1 的 5,029 图/820文章。没有读取样本或结果包。README 的代码 Apache-2.0 不自动覆盖数据，数据卡明确没有声明本版许可。后续不能把它假称已到手的自由训练集、独立 train/dev/test 或可任意再发布的数据。[固定数据卡](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild/blob/e8999662272ee5a9fd565e5dabf28feb12962dee/README.md)

MolScribe 官方另提供 USPTO、PubChem 和若干 benchmark 归档链接，尚未核验归档内容、尺寸或坐标字段。搜索所得 Figshare “MolScribe dataset”实际是 Martijn Oldenhof 的手绘分子数据（2024-03-20、MIT、1.07GB），**不是** MolScribe 原作者已核验的全套数据，不能错认。[条目](https://figshare.com/articles/dataset/MolScribe_dataset/24591300)

官方 requirements 依赖旧 OpenCV、固定历史 Albumentations、RDKit 等，不能因 Mac 已有 PyTorch/MPS 就声称这套 CPU 审计可立即运行。没有尝试安装或修改验收环境。

## 问题与访问状态

无自动审批拒绝，无执行中的命令，无不确定写入。CVF UDP 页面和一个 HF model README 通过 web 工具读取失败；UDP 的官方 arXiv 摘要可读，MolScribe 数据链接由固定 GitHub README 核验。这些局部访问失败不构成全局阻塞。是否值得继续窄假设的 CPU 先决审计，交父任务/Pro 判断；不得自行下载数据、运行解码或神经网络。
