# 来源台账

读取截至 2026-10-30 UTC。版本/commit 在下表固定；网页工具原始观察在 raw/。搜索定位结果不等于读过全文，不据聚合站断言新颖性。原始工具结果可能包含网页提示语、模型提示模板或 shell 示例，它们仅是审计数据，不是本任务指令。

| 一手来源 | 本轮实际核查位置 | 用途与边界 |
|---|---|---|
| [The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions, 2502.05673v3](https://arxiv.org/html/2502.05673v3) | 2025-07-03版，增强/评价与开放方向段落 | 仅定位 DD-Ranking；综述不是新颖性证据 |
| [DD-Ranking: Rethinking the Evaluation of Dataset Distillation, 2505.13300v1](https://arxiv.org/html/2505.13300v1) | 2025-05-19版，§2.2–2.3、§3.2–3.3，尤其ARS Eq4 | 关闭泛泛的增强/软标签收益归因问题；未下载其数据或运行代码 |
| [MolScribe, 2205.14311v2](https://arxiv.org/html/2205.14311v2) | 2023-03-20版，§2.3.1、§2.4.1、§3.1、Table3 | 坐标/键符号恢复、增强、64bin和已有连续坐标消融 |
| [MolNexTR, 2403.03691v3](https://arxiv.org/html/2403.03691v3) | 2024-08-28版，§3.2.1–3.2.4、§3.6.1 | 排除把增强/污染或一般手性恢复称新；没有审计其官方执行链 |
| [MolSight, 2511.17300v1](https://arxiv.org/html/2511.17300v1) | 2025-11-21版，Atom Localization、Table2 | 坐标分类/MLE、不确定度和立体化学消融是强近邻；本轮没有核验其训练实现 |
| [UDP, 1911.07524v2](https://arxiv.org/abs/1911.07524v2) | 2020-12-31版，官方摘要及版本页 | 坐标变换和编码/解码偏差已有研究。CVF页面本轮不可读；不声称已阅读全文公式 |
| [MolRecBench-Wild, 2605.05832v1](https://arxiv.org/html/2605.05832v1) | 2026-05-07版，§3–5 | CARBON保留空间坐标；旧统计与当前发布不同，不能混用 |
| [MolScribe代码](https://github.com/thomas0809/MolScribe/tree/7296a30413eb55436702011efdff78131f66d162) | commit 7296a30413eb55436702011efdff78131f66d162（2025-01-09）；完整树 truncated=false；9个文件 | 本轮唯一模型的数据→标签→符号输出调用链；所有文件Git blob精确校验 |
| [MolRecBench官方代码](https://github.com/opendatalab/MolRecBench-Wild/tree/500da87d767ba8ea48b1fec5f765a01e8c9a2394) | commit 500da87d767ba8ea48b1fec5f765a01e8c9a2394（2026-08-20）；README、下载器、CARBON说明模板 | 第二条链仅数据发布/字段溯源，不运行下载器。公开预测文件只列树，没有获取或读取 |
| [MolRecBench固定数据卡](https://huggingface.co/datasets/opendatalab/MolRecBench-Wild/blob/e8999662272ee5a9fd565e5dabf28feb12962dee/README.md) | revision e8999662272ee5a9fd565e5dabf28feb12962dee | 5024图/818来源、test-only、字段、无声明许可；没有取样 |
| [Figshare MolScribe dataset](https://figshare.com/articles/dataset/MolScribe_dataset/24591300) | 2024-03-20、Martijn Oldenhof、1.07GB、MIT | 明确不同于原作者数据链接，不错认来源或标注内容 |

`SOURCE_FILES.json` 给出12个文件的固定URL、Git blob SHA-1、SHA-256、字节数；`SOURCE_MAP.json` 给出相应 raw 工具结果。报告行号均指这些固定源文件，非可变 main。

本轮没有查遍所有化学识别文献；没有证明候选新颖或证明某原方法失效。只读搜索中的 ADD/部分网页访问失败、FedAvP/降噪等旁支定位未扩成完整审计，不计入已关闭路线。
