# CNS-Bench 两种 shift 的零训练筛选协议 v1.0

本协议在读取本地预测结果、计算实际效应之前锁定。已知原论文 Figure 8 的论述；这不是完全盲于已发表结果的预注册，也不预先宣称新贡献。演讲与可视化工作暂停。

## 问题与固定范围

cartoon_style 的首次错误是否集中于 scale=1.5，而 heavy_snow 更渐进？区分类别构成、共同源图、首次错误后的恢复及末段持续错误后，原解释改变多少？固定两种 shift，不扩展或挑换。

固定 `resnet18.a1_in1k` 与 `vit_base_patch16_224.augreg_in1k`：这两个名字分别直接出现在官方 `benchmarking/results.ipynb` 示例中；前者是较小 CNN，后者是不同架构的监督 ViT-B AugReg。选择不依赖重分析结果。文件缺失则报告并停止对应分析，不找结果更有利的替代模型。官方代码快照：`3c09a993007b5844ccd799f8c523a00985aa31ad`。

## 资产与访问边界

只读取预计算预测、元数据及相关公开源码。不得训练、运行模型、启用 GPU、分配 Colab、下载权重或完整图像包。官方 Edmond 页面在本机正常浏览器访问时出现 Anubis 挑战，已停止该下载；不更换代理、隐藏 API 或绕过挑战。预测包的版本、许可和发布方 checksum 在本轮尚未亲自验收；72.8 MB、CC0 只作为委托方已查信息，不能标成已验证。GitHub 官方元数据是明确允许的独立来源。

原始预测和元数据置于忽略的 `raw/`，不提交、不 push。每项取得的资产记录 URL、抓取时间、字节数、SHA-256；GitHub 文件另核 Git blob SHA-1。只执行本任务的新分析代码，不导入历史训练入口。

## 验收、key 与完整性

预期 CSV 字段来源于官方 `prepare_csv_files.py`，须由实际包再次验证：`pred_class,gt_class,shift,class_str,img_name,class_name,seed,scale,<model correctness bool>`。允许 CSV 导出的无名 index；不静默猜字段或布尔值。逐行验证 correctness 与预测/真值一致。

唯一 key 为 `(shift,gt_class,seed,scale)`；类别名字与数值 id 的映射须一致，跨模型相同 key 的元数据须一致。模型用 key 的 outer merge 对齐，报告缺行，不按行号合并。官方示例 notebook 中的按 index 合并不沿用。

轨迹为 `(shift,gt_class,seed)`，显式按六个固定 scale `[0,.5,1,1.5,2,2.5]` 排序。重复 key、额外 scale、矛盾标签/元数据直接失败。缺 scale 的轨迹整条排除，逐类逐 shift 报告；这是完整性排除，不是右删失。发布前已被 OOC 过滤而未出现的轨迹不可从预测包推断，更不可编成删失记录。

## 原定义与状态

原论文第 3.3 节把 failure point 定义为最小错误 scale。官方 `EvalFilter.compute_failure_points` 实际取分组内第一条 False，依赖输入顺序；我们在单个 shift 内先排序以实现论文定义，并用无第三方依赖的等价实现核对。

原 notebook 的 histogram 分母是该 shift 的全部 scale=0 行数；baseline 错误计入 scale=0，全对轨迹不占任何 failure bin，但仍留在分母。官方 margin 计算将全对轨迹临时赋值 3；这仅是实现中的汇总约定，不当成真实观测失败点。

每条轨迹输出：首个观测点是否错误；首次错误位置 F（全对为 null/∞）；correct→wrong 与 wrong→correct 的次数；首次错误后是否曾恢复；终点是否正确；若终点错误，末段全错的起始 scale T（最后一次正确之后的位置；全错为 0），终点正确则为 null/∞。只有最后一点错误时 T=2.5。T=1.5 不保证 F=1.5，因为此前可能错误后恢复。所谓“持续”仅指剩余已观测点，不断言永久、不可逆或两点之间持续错误。

恢复率必须按首次错误位置分层。首次错误出现在终点的轨迹没有后续恢复机会，恢复率记 NA，并另报该层分母；不将其当作零恢复证据。另报逐 scale accuracy 和两方向转换。

## Cohort 与同源资格

1. **原发布完整 cohort**：每个模型/shift 各自的全部完整轨迹，包含 baseline 错误。复现原 failure histogram 分母。
2. **类别等权敏感性**：先按完整性而非正确性固定两模型、两 shift 均有完整轨迹的共同类别集 K，再算各类别比例并对 K 等权。K 固定用于所有 cohort；若任何 cohort 在 K 中出现空 eligible 类别，则该标准化结果支持不足，不各自删类重新定义 K。报告单侧类别和空 eligible 类别的数量。对原 cohort 不进行结果依赖重采样。
3. **按 key 匹配的完整 cohort**：两 shift 共同 `(gt_class,seed)`。报告原始/完整/交集/排除数及类构成。
4. **该模型的共同 baseline-correct cohort**：同一个 key 在两个 shift 的 scale=0 都正确，并且两条轨迹完整。两侧严格使用同一 eligibility 分母。不能各自选择 baseline-correct 后宣称配对。
5. **跨模型共同 baseline-correct 敏感性**：同一 key 在两个模型、两个 shift 的四个 baseline 均正确且完整；仍分别报告模型，不把模型当独立随机重复。

先在所有完整匹配 key 上比较两个 shift 的 scale=0 `pred_class`、`gt_class`、图片标识，报告不一致；预测相同不是同图证明。严格同源主分析还需发布元数据中的可核共同基图身份或两侧真实 baseline 文件的相同 SHA-256。当前 harness 用可追溯的 baseline SHA-256 manifest 验证，不下载整包图像。

若不能证明源图身份，输出只能称“(class,seed) 匹配的探索性分析”；严格配对主分析及其继续投入判定标记 unavailable，不把 seed 相同写成同一图。在未证实 scale=0 确为 clean 基图之前，资格条件正式叫 **first-observed-scale-correct**，不叫 clean-baseline-correct。首点错误在全轨迹敏感性中明确保留，不悄悄删除。

首轮只允许每个 source/shift/model 一个六点轨迹，采用 equal-source 权重。多变体或一个基图哈希对应多个 seed/key 则报告重复源或别名并停止严格主分析，不将它们偷偷作为独立样本；扩展变体需另定协议，不在本轮据结果临时选择。

## 固定端点与分解

对每个模型分别定义 `Delta = P(cartoon first_error=1.5) - P(snow first_error=1.5)`，比例的分母是相应 cohort 的全部轨迹，不是只在失败者中条件化。

报告原发布样本 Delta、类别等权 Delta、共同 baseline-correct 的首次错误 Delta；再在完全相同的共同 cohort 上把 first error 替换为末段连续错误起点=1.5，得到 terminal Delta。所有图表同时保留六个 scale，不只展示主位置。

固定、分别报告两个主要变化：`D_C = Delta_common_first - Delta_released_first`（cohort 变化）与 `D_E = Delta_common_terminal - Delta_common_first`（端点变化）。D_E 的 cohort 和权重严格相同；两者不能包装为因果中介。可另列代数总变化 D_C+D_E，但不作为主要端点或掩盖两者方向抵消。单位为轨迹比例的百分点，不是准确率差。类别等权版本和跨模型共同首点正确版本是敏感性，不替换主端点。

## 不确定性

4000 次 bootstrap，固定种子 20261030。抽样单位是 source key 的整组记录，保留六点、两 shift 和两模型的一起出现/缺失状态；不得把 6 个点或多个模型拆成独立样本。为了比较有重叠的原 cohort 与共同 cohort，从两种 shift/两模型的 source-key 并集按类别分层抽样，在同一 bootstrap replicate 内重新计算所有固定 cohort 的统计量。类别样本数固定；报告类别等权结果。

敏感性另以整类别为 cluster 做有放回抽样，保留类内全部 source key。这涉及不同推断目标（将类别视为可重采样对象），不任意选择较窄的区间。bootstrap 是对已过滤发布 cohort 的条件性重采样，不估计 OOC 删除机制，也不恢复总体代表性。

两个固定模型的 D_C、D_E 采用同一 bootstrap 样本，以四个固定坐标的最大绝对居中偏差的 95% 分位数构成同时区间；另报各端点 percentile 区间。不分别算两个 cohort 的区间后再相减上下界。小样本导致空分母的 replicates 明确报告，不静默补零；有效重复比例不足 99% 时判定未定。零事件或无变异 bootstrap 可退化为 [0,0]；必须标记退化、不能据此证明无事件或稳定，小分层只作谨慎描述。主要同时区间半宽为零时资源判断固定为未定。

## 预先固定的资源停止门

阈值为 **5 个百分点的主差分解释变化**，这是是否继续投入的资源门，不是统计真理，不等于 5pp accuracy，也不作为论文显著性宣称。

在同源资格成立、验收通过且有效 bootstrap 足够时：若两个模型的 D_C、D_E 的整个同时区间都落在 (-.05,.05) 内，关闭此候选、报告在预定资源门下的稳定性；若至少一个区间整体位于该范围之外，最多进入独立语义核查，不自动称新贡献；任何区间跨门则未定。不是检验不显著就等于无影响。类别 cluster 敏感性与主判定冲突则降为未定。不能因结果改变阈值、模型、shift 或主要端点。

恢复很少或原解释保持也是有效终点。ImageNet-P 已研究 prediction flips；统计翻转本身不算新机制。若发现值得继续的变化，还必须独立检查源图身份、语义质量、过滤偏差和既有文献。

## 已知限制

六点离散轨迹不是连续物理过程；相同 scale 不代表两 shift 的相同物理强度。匹配和 baseline-correct 条件化可能产生选择偏差。结论只适用于发布后保留的 cohort、两个既定模型和两种 shift。无训练、无新模型性能、无语义因果解释。

来源：
- https://arxiv.org/html/2507.17651v1 （3.3；Figure 8）
- https://github.com/odunkel/CNS-Bench/blob/3c09a993007b5844ccd799f8c523a00985aa31ad/benchmarking/results.ipynb
- https://github.com/odunkel/CNS-Bench/blob/3c09a993007b5844ccd799f8c523a00985aa31ad/evaluation/metrics_computer.py
- https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MAGNKP （本轮遇挑战，未下载）
