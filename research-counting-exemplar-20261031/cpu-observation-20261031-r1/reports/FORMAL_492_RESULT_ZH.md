# 固定 CountGD++ CPU 观察：492条完整完成，预定两项门均未通过

本轮得到完整、可复核的观察结果，但**没有把此候选推进为可靠的新贡献**。执行已终止，查询完整，数值门未通过；这不是工程失败造成的科学阴性。

| 预定指标 | 实测 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| G：逐pair取C/V较小误差后再减F，跨41pair平均 | 0.02868010 | ≥0.10 | 未通过 |
| F≤0.10且C≥0.30、V≥0.30的pair数 | 1 | ≥10 | 未通过 |

| 模式 | 输入 | 41pair平均归一化误差 |
| --- | --- | ---: |
| F | 完整正负细类名称、原3正3负示例框 | 0.47141944 |
| C | 共同物体名称、负文字空、原3正3负示例框 | 0.56917801 |
| V | 正负文字均空、原3正3负示例框 | 0.61821953 |

G不能由上表三个总体均值直接相减得到：先对每一pair取min(C,V)，再跨pair平均。这一选择使用GT，是oracle包络，不是可部署的选择器。误差未截断。

每图每模式 E=(|预测A−真值A|+|预测B−真值B|)/(真值A+真值B)，先合并A/B两个查询；pair误差为两张固定图的E均值。全部阈值使用Fraction精确比较，边界包含等号。真值只绑定冻结labels。

唯一满足单pair条件的是 `FOO_INTRA_PEP1_PEP2`（黑/白胡椒粒）：F=0.05943993、C=0.48816621、V=0.49060524。该单例不足以满足跨pair门。双门未触发，因此本次未开展支持性目标选择框图审查；没有声称其框定位正确，也没有事后删除pair或重算门。所有35754个预测框和分数均完整保留。

## 完整性与执行证据

- 原47pair/94张固定选图元数据保留；效果前冻结41pair/82张图进入正式观察，F/C/V×A/B共492个唯一查询，无缺失、重复或额外查询。
- 492次root forward全部返回；重试0、训练0、GPU操作0、网络调用0、pickle调用0、新增CU 0。
- 200项冻结文件/资产的字节数和SHA256通过；992项输出在worker原子落盘副本与stdout收集副本中逐项一致。
- 独立脚本直接从冻结labels和保存的预测计数用Fraction重算，不导入生产metrics.py；逐图、逐pair、全局G和门结果与生产输出完全一致。
- 每query核对原图hash、查询hash、原注释身份、F/C/V文字、A/B及6个原框身份、计数/框/分数长度与有限性。padding logits为预期的负无穷；active logits、坐标和分数有限。
- 无查询达到800预测；基本app路径的无adaptive crop饱和情况为0。本轮仍不是完整论文adaptive-crop/pseudo-exemplar路径复现。
- 主控23941、worker23943、caffeinate23944均已确认退出。没有遗留CPU模型进程。

主控通用终态字符串为`COMPLETE_492_PENDING_BOX_REVIEW`；本轮完整性已通过，且双数值门未触发，未据此作正确目标选择解释。此字符串不表示缺少查询或仍在推理。

## 资源与实现范围

正式阶段2709.205781秒（45.153分钟）；含此前工程验证的正式退出累计3310.373825秒（55.173分钟），低于7200秒。额外独立核验及报告写入另记入[最终预算账本](FORMAL_492_BUDGET_LEDGER.json)。峰值RSS 4.445 GiB，低于8 GiB。实际加载后2倍成本准入预测7106.277093秒，已通过。此前90分钟草案已在任何真实效用前被累计120分钟最终决定替代。

范围固定为Mac CPU、FP32、线程2/1，CountGD++ HF revision `06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1` 的基本app路径，confidence .23、短边800/max1333，纯PyTorch attention，6个示例框及名称均未改变。主权重SHA256 `94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5`。严格加载1042物理键、66个真实共享别名、1108逻辑键，缺键/多键0。CPU重复性通过不能证明CPU比CUDA更正确；本结果不宣称与官方CUDA或论文完整实现等价，也不泛化到CountEx或整个领域。

冻结SHA256：`649a171fb5bd1d659aa43675e072024ad6c5aa7fba75b0efefa70d481517a79b`。原MPS/CUDA工程失败、旧freeze和原始日志全部保留；它们没有被当成科学阴性。此次未新增GPU分配、下载、安装、上传、提交或推送。

## 效果前NA的6pair

以下6pair均保留为NA，未跑来充当干净观察，未替换图、修框、修名称或将误差记为0。

| pair | 冻结原因 |
| --- | --- |
| FOO_INTRA_SED1_SED2 | 一张固定图的整组种子方向与另图/作者类名相反 |
| FUN_INTRA_PLC1_PLC2 | 一张固定图B3为红心牌，负视觉组不纯 |
| HOU_INTRA_COI1_COI2 | 不能从现有原图逐一确认每个框的硬币面额 |
| HOU_INTRA_CTB1_CTB2 | 杆材质与重叠框无法逐一独立确认 |
| HOU_INTRA_TPK1_TPK2 | 牙签/牙线棒的共同物体名未充分确定 |
| OTR_INTRA_BUT1_BUT2 | 官方文字与框相反，计数真实类别归属未核定 |

全部47pair数值/NA表：[ALL_47_PAIRS.csv](ALL_47_PAIRS.csv)。完整逐图预测计数、pair误差、精确分数、运行范围及退出证明：[FORMAL_492_RESULT.json](FORMAL_492_RESULT.json)。

结果证据位于[逐query结果目录](../formal/artifacts/results)、[原始汇总](../formal/artifacts/science/summary.json)、[完整输出hash索引](../formal/artifacts/artifact_index.json)和[主控终态](../formal/controller_report.json)。复核代码：[verify_formal_492_independent.py](verify_formal_492_independent.py)。交付清单：[FORMAL_492_DELIVERY_MANIFEST.json](FORMAL_492_DELIVERY_MANIFEST.json)。冻结源码及权重许可沿用已审资产记录。

当前无执行阻塞。科研判断是：此观察候选未通过原定投入门，尚不足以建立可靠新贡献；不等于证明相关科研问题不存在。
