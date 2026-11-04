# Ctrl-A 冻结机制筛选实测报告

完成时间（UTC）：2026-10-30T12:45:37.918127+00:00

资源门结论：**NO_GO_FOR_ONLINE_FORK**。未启动在线四臂；冻结实验没有检验后续全局反馈是否补偿。

同参数F相对S的改善为 seed0 0.470 [0.287, 0.663] pp、seed1 0.513 [0.307, 0.710] pp；均为小幅正向结果，但未达到预设+2pp资源门。这不是零效应结论。分别重新匹配后，完整audit区间验收为 seed0 3/3、seed1 0/3 通过；未通过者不能声称已排除平均难度差异。

## 共同 checkpoint 与预算

| seed | 初始 controller acc | epoch10 controller acc | epoch10 clean CE | 实际 epoch | 原46k曝光 |
|---|---:|---:|---:|---:|---:|
| 0 | 9.90% | 89.70% | 0.2845 | 10 | 20 |
| 1 | 9.40% | 90.60% | 0.2926 | 10 | 20 |

模型后端为 MPS / float32，Apple M5 Pro、20 GPU 核、64 GB；CPU Pillow 只负责官方预处理。真实训练 1,840,000 张（上限1,840,000），推理账本 4,236,891 image-forwards（上限8,000,000），另有2,008个合成训练/梯度前向。从首次完整模型测速到最后阶段账本的连续墙钟 42.94 分钟（硬上限480分钟）。Colab **0 CU**。

## 预注册主资源门

Δ_err = error(S) − error(F)，原 ASD、N5、同参数；正数表示F改善。95% CI按原图配对重采样，保留其全部15视图。每个 seed 都须点估计≥+2pp且CI下界>0；此门只用于资源分配。

| seed | Δ_err pp [95% CI] | 该seed过门 |
|---|---:|---|
| 0 | 0.470 [0.287, 0.663] | False |
| 1 | 0.513 [0.307, 0.710] | False |

## N2 冻结参照

| seed | renderer | N2错误率% [95% CI] | 平均CE |
|---|---|---:|---:|
| 0 | S | 18.533 [17.310, 19.694] | 0.533705 |
| 0 | E | 18.570 [17.350, 19.737] | 0.533633 |
| 0 | F | 18.537 [17.320, 19.707] | 0.532394 |
| 1 | S | 17.967 [16.860, 19.090] | 0.519623 |
| 1 | E | 17.920 [16.797, 19.053] | 0.519562 |
| 1 | F | 17.860 [16.743, 18.980] | 0.517513 |

## 冻结错误率、交互与难度匹配

表中为按官方末端操作概率1/15汇总。原始参数的三种渲染同参数；shared系数来自S的strength-match选择并共用于S/E/F；rematched分别调幅，不能称几何相同。

| seed | family | renderer | 系数 | 错误率% [CI] | E pp [CI] | 相对N2 CE差% [CI] | audit匹配合格 |
|---|---|---|---:|---:|---:|---:|---|
| 0 | raw | S | 1.00000 | 31.930 [30.740, 33.040] | 1.453 [0.803, 2.117] | 74.948 [69.540, 81.412] | False |
| 0 | raw | E | 1.00000 | 31.763 [30.567, 32.930] | 1.447 [0.810, 2.113] | 74.687 [69.305, 81.098] | False |
| 0 | raw | F | 1.00000 | 31.460 [30.270, 32.613] | 1.313 [0.676, 1.977] | 72.978 [67.667, 79.112] | False |
| 0 | same_parameter_matched | S | 0.58125 | 18.370 [17.173, 19.550] | 0.310 [-0.167, 0.810] | -1.931 [-3.546, -0.231] | True |
| 0 | same_parameter_matched | E | 0.58125 | 18.320 [17.107, 19.497] | 0.373 [-0.100, 0.887] | -2.057 [-3.671, -0.344] | True |
| 0 | same_parameter_matched | F | 0.58125 | 17.990 [16.797, 19.184] | 0.217 [-0.250, 0.720] | -3.549 [-5.162, -1.887] | False |
| 0 | renderer_rematched | S | 0.58125 | 18.370 [17.173, 19.550] | 0.310 [-0.167, 0.810] | -1.931 [-3.546, -0.231] | True |
| 0 | renderer_rematched | E | 0.58750 | 18.547 [17.337, 19.713] | 0.363 [-0.130, 0.870] | -1.331 [-2.966, 0.361] | True |
| 0 | renderer_rematched | F | 0.60000 | 18.450 [17.270, 19.647] | 0.110 [-0.367, 0.620] | -1.242 [-2.873, 0.449] | True |
| 1 | raw | S | 1.00000 | 31.247 [30.127, 32.393] | 1.377 [0.747, 2.033] | 76.324 [70.517, 82.860] | False |
| 1 | raw | E | 1.00000 | 31.113 [30.020, 32.227] | 1.243 [0.613, 1.900] | 76.157 [70.465, 82.739] | False |
| 1 | raw | F | 1.00000 | 30.733 [29.640, 31.873] | 1.187 [0.543, 1.827] | 73.624 [67.983, 79.916] | False |
| 1 | same_parameter_matched | S | 0.54375 | 17.270 [16.070, 18.460] | 0.350 [-0.097, 0.807] | -3.761 [-5.619, -1.862] | False |
| 1 | same_parameter_matched | E | 0.54375 | 17.223 [16.016, 18.387] | 0.320 [-0.120, 0.783] | -3.843 [-5.707, -1.963] | False |
| 1 | same_parameter_matched | F | 0.54375 | 16.913 [15.700, 18.103] | 0.247 [-0.187, 0.700] | -5.517 [-7.377, -3.692] | False |
| 1 | renderer_rematched | S | 0.54375 | 17.270 [16.070, 18.460] | 0.350 [-0.097, 0.807] | -3.761 [-5.619, -1.862] | False |
| 1 | renderer_rematched | E | 0.55000 | 17.267 [16.047, 18.417] | 0.300 [-0.140, 0.783] | -3.211 [-5.060, -1.372] | False |
| 1 | renderer_rematched | F | 0.56250 | 17.173 [15.973, 18.327] | 0.160 [-0.283, 0.637] | -3.534 [-5.312, -1.757] | False |

匹配合格要求 audit 相对CE差的点估计和完整95%CI均在±5%内。未通过的比较不能用来排除总体难度。匹配只约束平均CE，不保证损失分布、各类难度或语义保真相同。原始N2只是同一冻结模型的难度参照，未训练额外N2模型。

完整 conditional R、同底图 single R、ROR gap、E、新增错/纠正错及每项CI见 operation-metrics.csv 和各 seed JSON；低分母明确标记无效，不挑最坏组合。官方原始图 ROR 独立保存为151个条件，配对CI见official-ror-metrics.csv，防止把基础crop/flip上下文当成组合效应。这里的条件比值按固定抽样的前缀上下文分布汇总，不是每种固定组合的穷举结果。

## 审计、故障与限制

19项预训练测试通过；另有2-worker spawn冒烟测试。矩阵原32×32坐标、单操作、整数平移/逆操作、边界、顺序、光度屏障、分层数据与统计方向均有测试。首次MPS channels-last反向失败后仅改连续布局，CPU/MPS梯度相对L2差约2.02e-6；第二项故障是动态模块的spawn序列化，真实训练曝光0时修复。两次失败与恢复回执均保留，硬时限未重置。
冻结前后模型参数和BN状态逐项字节一致。无相邻仿射块的挑战上S/E/F预测及CE一致，见result-integrity.json。推理读取方式0/6 workers逐字节哈希一致，改0 workers减少启停开销，见reader-benchmark.json。源代码未修改，历史tracked文件保全检查另见preservation-after.json。
这是缩短机制实验：airbench94结构，两个seed各10实际epoch，非论文最终精度复现。代码提交e77a4c1早于v2；新分层不代表独立确认。图像bootstrap不代表训练seed不确定性。覆盖率不代表语义保真；E/F的prefix是裁回后的读出，full在连续块内保留渲染状态，细节见PRE_RUN_NOTES_ZH.md。扩展画布也改变边界填充参与插值的方式；Pillow uint8量化保留，不能把所有像素变化都归为纯粹信息恢复。非零交互不说明Ctrl-A论文承诺被推翻。
官方test批次未解包、未实例化、未评估；公开归档含test成员的压缩字节但从未读取其图像/标签。数据、权重、原始预测、日志和第三方源码受本目录.gitignore保护；未commit/push或上传。

## 来源与运行

- Ctrl-A v2： https://arxiv.org/html/2603.21819v2 （2026-10-26）。
- 官方代码： https://github.com/jbcdfm/ControlAugment/tree/e77a4c19a3db893c120fc18b774fa290e826592b （MIT，版本固定）。
- Lazy Resampling 的融合重采样先例： https://monai.readthedocs.io/en/stable/lazy_resampling.html 及 https://github.com/KCL-BMEIS/LazyResampling 。单纯仿射融合获益不是新贡献。
- CIFAR-10 官方下载与训练成员校验： asset-manifest.json；源码hash： source-manifest.json；接口差异： patch-manifest.json。

执行入口：`../.venv/bin/python -B run_stages.py`（本次已完成，不应直接重复执行）；独立阶段为`train_common.py`、`frozen_diagnostics.py`；汇总为`summarize_results.py`。协议在`protocol.json`与`protocol-lock.json`，采样在`sampling-manifest.json`，实际命令与文件hash在`run-manifest.json`及恢复/最终清单。所有运行均需当前获准且可见MPS的本机执行环境。

## 图与原始结果

- figures/training-health.png：学习健康度。
- figures/rendering-error.png：三组渲染错误率。
- figures/paired-rendering-gain.png：两个seed各操作的同参数收益。
- figures/conditional-interaction.png：各操作E及CI。
- figures/official-ror.png：原始图单操作ROR。
- raw/*.npz：全部固定样本预测、标签、CE；weights/*.pt：完整epoch模型/BN、optimizer、schedule、RNG、ξ/Γ/α。
