# LoRA位置 × RandAugment：四格开发实验结论

四格已在Mac的真实MPS设备上完成，每格30 epochs、1110次更新。预先冻结的主交互为 **+0.33956个百分点**，未达到原定 **1个百分点工程投入门**。两个位置中RandAugment都有小幅开发集收益，但收益差不足以支持继续自动扩跑这一版定向假设，也没有成立的新贡献。此结论保留为单seed、共同优化配方的开发结果。

## 问题、协议和公平性

本轮问题是：相同可训练参数量下，ViT前部和后部Q/V-LoRA的位置，是否改变固定RandAugment对干净照片错误率的收益？这是用户扩展范围后的普通增强研究；没有把随机增强称为因果方法，也没有宣称识别了数据生成过程中的因果干预。

主量预定义为：

`G_front = E_front,weak − E_front,RA`

`G_back = E_back,weak − E_back,RA`

`I = G_front − G_back`，预期方向为正。

协议于2026-10-22 **14:51:37 UTC**冻结，科学训练于 **14:52:39 UTC**启动。协议SHA256为 `9386fc378f876cf12c68ef5d7248f073d7313433d158350b4143942ac1b52a3f`；锁、数据清单和七个核心源码的哈希均保留。详见[中文协议](protocol-zh.md)、[机器协议](protocol.json)、[协议锁](protocol-lock.json)与[锁定时序](lock-start-timeline-check.json)。

- 相同torchvision ViT-B/16、ImageNet1K_V1底座；前部blocks 0/1/2与后部9/10/11，均Q/V LoRA r4、缩放1。Adapter 36,864参数，37类头28,453参数，合计 **65,317**。K增量为零，其余底座冻结。
- 相同按slot的A初始化、零B、分类头、FP32、AdamW、学习率、调度、batch16、源照片及其顺序。每个增强条件下，跨位置逐epoch的RGB、归一化输入及source-order SHA全部相同。图像patch16/token序列与预训练位置嵌入相同，无文本分词器。
- Weak为温和随机裁剪及水平翻转；RA在同一weak图像后加入torchvision完整14操作池的 **N2/M9**。没有选择性删除操作或按结果调强度。
- 同一seed **20261022**，每格30 epochs、17,730次源照片呈现；总计 **4440次更新、70,920次呈现**。固定最终epoch30，保留epoch0/5/10/20/30输出；没有挑最佳epoch、类别或替代主指标。
- 执行顺序为front-weak、back-ra、back-weak、front-ra。共同学习率只定义这一配方下的交互，不保证不同位置都处于各自最佳学习率；位置相关的梯度尺度/优化敏感性未被排除。

草案10 epochs在看任务结果前按实际dummy步耗时规则统一选择30 epochs。Dummy使用人工目标、模型丢弃，不产生任务准确率；预测计算及节奏时间2932.27秒，加25%余量3665.34秒，符合原7200秒cap。没有看到结果后延长某个cell。

## 固定开发结果

评估照片固定为589张，指标单位为源照片。错误率越低越好；NLL使用自然对数。

| 位置/增强 | 错误数 | 错误率 | NLL（nat） |
|---|---:|---:|---:|
| front / weak | 42 | 7.1307% | 0.237033 |
| front / RA | 36 | 6.1121% | 0.234124 |
| back / weak | 40 | 6.7912% | 0.211945 |
| back / RA | 36 | 6.1121% | 0.209978 |

因此，`G_front=+1.01868`个百分点，`G_back=+0.67912`个百分点，**`I=+0.33956`个百分点**。两组RA的最终错误数相同。弱增强的front−back基线差为+0.33956个百分点，RA差为0；主量只相当于收益差的净2个错误计数单位/589，不能说成两个独立动物受益。

工程门在拟合前已固定为 `I ≥ 0.01`，用于决定是否值得投入更完整的验证，量级约为每592张净6个错误计数单位。它不是显著性、功效、等效性或新颖性标准，也没有事后加上区间下界门。**本轮点估计未过门，停止自动追加seed、学习率搜索或确认集读取。**

在固定四个模型、保持37个观察breed样本数的条件下，5000次配对源照片bootstrap给出区间 **[−0.84890，+1.69779]个百分点**。它只表示该开发池的照片重采样敏感性，不包含训练随机性、未知动物身份依赖或总体覆盖保证。区间同时包含0和1个百分点，不能确认一般正交互，也不能证明等效或排除有实际量级的差异。

两个RA收益均为正，因此本轮也不是“增强无用”的证据。次指标NLL交互为+0.00094121 nat；不改用它、macro或某类成绩挽救工程判定。全部绝对结果、37类结果和逐照片结果见[主分析](analysis/summary.json)、[逐类CSV](analysis/per-class.csv)、[逐照片CSV](analysis/per-image.csv)及[配对数组](analysis/paired-errors-and-bootstrap.npz)。

最终训练CE按front-weak/back-ra/back-weak/front-ra顺序约为0.017417/0.023309/0.013391/0.019025。epoch20→30错误数分别为41→42、37→36、40→40、36→36。没有明显的数值失败或完全未学会迹象，但低loss、非零更新及晚期变化不能证明充分收敛或排除优化差异。每格完整曲线、1110步日志、小型可训练权重和输入哈希均在[results](results)中。

## 数据许可、标签与独立单位

使用Oxford-IIIT Pet原trainval中的固定公开镜像版本。当前官方页面的CC BY-SA4.0说明与旧归档的研究/原网站说明一并保存；仅在本Mac研究使用，没有发布或重新分发。来源固定revision、范围读取、CRC、字节数及SHA有回执，详见[许可](data/license-receipt.json)、[官方页面快照](data/official-pets-page.html)、[来源核验](data/photo-mask-receipt.json)和[署名](qa/attribution.md)。

先排除旧开发资料177个JPEG ID，再从新候选中保守移除两个相同拍摄簇的全部4张、不补图：Bombay_132/Bombay_107，以及keeshond_103/keeshond_167。最终训练591张，开发589张；各格源照片及呈现次数完全相同。初始1184张train/dev照片和对应mask保留作QA追溯，模型只使用清理后的1180张。已知重复已剔除，动物身份和ImageNet预训练重合仍未验证，不能把照片、增强draw或多视图当独立物体。

品种身份是原标注目标。拟合前37个预选source/weak/RA例子分别由主线和独立分工查看，其中一例后来因重复簇移除，剩36例；这支持可见性检查，不能证明所有draw保留细粒度品种线索。Solarize等可能改变颜色/纹理诊断线索，裁剪可能损伤局部证据。几何mask检查没有完全前景丢失，也不能替代语义标签证据。

三个全背景trimap的照片仍清楚可见宠物，保留分类标注；mask仅作不可用的辅助代理，不按结果删图。原QA失败及修正前源码保存在[首次QA尝试](qa-attempt01)；mask问题和视觉裁定见[QA](qa/source-qa.json)、[重复簇裁定](qa/duplicate-adjudication.json)、[视觉检查](qa/visual-review.json)。没有把可见性代理改称真实标签保持证书。

592个预留确认候选成员没有按名请求、提取、保存、解码、评分或用于选择；公开metadata的ID与类别在预分层时已知，不能声称盲标签。ZIP范围缓冲包含512字节头部余量，可能短暂触及相邻归档字节后丢弃，不能声称任何预留成员的传输字节都未触及。此边界在拟合前已明确记录于[data/range-transfer-scope.json](data/range-transfer-scope.json)。旧冻结测试保持封存。

## 独立复算与严格失败的保留

独立分工使用CPU1线程，核对全部20份开发NPZ、IDs/labels/predictions、四格risk/NLL、G、I及所有5000个bootstrap值；bootstrap最大差0，NLL汇总最大差约1.94e−16。还核对参数预算、初始化、变化、冻结回执、步数、呈现次数、输入配对、协议锁及原始文件SHA。没有重训、重新做ViT前向或独立重演梯度/优化器/底座重建，结论限于保存输出、CPU参数算术、源码和运行回执。

**原预存严格验证脚本没有通过。** 它在首个Q增量Frobenius范数的FP64 oracle比较处失败，原脚本、stdout/stderr和失败JSON完整保留。保留原容差的续查发现24个Q/V范数中18个超过该FP64比较容差；不是把整套原脚本重写成通过。

诊断显示训练代码对Q/V增量转CPU后使用FP32 norm。用同一保存参数执行CPU FP32乘积及FP32 norm，**24个报告值全部精确复现，差0**。只把同一CPU乘积的norm改用FP64后接近完整FP64乘积范数，最大差约8.35e−10；原FP64 oracle与报告值的最大相对差约7.75e−5。此证据说明归约定义差异，未逐元素验证MPS矩阵乘积。

另存的同定义v2仅更正reported Q/V norm的oracle定义、增加精确相等要求，同时保留FP64数值、原B范数谓词和全部科学统计容差；**897项检查通过**。原严格失败、18项超容差与v2修订diff一起保留，训练、数据、主量、bootstrap和工程门未改变。

完整证据见[独立结果审查](independent/finding-review-zh.md)、[原失败](independent/result-verification-failed.json)、[保留差异的续查](independent/result-verification.json)、[范数定义核验](independent/norm-definition-verification.json)及[同定义v2审计](independent/result-verification-v2.json)。

## 真实设备、时间和后台

实际使用Apple M5 Pro/64GiB上的 **mps:0**，CUDA不可用、MPS fallback关闭；CPU1线程、进程nice10、串行小任务，按MPS推荐工作集设置0.1比例上限，并在同步计算段后暂停其耗时的2倍。没有测得GPU利用率，不能把节奏规则写成实测占用率。

2026-10-22 14:52:39开始，15:53:57完成，墙钟 **3679.29秒（61分19秒）**，包含为8A优先临时SIGSTOP的 **334.00秒**。15:27:29对同一PID49366恢复，没有重启、重置优化器或预算。扣除已知暂停为3345.29秒；原7200秒cap未延长，完成时余3520.71秒。父线程确认8A的CPU任务于15:32 UTC分钟内完成；未访问8A文件、环境或进程。

进程RSS峰值约2.47GB，采样MPS driver分配最大约4.93GB。训练同步段合计956.55秒，评价同步段84.19秒；这些含host检查、同步和传输，不是GPU kernel纯时间。CPU增强、哈希、节奏、评价和暂停影响墙钟，不能把cell时间当无竞争的模型速度排名。[运行核算](runtime-accounting.json)与[资源暂停/恢复](resource-resume-01.json)保存了逐格数字。

原运行工具已返回退出码0，15:54:28关闭回执确认PID49366不再存在。独立审查已结束，无额外seed、备选训练、调度或本轮后台训练。[退出回执](execution-exit-receipt.json)、[后台状态](background-status.json)、[最新状态](LATEST_STATUS.json)可检查。额外一次收尾`ps -p 49366 -o pid=,stat=,etime=,command=`被沙箱拒绝（exit127，operation not permitted），未重试或换路；不把这次未执行的检查写成成功，后台结论依据前述原会话退出及关闭回执。详见[收尾检查边界](closeout-checks.json)。

所有代码、数据、实验和报告留在Mac。无新依赖、A100/Colab、ARiS、付费、上传或发布。项目目录不是Git仓库，分支不适用。请求配置保留GPT-6.1-Sol / max / Fast；没有暴露实际后端回执，不能声称逐轮验证档位。

## 查新、停止判定和下一检查点

本轮依照提案阶段已读的Surgical Fine-Tuning、Hint-Aug、DynamicAug、CoCa主来源章节判断最近邻，具体内容、精确链接及已读范围见[查新边界](novelty-boundary-zh.md)。它们已覆盖位置适应、PEFT增强敏感性、动态RA/LoRA及LP/LoRA增强响应。本轮没有新调度算法，四格未在已读部分见到不等于全领域无人做过。普通优化、预训练特征、有限样本正则化及标签线索破坏仍足以解释潜在收益。

Geiping的OpenReview正文仍因历史浏览器验证页未取得，不能说精确先例查清。其他历史验证/403/受限页面及Flowers README不可重试失败均未绕过。本轮Pets许可和公开数据路线可用，当前执行没有等待许可或下载的阻碍；剩余缺口是跨seed稳定性、布局相关优化敏感性、独立动物/标签证据与更强参照，尚不足以支撑原创结论。

按原工程门，**停止这一版共同配方定向假设的自动扩张**。不为救符号补seed、调阈值、改主指标或开确认集；不同时训练备选。该决定不证明所有位置×增强交互都不存在。若日后有独立科学理由继续，需要另行落盘有公平学习率敏感性、预定配对seed、全层LoRA/全量微调参照及新确认隔离的协议；本轮未自动启动这些工作。

旧smallNORB阴性45个CPU读出及19份交付、此前J*解释/被否决代理候选/机制图、11月4日课堂原论文与两个可视化页面均按字节/SHA保留；没有改写旧预注册结果。旧J*近零表示两组增强收益差近零，不等于增强本身没有收益；旧工程量级门也不是统计证据。

入口为项目根目录的[最终检查点](../../CURRENT_CHECKPOINT_20261022_LORA_PILOT.md)。完整本轮文件清单和字节/SHA核验见[delivery-manifest.json](delivery-manifest.json)、[delivery-verification.json](delivery-verification.json)；[preservation-final.json](preservation-final.json)核对旧17份提案、58份研究及25份其他受保护材料。代码入口为[train.py](train.py)、[common.py](common.py)、[analyze.py](analyze.py)，独立脚本及版本差异留在[independent](independent)。本轮结论可检查，下一步处于停止自动扩跑状态。
