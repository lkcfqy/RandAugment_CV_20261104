# TTA-conformal 机制诊断：关闭本候选：未达到预定新机制继续门

本次完成了独立CIFARResNet20的80 epoch训练、20k图像十视图缓存、四臂全部7温度/4评分/2alpha的评估与999次配对分组重采样。结果仅针对固定seed0这一模型及内部heldout，不能推广成多个模型/训练seed结论。候选结束不表示整体研究停止；没有自动追加本候选实验。

## 主结果与停止决定

预定A门：**未通过**。Rp/Rq未达到预定异号且两侧均超过0.25标签的同时区间要求；至少一臂覆盖率同时下界未达到89%；存在严格数值饱和标记。

预定B主门：**未通过**；计入反证后的B门：**未通过**。至少一个预定参数或温度的tuning覆盖/类覆盖约束不可行；至少一个原输出温度的缩集优势不超过同时下界0.25；类覆盖Dhat可比性未全部证实；边际覆盖及1pp容差未全部证实；恢复原排序后的实质优势消失未证实；恢复原排序造成至少0.25标签衰减未证实；未建立超出SAPS/rankonly已知机制的优势。

APS、T=1、alpha=.1：

| 臂 | 平均集合大小 M | 覆盖率 | Dhat | 校准阈值 |
|---|---|---|---|---|
| h00 | 4.9043 | 90.83% | 0.000592 | 0.957422 |
| h10 | 4.6714 | 91.11% | 0.000459 | 0.953088 |
| h01 | 5.4104 | 90.48% | 0.000543 | 0.922188 |
| h11 | 5.1075 | 90.69% | 0.000332 | 0.914848 |


Rp=M00-M10，Rq=M01-M11，J=Rq-Rp；正值表示替换rank后缩集，单位均为标签数。四臂都独立calibrate，共用每样本U。下面的区间控制整个预注册检验族，不是逐格普通95%区间。

| T | Rp及95%同时CI | Rq及95%同时CI | J及95%同时CI |
|---|---|---|---|
| 0.5 | 0.129 [-0.463, 0.721] | 0.346 [-0.232, 0.925] | 0.218 [-0.441, 0.877] |
| 0.75 | 0.204 [-0.464, 0.873] | 0.437 [-0.224, 1.097] | 0.232 [-0.570, 1.035] |
| 1 | 0.233 [-0.482, 0.948] | 0.303 [-0.271, 0.877] | 0.070 [-0.724, 0.864] |
| 1.5 | 0.269 [-0.405, 0.944] | 0.446 [-0.307, 1.199] | 0.176 [-0.603, 0.956] |
| 2 | 0.464 [-0.452, 1.381] | 0.278 [-0.547, 1.104] | -0.186 [-1.144, 0.772] |
| 3 | 0.264 [-0.549, 1.077] | 0.402 [-0.408, 1.212] | 0.138 [-0.661, 0.937] |
| 5 | 0.253 [-0.479, 0.985] | 0.402 [-0.322, 1.126] | 0.149 [-0.507, 0.806] |


90%同时区间完整位于[-.25,.25]的局部等效项：无。其余不能仅因“不显著”写成等效或无作用。

![机制诊断图](analysis/mechanism-diagnostic.png)

图中保留所有温度点。灰带为±0.25标签；右图只是有限网格点，不代表连续温度族前沿。描述性top1准确率：原输出65.96%，十视图均值68.14%；这不是官方test benchmark。

## 调参、温度前沿与负控

RAPS/SAPS参数仅用2k pseudocal+2k validation在原输出T1选择，四臂/温度/alpha共享；完整候选含失败项见 [tuning-only-selections.json](analysis/tuning-only-selections.json)。正式cal只估阈值，eval没有选择参数/温度/子群。APS tuning选T_p*=5、T_q*=0.75；B仍比较全部7个原输出温度，不在eval丢掉不可比或不确定点。

| 评分 | 共享参数 | tuning选q温度 | Rp及95%同时CI | Rq及95%同时CI | J及95%同时CI |
|---|---|---|---|---|---|
| raps | {'lam': 0.1, 'kreg': 5} | 2 | 0.271 [-0.185, 0.727] | 0.320 [-0.220, 0.861] | 0.049 [-0.380, 0.479] |
| saps | {'lam': 0.2} | 2 | 0.662 [-0.403, 1.728] | 0.713 [-0.409, 1.836] | 0.051 [-0.459, 0.561] |
| rankonly | {} | 1 | 0.674 [-0.549, 1.897] | 0.674 [-0.549, 1.897] | 0.000 [0.000, 0.000] |


SAPS使用原论文Eq10；rankonly=rank-1+U。RAPS按预定诊断公式，正则项不随机，不能混称原论文所有随机化实现的精确复现。SAPS/rankonly的原输出tuning温度与q温度不同的点若不在预注册同时检验族中，其不确定性会阻止B升级，未事后扩大检验族或选更好点。负控排除同时要求参数/温度tuning可行、点估计可比、现有同时CI端点给出的覆盖/D可比性成立及size优势下界超过0.25；不可比或不确定均不得升级。缺失Dhat传播为不可估。没有扩充CI族。逐项门判据见 [decision.json](analysis/decision.json)；预先静态反例测试见 [decision-guard-tests.json](decision-guard-tests.json)，补丁与前后哈希见 [decision-guard-patch.json](decision-guard-patch.json)。

alpha=.05完整结果仅作敏感性，未用于救题。所有224个组合保留于 [CSV](analysis/all-point-estimates.csv) 和 [含逐类覆盖的JSON](analysis/all-point-estimates.json)。

## 数据、数值和推断验收

- 公开CIFAR100官方Python包MD5匹配 `eb9058c3a382ffc7106e4002c42a8d85`；SHA256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`。数据来源和引用见 [SOURCES.md](../SOURCES.md)，未标MIT许可。
- 60k图像发现26个精确重复组、52张图；10组跨官方train/test。按训练前标签盲分组规则分配，cross-role duplicate=0；train40k、tuning4k、cal6k、eval10k均精确满足。图像ID、原来源、内容SHA、角色见 [split manifest](results/split-manifest.jsonl)。近重复未检测。
- CPU数学验收9项全部通过，涵盖退化恒等、温度/置换、同谱集合大小、rankonly、精确次序统计量与尾部/空集政策，见 [math-acceptance.json](../math-acceptance.json)。
- FP32 logits按image/view/class保存，baseline为中心原图view4；GPU输出流SHA与读回SHA一致，逐样本均值验收通过。后续评分均float64，允许空集、不补top1，稳定classID并列与donor rank保留。细节见 [cache acceptance](results/cache-acceptance.json) 与 [数值诊断](analysis/numerical-diagnostics.json)。
- 999次cal/eval分别重采样重复组、同次跨臂配对、重新校准；cal有5995组，eval有9996组。预注册族共304个维度，95% max-|t|临界值6.3821，90%为6.0261。没有独立重复训练。置信区间是近似bootstrap，同时控制不是有限样本精确保证。
- eval逐类n范围75–120，不足以给每类1pp精度保证；逐类Wilson95%区间只是描述性而非100类同时保证。任何类n<2会标不可估，本次D不可估的variant-resample数为0。Dhat负值保留。

## 可靠性修复、资源与失败账本

本轮是第三次分配、第二条正式训练轨迹，前一次恢复分配仅完成验收、正式训练为0。科学设置、评分公式、阈值、304维CI族和重采样方法均未改变。正式统计前主线静态审计修复了参数/温度不可行漏阻断、负控可比性漏检查及Dhat不可估时的异常。原90分钟预测门停止记录保留，之后主线/Pro批准105分钟启动门，120分钟硬限、6CU、原连续6小时和2GiB未变。分配前完整累计预测100.37分钟；实际准备48.61秒，通过8分钟准备限制，临正式开跑累计预测97.22分钟。105是启动门，未被用作运行中停止线。

实际GPU为Tesla T4，现有官方CLI0.7.5及已有授权；费率1.15CU/小时。本次分配至释放核验上界36.488分钟，三次累计上界64.034分钟。按费率估计累计1.227CU；余额41.94→40.79，显示余额差约1.15CU，不当作精确计费。

本次正式25040步、320万呈现、200000views；本次额外验收0步0views。前两次验收共400步/51200呈现（参考与50步重放双方均计账），已耗尽验收上限。第一轮最后确认23475正式步，未知尾段仍按完整25040步/320万呈现上计。累计正式上界50080步/640万呈现；含验收共50480步/6451200呈现；推理保守上界410240views。没有排除失败计算，没有用验收模型续训。

正式每5epoch保存完整模型、优化器momentum、手动cosine状态、Python/NumPy/Torch CPU/全部CUDA RNG、step/epoch/采样状态和归一化；epoch0及5–80共17个检查点全部下载、fsync、文件SHA及完整状态重新打开校验。下一块恢复RNG在模型/优化器构造和加载之后，原GPU randperm/增强流程保持不变。正式40个512图logit块（末块32图）均逐块验证image/role/source/view/class/dtype/最终checkpoint身份和哈希，再推进下一块。全部正式logits取回后立即释放GPU，后续组装均值缓存与统计在本机CPU。

推理第23块曾出现单个CLI调用不返回。只读状态/内核探针确认会话仍存活、epoch80与112640views、该块文件及完成标记均不存在后，在同一会话对原幂等函数执行一次有界恢复；完成标记确认117760views且匹配最终模型后，才结束卡住的旧本地CLI客户端。原控制器读取完成标记后继续剩余分块。没有新分配或重复模型计算；等待时间全部计入累计GPU预算。旧CLI的-15退出码是这一主动客户端清理，不是GPU中止。内部连接卡顿原因未确定，证据及恢复命令见 [INFERENCE_CONNECTION_RECOVERY.json](INFERENCE_CONNECTION_RECOVERY.json)。

释放后：`[colab] No active sessions found on server.`，额度回执：

```
Current balance: 40.79 compute units
Usage rate: 0.00/hr
Active assignments: 0
```

GPU释放后的完整本地状态和轨迹复核见 [durability-final-verification.json](durability-final-verification.json) 与 [trajectory-continuity.json](trajectory-continuity.json)。首次下载至本轮释放185.98分钟；统计及最终归档仍计入原23:53:27UTC墙钟截止，最终时长/目录字节计数见 `archive-receipt.json`。累计账本见 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)。

全部失败分支保留：第一轮使用detached Popen，75epoch后远端分配消失，缺少中途持久化导致正式产物未取回；平台内部原因未知，不能把本地pruned当平台根因。第二次分配恢复验收通过，96.16分钟预测超过当时90分钟门，主动释放，未训练正式模型。[第二轮报告](../attempt02/REPORT_ZH.md)保留原预测细目及首次审批拒绝/随后直接授权证据解锁的记录。第三次分配经透明门修订后执行本次完整正式轨迹；旧报告和各次CLI账本均保留。

完整版本链：[原协议](../PROTOCOL_ZH.md) → [第一次恢复修订](../attempt02/AMENDMENT_APPROVED.json) → [105分钟门修订](AMENDMENT_APPROVED.json)；[完整成本表](PREALLOCATION_COST.json)与[8分钟准备限制](PREPARATION_LIMIT_ADDENDUM.json)均归档。没有新OAuth、持久凭据、购买、MacGPU、commit/push或上传历史私密项目。迟到第一轮产物不得按效果挑选：本次正式轨迹已按训练前规定指定采用。

## 解释边界

已有 [TTA-conformal论文](https://arxiv.org/html/2505.22764v1) 已讨论rank改善与概率平滑；[TS4CP/ConfTS](https://arxiv.org/abs/2402.05806) 和 [SAPS](https://arxiv.org/html/2310.06430v2) 已覆盖温度/rank机制。因此普通缩集、普通rank收益和四格分解不能独自作为新贡献。本次没有复用其未经核实许可的研究代码。

这是内部heldout重分割，不是官方test成绩；组原子分配使图像级抽样不再完全独立，cluster bootstrap用于反映已知重复依赖。Dhat的经典无偏修正在独立同分布类内观测下成立；在重复组存在时应视为常规修正估计而非宣称严格无偏。单模型、一个增强协议、有限温度族、保守同时CI及有限样本限制了功效。未通过预定门只关闭此候选，不证明所有TTA机制不存在。

完整协议：[PROTOCOL_ZH.md](../PROTOCOL_ZH.md)；训练前哈希：[protocol-lock.json](../protocol-lock.json)；所有产物的SHA256见 `SHA256SUMS.json`。若无新指示，本候选不追加seed、模型或视图。
