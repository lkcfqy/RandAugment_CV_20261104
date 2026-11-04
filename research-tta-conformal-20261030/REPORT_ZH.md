# TTA-conformal：运行时丢失，科学结论尚未取得

**执行状态为资源中断（NOT_EVALUATED），不是机制被否定，也不是候选科学失败。** 独立T4任务已真实启动并通过数学/数据/性能门，最后确认到75/80 epoch、23,475正式updates。运行时随后消失，未取回最终checkpoint或正式logits，因此没有正式四臂结果、A/B判定或bootstrap结果。整体研究可继续其他候选；本分支没有自动追加训练。

## 已完成的工作

训练前协议于 `2026-10-30T17:51:23.641124+00:00` 冻结，完整参数、调参可行约束、tie break、999次分组paired bootstrap、同时CI族及A/B继续门均在 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。冻结源文件逐项SHA仍匹配 [protocol-lock.json](protocol-lock.json)。SAPS按原论文Eq10独立实现，RAPS采用题定的非随机正则项；没有复制未核实许可的研究实现。

CPU数学验收9项全部通过，运行约0.017秒，准备及验收在15分钟内完成：p=q四臂一致；纯温度rank退化；纯标签置换谱退化；同谱/同阈值/同U逐图set size相同；所有样本共享严格正谱时APS/RAPS跨温度集合不变；rankonly正确；ceil次序统计量不插值且k>n为inf；稳定并列/保留donor rank/允许空集；float64尾部诊断。见 [math-acceptance.json](math-acceptance.json)、[数学说明](MATH_NOTES_ZH.md)。这些是实现验收，不是研究贡献。

实际分配Tesla T4标准档，费率1.15CU/小时；PyTorch2.11.0+cu130、CUDA13.0、NumPy2.1.3。使用原有授权，没有新OAuth、购买或Mac MPS。只上传独立实验脚本，未上传私密历史项目。

性能门完成200updates/25,600训练呈现及512张图×10views。最慢50步段0.086707秒/步，稳定段约0.0526秒/步；完整十视图512图链路0.550秒。测速权重销毁、正式重新seed0。最慢段1.5倍加剩余推理及240秒取回余量的保守总预测为60.72分钟、1.164CU，满足90分钟启动门。记录见 [performance gate](partial-evidence/performance-gate-from-log.json)。

## 中断的精确时间与已知原因

- 17:52:19 UTC：分配请求；T4成功分配，实际费率随后核实。
- 17:53:27 UTC：官方数据下载开始；17:53:56数据/分组检查通过。
- 17:54:14 UTC：性能门通过，正式80epoch训练启动。
- 18:02:39 UTC：一次Jupyter只读状态查询开始，180秒后超时。会话/额度接口正常，改用官方CLI下载process.log，确认训练持续；未重启模型或增加推理。
- **18:16:23 UTC**：最后确认epoch75、23,475步、loss=0.580653，正式训练已呈现至少3,000,000图次。18:16:39–40该日志仍成功下载。
- **18:16:40–42 UTC**：usage首次显示0active assignments、0CU/小时。此时本任务未发送stop，未触发110/115分钟时间后备。
- 18:17:12 UTC：下一次process.log下载返回“File or directory not found”。
- 18:17:27–28 UTC：服务器sessions确认无会话，CLI清理本地失效记录。官方终止事件reason为 `pruned`，不是本任务的 `user_requested`。
- 18:18:20 UTC：再次核实余额41.47CU、0会话、0CU/小时。

**运行时消失的根因未确定。** 不能据此指称额度耗尽、预算超限、用户叫停或审批拒绝；现有证据不支持这些说法。最后日志之后具体完成到哪一步未知，正式80epoch是否完成无可验证证据；完整产物均不可用。没有本任务主动stop的调用回执。证据见 [最后训练日志](monitor/remote-process-019.log)、[下载失败](cli-ledger/collector-log-020.json)、[空会话](cli-ledger/reconcile-sessions-1817.json)、[终止事件](termination-events.jsonl)、[最终usage](cli-ledger/final-usage-interrupted.json)。

## 仍然可恢复和交付的数据来源

官方CIFAR100 Python包MD5为 `eb9058c3a382ffc7106e4002c42a8d85`，SHA256为 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`。原运行检测到26个精确重复组/52张图，其中10组横跨官方train/test；按预先锁定的标签盲group分配规则，跨role重复0，所有固定规模精确满足。

| 角色 | 数量 | 官方train来源 | 官方test来源 | 原日志角色ID SHA256 |
|---|---|---|---|---|
| train | 40000 | 40000 | 0 | `c8e950be5217aee105ed6da37ea010f367e50b48ae2c6b676345da0e0c56ef77` |
| pseudocal | 2000 | 993 | 1007 | `16fd54f7b0526f0e8da73859cb35bfb00826a876581855a47708d650516d7145` |
| validation | 2000 | 994 | 1006 | `1f4992872284503e831b5cb4d67b8c759ac0ae166b0d98e715dc8cb1f00fb0a5` |
| cal | 6000 | 2946 | 3054 | `015d5770b743af4106a6bb23fe069e51797a94fe8a84a4262d71b8b7b192706d` |
| eval | 10000 | 5067 | 4933 | `3963cfc22f4e50989199e20ec16fc53bc99d72cc30c3f8d651da5f3dee514c49` |

为保住来源清单，运行时消失后只用Mac CPU重新下载同一官方公开包，抽取冻结脚本中相同的数据/划分函数重建清单；**没有模型训练、推理或GPU分配**。官方包SHA及全部5个角色ID SHA与原Colab日志逐项一致。此清单是可验证的重建品，不冒充从原运行时取回的文件。完整图像ID/来源/图像SHA/角色清单在 [split-manifest.jsonl](partial-evidence/split-manifest.jsonl)，重建回执在 [cpu-provenance-recovery.json](partial-evidence/cpu-provenance-recovery.json)。公开数据原包大小169,001,437bytes，保留在recovered-data/；没有将数据许可标MIT。

## 预算与最终清理

从分配请求至首次usage确认无分配的保守上界为 **24.37分钟**，按1.15CU/小时约 **0.467CU**。账户显示余额41.94→41.47，差约0.47CU；两位小数且结算可能延迟，非精确账单。均在120分钟/6CU上限内。首次下载到最终归档时长和全部本地新增字节数见 `archive-receipt.json`，均须满足6小时/2GiB。

已确认至少75epoch×40k=3,000,000正式呈现，外加测速25,600；已确认训练步23,475+200=23,675、测速推理5,120views。最后日志后实际计数未知；不可伪填为完整25,240步、3,225,600呈现或205,120views。冻结代码仍设这些硬上限。正式20k×10views未取回，不报告正式校准、测试准确率或机制结果。

`released.json`明确记录“运行时已消失后核验无遗留”，不是成功执行stop的虚构回执。本地后备计时器收到该核验标记后退出。没有剩余GPU会话或计费，见 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)。

## 剩余阻塞与安全恢复边界

阻塞动作：完成固定80epoch训练，并取回 `final-model.pt` 与20k×10-view FP32 logits，以运行已锁定统计。目标为本人Colab会话 `tta-conformal-20261030`、远程 `/content/tta_conformal_20261030/`。原因为运行时消失且文件通道失效。**这不是审批拒绝，也没有待审批动作；根因和最后完成步数不确定。**

没有可用的中途model/optimizer/RNG恢复checkpoint。直接重跑80epoch会超出本候选25,240训练步、3,225,600呈现上限，所以没有新建第二台GPU或暗中重跑。若有匹配的完整中途状态才可在剩余步数内恢复；当前没有。另一次完整尝试需要单独调整实验预算/协议，本报告不假定此授权，也不自动发起。

未来类似执行的具体改进是持久化中途恢复状态与完整随机数/优化器状态，并在仍有运行时时定期取回；最终科学模型仍可按预定末epoch选取，无需改成best。这个改进建议没有被用于追加本候选实验。

统计与成功报告脚本已编写并通过语法检查，但**没有在正式logits上执行**；不能把脚本存在等同于bootstrap完成。`analyze.py`、`build_report.py`和`finalize_archive.py`是原成功路径的准备代码；本报告由独立中断归档脚本生成。主线应将此候选标“资源中断/未评价”，继续其他独立方向，而不将它写成科学阴性结果。

## 研究范围与引用

[TTA-conformal原文](https://arxiv.org/html/2505.22764v1) 已有rank改善和尾部平滑；[TS4CP/ConfTS](https://arxiv.org/abs/2402.05806) 与 [SAPS Eq10](https://arxiv.org/html/2310.06430v2) 已涉及温度/rank机制。本任务按这些事实锁定更强A/B诊断，没有把四格工具本身或普通rank改善当贡献。原始数据见 [官方CIFAR页面](https://www.cs.toronto.edu/~kriz/cifar.html)，架构独立实现依据 [ResNet论文](https://arxiv.org/abs/1512.03385)；来源和许可说明在SOURCES.md。

产物总哈希见 `SHA256SUMS.json`，完整阻塞结构见 [BLOCKER.json](BLOCKER.json)。没有commit/push、外传私密历史内容或自动追加实验。
