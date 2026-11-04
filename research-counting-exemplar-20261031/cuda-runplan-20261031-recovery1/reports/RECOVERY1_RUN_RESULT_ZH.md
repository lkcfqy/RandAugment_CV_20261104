恢复执行已结束：CPU三模式参考完成，但CUDA F热身的原数值门失败，正式492查询未启动。会话已释放；独立官方snapshot与sessions均确认无活动会话、费率0，有限生命周期watchdog已退出。没有再次分配或调参。

本次freeze：`dc6ce7deb6a8ba9a0f3ead691833ad734865a2e2be31d6b3f10602684277185e`；T4 endpoint：`gpu-t4-s-use1c0-am16sfohvzuy`。13块包上传成功，两份官方safetensors各下载一次且hash正确；未新建认证，未上传凭据或新增私有文件类别。

执行了4次人工root forward：CPU F/C/V完整查询分别73.3114、70.2400、70.4179秒，加载/快照加CPU阶段245.3922秒；CUDA F热身forward返回，在比较时失败。第4条attempt文件仍为STARTED，因错误发生在COMPLETE持久化前；最终entered/returned计数均4，不应解读为CUDA未返回。后续CUDA F/C/V均未执行，完整吞吐和正式预算门没有得到评估。

严格加载1042物理键、1108逻辑键和66个实际共享别名通过，CPU→CUDA迁移前后参数/buffer/别名完全相同；F归一化输入、mask、boxes与token哈希一致，形状1×3×800×1333。FP32、AMP关闭、TF32关闭、原生纯PyTorch attention保持。资源确认2个CPU、affinity[0,1]、torch线程2/1；相关OMP/MKL环境未设置。

原容差仍为 `1e-4 + 1e-3*abs(CPU)`。CUDA F的最大活跃logit误差2.94157362，框坐标最大误差0.879148066，原生筛选mask不相同；活跃logit与框坐标本身有限。原数值门因此失败，没有放宽、换fixture或排序补救。

保存的定位线索：encoder分数最大绝对差0.00058651，仍在原容差内；CPU/CUDA选中同一900元素集合，但10个rank位置不同（5对相邻交换）。这不是已证实根因，也不能因此宣布输出等价。完整原始topk顺序在工程状态文件中。本轮未再执行诊断forward。

| 资源 | 已记录结果 |
|---|---:|
| 本次allocation壁钟 | 383.690479 秒 |
| 费率 | 1.15 CU/小时 |
| 计算累计 | 545.684807011 / 1800 秒 |
| 计算剩余 | 1254.315192989 秒 |
| 本轮保守费用（含0.05余量） | 0.177486104 CU |
| 两轮保守费用结转，向上取5位 | 0.32145 / 2 CU |
| 保守剩余额度 | 1.67855 CU |
| 累计应用载荷与控制余量 | 4,636,108,808 / 5,368,709,120 字节 |
| 远端owned disk | 1,972,325,691 / 6,442,450,944 字节 |
| 父进程+worker峰值RSS | 4,366,069,760 字节 |
| CUDA allocated峰值 | 2,479,433,216 字节 |
| CUDA reserved峰值 | 2,896,166,912 字节 |

费用采用 `max(本轮前后余额差, 费率×allocation时长)+0.05` 再加原0.14396结转。最后余额40.646107910965334；余额可能继续结算，不称最终账单。模型准备计算、CPU参考、CUDA尝试、远端收尾与本机校验全部记入原计算账本；安装10.663640秒和官方权重下载13.500651秒另计allocation壁钟。

9个索引工程文件均在释放前完整落到本机，事后再次核验大小与hash；源freeze全部244文件未变。GPU worker清理时仍有约9.57MB allocated/16MiB reserved，最终官方VM释放已独立确认，未把worker退出当资源释放。

原研究协议、41pair/82图/492query、原47pair及6NA、labels/图/六框/文本和数值科学门未改；真实query、真实效用指标和训练全部为0。此次失败只能记为工程数值阻塞，不能称科学阴性或可靠贡献。

[结构化结果](RECOVERY1_RUN_RESULT.json) · [计算与费用账本](compute_credit_ledger_after_recovery1.json) · [传输账本](transfer_ledger_after_recovery1.json) · [独立回收核验](POSTFLIGHT_RECOVERY1.json) · [工程原始证据](../run/artifacts/engineering/status.json) · [主控报告](../run/controller_report.json)

当前没有活动GPU，也没有待执行的自动重试。本次GO已经使用完毕，下一步仅待父线程审阅数值失败证据并决定研究推进方式。
