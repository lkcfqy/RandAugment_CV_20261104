本次唯一 T4 执行已结束并确认回收。工程预检触及 180 秒外部墙钟限制，正式观察未启动；这不是科学阴性，也不能证明候选贡献。

冻结 SHA256 `915d27011f9f948150de676b08f5830d5811066b993fc500aad729f0f284849a`，使用获准的同一份包、fixture、阈值、FP32 与纯 PyTorch attention 路径。两份官方 safetensors 完整 hash 匹配；严格加载 1042 个物理键、1108 个逻辑键、66 个真实共享别名，缺键/多键均为 0、加载值逐项一致。原生 Python 3.13.16、Torch 2.11.0+cu130、torchvision 0.26.0+cu130；GPU Torch 未重装。

两次预检尝试：CPU F 完整返回，完整 query 耗时 80.582 秒；CPU C 仅有 STARTED 证据，随后被外部 watchdog 中止（进程返回 -15）。CPU V、CUDA warmup/F/C/V 均未进入。最近一次完整状态记录的 root 返回数为 1；中止前第二次尝试没有完整返回记录，不能把旧状态计数当最终计数。实际外部 worker 墙钟 180.727 秒含 watchdog 轮询和进程终止开销，停止原因明确是预设 180 秒预检截止。

真实 query 为 0，正式 metrics 未计算。CPU/CUDA 数值等价与 CUDA 吞吐均未评估。没有放宽容差、替换 fixture、换 GPU、训练或第二次 allocation。

allocation 墙钟 294.128 秒（约 4.90 分钟）；实际费率 1.15 CU/h，按持续时间估计 0.0940 CU，加 0.05 CU 余量为 0.1440 CU，低于 2 CU 上限。余额快照 40.78554782 → 40.77359402 CU，差额不作为最终账单，因为服务可能稍后结算。

执行器累计计算扣账 259.686473 秒（含先前 54.915785417 秒），剩余 1540.313527 秒；预算未增加。准备计算 22.091 秒，安装 11.649 秒、权重下载 20.503 秒另记 allocation。父子进程 RSS 峰值 4.059 GiB；远端自有磁盘约 1.837 GiB，均在护栏内。

回收事实：主 unassign 命令退出 0。第一次核查输出先有 `[colab] Pruned 1 stale local session(s).` 再有 JSON，冻结解析器因此报“Expecting value”并标 UNCERTAIN。独立 watchdog 随后的官方 snapshot，以及额外一次只读 snapshot，都确认 assignments=[]、active_assignments=0、rate=0；watchdog 进程也已退出。故最终资源状态是“已确认回收”，旧 controller_report.json 不改写。需在未来版本中修正 snapshot 混合 stdout 兼容，当前不改变冻结代码或启动新实例。

远端索引列出的 6 个证据文件均在释放前回传并逐一核对字节数/hash，包含两次尝试、准备报告、CPU F 工程状态、安装日志和 supervisor 超时报告。worker 被中止后没有正常 completion.json，这是已记录的预检中止，不是完整观察。

证据入口：`run/artifacts/supervisor.json`、`run/artifacts/attempts/`、`run/controller_report.json`、`run/release_watchdog.json`、`reports/POSTFLIGHT_GO_20261031.json`、`reports/compute_budget_ledger_after_cuda.json`、`reports/CUDA_RUN_20261031_RESULT.json`。剩余阻塞：固定 CPU 参考预检超时；本次单 allocation 授权已用，未自动重试。
