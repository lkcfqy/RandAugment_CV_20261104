MILL proxy 的隔离代码与人工小数组 oracle。真实拟合、真实合成、CIDNet forward、效用、全分辨率预检和 GPU 均未执行。

完整数学定义在 [PROTOCOL.md](PROTOCOL.md)，固定设置在 [protocol.json](protocol.json)。本轮采用最终澄清：synthetic PSD/Pearson 使用已有 1e-12 centered-variance 数值退化门，退化差异记 2；真实 STAT_UNIDENTIFIABLE 不进入继续条件；已知 proxy 足够的 CLOSE 保持更高优先级。无 null 科学配置。

| 模块 | 职责 |
|---|---|
| core / fit | Q8、精确源亮度分箱、固定 RNG；A 网格与 NNLS；B PAV/条件残差 |
| spectral / synth | 双边 ortho FFT、矩阵主平方根、共相位轨道、固定十轮 rank map |
| metrics / diagnostics | 全 RGB PSNR、指定 Lab DeltaE76、每 cell 五类统计门与复相干度 |
| decision | D、C_meanD、C_anchor、同集合对照、独立 R 排序、关闭优先、可选 cluster CI |
| audit / pipeline | 严格 326 身份、无 pickle 缓存、冻结后才开放测试 Y/GT 的流程 |
| model | 固定作者四源码哈希、strict safetensors、只接输入的延迟 CIDNet 适配器 |
| runtime / scheduler | 持久累计预算、2 GiB 写前磁盘检查、外部 watchdog、后续调度入口 |

人工测试及元数据冻结命令：

```sh
../.asset-venv/bin/python run_oracles.py
../.asset-venv/bin/python freeze.py
```

oracle 最大人工数组为 8×10；4×6 流程使用普通 NumPy 函数检查 326 条输入身份，未加载 CIDNet。测试进程不会导入 torch，历史红/绿运行都计入 `reports/oracle_runs.json`，累计 cap180秒。账本记录实测 wall，并逐运行向上取整后从未来原始1800秒总预算扣除，包含启动/记录余量。

调度入口是 `python -m millproxy.scheduler`。`status --asset-root <资产目录>` 只读状态，不创建科学进程。未来 `preflight`/`formal` 必须给 `--asset-root`、`--release`、`--device`、`--post-bound-seconds`，release 要有当前 freeze SHA、独立静态审查通过、最终协议字段和对应 scope。当前未创建这样的真实执行 release。

preflight 在单独 worker 中使用固定人工 600×400 图，完成1次模型 warmup、3次计时取max、A/B各4draw整cell生成与统计，只记录时间与内存，不计任务效用。tpost 必须明确覆盖未计量的拟合、汇总与持久化，不能用小 oracle 速度代替全尺寸测量。formal 要先通过同代码哈希、同设备的 preflight 和最新累计预算的资源式；父进程持有不可静默重置的 ledger/lock，外部 watchdog 覆盖 native 调用。留1秒原预算用于父进程落盘，阶段耗时向上取整记账。未完成 session 或遗留 lock 必须检查，不能以重启清零。

正式 worker 在同一连续剩余预算内依次进行 train-only fitting、test-X-only generation、256张Q8生成图冻结、326次任务模型调用、最终诊断/控制/决策、结果落盘。每次参数或图缓存写入前检查新增字节加元数据余量不会使整个资产目录超过2GiB。空训练箱的经验Q仅保存原非空箱全集，空箱通过 donor索引恢复，避免无意义重复占盘；数值与完整复制相同。模型输出不经 PNG。

代码、测试、数学协议和固定配置共同参与 freeze hash；入口重新计算当前内容，拒绝旧哈希。所有源码哈希逐一等于作者固定 commit 的值，不是“仅记录当前文件自身哈希”。没有新增下载、包安装、Colab、凭据、发布或推送。独立审查与后续真实预检仍由上游开启。
