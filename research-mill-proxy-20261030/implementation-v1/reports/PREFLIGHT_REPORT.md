一次授权的全尺寸人工预检已完成；资源门 PASS，正式实验等待上游决定。冻结的18个文件哈希保持不变：`e7b5ddfaf0beef296612e90138fd37d93337066d520576a6e9a9ba638806f6c7`。

| 项目 | 结果 |
|---|---:|
| 设备 | Apple M5 Pro / MPS |
| 图像 | 固定人工 400×600 RGB |
| 模型调用 | 1 warmup + 3 同步计时，共4次 |
| 三次计时 | [0.09217962500406429, 0.08799987500242423, 0.08746595900447574] 秒 |
| tF（最大） | 0.092179625004 秒 |
| tcell（A/B各4draw及最终输入统计） | 4.511538709005 秒 |
| tpost（预先声明） | 600 秒 |
| 完整预检会话 | 7.410309583000 秒 |
| 预检扣账 | 8 秒 |
| 总扣账 / 原预算剩余 | 30 / 1770 秒 |
| 进程峰值RSS | 968441856 字节，923.578 MiB |
| 资产目录占盘（预检后） | 259112960 字节，247.109 MiB |

完整资源式：`30 + 1.25 × (326 × 0.092179625004 + 32 × 4.511538709005 + 600) = 998.024745549 秒 <= 1800 秒`。

拟合成本依据是冻结实现的静态工作量：每档 A 使用4097×256共享power表、6×3×240000像素sufficient statistics及残差分箱；B使用3个256节点PAV、6×3×240000残差条件排序、18次400×600 FFT、240000个复数Hermitian 3×3主平方根。预先给每档A+B拟合120秒、四档共480秒；另外120秒覆盖输入解码、模型初始化、逐图标量指标、缓存/哈希、汇总和落盘。600秒是保守资源预留，不是已经测得的拟合成本或已证明的运行上界；没有从微型oracle速度外推，也没有执行真实拟合。资源PASS以这份预留为前提。

冻结调度器的MPS字段是worker末尾当前live tensor分配7926784字节，不是MPS/driver峰值；可用的峰值证据是进程RSS。为遵守一次预检限制，没有重跑补测。进程已经成功退出，MPS上下文随worker释放；ledger active=null，无lock，无generated目录。

真实test Y/GT访问=0，真实fit=0，真实synthesis=0，正式模型forward=0，任务效用metric=0。四次CIDNet输入均为人工图像。没有启动正式流水线，也没有科学结论。

证据见 `run-state/preflight_launch_output.json`、`run-state/preflight_result.json`、`run-state/compute_ledger.json`；完整参数、限制与证据哈希见 `reports/PREFLIGHT_REPORT.json`。辅助命令 `sysctl -n hw.model hw.memsize` 被OS/sandbox拒绝（Operation not permitted），未重试，不影响预检；GPU型号来自成功的system_profiler查询。
