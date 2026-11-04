本地 CUDA 执行包已冻结，46 项人工测试通过。尚未分配 T4、上传或运行模型；正式查询仍为 0，不能据此宣称科学贡献。

冻结 SHA256：`915d27011f9f948150de676b08f5830d5811066b993fc500aad729f0f284849a`。`FREEZE.json` 覆盖 244 个代码、输入、依赖与包文件；独审后必须用匹配该 SHA 的明确 GO 才能运行。

- 小包 `100,869,649` 字节（约 96.20 MiB），13 个上传分块，每块不超过 8 MiB。包含官方源码、15 个固定 wheel、固定 94 张图、输入元数据和人工 fixture。
- 两个官方 safetensors 合计 `1,378,009,960` 字节，在一次 allocation 中按固定官方 revision 下载一次、验证完整 hash，并在预检及正式阶段复用。GPU Torch/vision 先检查原生版本，不重装；缺依赖或不兼容则结束。
- fixture 为固定 `1333×800 RGB` 纹理、seed `20261031`、固定六框及 F/C/V 字符串；像素 SHA256 `311764e9a1f62545d97d904d0ecd377049a19645a87c478a1af68d8b7a15e87b`。旧常色失败完整保留，不允许换图追通过。
- 同一远端 CPU F/C/V，CUDA F warmup 后 F/C/V，共最多 7 次 root forward；外部预检墙钟 180 秒。FP32、AMP/TF32 关闭，官方 `_C` 必须自然为 None。原逐元素容差 `1e-4+1e-3*abs(CPU)` 与原生筛选 mask 一致性保持不变；原生未使用 token 的 `-inf` padding 按此前约定单独验证。
- 预检和完整 492 查询预算门均通过才进入 41 pair、82 图、492 query。每条结果保存输入身份、预测数量/框/分数与饱和标记；任何缺失均不能产生完整科学结论。六个 NA 与原 47 对一起报告。

实际协议是 `payload/execution_contract.json`（v3），worker 输出它作为 `inputs/protocol.json`。其明确使用 C、V ≥ 0.30、先前计算 54.915785417 秒及新 metrics hash；旧 v2 原字节保留，输出时标为 `historical_protocol_v2_NOT_CURRENT.json`。E=(|A预测−A真值|+|B预测−B真值|)/(A真值+B真值)，truth 只由冻结 labels 绑定。pair 平均后计算 GT oracle min(C,V)，不可部署；数值过门仍需审预测框排除错计抵消，不能删样本改门。

计算预算从 54.915785417 秒继续累计。运行时本地核验、远端解压/导入/严格加载、CPU/CUDA、后处理与收尾共享计算账本；安装和下载墙钟另记 allocation。正式前预算式为“累计已耗 + 0 额外加载热身 + 2×492×最大完整 CUDA query 时间 + 60 秒校验收尾 ≤ 1800 秒”，另核 allocation 剩余时间。正式时间测量含图像读取、token/group、传输同步、数值校验、序列化、首次写盘/流式输出。正式输出前逐次磁盘检查，父 watchdog 同时检查内存、GPU 分配、磁盘及截止。

一次 T4 allocation 最多 45 分钟，严格 <2 CU；采用实际精确 rate 加 0.05 CU 余量，不按历史报价直接开跑。历史 1.15 CU/h 对应 45 分钟约 0.8625 CU，仅用于估计。当前实例速率、下载/安装耗时及 CUDA 吞吐均未测得，完整 492 查询能否在余量内完成仍待工程门。

主控制器和独立有限生命周期的本地 watchdog 都使用已有 CLI 凭据，提前 90 秒回收。官方 Colab 提供原生 `runtime.unassign()`（[官方源码](https://raw.githubusercontent.com/googlecolab/colabtools/main/google/colab/runtime.py)）；包内另准备了到期自释放兜底，是否有内部管理服务需在授权实例上确认。不会上传凭据或新建 OAuth。管理服务/网络同时不可达时不能保证已释放，必须报告 UNCERTAIN，最终以官方 assignment 列表确认。进程退出不等于 VM 释放。

本地 stdout 每条立即 fsync；远端输出索引中的文件必须在释放前完整落到本地。缺包则在同一实例退出前取回；不用 `colab run` 的默认自动释放，也不在释放后补下载。CLI 默认 assignment retries 已关闭，claim 文件阻止第二次 allocation。

只读核验命令（不联系 Colab）：

```sh
PYTHONDONTWRITEBYTECODE=1 ../cpu-readiness/.venv/bin/python payload/scripts/controller.py --check
```

收到父线程明确 GO 后，单次执行命令（当前不要执行）：

```sh
PYTHONDONTWRITEBYTECODE=1 ../cpu-readiness/.venv/bin/python payload/scripts/controller.py --run --authorization /absolute/path/to/explicit-go.json
```

以上命令 cwd 为本目录。`authorization.template.json` 明确为 false，不能自行改为批准。执行后看 `run/controller_report.json`、`run/artifacts/` 和 `run/stdout.log`。

证据：`reports/verification.json`、`reports/verification_tests.log`、`reports/HANDOFF.json`。46 项测试包含模拟 stop 异常后 reconciliation、实际人工卡住的 preparation 子进程被父 watchdog 杀死、越界写盘在落文件前拒绝、缺 query 拒绝完整汇总、计数公式反例和原始容差。测试里的 Colab/分配均为 stub，外层同时禁止网络、模型 forward、GPU API 和 pickle，计数全为 0。固定 492 查询合同和 82 张图 hash 均通过，未解码真实图做效用检验。

当前阻塞为父线程指定的代码冻结后明确 GO，属于 pending，未遭工具拒绝。CUDA 原生环境与数值工程门尚未验证；候选观察仍不是已证贡献。

收尾截止修复：远端索引收尾、本机补传和复算共用原累计截止，不在交接处重置预算。补传 timeout=min(原 timeout,剩余计算时间)，同步核验由 OS ITIMER_REAL 墙钟定时器打断；到期保留已存证据、标记 incomplete，停止进一步补传/复算并执行既有回收。新增一个人工测试（补传阻塞、核验阻塞两个子情形）通过。精确 diff 见 `reports/finalization_deadline_fix.diff`，前后 freeze 与不变输入证明见 `reports/FINALIZATION_DEADLINE_FIX.json`。
