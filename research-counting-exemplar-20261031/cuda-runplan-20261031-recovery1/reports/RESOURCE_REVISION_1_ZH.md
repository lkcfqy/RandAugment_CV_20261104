已完成一次效用前资源修订，等待独审与绑定本次 freeze 的 GO；本目录尚无 run 或 allocation claim。本轮没有分配、上传、下载、模型前向、GPU 调用或训练。

Freeze：`dc6ce7deb6a8ba9a0f3ead691833ad734865a2e2be31d6b3f10602684277185e`。包为 100,871,335 字节，13 块；与原包相比仅增加 1,686 字节。原有216个清单条目及清单文件，共217个归档文件，路径集合完全相同。原244个冻结文件的对应结构保留；旧目录全部320个文件哈希未改，原 claim 未覆盖。

[完整资源核验](reports/RESOURCE_REVISION_1.json) · [逐文件差异](reports/RESOURCE_REVISION_1.diff) · [48项人工测试结果](reports/verification.json) · [测试日志](reports/verification_tests.log) · [执行计划](payload/plan.json)

资源预算按以下顺序共用原1800秒截止，安装与下载仅在 allocation 壁钟中另计：

| 项目 | 上限或结转 |
|---|---:|
| 历次已计计算 | 259.6864732820104 秒 |
| 剩余总计算 | 1540.3135267179896 秒 |
| 准备计算，包含组包/解压及检查 | 60 秒 |
| 加载、快照、同环境 CPU F/C/V | 360 秒 |
| CUDA迁移、F热身、F/C/V比较 | 90 秒 |
| 统一收尾与本机复算预留 | 至少60 秒 |
| 本次 T4 allocation | 最多45分钟，留90秒回收 |
| 历次 CU 保守结转 | 0.14396 CU |
| 总2 CU下的剩余 | 1.85604 CU |

正式门仍为 `实际累计计算 + 2×492×max(CUDA完整query时间) + 60 ≤ 1800`，包含本次加载、全部三次CPU参考、热身及所有准备实耗；不缩图、不缩模式、不省CPU参考。若所有阶段都用满上限，CUDA三模式最大完整query时间必须不超过约0.986091秒。CUDA吞吐尚未测得，因此这里只证明计划受预算限制，没有保证能执行正式观察。

首轮资源审查：CPU F已经实际进入并返回，完整query耗时80.5821秒；其中前80.3289秒含fixture读取/解码/哈希、token与原生预处理、CPU模型、hooks、有限性/原生mask检查和序列化，余约0.2532秒含参考哈希与首次证据持久化。下载20.5027秒与安装11.6492秒均在此计时之外；没有细分纯模型时间或CPU忙等比例。首个worker状态至F启动约33.3453秒，主要路径为导入、加载、严格核验、快照及token准备。F完成后C启动，约66秒后被总180秒预检截止终止，V和CUDA都未运行。若仅按F估算三次参考再加准备即约275.09秒；这是资源预测，不是已测得的C/V时间。

旧日志没有记录 vCPU 数、affinity/cgroup quota、实际 torch getter 或 OMP/MKL 环境，已释放VM无法补读。源码在成功的CPU F前调用2线程/1互操作线程setter，但没有证据支持增加线程或归因为过度线程。修订固定2/1，不调优；只补记这些资源字段。

下载账本从已核资产1,608,332,982字节，加首轮官方权重1,378,009,960字节、可重建上传JSON body134,494,652字节、已存stdout75,990字节和补传base64上界3,684字节，共3,120,917,268字节；另列1MiB未测控制封套余量，结转3,121,965,844字节。未使用的旧预留没有算作实际下载。这是应用载荷账本，非抓包，不宣称测得TLS/TCP头字节。

第二次官方权重仍为1,378,009,960字节。预留200MiB用于包编码传输，以及627,397,976字节用于原有stdout上限、补传上限及其base64封装。最保守预测5,337,088,980字节，小于5GiB（5,368,709,120字节），余31,620,140字节。只计算原始公开资产和两次权重就有4,364,352,902字节，已超过旧3.5GiB。磁盘上限仍6GiB，本机当前约2.299GiB；首轮远端约1.837GiB。没有新增私有文件路径或凭据，也没有执行第二次上传。

快照改为显式 `CUCONTROL1:` JSON帧。主控和独立回收watchdog共用同一解析函数；CLI清理提示和合并stderr仍写入原始日志。缺帧、坏帧、重复帧均拒绝；首轮不准确的UNCERTAIN报告保留。人工测试覆盖带清理提示的快照及释放、畸形帧、60/360/90秒边界、CUDA阶段外部终止、跨远端与本机的原总截止，以及CU结转导致的拒绝边界。48项通过，模型/GPU/网络/pickle计数均0。

模型源码、权重身份、strict加载、纯PyTorch attention、FP32与原容差、固定fixture及六框/文本、492查询、GT绑定、F/C/V、0.23、800/max1333、41pair/6NA和科学门完全不变。仍是HF/app基本路径，未实现adaptive crop或pseudo-exemplars。CPU超时只构成工程阻塞，尚无真实效用证据。

本地只读校验命令：

```sh
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python /Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cuda-runplan-20261031-recovery1/payload/scripts/controller.py --check
```

独审后由父线程显式提供匹配新freeze、`allocation_allowed: true`的授权文件，才使用以下入口；现有template保持false，尚未创建新GO：

```sh
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python /Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cuda-runplan-20261031-recovery1/payload/scripts/controller.py --run --authorization <matching-GO.json>
```

原45分钟独立回收保护及官方runtime unassign补充保持不变。断网仍可能妨碍本机回收，不能将worker退出等同于VM释放；最终必须核对官方assignments/sessions。
