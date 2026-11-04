Mac CPU全量观察在单遍时间与内存上有可行性，但把累计计算上限改为90分钟并不能自动满足原来的2倍工程成本门。本次只读取已有记录和主机资源，没有运行任何模型、GPU、安装、上传或分配，也未改变代码、旧freeze或当前1800秒预算。

已有Mac CPU F/C/V原生路径计时分别为6.159160、6.139380、6.137315秒；独立诊断F为6.300725秒，线程固定2/1，输入均为1×3×800×1333。它们是常色人工输入的短时结果，包含原生预处理、模型与过滤，但不含真实JPEG解码/hash、后续validity检查、完整JSON/fsync。因此“约50分钟”是外推，不是完整观察实测。当前固定纹理fixture尚未在Mac CPU运行过，不能拿Colab CPU或旧常色结果假装新本机测量。

原worker导入约0.455秒，启动到严格加载完成约10.041秒；内部load计时0.513秒不能替代完整启动成本。可以在拟执行计划中给“本地hash/加载+可选三模式人工检查”60秒、“写盘与最终复算”60秒预留，这些是计划余量，不是已测上界。

以所有已有Mac CPU结果的最大值6.300725秒估算，492query需3099.9566秒（51.666分钟）。历史计算继续结转545.684807011秒。候选90分钟总上限剩4854.315192989秒（80.905分钟），不重置历史CPU/MPS/Colab成本。

| 工程耗时系数 | 累计预测，含历史和120秒预留 | 90分钟内 |
|---|---:|---|
| 1×单遍外推 | 62.761 分钟 | 是，非保证 |
| 1.5×余量示例 | 88.594 分钟 | 是，仅余约84秒 |
| 原2×准入门 | 114.427 分钟 | 否 |

保留2×门时，CPU最大完整query时间需≤4.811296秒，已有实测不满足。1.5×示例需≤6.415061秒，只有很窄余量；单遍硬截止下扣预留可容纳平均9.622592秒/query。没有自动采用任何新系数。若坚持90分钟，需在效用前明确新的CPU工程准入规则；E、G及≥10pair等科学门可完全不改。将跨设备数值等价从执行前提中移除，也应明确记录为CPU后端范围修订，不能抹掉原失败或声称CUDA等价。

主机只读核验为M5 Pro、18核、64GiB统一内存；当前AC供电、100%电量，AC空闲sleep=0，ChatGPT存在防空闲休眠assertion。旧CPU严格加载峰值2.556GiB；包含CPU和MPS的两段完整旧实验峰值3.634和3.800GiB，不能把后两项冒充CPU-only峰值。当前free页约9.865GiB、swap计数为0，磁盘可用约897GB，8GiB进程护栏及6GiB项目新增磁盘护栏均有现实余量。64GiB总内存不是本任务可无限使用的授权。

现有环境可直接复用：torch2.13.0、torchvision0.28.0、transformers4.49.0、tokenizers0.21.4、safetensors0.8.0、numpy2.5.4、Pillow13.0.0。版本通过本地metadata读取，未导入或运行模型。它与Colab的torch2.11.0/vision0.26.0不同，报告应明确本机CPU运行栈，不能称CUDA论文数值复现。

最小实现只需派生一个cpu_worker和一个本地launcher，放入新目录。原model_io严格加载、自然_C=None、hooks/token契约、numerics.validate、evidence原子写入、contracts中的GT绑定和完整性函数、Fraction metrics直接复用并锁hash。正式492query循环、A/B和F/C/V输入、3正3负框、.23/800max1333、41pair/6NA与预测框保存保留。worker删去T4检测、GPU迁移/内存/sync/清理、跨设备比较和Colab资源逻辑，把全部query固定CPU；不要重写模型或metrics。

本地launcher复用之前Mac版ps RSS采样和外部进程截止，替代Linux/proc；统一累计deadline贯穿加载、人工预检（若获准）、492query及最终复算。只改新增CPU入口，旧包不动。直接引用现有模型/图像/依赖，不再下载或复制整份权重，新增CU为0。新run/claim及逐条fsync证据独立，缺任一query标记incomplete，不删样本、不补充模型重试。

本机有/usr/bin/caffeinate；未来获准执行时可将防空闲休眠绑定run进程，不永久改变pmset。正常保持AC供电并避免强制休眠/退出主机即可持续本地工作；无法保证关机、强制休眠或主机任务终止后继续。无需联网或Colab维持运算。

[结构化评估](MAC_CPU_BACKEND_FEASIBILITY_20261031.json)；原始证据位于engineering-precheck/run/engineering.json、engineering-diagnostic/run/engineering.json及cpu-readiness/reports/cpu_verification.json。全部模块和证据hash已列在结构化评估中。当前仅待科学审查决定是否修改后端范围、总wall与工程余量规则，没有新增forward授权。
