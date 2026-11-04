本机 CPU 准备和有限工程检查完成。已有4个完整模型均通过安全、严格、逐张量加载核对；同一张预先按SHA固定的support图完成8次前向，全部输出有限。完整512次dev诊断未启动，本次没有效用结果。是否开展完整诊断仍待父线程审核。

实际使用项目专用 CPython3.11.14 ARM64 / Torch2.1.0 / torchvision0.16.0 / MMCV2.1.0 / MMDetection3.3.0 / MMEngine0.10.7，NumPy1.24.3、OpenCV4.11.0.86、transformers4.34.0及其余精确版本见 [ENVIRONMENT_LOCK.json](ENVIRONMENT_LOCK.json) 和 [pip freeze](logs/pip_freeze.log)。Python来自此前已允许的官方 Astral 固定发行；核心包仅来自PyPI/OpenMMLab。未修改系统环境、模型/算子源码或安装第三方fork；Fairscale仅构建普通Python wheel，CUDA扩展关闭。已有BERT safetensors在本机复用，未下载新模型权重。

原pip check返回1，不能写作“pip check通过”。诊断确认：Torch官方ARM64文件名与Mach-O/RECORD正确，内嵌WHEEL却写x86_64；Ninja的Tag出现在空行之后，标准header parser读不到。依赖缺失/版本冲突均为0。保留原失败日志和未修改的包元数据；官方实际导入、6项原实现文件pins与随后的完整模型检查通过。证据在 [WHEEL_METADATA_DIAGNOSIS.json](receipts/WHEEL_METADATA_DIAGNOSIS.json) 和 [OFFICIAL_IMPORT_GATE.json](receipts/OFFICIAL_IMPORT_GATE.json)。没有为了通过检查改wheel标签、任意白名单或降级pip。

每个checkpoint共有908个state张量、691,880,148个原始张量字节；均weights_only=True在CPU加载，strict=True，无missing/unexpected，全部键、dtype、shape与字节精确匹配，所有浮点值有限。四个文件hash、完整状态hash及原模型/数据hash详见 [PROTOCOL.json](PROTOCOL.json) 与 [4份回执](receipts/r1)。无需unsafe pickle、动态白名单或trust_remote_code。

测速只用support1892 / 9870_Gerres_f000022.jpg，SHA256=117ea3546c7b67e1fbc9a213cc7cd920b934b0edaea65af28b1d72c076c22071。原始1080×1920 uint8/BGR，经原pipeline得到3×749×1333输入。Color位于cv2解码后、原FixScaleResize前；Color(prob=1,level=0,min=max1或0.8)保持不变。A逐像素恒等，B逐字节匹配固定cv2饱和度混合，两者同样消费1个NumPy随机数，几何、GT和pack后的metadata一致。原始annotation未改。

以下为实测秒数，RSS为各新进程的最高值，未查询物理内存容量。模型eval、FP32、inference_mode，Torch线程4、interop1，没有autocast或compile。

| 模型 | 构建+安全加载+完整校验秒 | 原图管线+前向秒 | 0.8管线+前向秒 | 峰值RSS GiB |
|---|---:|---:|---:|---:|
| 20261031_A | 1.427 | 2.2001 | 2.2193 | 3.0066 |
| 20261031_B | 1.446 | 2.2120 | 2.2066 | 3.0077 |
| 20261101_A | 1.427 | 2.2116 | 2.2236 | 3.0068 |
| 20261101_B | 1.431 | 2.2164 | 2.2135 | 3.0078 |

成功四模型阶段用时33.666秒。此前CPU入口曾把普通dict传给官方需要ConfigDict属性访问的代码：A安全加载完成后首次前向中止，未返回预测。已保留失败记录，并在无效用结果时以原官方Runner同样的Config包装修正入口；另一次静态字段断言写错路径，在模型启动前已修正。合计5次模型构建/加载、9次前向调用开始、8次完整前向、1次中止；中止阶段另用28.983秒，没有隐藏重试。科学配置/权重/图像/颜色强度均未改，见 [INFRA_REVISION_R1.json](INFRA_REVISION_R1.json)。

线程有一项必须明确：八次模型测速的OpenCV是GCD后端，setNumThreads(4)并未形成4线程上限，getNumThreads返回18；这不是实测并发线程数。Torch的4/1线程设置已核实。为满足未来固定线程运行，额外只做了同一support图的无模型管线检查：官方setNumThreads(0)令OpenCV顺序执行且getNumThreads=1；两种颜色的输入张量、GT与metadata均同先前设置逐字节相同。没有再次运行模型。其依据是 [OpenCV4.11官方parallel.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp)，测试见 [THREAD_CONTRACT_CHECK.json](receipts/THREAD_CONTRACT_CHECK.json)。未来建议Torch4/interop1、OpenCV顺序执行，保留其他线程环境变量4。

成本：预先冻结公式给出2885.534秒（48.092分钟）。为纳入OpenCV顺序管线，将实测最慢Torch前向2.213462秒与单独实测最慢顺序管线0.012116秒相加，再与原8次最大合并时延取更大值，沿用2倍前向系数、1.5倍4次加载和600秒落盘/8次COCO复算余量，得到2887.588秒，约48.13分钟。这是分部实测支持的保守规划数，不是512次实测或硬上界；未测完整持续热负载、其他64图内容变化、完整复算/落盘时间，不能保证一小时内结束。已完成的环境准备与一次入口失败单独记账，不伪装成正式512耗时。详见 [COST_EVIDENCE.json](COST_EVIDENCE.json)。

归档前研究目录6,150,689,909字节，6GiB上限内余278.25MiB，物理磁盘可用892,953,817,088字节。仅stat旧预测文件大小，未读取其分数；8份新预测JSON按历史最大文件大小外推约22.42MiB。建议新诊断全部产物上限100MiB、保留80MiB研究目录余量，不复制checkpoint。最终磁盘数见归档回执。

正式计划见 [FULL_DIAGNOSTIC_PLAN.json](FULL_DIAGNOSTIC_PLAN.json)：同四checkpoint×同64dev×1.0/0.8，共512前向、4次加载；原图与变换图都在同一CPU环境重算；保留原COCO口径，主AP maxDets100，AP50/AP75等原报告项maxDets1000，完整top300预测，不做阈值筛选。4195/4660原bbox/area和官方先resize后加载标签路径保持。每seed预先比较Delta(c)=AP(B,c)−AP(A,c)及Delta(0.8)−Delta(1.0)，两侧描述，不依据结果改条件或数量。只有全8条件完成后汇总效用，官方test不进入入口。

本轮模型/算子未移植，未训练、未新增seed、未读dev/test预测分数，未上传、未使用GPU/MPS/Colab；新增CU=0。历史账本仍1.8088330815795528CU、5443.899788473129GPU秒，未清零。此前被拒绝的sysctl物理内存读取未重试，也未换路径查询。所有CPU子进程已经正常结束。

无需模型即可确认：B固定0.8只让同一原图每次曝光经过同一颜色映射，没有新增随机颜色选择；标准随机几何仍在。它不代表一般随机增强。未来交叉输入最多回答固定训练颜色与人工输入颜色匹配是否改变A/B差距，不能当真实域偏移、部署预处理、机制证明或可靠科研贡献。

剩余事项是父线程对完整本机512诊断的GO，以及据已审核计划增加小型正式评估入口。当前没有已启动、待重试或执行结果不确定的模型作业；旧support测速入口带一次性保护，不应再次运行。
