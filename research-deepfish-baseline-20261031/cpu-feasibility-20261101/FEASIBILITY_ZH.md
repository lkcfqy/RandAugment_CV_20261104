结论：存在可保留官方原始 GroundingDINO 实现的 Mac CPU 路径，且找到了官方 ARM64 二进制候选；目前不能声称本机已可运行。现有环境需要独立准备，但没有发现必然需要模型移植的证据。本轮仅完成只读核查，没有安装、导入执行框架、实例化模型、推理、上传、Colab 调用或 GPU 分配。

本机实测为 macOS 27.2 / arm64，Apple Clang 21.0.0 与 Command Line Tools 已在。检查到的主要项目环境为 Python 3.14.10 / Torch 2.13.0 / torchvision 0.28.0，均无 MMCV、MMEngine、MMDetection；另一个旧环境为 Python 3.9.6 / Torch 2.5.1，也无上述包。已检查的 PATH、Miniforge envs 和 Python.framework 路径未发现 Python 3.11。当前 CPU 分析环境的 NumPy 2.5.4、OpenCV-headless 5.0.0.93 也不是实验锁定版本，不能直接拼接旧 Torch2.1/MMCV2.1。

官方支持证据分三层：

1. [MMDetection v3.3.0 文档](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/docs/en/get_started.md#install-on-cpu-only-platforms) 明确支持 CPU 推理，同时列出部分不支持算子；这不是 GroundingDINO 在本机的运行证明。
2. [MMCV v2.1.0 注意力源码](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/multi_scale_deform_attn.py) 的非 CUDA/MLU 分支直接调用其官方 PyTorch CPU 实现，内部使用 bilinear grid_sample、zeros padding、align_corners=False；[官方测试](https://github.com/open-mmlab/mmcv/blob/v2.1.0/tests/test_ops/test_ms_deformable_attn.py) 包含 CPU 参数。已静态追踪 GroundingDINO encoder/decoder 使用这个类，预测头采用 top-k，当前 max_per_img=300。没有自行重写注意力或 NMS。模块导入仍加载 mmcv._ext，因此 mmcv-lite 不能替代带算子的 MMCV。
3. [官方 CPU/Torch2.1.0 发行索引](https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html) 有 cp311 universal2 wheel。对其 1,872,544 字节归档只做 ZIP 和 Mach-O 静态读取：扩展确有 x86_64 与 ARM64 两个切片，Python 注意力/颜色文件与官方 v2.1.0 源码 SHA256 相同。wheel SHA256=`6bc8018e724dc53dbea36fe4db5777da25cd0b67a912b2e802d2cba7b4ce3894`。没有安装、import 或 dlopen，ABI 成功尚未证明；不能只凭 universal2 文件名宣称已兼容。

优先候选依赖如下；完整精确版本在 [requirements_mac_cpu_candidate.txt](requirements_mac_cpu_candidate.txt)，来自已成功 GPU 环境冻结表，去掉 Linux-only Triton，将 CUDA 发行替换为官方 Mac 发行。它是尚未解析/安装的候选，不是已验证 lock。

| 部件 | 候选版本/来源 | 需要做的准备 |
|---|---|---|
| Python | CPython 3.11.9 ARM64；[Python.org universal2 installer](https://www.python.org/downloads/release/python-3119/) | 新独立环境；3.11.9 是有确切官方安装器的候选，并非最新版本声明。未来安装前记录解释器制品哈希 |
| PyTorch / torchvision | 2.1.0 / 0.16.0，原生 Mac ARM64 wheel | 官方 [PyTorch 历史发行](https://pytorch.org/get-started/previous-versions/)；不用 +cu118，也不切 MPS |
| MMCV | 2.1.0，指定 CPU/torch2.1.0/cp311 universal2 wheel | 若 ABI 正常，免 MMCV 源码编译；不能直接使用 PyPI 的 mmcv-lite |
| MMDetection / MMEngine | 3.3.0 / 0.10.7 | 保留官方实现及已有文件 pins；MMDet3.3 要求 MMCV <2.2、MMEngine <1 |
| 文本部分 | transformers4.34.0、tokenizers0.14.1、huggingface-hub0.17.3、safetensors0.5.3 | 沿用离线、安全加载路径 |
| 图像/数值/评估 | NumPy1.24.3、opencv-python4.11.0.86、Pillow11.1.0、SciPy1.13.1、pycocotools2.0.8 | 维持 cv2、uint8、BGR；不用现有 OpenCV5/NumPy2 代替 |
| 其余 | fairscale0.4.13 及原完整冻结表 | fairscale 发行仅 sdist，需常规打包；是否出现平台依赖问题未验证 |

如官方 wheel 无法加载，[官方源码构建说明](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/build.md#build-on-macos) 提供 MMCV_WITH_OPS=1，Apple Silicon 要求 Torch≥1.13；可选 ninja1.11.1.1、psutil5.9.8。当前 Clang 存在并不等于构建成功。v2.1 的 setup 会自动检测 MPS，FORCE_MPS=0 不能保证只编 CPU，可能涉及 Objective-C++/Metal/Foundation；这与未来明确 device=cpu 的推理选择不同。若需要改模型/算子源码、第三方 fork、架构翻译、重编 PyTorch 或长时间跨后端一致性工程，这个诊断不值得继续。现有证据支持先考虑官方二进制的有限验证，不支持直接宣布要大规模移植。

已找到本机先前保存的官方 BERT 七个文件，共 441,169,829 字节，逐项 SHA256 全部匹配 DeepFish 的原资产 pins，包括 440,449,768 字节 safetensors。可以按已有本地路径复用，不必为本诊断再传 BERT。四个完整 checkpoint 均在 Mac，总计 2,768,703,848 字节；本轮只核文件存在/大小并引用既有哈希核验记录，没有重新加载权重。这些 checkpoint 从未包含在此前 13 MiB 上传授权中，本轮上传量为 0。

成本仍有明确未知项：只有六个核心发行包的已知体积合计 114,373,068 字节；这不包含 Python 安装器、全部间接依赖及解压/构建缓存，因此不是总下载或磁盘预算。CPU 吞吐、完整预测的峰值内存及 ABI 加载未测量。一次 sysctl -n hw.memsize 读取被 OS 以 Operation not permitted 拒绝，未得到 RAM 值、未重试，不影响静态源码结论。现有 GPU 每图时间不能外推到 Mac CPU，也不承诺两小时完成。

若未来获准执行，必须对四个冻结模型各做原图与固定0.8输入两条件，均使用同一个 CPU runtime、同一64dev/187GT、原 dtype/resize/标注契约和评价口径，共4×2×64=512次图像前向。原图条件也必须在该 CPU runtime 重评，不能把旧 GPU 原图分数直接与新 CPU 变换分数相减。颜色应在原图 cv2 解码后、同一 FixScaleResize 之前按原 uint8/BGR 实现施加；保留官方先resize后加载标签的路径以及4195/4660原框/area。四个最终 checkpoint 固定、安全 weights_only 加载、不重新训练、不改阈值、不选 checkpoint、不碰官方test。当前 CUDA-only worker 有明确 CUDA 设备/版本门，不能直接删门后称同一次复现；未来需要一个很小的独立 CPU 评估入口，仅调整设备/运行路径，模型、算子与科学参数不动。

CPU 交叉输入结果最多是独立后验诊断：对每个模型比较 CPU 上 AP(X0.8)−AP(X1.0)，并可在种子内观察 A/B 输入响应差异；即使方向符合预期，也不能单独证明域适应、正则化或一般增强机制。总时长必须包含环境准备、四模型加载、512次实测前向、八次评价和落盘，不能只计模型前向。

无需运行模型即可证实的事实：A 的 Color 参数唯一值1.0，B唯一值0.8；prob=1，level=0，min_mag=max_mag。官方 Color 每次虽然消费一个 NumPy 随机数做开关判断，但此配置始终启用，幅度不随机。B 对同一解码图执行确定的 0.8×BGR+0.2×灰度广播（cv2 uint8 舍入），每次曝光都只有这一种颜色版本；A 则恒等。颜色位于随机 flip/resize/crop 之前，既有几何随机性仍然存在。这里的“颜色多样性”指同一图跨曝光的颜色变换选择，并没有测量或宣称像素直方图中不同值的个数单调减少。[官方 Color](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/datasets/transforms/colorspace.py) 与 [adjust_color](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/image/photometric.py) 支持这一静态判断。

所以此前两臂严格说比较的是“相同标准随机几何＋固定颜色映射”与“相同标准随机几何＋颜色恒等”，B 只施加固定训练颜色分布偏移，没有新增随机颜色变换选择，不能代表常规随机 ColorJitter/RandAugment 的一般效果。这项事实不需要用后续诊断追认，也不构成新机制或科研贡献。

当前终态：只读可行性检查完成；安装、ABI/算子验证、完整模型运行与交叉输入结果全部未做。没有发起后续执行。GPU/CU 历史账本不变（1.8088330816 CU、5443.8998 GPU秒）。机器证据见 [FEASIBILITY.json](FEASIBILITY.json)、[本机环境](LOCAL_ENVIRONMENT.json)、[wheel静态核验](OFFICIAL_WHEEL_INSPECTION.json)、[源码/包元数据清单](SOURCE_MANIFEST.json)。
