# 只读环境检查与待批准CPU编码器方案

2026-10-21。**在已检查的位置，没有找到可直接使用的PyTorch/torchvision、ONNX Runtime或预训练视觉权重。未安装软件、未下载权重，新的识别正控尚未运行。旧阴性结果全部保留。**

## 已检查的实际环境

| 位置 | 实际结果 |
|---|---|
| `/usr/bin/python3` | CLT Python3.9.6；torch、torchvision、timm、NumPy、Pillow、ONNX Runtime、transformers、sklearn、OpenCV、coremltools均不可用 |
| `/Users/lkc/miniforge3/bin/python` | Python3.14.7；上述视觉/数值模块均不可用；没有其他Miniforge env；已有packaging26.3可作元数据解析 |
| 现有uv工具解释器 | Python3.14.7；未发现torch、torchvision、NumPy、Pillow、ONNX Runtime、transformers |
| uv已展开缓存 | 找到NumPy2.5.4、Pillow13.0.0、scikit-learn1.9.1缓存；未将缓存路径注入环境，也没有加载这些库或修改它们 |
| 标准torch/huggingface/timm权重缓存与命名顶层权重文件 | 在已列位置未发现可用视觉权重 |

这是对常见运行时/缓存的有限检查，**不声称穷尽Mac所有应用、项目或目录**。未遍历8A项目、读取其代码或修改其依赖；没有调用远程Colab/A100。数字库缓存不是已经具备语义识别能力的视觉编码器，不能据此宣称正控已解决。

Vision Code8是旧轮已记录的运行故障。本轮没有再次调用Vision、更改安全设置或提权绕过。

## 一个具体的新增资源方案，等待批准

选择官方 **ResNet-18 IMAGENET1K_V1** 作为固定的CPU编码器，使用现有Miniforge Python3.14.7，在本研究子目录建立新venv及独立缓存。现有环境与8A依赖不变。只做冻结编码器推理及小型类别中心读出，不训练骨干、不使用GPU/MPS/ANE。

| 资源 | 固定版本/文件 | 精确下载字节 |
|---|---|---:|
| torch | 2.14.1，cp314 macOS14+ ARM64 wheel | 127,325,330 |
| torchvision | 0.29.1，同平台wheel | 1,813,852 |
| NumPy | 2.5.4，macOS14+ ARM64 wheel | 5,455,617 |
| Pillow | 13.0.0，ARM64 wheel | 4,852,269 |
| 其余9个完整依赖 | filelock4.2.4、fsspec2026.9.0、Jinja2 3.1.6、MarkupSafe3.0.4、mpmath1.3.0、networkx3.7、setuptools84.2.1、sympy1.14.0、typing-extensions4.16.0 | 10,430,763 |
| **13个wheel合计** | 全部固定URL与SHA256 | **149,877,831** |
| ResNet-18权重 | `resnet18-f37072fd.pth` | **46,830,571** |
| **总下载载荷** | 不假设复用缓存，不含少量HTTP/元数据开销 | **196,708,402（约196.7MB）** |

轮子来自[官方PyPI torch](https://pypi.org/project/torch/2.14.1/)和[torchvision](https://pypi.org/project/torchvision/0.29.1/)及其依赖的PyPI发布元数据，针对当前解释器和平台解析全部有效依赖与兼容wheel；Linux CUDA依赖均被平台标记排除。`proposed-resource-budget.json`列出每个文件的确切官方CDN URL、字节数与SHA256；`proposed-requirements-lock.txt`使用直接wheel URL与完整哈希，不依赖未固定的解析结果。现有NumPy/Pillow缓存不计作可节省的下载，以便隔离缓存并给出保守载荷预算。

权重来源由[官方ResNet-18文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)及其源代码确认：[官方权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)。只发出了HEAD，返回200与Content-Length=46,830,571，正文读取0字节；完整SHA256要在批准下载后计算，当前仅知官方文件名中的哈希前缀f37072fd。后续只按tensor-only方式载入。代码/依赖许可见PyPI元数据；官方也提示预训练模型可能具有训练数据相关条件，不能将库代码许可替代为所有权重权利。本方案仅本地科研，不访问或再分发原ImageNet图片。

预计新venv、独立缓存、权重与中间数据合计预留约1.5GB磁盘；运行内存预计不超过2GB。CPU单进程、1线程；校准估计1–2分钟，完整特征估计3–10分钟。**这些耗时/内存是预算，尚未实测**；先用训练侧图像计时，主轮超过10分钟则停止报告，不切GPU。

待批准后可执行的确切命令已写入 `install-after-approval.sh`，目前没有执行：

```sh
/bin/zsh /Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/encoder-readiness-20261021/install-after-approval.sh
```

该文件会用现有`uv`与现有Python，在新`runtime/`内建立venv、以完整哈希和`--no-deps`安装已核齐的13个直接wheel、分取官方权重并核对字节数/哈希前缀。不会自动下载新Python、使用全局配置、改变全局环境或共享缓存；若runtime已存在，先停止并保留其内容。脚本本身只是可审查提案，不构成用户授权。

## 先修测量能力，再冻结新轮

`source-train-calibration-plan.json`已在任何新编码器结果之前确定：只从原始源训练侧16张前景照片中，按新固定哈希每类抽2张校准，共4张；其余12张拟合类别中心。没有使用原外部留出图像或模型成绩来选校准源。

1. **基本正控A**：保持整只动物及足够边距，置于中性底，测试4张校准来源照片；至少3张正确。
2. **机制正控B**：同一完整缩放/裁剪规则产生的192个训练中性前景状态拟合读出，在校准4照片的64个F/T状态上按来源等权评价。总准确率至少75%，每个来源至少50%。这是测量能力门槛，不是4个来源就足以证明群体性能。
3. 固定编码器为官方ResNet-18 avgpool的512维特征、L2归一化、归一化类别中心，logits=10×cosine；预处理使用该权重官方Resize256→CenterCrop224和ImageNet均值/标准差。不根据外部留出结果换模型或调参数。
4. 两项正控通过后，先保存完整权重SHA256、版本、预处理/读出、校准结果和来源ID作为冻结记录，再对同一机制拟合两种策略并评价原四个留出组。若仍弱，先明确是基本识别还是强裁剪识别失败，停止，不为拿正结果继续换模型。

这将是**新轮探索实验**，不覆盖旧轮。原外部留出已经被旧弱模型查看过，不能称为全新确认性测试；也不能保证预训练数据与Oxford照片完全不重叠。任何正结果仍不等于新颖性、真实自然OOD因果识别或ForNet原方法反证。

## 当前状态与限制

新增依赖/权重尚未获准，故**停在安装之前，未运行新正控或新机制评价**。这是本轮“若必须新增依赖/下载权重，先报告供批准”的范围限制，不是自动审批拒绝。

另有一项只读来源访问被拒绝：网页工具读取 `https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet` 时返回“Failed to fetch restricted URL”。没有以raw/API/镜像路径重试同一目标。上述ResNet-18是独立的官方PyTorch模型方案，其文档、PyPI元数据和权重HEAD均已成功核实；没有取得被限制的ONNX模型。

本轮未使用ARIS、GPU/A100、收费服务、上传或对外通信。旧数据、模型、阴性结果及课堂文件保留。
