# 下一阶段单 CUDA 环境吞吐检查：需求与风险（未执行）

本任务止于资产事实。未开启 CUDA 环境、安装依赖、运行模型、训练或测试；本文件不是启动指令或时间承诺。

## 需要固定的依赖

detector 代码属于 Domain-RAG 内的 MMDetection 3.3.0 快照。仅运行 detector 不需要根目录的 Flux、diffusers、检索库或生成模型。

| 组件 | 静态证据 / 需要核实之处 |
| --- | --- |
| Python + PyTorch + CUDA + torchvision | vendored environment.yaml 记录 Python 3.9、PyTorch 2.0.0、CUDA 11.7、torchvision 0.15.0；这是历史文件内容，不是当前可运行环境的保证 |
| MMCV | mmdet/__init__.py 要求 >=2.0.0rc4 且 <2.2.0；environment.yaml 为 2.0.0rc4。必须带匹配 CUDA 的编译算子，mmcv-lite 不足以满足 MultiScaleDeformableAttention |
| MMEngine | >=0.7.1 且 <1.0.0；历史 environment.yaml 为 0.10.7 |
| Transformers / BERT | 历史 detector 环境写 4.34.0；BertModel 构造会读取 tokenizer/config 并调用 HFBertModel.from_pretrained。现有本地 BERT 资产已哈希，未来应固定路径并验证离线构造 |
| fairscale | encoder.num_cp=6，代码实际使用 checkpoint_wrapper；历史环境版本 0.4.13 |
| 图像与评估 | Pillow/OpenCV、numpy、scipy、pycocotools、shapely、six、terminaltables、tqdm、matplotlib 等；基础和 multimodal requirements 不能直接当作完整锁定环境 |
| multimodal requirements | 还列 jsonlines、nltk、pycocoevalcap；应逐一确认 detector import/初始化是否需要，避免隐式联网取资源 |

上述信息来自固定源码的 environment.yaml、requirements/runtime.txt、requirements/multimodal.txt、mmdet/__init__.py、models/language_models/bert.py 和 models/layers/transformer/grounding_dino_layers.py。根目录 requirements.txt 另写 torch 2.7.0、torchvision 0.22.0、transformers 4.46.3，偏向生成阶段；两份环境不能直接混装。当前 Mac 用来读取权重的 Python 3.14 / torch 2.13 也不是 CUDA 验证环境。

## 单一环境检查前的具体输入

先形成一份独立配置，固定此处原始 Swin-T 权重哈希、FISH support JSON 哈希、BERT 本地资产哈希、模型类别/DN 初始化策略、pipeline、单 GPU 的 batch 与梯度累积、精度、种子及固定更新数。明确 train_cfg.max_epochs，核对 scheduler 同步；不要运行 auto_modify_swin_t_config.py，也不要改用 mm_grounding_dino 的另一权重。

吞吐检查只需已公布 support 图像，不需要打开 test evaluator。关闭 test.json 驱动的 val、save_best 和任何按 test 挑配置的逻辑。若将来进行效用实验，先决定独立 dev 或预定最后 checkpoint 的策略；这不是吞吐数值本身能够解决的问题。新增 train 图片建 dev 会改变 few-shot 数据使用协议，不能默默这样做。

## 后续应收集的证据

在另行允许模型运行后，一次固定 CUDA 环境先验证库版本、GPU/驱动、算子 ABI 和运行时配置；记录 checkpoint missing/unexpected keys、最终可训练模块、BERT 离线加载是否成功、实际输入尺寸与 batch。

随后用有明确上限的少量 support 批次测前向/反向/optimizer step，包含预热与稳定计时、CUDA 同步、峰值显存、数据加载时间、loss/梯度有限性、checkpoint 写盘开销。计时与任何模型效用评估分开；不在此阶段扫方法、测试集或多个 GPU 环境。不把“安全读取权重”当成算子、反向或数值正确性已过关。

## 不能预先承诺的部分

- CUDA 算子编译/预编译 wheel 与 GPU 架构、Torch ABI 的匹配；旧历史环境可能不能直接落在新的运行时。
- 高分辨率输入、900 queries、激活 checkpointing、BERT 以及 unfrozen backbone/text 的显存和速度。
- batch=4 在目标 GPU 上是否容纳；改变 batch/梯度累积会改变有效更新数，不能仅按 epoch 比较。
- 初始化、数据上传/读取、模型写盘及评估的额外时间。直接原配置每轮 909 张 test 的 val 可能远大于 1–3 个训练更新，这个行为必须在新协议中明确处理。
- 仅有一个发布 support 抽样；三个 shot 集不嵌套，且存在相邻帧和同前缀组，结果适用范围受限。
- 检测标签制作链、派生许可说明和独立 dev 策略尚未补齐。

因此目前没有“能在两小时内完成训练/实验”的依据，也没有新付费资源或 Colab 额度申请。下一步若执行，首先产出该固定环境的实际初始化、单步耗时和显存证据，再据此估算预算。
