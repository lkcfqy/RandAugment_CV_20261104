# 单 episode、单 seed 两臂探索计划（未执行）

定位：官方 5-shot 的一个固定 episode，额外固定前缀隔离 dev；比较普通颜色处理是否产生值得继续检查的信号。不是 Domain-RAG 复现、正式跨场景基准或科研贡献证明。当前只准备本机文件，未安装、实例化模型、启动 CUDA、分配资源或上传。

## 唯一实现与参数

采用 **OpenMMLab 官方 MMDetection v3.3.0**，固定 commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`，原始 `configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py`。仅保留一个 detector 实现。它的 Swin-T / 原始转换权重与本地已核资产匹配；不要运行 Domain-RAG 的自动修改脚本、Swin-B、MM-Grounding-DINO、Flux 或检索路径。

静态对照发现 Domain-RAG vendored config 改了 dataset base 和顶层 max_epochs；其 GroundingDINO 类另外增加了可选 autocast 分支。所读 COCO loader、DN generator、formatting 和版本约束与 v3.3.0 相同。对照摘要在 `OFFICIAL_VENDOR_COMPARISON.json`，不是对整份仓库的等价性声明。[官方配置](https://github.com/open-mmlab/mmdetection/blob/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py)

| 项目 | 两臂共同固定值 |
| --- | --- |
| 支持集 / seed | 官方 5 图 5 框；ID 930,1892,1900,2244,2363；seed 20261031 |
| 初始化 | 本地 SHA-256 为 `822d7e9d...424e3` 的 691,901,857 字节原始 Swin-T 权重；BERT 资产和代码也锁哈希。两臂开始前比较全部模型张量哈希 |
| 类别 | `fish` 唯一类别，COCO id 1 → 内部 label 0；新实验显式 `bbox_head.num_classes=1`，900 queries 和 max_text_len 256 保留 |
| 新初始化参数 | 已核 checkpoint 不含 `dn_query_generator.*`。DN 的 1×256 label embedding 等运行时新增状态须以同一 seed 初始化并比对；只接受预先解释清楚的 missing keys，其他缺失/形状不符先停止报告 |
| 优化 | FP32、batch=1、梯度累积=1；AdamW lr 1e-4、weight decay 1e-4、backbone lr multiplier 0.1、clip norm 0.1；不冻结新增模块、不增添源域训练；LR 恒定 |
| 曝光与顺序 | workers=0；DefaultSampler shuffle=True、固定 seed；每 epoch 正好 5 张；U=5E，源图曝光每臂恰好 U 次，每张 U/5 次。无 MixUp/Mosaic/重复扩增图集 |
| 更新数 | **尚未冻结**。只在吞吐预检后按计算预算选一个 U；两臂完全相同，不按 dev/test 分数选择 |
| checkpoint | 固定第 U 次更新后的最后 tensor-only state_dict；通用 CheckpointHook 关闭，训练期间不评估 dev/test，无 save_best，不按分数早停或挑轮次 |
| 最终评估 | 两个固定最终 checkpoint 各在同一 64 图 / 187 框 dev 上评估一次；报告 COCO bbox AP、AP50、运行日志和差值；不作单 seed 的显著性/泛化推断 |

训练几何变换沿用官方 Swin-T：RandomFlip p=0.5；二选一直接多尺度 resize，或 resize→absolute-range crop(384,600，允许空框)→最终多尺度 resize；尺度 480–800（步长 32），长边 1333。几何参数和变换后 GT 在两臂逐图核对一致。所有 dev 只做固定 resize，不做颜色增强。

**唯一实验差别**：加载图像和标注后、几何变换前加入官方 `Color`。A 为 `prob=1.0, level=0, min_mag=max_mag=1.0`（identity）；B 为同样设置但 `min_mag=max_mag=0.8`，固定将原 BGR 与灰度图作 0.8/0.2 混合，不搜索强度、不用生成图。

两臂都保留该 Color 操作，因为它即使 prob=1 仍消耗一次 NumPy 随机数；只给 B 添加它会改变后续随机几何。固定 level/min/max 不再采样幅度。A 必须在未来环境中实测为逐像素恒等；两臂 RNG 重置、采样顺序与几何轨迹也需核对。证据在 `sources/color_transform_evidence.json`。[官方 Color](https://github.com/open-mmlab/mmdetection/blob/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a/mmdet/datasets/transforms/colorspace.py)；[MMCV adjust_color](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/image/photometric.py)

`templates/two_arm_config.py.in` 已准备，只有 U、颜色幅度和 work_dir 三处待填。它只是模板，未被 import/执行，也未用 MMEngine 解析。两臂训练参数除了颜色幅度及输出目录应无差异。模板的 `test_dataloader` 是官方测试接口名称，但唯一输入文件为本次 `dev64.json`；运行包不包含官方 test。

## 单一 CUDA 候选环境

| 组件 | 固定候选与证据 |
| --- | --- |
| 系统 / Python | Linux x86_64 / CPython 3.9.21；一个 CUDA 11.7 支持的 GPU；显存是否足够留给实测，不能由模板保证 |
| Torch / torchvision / CUDA | torch 2.0.0 / torchvision 0.15.1 的官方 wheel 配对 / CUDA 11.7；不直接使用本机 Python 3.14 CPU 检查环境。[PyTorch 历史版本配对](https://pytorch.org/get-started/previous-versions/#v200) |
| MMCV / MMEngine | mmcv 2.1.0 CUDA wheel + mmengine 0.10.7；满足官方 mmdet 约束 `>=2.0.0rc4,<2.2.0` 与 `>=0.7.1,<1.0.0`。官方 cu117/torch2.0.0 索引中确有 cp39 manylinux wheel；只读索引，没有下载/安装 wheel |
| 文本 / 激活 checkpoint | transformers 4.34.0、tokenizers 0.14.1、huggingface-hub 0.17.3、safetensors 0.5.3、fairscale 0.4.13；BERT 指向已有本地资产并禁用隐式联网 |
| 图像 / 数值 / COCO | numpy 1.24.3、Pillow 11.1.0、OpenCV 4.11.0.86、scipy 1.13.1、pycocotools 2.0.8、shapely 2.0.7；其余运行支持依赖列于 runtime_candidate.txt |

核心轮子匹配和版本范围有静态依据；辅助版本大部分取自 Domain-RAG 历史 detector 环境，**整个候选列表尚未做 resolver/pip check/import/算子验证，不是已验证 lockfile**。唯一有意变动是选择已发布稳定 mmcv 2.1.0 wheel，以及官方 pip 配对的 torchvision 0.15.1。所有传递依赖和环境镜像 digest 要在实际运行前冻结。

必须有编译的 MultiScaleDeformableAttention；mmcv-lite 不足。当前未核目标 GPU/driver 和 CUDA 算子前反向 ABI，较新 GPU 不能直接假定兼容这套历史 CUDA。若此单环境不匹配，报告具体失败，不自动切换另一模型或扫多个环境。[MMDetection 安装说明](https://mmdetection.readthedocs.io/en/v3.3.0/get_started.html)

明确类文本 `('fish',)` 与 `return_classes=True` 避免自由句子 NER；源码的 NER 分支会调用 `nltk.download`，计划不走该分支。jsonlines、NLTK、caption metric 依赖没有放入最小候选；未来离线 import 检查如显示必需，只报告最小缺项再固定版本，不运行下载资源或加入整个生成环境。

## 最小吞吐预检与冻结顺序

1. 另行获准环境运行后，在**一个**指定环境检查版本、GPU/driver、CUDA op 前反向，执行离线 BERT 加载和上述数据/颜色/初始化契约检查。记录 missing/unexpected keys、参数 trainable 状态、初始权重张量哈希。此前不进入效用比较。
2. 每臂从同一初始化与 RNG/采样状态开始，预热 5 次 optimizer updates，再计时 10 次；合计每臂 15 次、两臂 30 次的独立预检。使用实际 5-shot 和正式几何策略，CUDA 同步计时；记录每步、P50/P90、加载时间、输入尺寸、峰值显存、loss/梯度有限性及写 checkpoint 耗时。预检 checkpoint 全部弃用，不作为最终初始化/结果。可以用 5 张 support 测纯推理时间估算 dev 开销，不算 AP。
3. 基于较慢臂的测量、初始化/写盘/64 图评估开销和保留时间，确认单环境墙钟预算，再从 `{25,50,100,200}` 中冻结**一个**能容纳两臂的 U。这个集合是预算选项，不逐档训练；若连 25 次/臂都不合适则只报告吞吐。没有“两小时可完成”的预先承诺。
4. 冻结 U、精度、batch、代码与输入哈希后，把预检状态完全丢弃，重新从原始初始化各运行 A/B 一次。记录每次 image ID、几何参数与 source 曝光；任何参数或预算变化都在两臂开始前共同确定，不让一个臂多看数据或多更新。
5. 仅对固定最终 checkpoint 做一次 dev 比较；官方 test 封存，无 test/save_best 选优。单次差值只用于决定是否值得扩大独立重复，不宣称有效性已被证实。

未来执行复用官方 `tools/train.py` 的 Config/Runner 入口（launcher none，单 GPU），只加一个很短的驱动：`runner.train()` 完成且 `runner.iter==U` 后，从内存导出一次 `model.state_dict()` 为最终文件，再用官方 TestLoop 对 dev 评估。两份渲染配置加这个输入/日志核对驱动即可，不搭新训练框架。当前只有配置模板和计划，尚未创建或运行该 driver/runner。

安全加载契约也须保留：MMEngine 0.10.7 的本地 checkpoint loader 默认调用 `torch.load`；PyTorch 2.0.0 官方源码支持 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`，未来进程必须设置它，并对原始权重显式做 `weights_only=True,map_location='cpu'` 检查。若安全加载失败则停止，不能 unsafe fallback 或动态加白名单。BERT 采用现有 safetensors。通用 MMEngine checkpoint 可能附带 MessageHub/HistoryBuffer 对象，因此模板关闭该 hook，最终只存张量 state_dict，不为重新加载一般训练状态放宽反序列化限制。该路径目前只是静态核对，尚未在候选旧版 Torch 中实测。[PyTorch v2.0.0 serialization](https://github.com/pytorch/pytorch/blob/v2.0.0/torch/serialization.py#L775)；[MMEngine v0.10.7 loader](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/runner/checkpoint.py#L347)

## 未来输入类别与大小（未上传、不是上传许可）

| 类别 | 已测字节 |
| --- | ---: |
| support5 + dev64 的 69 个原始 JPEG | 11,756,375 |
| 官方 5_shot.json | 1,827 |
| 仅复制既有记录的 dev JSON | 24,814（只在内存序列化测量，未写训练文件） |
| 上述数据合计 | **11,783,016**（约 11.24 MiB） |
| 已核原始 Swin-T 权重 | 691,901,857 |
| 已有 BERT 候选 7 文件，含许可和 README | 441,169,829 |
| 全部上述输入合计 | **1,144,854,702**（约 1.066 GiB） |

新增配置、清单和最小开发脚本另给 1 MiB 上限；官方 detector 源码/依赖若在远端从固定公开版本取得，则属于另行批准环境准备，不需要上传整个 Domain-RAG。若公开权重在目标环境直接下载/已有缓存，可减少本机传输，但当前没有进行这些操作。

不包含 FISH 整包、官方 test、未选 train、计数项目、凭据或生成器。确切类别和计量规则见 `PLANNED_INPUT_SIZES.json`。此前计数实验的 96 MB 上传许可**不适用于这个新实验**。现在只完成本机准备，运行/上传仍未获准也未尝试。
