本轮 DeepFish 两臂探索实验已完成，资源已释放。固定 Color factor=0.8 没有改善主指标：dev AP 从 26.8 降至 25.2，差值 −1.6 个百分点。AP50 的上升不足以支持“该增强有效”或“已找到可靠科研贡献”。

| 固定 dev 指标（百分制） | A：factor=1.0 | B：factor=0.8 | B−A（百分点） |
| --- | ---: | ---: | ---: |
| AP，IoU 0.50:0.95，主指标 | 26.8 | 25.2 | −1.6 |
| AP50 | 45.7 | 48.1 | +2.4 |
| AP75 | 26.7 | 23.2 | −3.5 |
| AP small | 0.0 | 0.0 | 0.0 |
| AP medium | 19.1 | 18.1 | −1.0 |
| AP large | 39.3 | 37.0 | −2.3 |

数值来自官方 MMDetection CocoMetric 的固定 dev 输出，按报告值换算为百分制。完整机器记录在 [reports/RESULT.json](reports/RESULT.json)，原始指标在 [A](gpu-run/artifacts/evaluate_A.json) 和 [B](gpu-run/artifacts/evaluate_B.json)。不从 AP50 的正差值另选主指标，也不据此推断普遍机制。

采用一个官方 5-shot episode、seed=20261031，support 为 5 图/5 框，ID 为 930、1892、1900、2244、2363。dev 为先前固定 hash 选定的 8 个前缀组、64 图/187 框；这 187 框是额外 dev 标签预算。support 的 3 个完整前缀均被排除，但前缀不等同于已验证的新视频或新场景；7463 habitat 编号仍有交集。

A/B 均执行官方 Color(prob=1, level=0)，仅 factor 不同，固定 cv2、uint8、BGR，保留同一标准几何增强。两臂各先做 15 次预检更新，每张 support 使用 3 次；全部预检模型、优化器和 RNG 状态丢弃。固定吞吐规则在任何 dev 指标出现前选择 U=200，即每臂每张图恰好 40 次正式曝光。正式训练各自重新从原权重及空优化器开始。全部完成的优化器更新为预检 30 次、正式 400 次，分开记账。

[U_FREEZE](gpu-run/artifacts/U_FREEZE.json) 记录 p90 更新耗时 A=1.5308 秒、B=1.5918 秒；冻结时完整剩余流程保守估算 1855.39 秒、工作预算 3941.66 秒。没有用 loss 或 dev/test 效用选择 U。两臂只评估第 200 次更新后的固定最终权重，没有 save_best 或中途 checkpoint 选优。此次包内没有官方 test，也没有运行其评估。较早的资产 QA 曾查看 3 张 test 图像，但没有模型分数；这些图像未进入本次运行包。

完成的真实性检查包括：44 项本机非模型测试；实际 CPU 与 CUDA 人工算子检查；A 逐像素恒等与 A/B 随机数消费相同；四个训练 worker 的完整初始化哈希、可训练参数名哈希一致，优化器初始为空；两臂完整 200 次图像顺序、几何参数、GT 和记录的 RNG 轨迹完全一致。正式配对摘要为 `f7c060a4c367478bd96dc6d8d663aa7cd29d488761086024bf1e0fc04bf7bed1`，见 [PAIR_CHECK_OK](gpu-run/artifacts/PAIR_CHECK_OK.json)。

每臂保留 19,200 条原始预测，覆盖同一固定 dev 的 64 个 image ID，类别为 fish；分数与坐标有限。原始预测见 [A bbox JSON](gpu-run/artifacts/evaluate_A_predictions.bbox.json) 和 [B bbox JSON](gpu-run/artifacts/evaluate_B_predictions.bbox.json)。全部 70 个远端最终产物已在释放前逐阶段原子落盘、哈希核对并回传确认，随后本机再次完整重算校验，总计 1,400,005,868 字节；见 [最终产物核验](reports/FINAL_ARTIFACT_VERIFICATION.json)。

两份最终权重均为 692,175,962 字节，已在本机使用现有 PyTorch、CPU、weights_only=True 安全读取，908 个张量全部有限，没有实例化模型、unsafe fallback 或动态 pickle 白名单。

| 权重 | SHA256 |
| --- | --- |
| [A final](gpu-run/artifacts/final_A_U200.pth) | `5a2a8d4955c0a91641c9e66ac879740077fc4640f06379e850640a46da2268f0` |
| [B final](gpu-run/artifacts/final_B_U200.pth) | `d14239f71925ac4f813eb80c8241ce7ca1670d1017bcf047ecae994327870778` |
| 原始官方 Swin-T checkpoint | `822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3` |
| 本轮实际运行包 | `dd122a9217321602d79f352a0165980b0b893a1bd1f824ddc64cdc68d2a10cd4` |

成功运行的实际环境为单 T4、隔离 CPython 3.10.19（SHA 固定的 Astral python-build-standalone）、Torch 2.1.0+cu118、torchvision 0.16.0+cu118、MMCV 2.1.0、MMDetection 3.3.0、MMEngine 0.10.7。保留确定性算法。五个核心 wheel 的 SHA 与先前 CPU 验证所得完全一致；来源为 PyPI、PyTorch 与 OpenMMLab 官方分发。原始 Swin-T 和 BERT 资产由 Colab 从固定官方来源直接下载并核已知 SHA。完整安装、版本和导入回执在 [environment](gpu-run/artifacts/environment)。

工程修订均在真实 dev 效用为 0 时完成：隔离 Python 解决原内核版本不兼容；setuptools 69.5.1 修复旧 Torch 导入；按官方配置证明并精确核验 7 个可学习 log_scale 的零初始化、未使用的 BERT pooler 和旧 position_ids，而非泛化放宽键检查；Torch 2.0 的确定性 CUDA 标量索引赋值失败通过官方 2.1.0/cu118 及匹配 MMCV 修复，未关闭确定性或修改上游检测源码。一次 CLI RuntimeError 的原始具体文本未被旧包装器保留，仍未知；该会话经远端清单缺失和 sessions/usage 核查后释放。此后修复本机失败进程挂起、加入启动回执与安全诊断，本轮新会话完成。详见各历史 recovery 目录，本轮没有掩盖或删除失败记录。

累计保守费用账本为 **1.1001052668 CU**；累计分配 GPU 时长上界 **3225.2736 秒（53.75 分钟）**，包含准备、失败尝试、预检、训练、推理和回取，CPU 检查费用也计入 CU。账本始终结转，没有清零，低于 2 CU / 7200 GPU 秒上限。这是带初始缓冲的保守账本，不冒充已结算账单。最终官方检查为 **sessions=0、rate=0**，余额 39.8021605514 CU。见 [controller_report](gpu-run/controller_report.json)。未购买额度、创建凭据或推送 GitHub。

运行包实际 11,678,781 字节，小于 13 MiB；仍仅含 69 张公开图片、support/dev 标注及必要运行文件，没有官方 test、完整仓库或私人文件。重试导致同一公开包重复上传；五次 GPU 尝试记录的输入合计 58,405,321 字节，不能称为“整个实验网络上传只有 13 MiB”。

本轮工程任务无剩余运行阻塞。科研边界仍然明确：仅一个 episode、一个 seed、一个固定颜色强度；dev 是前缀组隔离，尚无跨场景或跨种子证据；检测框派生方法与单独许可说明仍未从所核公开来源补齐。资产事实见 [DATA_MANIFEST](../DATA_MANIFEST.md)，分组与来源证据见 [GROUP_PROVENANCE](../followup-group-provenance-20261031/reports/GROUP_PROVENANCE_ZH.md)。本轮保留这个负主指标结果，不继续增开实验或调用官方 test。
