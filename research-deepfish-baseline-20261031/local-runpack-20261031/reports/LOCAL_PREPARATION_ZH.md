# DeepFish 最小运行包：本机准备状态

范围只有官方固定 5-shot 的 A/B 两臂。已形成 69 张公开图像、support/dev 原始记录、两份显式 JSON 配置、最小 driver/worker、候选依赖、官方权重 URL/SHA pins 与完整文件清单。没有上传、分配 GPU、安装依赖或实例化任何模型；原研究目录文件未修改。

## 本机验证边界

当前完整 suite 为 37 项，涉及授权失败无外部副作用、资源费率/截止时间、U 仅由计时决定、逐源图曝光、初始化与可训练参数一致性、配对 GT/RNG 漂移、原子落盘、释放结果不明、显式配置、边界框保留、包成员与隐私、ZIP 回读哈希。

颜色行为用人工 uint8 BGR 数组检查：A 逐像素恒等；A/B 都只消费一次 NumPy 随机数；B 等于 cv2 的 0.8 原图 + 0.2 灰度混合。执行的是固定官方 Color/adjust_color 的隔离纯方法，测试 fixture 没有导入 MMDetection/MMCV，也没有模型。缓存装饰器在独立未启用缓存路径下用 identity 代替；未来实际环境仍必须调用真正已安装的官方 Color 再通过同一契约门。当前本机 cv2 5.0.0、NumPy 2.5.4，与未来候选 cv2 4.11.0.86、NumPy 1.24.3 不同。

JSON 配置已静态核对，并在实际加载前恢复官方要求的 classes、scale、scales、crop_size 元组类型；没有用 MMEngine 构建 runner/model。所有 69 张图仅核原字节 SHA 和 JPEG 头部；没有针对这些图运行任何检测或效用分析。官方 test 不在包内。dev 的 annotation 4195、4660 的 bbox/area 与原记录相同；评估顺序保留 Resize→LoadAnnotations，不另裁边。

## 状态隔离与冻结规则

预检分别在独立进程中运行 15 次更新：5 次预热、10 次计时，每张 support 正好 3 次；预测计时只用 5 张 support，不算 AP，不保存预测或模型状态。正式 A/B 又分别创建全新进程，从原权重和空优化器开始。

正式 U 只从 {25,50,100,200} 按 README 中固定计时成本式选择。该式计入两次正式初始化、两次最终评估初始化、两臂更新、128 次 dev 图像推理、写盘及余量。U 冻结文件写出后才开始正式训练；函数拒收 loss/metric 字段。每臂 U=5E，每张 support 恰好 E 次；任何曝光/几何/GT/RNG 轨迹不一致都阻断评估。最终 checkpoint 仅为固定 U 的张量 state_dict，没有 save_best。

## 候选环境与资源未决

单一候选是 Linux x86_64、已存在的 Python 3.10、torch 2.0.0+cu117 / torchvision 0.15.1+cu117、mmcv 2.1.0、MMEngine 0.10.7、MMDetection 3.3.0。候选依赖尚未安装或做完整 resolver/ABI 验证。未来只允许 PyPI、官方 PyTorch/OpenMMLab 发行来源；关键 MMDetection 源文件另核 SHA。所有模型资产只在未来 Colab 从固定官方来源下载并核已知 SHA；BERT 用 safetensors，PyTorch 强制 weights_only，无不安全 fallback。

本次提出的上限为总 2 CU 和 2 个分配 GPU 小时，含全部准备、预检、两臂、推理和落盘。没有实际报价、环境或吞吐证据，因此当前是 **RESOURCE_UNDETERMINED**。需要未来控制器记录实际分配时刻与经核实的 CU 费率上界，并从分配起计算硬截止时间；预留 180 秒释放余量。若任何候选 U 不适配预算，就只交预检结果，不缩小 64 图 dev 或改变支持集。

实际运行前还需：明确的新上传/模型运行授权、兼容 Python/GPU、已挂载的持久存储、独立且已验证的 owned-session 释放 watchdog，以及实时 CU 报价。当前都没有伪造为 true；包内 context 示例明确关闭授权。driver 不负责偷偷申请新资源，旧计数实验许可不被沿用。

## 保留的可靠性代码

只提取旧代码的必要行为：同目录临时文件→flush/fsync→原子 rename→回读 SHA；小型 JSON 的 content-addressed stdout 回执；独立有限寿命的原生 Colab unassign fallback；外部控制器先尝试释放，再核对自己拥有的 assignment，仅在确认仍 active 时重试。没有复制旧实验结果、凭据、用户名路径或旧 controller 的硬编码解释器路径。

`release_owned` 只提供外部控制器的所有权/核对语义，不是 Colab 分配器。原生 unassign 返回也不宣称释放成功；最终必须由外部控制器核对 endpoint 消失和实际用量。当前未调用这些服务。

## 交付和唯一待执行命令

实际 ZIP 字节、SHA-256、清单 SHA、文件数和隐私核查见 `PACKAGE_RECEIPT.json`。封包器在重新生成清单前强制核对两份冻结标签的 canonical SHA、选中记录身份以及两个边界框。包中没有完整 FISH、test、模型权重、完整仓库、旧实验、个人路径或密钥；69 张原图都无 EXIF/comment/XMP 元数据。

未来获得授权且外部前提都成立后，在 Colab 解包目录执行唯一命令：

```sh
python -B driver.py --run-context /content/deepfish-run-context.json
```

现在不能执行。活动 context 在包外，由未来控制器根据真实授权、分配及清单摘要填写；包内示例不能直接启动。当前结果只说明本机准备与非模型检查完成，不证明 CUDA 可运行、预算够用或增强有效。
