# CountGD++ CPU严格加载与无forward接口核验

**本阶段通过：独立兼容依赖环境、CPU完整严格加载、原生概念token和人工框包装层检查。模型forward尝试/完成、所有nn.Module forward、GPU/MPS/CUDA、训练、pickle读取和模型阶段网络连接均为0。** 没有读取PairTally图片或计数做效用实验；按钮组标签未修改，仍等独立审查。

## 环境

新环境：`research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python`。基于已有CPython3.14.10，选择性只读链接46个旧分发包，复用torch2.13.0、torchvision0.28.0、safetensors0.8.0、huggingface_hub0.36.2等。未更新旧项目环境；新环境优先拥有自己的transformers/tokenizers，旧版本没有混入其导入路径。

从官方PyPI JSON解析并固定版本，实际wheel均来自 `files.pythonhosted.org`，逐个SHA256与PyPI元数据一致后离线安装；没有源码构建、非官方源或远程安装脚本。

| 新安装 | 版本 |
|---|---|
| transformers / tokenizers | 4.49.0 / 0.21.4 |
| timm / addict / termcolor | 1.0.30 / 2.4.0 / 3.3.0 |
| yapf | 0.40.1 |
| opencv-python-headless / pycocotools | 5.0.0.93 / 2.0.11 |
| importlib-metadata / platformdirs | 10.1.0 / 4.14.0 |
| tomli / zipp | 2.5.0 / 4.1.2 |

`pip check` 为 `No broken requirements found.`。新下载HTTP响应体66,609,147字节，约63.52 MiB。完整包hash与URL、只读复用路径、解析元数据和安装日志在 `new_wheel_manifest.json`、`reused_package_links.json`、`dependency_downloads.json`、`../sources/`、`../logs/`。这里的“兼容”仅限已完成的构造/严格加载/辅助函数测试，不声称Python3.14+新PyTorch复现作者全部forward数值或性能。

## CPU严格加载

运行的是此前已下载并静态检查的官方HF源码，版本仍为 `06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1`；没有使用自动 `trust_remote_code`。直接构造原生 `CountGDPlusPlus(config=...)`；配置唯一实质变化是把 `bert-base-uncased` 改为固定本地BERT目录，device原来和实际都是cpu。架构源码和算法修改0。

先检查实际模型 `state_dict()` 与预期完整逻辑集合一致，再对checkpoint中66个去重别名逐一验证：别名和canonical路径确实引用**同一个已注册Parameter对象**，并且实际storage指针、offset、shape、stride、dtype和device一致；仅在这些条件全部满足后，将同一安全checkpoint tensor绑定到对应别名。不是仅凭字符串猜配，也没有补入随机参数。

随后检查完整逻辑键、shape、dtype、CPU设备和所有checkpoint值的有限性，调用实际 **`model.load_state_dict(expanded, strict=True)`**。结果：

- 1,042个物理tensor、66个已证实的共享别名，合计1,108个逻辑state键。
- missing keys = `[]`，unexpected keys = `[]`；没有使用strict=False加载主模型。
- 全部1,108个加载后tensor与其对应checkpoint tensor逐项精确相等，全部值有限；所有参数在CPU、requires_grad=False，991个模块均处于eval状态。
- 主权重SHA256仍为 `94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5`。
- 缺键、多键、错误shape和错误dtype四类故意构造的无效载荷，都在修改模型权重前被检查器拒绝。

证据：`cpu_verification.json`、`loaded_logical_state.json`、`actual_shared_aliases.json`、`strict_payload_rejection_checks.json`；完整可运行检查脚本为 `../scripts/verify_cpu_load.py`。

第一遍检查曾把全部registered buffers误认为都必须进入state_dict，停在检查器断言；源码与实际差异确认只有BERT的 `position_ids` 和 `token_type_ids` 两个显式 `persistent=False` 缓存。它们分别精确为0..511位置序列和全零int64 token type，不是缺失的学习参数。修正仅针对检查器区分持久性，并对这两个缓存的名称和值作显式断言；严格checkpoint条件没有放宽。首遍脚本、失败报告和日志保存在 `attempt01/` 和对应脚本/日志文件，不把它算作模型缺陷。

第二遍完整验证约2.79秒（含最后记录和hash），进程峰值RSS **2,744,745,984字节，约2.56 GiB**。这只是构造/加载/检查的成本，不是推理吞吐。

## C/V负视觉组检查

测试使用人工toy文字、占位256维向量及人工图/框，未使用冻结数据集标签或图片。

1. **实际原生 `add_exemplar_tokens` 六组检查**：F/C/V×A/B，分别插入三个正视觉占位向量与三个负视觉占位向量。实际V输出为 `[CLS,*,*,*,.,*,*,*,.,SEP]`，IDs `[101,1008,1008,1008,1012,1008,1008,1008,1012,102]`。C只额外保留共同类名文字；正负占位值在A/B互换后仍归属正确概念组，组间self-attention mask分离。
2. **实际HF `count()` 包装层六组检查**：F/C/V×A/B，使用320×200人工常色图及六个人工框。真实预处理产生800×1280，框坐标精确放大4倍；每次都传入一个负视觉图组和三负框，C/V显式使用 `negative_texts=['']`。网络位置由一个普通Python recording callable代替，该callable不是nn.Module，在调用边界立即抛出约定停止信号；没有生成任何logits、预测框或预测计数。
3. 单独检查 `negative_texts=[]` 搭配一个负视觉组，会在调用模型前因长度不匹配被拒绝，不能把空列表当作C/V。

证据：`native_concept_token_tests.json`、`native_wrapper_group_tests.json`。记录器被调用6次，单独计数为 `recording_stub_calls=6`；它们不是模型forward。真实模型及所有子模块均安装禁止forward的guard，实际forward计数始终0。

本阶段证明原生辅助接口保留了负视觉组及概念token，不证明完整网络一定利用这些示例，也不提供任何效用结论。BERT、backbone、transformer和ROIAlign都未进行forward，MPS兼容性尚未验证。

## 后续边界

`OBSERVATION_ENTRY_PLAN_ZH.md` 提供最小可复用入口方案，以及只使用人工图的吞吐预检提案：最多4次forward、单设备、墙钟180秒、RSS8 GiB，输出仅成本/兼容性。**这些预算只是提案，尚未授权，也没有执行。** 不分配Colab，不自动开启MPS，不运行正式564queries。

当前仍待：独审解决按钮语义和冻结共同类名；后续授权设备/forward/预检预算；再验证实际前向路径和耗时。没有操作被自动审批拒绝，没有不明结果的动作。原有资产报告保持其历史阶段含义，本报告是新增CPU准备阶段的结果。
