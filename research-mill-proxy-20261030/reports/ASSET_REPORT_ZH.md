本轮资产准备与 CPU 严格加载已通过。工作仅发生在 `research-mill-proxy-20261030`；旧实验、旧数据和旧环境未改动。没有启动 GPU、分配 Colab、执行任何模型 forward 或计算研究指标。数学协议及正式 326 次 forward 尚待上游审查，本报告不是科学阴性或新贡献结论。

官方来源与许可均保留了本地快照和内容哈希。MILL 的 [作者 GitHub](https://github.com/CVC-Color/MILL/tree/c954236652a319a622fc9055862704b4599c1dee) 与 [作者项目页](https://color.cvc.uab.cat/mill/) 均直接链接 `davidserra9/MILL`，HF 数据卡在固定版本声明 `license: mit`。只下载 MILLs-DSLR.zip，没有下载其它相机/分辨率数据；没有下载、导入或执行 MILL 仓库的研究代码，也没有把数据卡的 MIT 声明当作该仓代码许可。

CIDNet 的 [作者仓库固定 commit](https://github.com/Fediory/HVI-CIDNet/tree/eb43d7d91e9a336c66856824ff9e4603ae41f408) 有完整 MIT LICENSE（Copyright 2024 Fediory Feng）；[HF 模型卡](https://huggingface.co/Fediory/HVI-CIDNet-Generalization/blob/51481ef2546f870060c43eb6d6525399f5b3d2d3/README.md) 也声明 MIT 并反链作者仓库。四个实际导入的架构文件已完整阅读，Git blob SHA 与固定树一致，只有标准张量模块和本地架构依赖。保留了原始源码和 LICENSE，没有执行仓库的安装、训练或评估脚本。

| 资产 | 固定版本 | 实际字节数 | SHA256 |
|---|---|---:|---|
| MILLs-DSLR.zip | `761a8e0320fa8e818e5929e0cba269015cd60e06` | 204,658,447 | `cf14eaf7ac287a30822fd626cba76b255f39e416b95bcc089e48c7aa91503112` |
| model.safetensors | `51481ef2546f870060c43eb6d6525399f5b3d2d3` | 7,920,332 | `2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894` |

两项 SHA256 同时匹配委托指定值和固定 HF tree 的 LFS oid。精确版本、原始 API 响应和下载日志分别保存在 `reports/source_pins.json`、`sources/`、`logs/downloads.json`。MIT 数据声明的证据是作者 HF 数据卡；不扩展为其它资产的许可判断。

ZIP 包含 559 个条目，其中 549 个 PNG。所有条目先检查绝对路径、`..`、反斜杠、重复路径、符号链接、加密标志、单文件/总大小及压缩比，再逐文件验证 CRC。所有 PNG 的 IHDR 均为宽 600、高 400、每通道 8 bit、PNG color type 2（RGB）。仅固定 panel 的 84 张图被解压，计 29,411,749 字节。

| 官方目录 split | 完整 scene_ID 数 | input 数 | gt 数 |
|---|---:|---:|---:|
| train | 30 | 300 | 30 |
| validation | 12 | 120 | 12 |
| test | 8 | 79 | 8 |

split 取自压缩包目录，绝不从 `Scene`/`Test`/`Val` 文件名前缀推断。特别地，validation 内有 `Test1`、`Test2`、`Test3` 等前缀。输入 scene_ID 仅移除最后的 `-1` 至 `-10` 后缀，`_a/_b` 全部保留；GT stem 必须与完整 scene_ID 完全相同。完整文件级/场景级清单在 `reports/archive_manifest.json` 与 `reports/scene_manifest.json`。

按 `SHA256(UTF8('MILL-proxy-v1/' + scene_ID))` 的十六进制升序排序全部 30 个训练场景，选前六个，未进行任何内容质量选择：`Scene7`、`Scene16`、`Scene10_b`、`Scene12`、`Scene23_b`、`Scene15`。全部测试场景为 `Test13` 至 `Test20`。每个场景顺序为 L10 anchor、L1/L3/L5/L7、bright GT，共 14 × 6 = 84 张，14 个 GT 和 70 个 input。30 个候选的完整排名与哈希见 `reports/train_scene_selection.json`。

`reports/panel.json` 为 UTF-8、按 key 排序、两空格缩进、末尾 LF 的实际文件；它的 SHA256 为 `5b3da46d5849b0c3b8b71aa9604a87e666523a70e43f19b97a579e042528892b`，另有 `panel.csv` 与 `panel.sha256`。文件级 manifest SHA256 为 `bfcccb892be9337163f76e86c4d0ecde875e73c0912dac77e857267cc6390b80`。每张 panel 图均保留原始 PNG 哈希及 decoded RGB 哈希。

作者在固定 [MILL README](https://github.com/CVC-Color/MILL/blob/c954236652a319a622fc9055862704b4599c1dee/README.md) 中声明 `Test13-9.png` 采集后损坏并被移除。该路径确实不在 ZIP，是唯一缺失档位，也不属于本轮 panel；`Test13` 的其它档位及 GT 完整保留，没有删除整个场景。原始 PNG 字节和解码后 RGB8 字节的精确重复审计覆盖全部 417 张 train/test 图，跨 split 及合并集合内部的重复组均为 0。decoded hash 定义为 `SHA256(b'RGB8\0' + struct.pack('>II', width, height) + row_major_RGB8_bytes)`。这里 raw 指 PNG 原文件字节，不是相机 RAW 传感器文件。validation 仅做 CRC、原文件哈希、PNG header 和命名检查，没有解码像素或开展研究读取。

真实灯光与场景定义的依据是 [MILL 论文 v1 第 3 节及 Figure 1](https://arxiv.org/html/2511.15496v1#S3)，HTML 快照已保存。作者描述在固定室内布景中调整可编程灯光，固定相机参数；DSLR 使用 Nikon D5200、ISO 100、f/9、1/5 秒。L1 是最低灯光功率，GT 是最大功率下且控制饱和的亮参考，L2–L10 来自最低和 GT 之间的照度间隔；较小数字表示较暗，L10 是最亮 input，仍不同于 GT。原始采集为 RAW，经过 Camera RAW 处理，再以双线性方式制成 600×400 的 MILL-s。因此当前文件是处理过的 8-bit RGB，而非线性 RAW。

作者报告 50 个场景、6 种背景与 98 件物体，并说明 train/validation 与 test 的物体和背景不重叠；目录里的完整文件 stem 是官方可操作的 scene 单位。论文/README 没有解释 `_a/_b` 与同一物理布景之间更细的关系，不能把它们额外声称为统计独立实体。本文没有依据图像内容重划场景；精确重复为零也不能证明不存在语义相关或近重复。这个小 ZIP 没有 lux/曝光元数据旁文件，固定曝光和灯光事实来自作者文献，未逐图独立重测；不能把档位直接当作已知线性 RGB 乘子或逐图绝对 lux 值。

模型配置精确为 `channels=[36,36,72,144]`、`heads=[1,2,4,8]`、`norm=False`。使用 `safetensors.torch.load_file(..., device='cpu')` 和 `model.load_state_dict(state, strict=True)`。191 个 FP32 tensor、1,975,569 个参数，missing/unexpected keys 均空；所有权重有限，逐 tensor 与加载后 state 完全相等。没有 prefix 猜配、`torch.load`、pickle、remote code 或 allowlist 变更。加载后 `eval()`、参数冻结、所有参数在 CPU；model forward 数为 0。详细证据为 `reports/model_cpu_verification.json` 与 `reports/state_manifest.json`。

| 项目 | 源码默认 / 官方示例 | 本轮锁定及未来位置 |
|---|---|---|
| 饱和度 | `net/HVI_transform.py:13` 默认 1.3；`eval_hf.py:36,55` 示例参数 1.0 | `model.trans.alpha_s = 1.0` |
| 亮度 | `net/HVI_transform.py:12` 默认 1.0；CLI 名为 alpha_i | `model.trans.alpha = 1.0`（不是虚构 alpha_i 属性） |
| 两个 gate | `net/HVI_transform.py:10-11` 默认 False | `model.trans.gated = True`、`model.trans.gated2 = True`，与 `eval_hf.py:57-58` 一致 |
| gamma | `eval_hf.py:38,59`：外部输入幂次，非模型属性 | 未来调用边界固定 1.0；本轮不调用 |
| GT 均值修正 | `measure.py:119` 的 store_true 默认 False；91–94 行进行 GT 均值缩放 | `use_GT_mean=False`，所有 GT 均值缩放排除，GT 不进入模型 |
| 学到的 k | checkpoint 的 `trans.density_k` = 0.983546793460846 | 保持权重；`this_k` 是 HVIT 中更新的运行时缓存，不能随意设成 1 |

`reports/inference_contract.json` 固定 RGB、[0,1]、FP32、NCHW、eval，输入按原生 600×400（均为 8 的倍数），无需 resize/padding，无 ImageNet normalization；未来输出先 clamp 到 [0,1] 再计指标。当前没有实现或执行正式研究指标。官方 `eval_hf.py:27` 的 strict=False 路径未使用；官方 `eval.py`/`measure.py` 都没有执行。

依赖在本候选新建的 `.asset-venv` 中隔离。该环境只读复用项目 PyTorch 等包和旧资产环境中已安装的 safetensors/HF 依赖，没有修改旧环境。唯一新安装包为作者 requirements 指定的 `einops==0.6.1`，从 PyPI 官方 JSON 解析 wheel URL，在 `files.pythonhosted.org` 下载并校验 SHA256 后使用 `--no-index --no-deps` 本地安装。未执行未知安装脚本。新环境 Python 3.14.10，torch 2.13.0，safetensors 0.8.0；这与作者历史 torch 1.13.1 不同，strict load 已通过，但不能据此声称 forward 数值等价，正式运行前仍受协议检查约束。

资源细账见 `reports/resource_ledger.json`。GPU/MPS/CUDA 计算时间 0，Colab 分配 0；实际 HTTP 响应体下载 203.15 MiB，额外为网页/连接器查阅预留 5 MiB 后为 208.15 MiB，低于 500 MiB。新增文件逻辑大小约 232 MiB，文件分配空间约 246.66 MiB；保守峰值上界额外计入一份完整压缩包和 5 MiB 元数据余量，约 446.84 MiB，低于 1.5 GiB。数据检查约 1.79 秒，CPU 严格加载约 0.49 秒，模型加载峰值 RSS 约 238 MiB；整体墙钟从本任务工作区创建时间计录约 10 分钟，低于 25 分钟。

检查了项目及祖先路径、任务工作区、相关 checkout 的 `AGENTS.md` / `.agents/skills`，未发现额外适用文件。没有凭据创建、登录、购买、发布、推送或私人数据上传。项目页首次出现 TLS 证书验证失败，同一 HTTPS 地址在保留证书验证的有限重试后成功；未关闭 TLS 校验。HVI README 大小写首次请求得到 404，根据官方 tree 的实际 `Readme.md` 修正后成功。没有动作被自动审批拒绝，没有待定或结果不明的资产操作。

目前资产无阻塞。未确定项保留为：上游数学协议、正式 forward 的运行时与数值检查、逐图绝对照度/原始相机元数据，以及官方未解释的 `_a/_b` 物理语义。上述事项没有被伪装成研究结果，也没有触发额外模型计算。
