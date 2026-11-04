# PairTally × CountGD++ 资产核验

本轮完成官方资产获取、数据结构检查、固定选图和源码/权重静态核验；**尚不能称运行就绪，更没有新贡献或模型效果结论。模型构造、forward、GPU/MPS/CUDA、Colab 和训练均为 0。** 没有执行下载的仓库代码、`torch.load`、pickle 或 `trust_remote_code`，没有安装包、改动旧环境、提交或推送。

全部新增内容位于 `research-counting-exemplar-20261031`。工作前检查了项目及祖先 `AGENTS.md`、相关 `.agents/skills` 和已有环境记录，没有发现适用的额外文件。使用现有 `research-shared-theta-ad-20261030/.asset-venv/bin/python` 读取文件与 safetensors 元数据；下载的源码只作文本/AST 检查。Superpowers 的入口技能有针对受派具体任务的 `SUBAGENT-STOP`，本轮没有另开创意开发或训练流程。

## 资产与版本

| 资产 | 固定版本 | 核验 |
|---|---|---|
| [PairTally 作者仓库](https://github.com/bbvisual/PairTally_Benchmark/tree/4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1) | `4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1` | 官方树与已取文件 Git blob SHA 全部一致；CC BY 4.0 完整 LICENSE 已保存，README 声明学术研究用途 |
| [CountGD++ 作者 GitHub](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/2ecbc4331d6730d017f52a4a97bbcbae7d5d714a) | `2ecbc4331d6730d017f52a4a97bbcbae7d5d714a` | README 反链指定 HF，HF 也反链此作者仓库；根 LICENSE 为 MIT |
| [CountGD++ HF](https://huggingface.co/nikigoli/CountGDPlusPlus/tree/06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1) | `06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1` | 主权重 **937,560,192 字节**；SHA256 `94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5`，匹配委托值和固定版本 LFS oid |
| [官方 BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594) | `86b5e0934494bd15c9632b12f734a8a67f723594` | Apache-2.0；安全权重 **440,449,768 字节**，SHA256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`；config/tokenizer/vocab 与许可也已取得 |

CountGD++ 仓内复制的 GroundingDINO/DETR 文件仍有 Apache-2.0 等原始组件版权声明，均已保留；不能把根目录 MIT 简化为所有第三方组件统一许可。版本、逐文件 SHA256、Git/LFS 校验分别见 `source_pins.json`、`asset_manifest.json`、`source_blob_verification.json`。没有下载 `.pth` / `.bin` pickle 权重。

## 数据、选图和坐标

[官方 README 链接的图像 ZIP](https://drive.google.com/file/d/1TnenXS4yFicjo81NnmClfzgc8ltmmeBv/view) 实测大小为 **152,874,125 字节**，并非预估的约 2.1 GB。服务返回 HTTP 206、正确 Content-Range；先取 ZIP 尾和中央目录，再仅取固定 94 张图的压缩条目，压缩图像载荷合计 19,610,778 字节。没有下载完整 ZIP，**因此不声称有完整 ZIP 的 SHA256**；中央目录和每个 Range 响应有 SHA256，解压图片均核对 ZIP CRC32，并有原文件与解码 RGB 的 SHA256。

实际注释是以简化后完整 `.jpg` 文件名为键的字典，共 **681 图：313 INTRA、368 INTER；47 INTRA pair、50 INTER pair**。pair 键使用官方有序的 `super_category/test_type/positive_code/negative_code`，包含 super category 防止 FOO/OTR 的 BEA 等代码碰撞。47 组图数分布：6 组×2图、2组×4图、18组×5图、1组×6图、1组×8图、1组×9图、18组×10图。`pair_inventory.json` 列出每一组的所有文件、数量和哈希排名。

固定规则：每个 INTRA pair 内按 `SHA256(UTF8('PairTally-observation-v1/' + 官方简化后完整文件名含.jpg))` 十六进制升序，取前两张。原始旧文件名也保留为映射字段，但不用于排序。共 **47×2=94 图，三模式×A/B 最多564 queries**。未根据图像内容、计数大小、模型效果删除或替换图。

`selection_frozen.json` 的 SHA256 为 **`c7ba6b400d0ec6123eda4274625c4d5a64111a664f978001e31215b3ad27b6bc`**，CSV 与 `.sha256` 同时保存。这一冻结针对图片及六个框身份；共同类名另列候选，尚未冻结。A/B 指官方 positive/negative group；未来换目标时须互换整组框、文本和计数，而不改六个框本身。三模式共用这些框。

全部 681 图都恰有 A、B 各 3 个框。主注释的四角格式为 `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]`；可确定性转为像素 `xyxy`。全部四角为有限、正面积、轴对齐矩形，且与 `bbx_anno_valid.json` 按 `image_metadata.original_filename` 一对一对应的原始 `pos`/`neg` 框精确相等。94 张已解码图的 **564 个框全部在图像边界内**，没有六框内部的精确重复。未发现图片原文件或解码 RGB 的精确重复；全部没有 EXIF orientation。分辨率是多种长宽，不能假定固定方图。

**重要：`points` 和 `negative_points` 并非真实物体位置。** 官方 `convert_cvat_to_pairtally.py:785–807,894–896` 用均匀随机点填充长度，边界来自六个示例框估计的方形大小；本轮用独立的 `random.Random(42)` 重建，全部 681 图两类点坐标逐值一致。计数与 metadata、新旧文件名中两类数字一致。65/94 图存在这些随机点超出真实图像边界，523/564 框未通过“仅含一个同类点”的几何诊断，原因正是随机占位点，**不能据此认定示例框错误或模型失败**。后续可使用官方两类计数和真实示例框；不具备逐物体真实定位 GT，不能用随机点核验预测位置。详见 `point_semantics.json`。

## 文字与视觉对应的未决事项

`common_class_candidates.json/.csv` 和 `COMMON_CLASS_CANDIDATES_ZH.md` 给出全部47组待冻结表。来源是作者转换脚本中的短名称 taxonomy；采用 bean、button、coin 等普通物体类别，未把 FOO/FUN/HOU/OFF/OTR 五个粗标签充当 C 模式类名。没有依据模型输出选词。

- **按钮组 `OTR_INTRA_BUT1_BUT2` 有实质冲突**：官方 A/B 文本为 `Beige` / `Clear`，但两张固定图的 A1–A3 都是透明按钮，B1–B3 都是不透明浅色按钮，方向相反。原始文件名还写 black/white。已保存两张图六框放大图 `exemplar_inspection/OTR_INTRA_BUT1_BUT2.png`。目前没有交换框、计数或修改标签；候选 F 文本为 A=`clear button`、B=`opaque white or beige button`，**尚须独立解决文字、视觉与计数组身份，不能静默把原文当正确 F**。
- 珠子组 `OTR_INTRA_BEA1_BEA2` 的文字只有 `Blue and purple` / `Orange and pink`，缺物体名；两张图的示例框支持这两组颜色，原始文件名的 green/pink 已不准确。候选补全为 `blue or purple bead` / `orange or pink bead`，待冻结。
- HCP 的示例是发夹，作者英文写 `hair clipper`；候选改为 `hair clip`。TPK 包括直牙签和牙线棒，作者共同名 `toothpick` 是否覆盖两者需要语义审查，候选 `dental pick`。STO 的 `yellowstone` 候选改为 `yellow stone`。

仅对上述四组八张固定图做了放大语义检查，其他组完成了结构与身份映射核验；不声称人工重新数完所有物体或穷尽所有语义问题。上述问题均在任何模型运行前记录，未进行效果筛选。

## 代码闭包、权重覆盖和加载预期

HF 与固定 GitHub 的 61 个共同 Python 文件中，**60 个逐字节一致，唯一差异是 `ms_deform_attn.py` 的官方安全回退处理**：导入 CUDA 扩展失败时定义 `_C=None`，并仅在 `_C` 存在且输入为 CUDA 时调用扩展。HF 独有 `countgdpp.py`、`example.py`、`test_all_prompts.py`。模型架构 `groundingdino_app.py` 与 GitHub 完全一致；详见 `source_identity_comparison.json` 和 `github_to_hf.diff`。

从 HF wrapper 的 AST 导入图追出37个本地模块，含 package `__init__` 间接导入的原始 groundingdino、COCO 数据工具等；源码闭包全部取得。未导入其中任何模块。`import_closure.json` 区分导入阶段和延迟函数依赖，排除 `__main__` 调试分支；可选 CUDA 扩展不存在不阻止纯 PyTorch 路径。

主权重由 `safetensors.safe_open(framework='numpy')` 仅读 keys/shape，另解析安全格式 header。共有 **1,042 个物理 tensor，1,018 FP32 参数 tensor，24 int64 buffer；233,362,816 个独立参数元素**。依据已检查的配置和源码，独立列举了完整 Swin-B-384（深度2/2/18/2）、BERT-base（12层）、6层 encoder/decoder、融合/文字层、投影和输出头的预期 key/shape/dtype；**物理键 missing、unexpected、shape/dtype mismatch 均为0**。Swin 有355 keys，BERT有199 keys，含全部 embeddings/encoder/pooler。完整 schema 与 header 在 `state_manifest.json`；这属于静态覆盖核验，未冒充运行时 `load_state_dict` 成功。

66 个共享 bbox-head 别名按 safetensors 约定从物理存储去重，header alias 映射与原生 `bbox_embed`/`transformer.decoder.bbox_embed` 的共享关系完全一致，展开后应为1,108个 state keys。未来应在完整审核代码与兼容环境上构造原生模型，使用 **`safetensors.torch.load_model(..., strict=True, device='cpu')`** 识别真实共享存储并验证剩余 missing/unexpected 均为空，或等效的严格、已证共享别名恢复方案。不能用普通 `load_file` 后 `strict=False` 吞掉这66项及其他缺键。已安装 HF mixin 的 `from_pretrained` 默认 strict=False，还有缺安全文件时 fallback 到 pickle 的分支，故不可无参数照抄示例；本轮没调用这些接口。

主权重已经覆盖 backbone 和 text encoder，Swin builder 本身不再下载额外 backbone。**原生构造器仍会先调用 BERT `AutoTokenizer.from_pretrained` 和 `BertModel.from_pretrained`**；为保留原生构建途径，额外取得固定官方 BERT safetensors、config、tokenizer JSON/config/vocab 与许可证。该 BERT 文件含199个 bert tensor和7个预训练cls头，使用旧 gamma/beta 命名；标准 transformers 的 BERT loader 负责其转换，最终 CountGD++ 主权重仍必须严格覆盖。没有使用额外随机模块来弥补主权重缺键。

未来离线加载可把冻结 JSON 配置中的 `text_encoder_type` 指向已下载 BERT 目录，其他架构字段保持原值；设离线、只读本地安全权重并禁止 pickle fallback。不要直接执行 GitHub app/test 的 `torch.load(...)["model"]` + `strict=False` 路径。当前没有构造或修改网络，严格运行时加载和参数有限性尚未验证。

## 原生接口与 MPS 边界

HF `count()` 与 GitHub app 是同一原生模型、独立包装层。C/V 都必须保留 **一个负视觉组和 `negative_texts=['']`**；传空列表会失去该概念或报长度不匹配。HF 在 negative_texts 为 None 且有负框时自动补空字符串，但未来宜显式指定。GitHub app 的 `count_main` 在 negative_prompt 非 None 时保留空负文字组；`predict` 为每组添加句点。官方 `test_dataset.py:481–483` 却在不启用 `--neg_text` 时同时清空 neg_texts 和 neg_prompts，**该CLI开关不能实现本观察C/V**。

对官方 tokenizer JSON作纯CPU tokenization，并用普通 Python 列表独立复写已读的插入控制流（没有运行仓库函数），V 得到 `[CLS,*,*,*,.,*,*,*,.,SEP]`，token IDs为 `[101,1008,1008,1008,1012,1008,1008,1008,1012,102]`；C只在第一段额外包含共同类名。`native_prompt_static_trace.json` 记录了这一静态推导；**未来仍须对原生实际输出做无效果 smoke 断言，当前未通过模型运行验证**。

输入为 RGB，短边800、最大边1333，ImageNet mean/std；原图像素 xyxy 框按真实 resize 比例缩放，Normalize 不把 exemplar 框归一化。ROIAlign output_size=1、spatial_scale=1/8、aligned=True。阈值保持0.23，先正得分严格大于0.23，再正得分严格大于负得分；num_queries=900。F输入细类正负文字，C输入共同正类名及空负文字，V正负文字都空；三者仍各3正框/3负框。这里没有实现新方法、调整阈值或增加增强。

Mac 路径不要求自编译 CUDA deformable-attention：官方 HF 已有基于 `F.grid_sample` 的纯 PyTorch 实现，MPS 输入会走此路径。**这不等于已证明完整模型能在MPS执行**；还涉及 torchvision ROIAlign dispatch、BERT mask、设备标量/索引等。现有 torchvision 的 ROIAlign 有条件选择编译Python路径与原生op，本轮仅检查源码，没有探测算子或开启CPU静默回退。严格CPU加载、原生token插入断言、MPS算子兼容性/耗时均留待下一次授权阶段。

HF包装层和GitHub app都不包括 pseudo-exemplars 或 adaptive cropping，应命名为 **CountGD++ 官方 app/HF 基础推理路径的固定模型观察**，不是整篇论文评测复现。未来须记录900个query的限制及高计数/≥800预测的饱和情形；不自动丢图或改变阈值。本轮不能外推为 CountEx 或全部最新方法失败。

## 依赖与剩余阻塞

HF要求：torch、torchvision、`transformers<4.50`、`huggingface_hub>=0.23`、safetensors、timm、addict、`yapf==0.40.1`、termcolor、scipy、numpy、Pillow、matplotlib、opencv-python-headless、pycocotools、PyYAML。完整原始 requirements 与静态闭包已保存。GitHub app另外有Gradio/官方image-prompter/Spaces等UI及训练评测依赖；HF基础count路径无需启动该UI，也未下载或安装那些额外包。

现有环境是 Python3.14.10、torch2.13.0、torchvision0.28.0、transformers4.57.1、safetensors0.8.0、HF hub0.36.2。**transformers4.57.1不满足官方 `<4.50`，并缺 timm/addict/yapf/termcolor/OpenCV/pycocotools。** 原GitHub历史要求torch<2.6，HF要求并未作同样限制；不据此声称当前环境数值等价。没有重复安装或修改旧实验环境；下一阶段须选择隔离兼容环境并核验版本，尚未执行。

当前未决动作都不是审批拒绝：①按钮组正确F文字与官方A/B计数组关系需解决，其他标记候选类名需协议冻结；②隔离兼容环境和严格运行时加载待实施；③MPS运行时/原生token smoke待后续明确允许。没有新OAuth、购买、非官方源安装、未决权限请求或结果不明的下载。

上游对未来投资门的要求已保留：先按pair平均每组至多两图；再对每pair取 `min(E_C,E_V)`，跨所有eligible pair平均后与F比较，差须≥0.10；至少10pair满足pair均值F≤0.10且C、V各>0.30。等计数图保留，A/B预测数相同不能单独证明忽略示例；若达门，需人工看真实预测框排除错误抵消。**本轮未定义新指标或实现/执行该观察，仅记录约束。**

资源细账在 `resource_ledger.json`，实际下载约1.36 GiB，低于3.5 GiB；新增文件约1.4 GiB，低于6 GiB；资产工作低于45分钟。唯一最初连接失败是受限shell的DNS，已批准的官方资产下载路径随后成功；自动审批拒绝0。每次HTTP与字节数、所有哈希及生成脚本均已保存。

优先交接文件：`selection_frozen.json`、`COMMON_CLASS_CANDIDATES_ZH.md`、`point_semantics.json`、`model_static_verification.json`、`native_prompt_static_trace.json`、`source_pins.json`、`CRITICAL_SHA256SUMS.txt`、`asset_manifest.json`、`resource_ledger.json`。以上均位于本目录 `reports/`。
