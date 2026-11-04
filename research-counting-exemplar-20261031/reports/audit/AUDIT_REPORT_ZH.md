# CountGD++ / PairTally 独立资产与语义审查

审查结论：**资产一致性 PASS；原 47 对作为“干净三模式语义对照”整体放行不通过。** 已逐项查看 47 对、94 张图的六框裁剪，并直接查看两张按钮完整原图。共发现 3 对明确文字/视觉冲突，2 对细类身份未能逐框独立确认，1 对共同类语义不稳。建议效果前记录 6 对 NA，其余 41 对为可冻结候选；这不是擅自修改主协议、缩小原 47 对分母或调低原科学门。主选择、原图、标注、计数、主报告均未修改。

本轮仅标准库、Pillow 读取/裁剪/哈希和静态源码审阅；模型构造、安装、权重反序列化为网络、fit、forward、GPU 均为 0。没有读取模型效果，所有裁剪来自原始图像，未增强、修复或生成图像。

## 1. 身份错位定位

`OTR_INTRA_BUT1_BUT2_150_240_b5bef4.jpg`：官方 A/B 计数 150/240，原名 `black-button_white-button_INTRA_OTR_BUT1_BUT2_00150_00240_1_00226.jpg`。
`OTR_INTRA_BUT1_BUT2_130_220_44f428.jpg`：官方 A/B 计数 130/220，原名 `black-button_white-button_INTRA_OTR_BUT1_BUT2_00130_00220_1_00838.jpg`。

两张原图及 12 个框一致显示 A1–A3 透明纽扣，B1–B3 不透明白/米色纽扣。官方 `image_metadata.json` 和 `pairtally_annotations_simple.json` 却写 A=`Beige`、B=`Clear`。每个冻结框均精确等于原 `bbx_anno_valid.json` 的 `pos/neg`，简化注释四角转换也逐值一致；所以本地下载、简化文件名、框坐标转换没有造成交换。发布源自身已有冲突。

转换脚本 `convert_cvat_to_pairtally.py:280–283` 的 BUT1 描述米色、BUT2 描述透明；`:409–439` 用数字后缀索引描述，`:629–636` 用该表覆盖旧文件名描述；`:886–896` 从 metadata/文件名数字取计数并生成占位点；`:900–908` 按原 pos/neg 保留框。当前脚本的有效描述表为长描述，而发布 JSON 是短标签，因此不能声称当前脚本逐字生成了该版全部文字；但相反的 BUT1/BUT2 文字语义也存在于源码映射。

**能够确定的是官方文字映射与视觉组不一致；不能进一步证明“只有文字错、计数一定正确”。** 旧 black/white 文件名不能替代真实类别核验，随机点不能提供定位证据，按钮项也不在较小的 `parsed_annotations.json` 中。现有证据无法判定作者制作阶段究竟是代码类别名、原始 pos/neg 指派还是计数绑定哪一处先出错。建议按钮 pair NA；A=`clear button`、B=`opaque white or beige button` 仅是可观察视觉名，尚不能直接作为已验证的计数 F 标签。禁止为使数据看起来一致而翻转计数或框。

## 2. 新发现和 NA 政策

| pair | 证据与处理建议 |
|---|---|
| FOO_INTRA_SED1_SED2 | `411_377_cb5c01` 图 A 三框是黑色条纹长种子，B 三框浅色扁种子，与作者 pumpkin/sunflower 和第二图 `214_190_042c4a` 的方向相反。原 bbx 已如此，非本地转换。整 pair NA，不能只交换该图标签/计数，也不取第三张替换。 |
| FUN_INTRA_PLC1_PLC2 | `038_037_27c0e3` 的 B3 `[349.14,530.34,402.23,635.84]` 清晰框到红心 Q 及邻近红心 3；B1/B2 是黑花色。原 bbx 已含这个负例框，第二图六框相容。整 pair NA，不能把三负例改成两负例。 |
| OTR_INTRA_BUT1_BUT2 | 上述两图所有框与文字反向，计数语义未解决；整 pair NA。 |
| HOU_INTRA_CTB1_CTB2 | 共同物体 cotton bud 无疑；第一图 A1 白色杆及多个遮挡框不能支持逐框确定木杆/塑料杆。是身份不确定，未宣称作者标错。严格准入建议 NA。 |
| HOU_INTRA_COI1_COI2 | 共同物体 coin 无疑；多个框是模糊头像、背面或暗面，未能逐一辨认 5/10 分。不是错标判定，严格逐框身份准入建议 NA。 |
| HOU_INTRA_TPK1_TPK2 | A 直牙签、B 牙线棒。dental pick 是否覆盖两者存在对象类型歧义；不能为了保留 pair 改成过宽的“口腔卫生用品”。C 暂无足够依据，整 pair不进三模式主比较。 |

统一建议：任一固定图出现必需视觉组不纯、计数语义冲突或严格身份无法判断，整 pair 标 NA；保留原两图与计数，不重采样、不补框、不根据效果解除 NA。若主线程选择仅依赖作者来源而放行硬币等低清身份，须在任何效果前单独锁定这一较弱政策，并明确不是独立逐框证实；不能与 3 对已确认冲突混同。原 47 对投资门若坚持全体干净，当前仍阻塞；审查表本身没有重设分母、阈值或运行数量。

`PAIR_REVIEW_47_ZH.md` 和 `NAME_AND_ELIGIBILITY_PROPOSAL.{json,csv}` 包含完整 47 对逐项名称、理由及两个图 A/B 数字。`IMAGE_BOX_COUNT_AUDIT.json` 包含全部 94 张图和 564 个不变框的逐组身份说明。所有官方计数只是来源一致性核验，本轮没有独立人工重数全部 94 图。

共同名不使用 Food/Fun/Household/Office/Other。可冻结的名词修正：hair clip（非 hair clipper）；blue/purple bead 与 orange/pink bead；yellow stone；中国条形牌明确为 chinese playing card 并把红色“圆形标记”写清；ikea clip 采用 bag clip，peg 采用 clothes peg，stone 采用 decorative stone。TPK 不强行给出共同类。其他普通对象名称逐项在表内保留，不按模型输出挑词。

## 3. 数据与来源独立核验

- 交付清单 436/436 文件 SHA256 与大小通过；冻结选择哈希 `c7ba6b400d0ec6123eda4274625c4d5a64111a664f978001e31215b3ad27b6bc` 一致。
- 从 313 张 INTRA metadata 独立按预定文件名 SHA256 排序，47 对各取前两张，与冻结的 94 张完全一致。
- 94 图全部 A/B 各 3 框，冻结 xyxy 等于原 bbx pos/neg；四角形式、来源文字、metadata/旧文件名数字/points 长度全部一致。这保证传播没有错位，不保证源标注语义正确。
- 681 张图片共 1,362 个点列表、109,271 个点，用独立标准库 `random.Random(42)` 和源码相同边界规则逐值重现，0 差异。点坐标是均匀伪随机占位；只有列表长度承载作者计数。不得用于定位真值、框纯度验证或自动纠错。
- 较小的 `parsed_annotations.json` 只含选中 94 图中的 38 图；这 38 图的 pos/neg 坐标全部与 bbx 一致。3 个条目的差别仅为 bbx 多出的 `frame_path` 元数据，绝非框修正；56 图没有 parsed 版本，不能虚称全部原始 CVAT 已验证。
- 已读固定 commit `4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1` 的 README 和更新元数据。该次 merge 只新增 CountGD++ 评估及 README 修改，没有标注修复；README 使用点列表长度为计数，未提供上述身份勘误。公开 issues 页此次只显示 filename_mapping 缺失问题，不能据此声称作者历史上从未纠错。GitHub commits 历史页此次 web 工具返回 restricted URL，未换通道绕过；历史核验限于本地固定 commit 证据。

## 4. 权重与源码路径

主权重 `models/model.safetensors` 大小 937,560,192 字节，SHA256：
`94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5`。

仅解析 safetensors JSON header、流式哈希：1,042 个物理张量 = 1,018 F32 + 24 I64；233,362,816 个唯一 F32 元素、497,664 个 I64 元素。66 个 metadata 别名全部指向已存在物理 bbox head，精确对应 6 个 decoder layer 共享 MLP、同时注册在 `bbox_embed` 和 `transformer.decoder.bbox_embed`，展开为 1,108 keys。原生代码 `groundingdino_app.py:221–235` 与 cfg `dec_pred_bbox_embed_share=True` 支持这一解释；不是缺失 66 个已训练张量。对已有静态 schema 比较无形状/dtype 缺项，**尚未构造模型，不等于 strict runtime load 已过**。

HF/GitHub 61 个共同 Python 文件中 60 个逐字相同；唯一差异为 `models/GroundingDINO/ms_deform_attn.py`：HF 导入失败设置 `_C=None`，只有 `_C` 存在且 CUDA tensor 时走自定义扩展，其余走 PyTorch 路径。HF 另有 `countgdpp.py/example.py/test_all_prompts.py` 包装；GitHub 的训练/分块/其他评估路径并非全部同一入口。HF/GitHub `groundingdino_app.py` 相同，支持沿官方 app 的结构观察，但不证明 GPU 数值等价。

下一执行阶段应保留主架构、锁定本地安全权重及离线 BERT，并做共享存储感知的 strict CPU 加载。禁止把 GitHub `torch.load + strict=False` 或 HF mixin 的默认宽松/pickle fallback 当验证。此次没有执行这些路径。兼容环境、strict 加载、参数有限性、CPU/MPS 实际算子运行仍待独立授权阶段。

## 5. 原生空文字与三模式合同

HF `countgdpp.py:139–165` 支持 `negative_texts=['']` 加 `negative_exemplar_boxes=[B_boxes]`，保持一个负类组含三个框；`negative_texts=[]` 不等价，会造成组数不匹配或丢失该类语义槽。GitHub app `app.py:334–336` 在 negative_prompt 非空时保留空字符串；`test_dataset.py:481–483` 却在关闭负文字时同时清空负视觉组，因此不能以该 CLI 开关实现 V/C。

| 目标方向 A（B 时整组互换） | positive text | negative_texts | positive boxes | negative_exemplar_boxes |
|---|---|---|---|---|
| F | F_A | [F_B] | A1,A2,A3 | [[B1,B2,B3]] |
| C | 共同名 | [''] | A1,A2,A3 | [[B1,B2,B3]] |
| V | '' | [''] | A1,A2,A3 | [[B1,B2,B3]] |

A/B 换向时保持原目标对应计数、文本和整组三框同步换位；C 的共同正文字不随方向变，负文字始终空。三模式图像和框不变。

源码先编码文本、抽取正三框再抽取负三框，分别插入 label 0 / 1 的词段（`groundingdino_app.py:270–319,470–522`）。V caption 是 `' .  . '`，预期 ID 为 `[101,1008,1008,1008,1012,1008,1008,1008,1012,102]`；“空文字”指没有对象语义词，并非完全没有 token 或 BERT。固定 threshold 0.23，并保留正分数严格大于负分数的比较；不得看效果改阈值。

现有 token 回执是 tokenizer + 列表逻辑模拟；本轮静态确认了代码路径，没有把它升级为实际网络 token/负视觉注入测试已通过。真实执行前需在既定无效用预检中断言两段各有三个视觉 token、正负框组数正确、换向保持身份；不能静默丢负框。

## 6. 解释边界与交付

后续仅能称“CountGD++ 官方 app/HF 基础路径的固定模型观察”；没有执行论文完整 adaptive crop/pseudo-exemplar 评测，不能代表论文最佳系统，也不能外推所有当前计数模型失败。三个明确标注冲突是资产审查发现，尚无模型行为或研究贡献证据。

本审查没有执行、安装、下载额外权重或分配 GPU；没有改原文件。`audit_manifest.json` 记录独立审查产物哈希。需要主执行线程在效果前落实名称和 NA 政策，现有 47 对尚不能无条件作为完整干净观测队列。
