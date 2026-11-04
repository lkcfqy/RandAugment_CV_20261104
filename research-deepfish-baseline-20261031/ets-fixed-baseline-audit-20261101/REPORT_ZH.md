现有 A 已包含官方随机几何增强；ETS 固定示例主要增加 CachedMosaic、随机 HSV 和 CachedMixUp。可以把这组已发表算子作为一个普通增强基线候选，但它同时改变图像组合、框数和来源曝光，不能把差值归因于单个算子。固定饱和度探针及 CPU 交叉输入诊断已结束，原结果不改、不追加强度或种子；此次未调用模型、训练、安装、上传或分配 GPU。

本次固定 ETS 提交为 [`99cfcff8e17844c328e43f1417153b41afb70f9d`](https://github.com/jaychempan/ETS/tree/99cfcff8e17844c328e43f1417153b41afb70f9d)，提交日期 2025-10-24。只读取源码；[来源与哈希](SOURCE_MANIFEST.json)、[逐项静态结果](PIPELINES_STATIC.json)、[28 项核验](STATIC_VERIFICATION.json)一并保存。

**A 与 ETS 固定配方的可执行差异**

这里的“ETS 固定配方”指 README 和训练数据基类中实际写出的示例，不指搜索获胜配置。MMDetection 官方 Swin-T 配置已与本机安装版本逐字节比对；A 的模型配置只改变类别数 80→1 和本地 BERT 路径，优化器参数保持官方值。[官方 Swin-T 配置](https://github.com/open-mmlab/mmdetection/blob/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py)

| 项目 | 已完成实验的 A | ETS 固定示例 / 移用时必须明确的差异 |
|---|---|---|
| 解码与顺序 | cv2、uint8、BGR；LoadImage→LoadAnnotations→Color(1.0)→Flip→RandomChoice→Pack | LoadImage→LoadAnnotations→CachedMosaic→HSV→Flip→CachedMixUp→同一 RandomChoice→Pack。CopyPaste、RandomErasing 是注释，未生效。 |
| Color | 项目额外插入 `prob=1, level=0, min_mag=max_mag=1`；像素恒等，但消耗随机数 | ETS 没有此操作。未来若保留 A 并在对照两臂共同保留它，必须称“ETS 风格固定配方适配”，不是原样配置。 |
| 几何 | 官方默认水平 Flip(.5)；RandomChoice 两分支各 .5；直接随机 resize，或 resize→crop→resize | 列表和值与 A 完全相同，可直接沿用。最终候选短边 480:32:800、长边约束 1333；裁剪支路先选 (400/500/600,4200)，绝对范围 crop(384,600)，允许裁后空框。共享配置不等于共享实际几何随机轨迹。 |
| Mosaic | 无 | `img_scale=(640,640), prob=.6, pad_val=114`；默认中心范围(.5,1.5)、clip=True、缓存40、random_pop=True。输出画布1280×1280，当前图与3个缓存样本拼接；缩放、平移、合并及裁掉画布外框。 |
| HSV | 无随机颜色操作 | `YOLOXHSVRandomAug(hue_delta=5,saturation_delta=30,value_delta=30)`；每次调用各通道独立随机启用，增量来自 uniform(-1,1) 后转 int16；H模180、S/V截到[0,255]，BGR↔HSV。不是 Color(.8)，不会增加来源图/标签。 |
| MixUp | 无 | `prob=.3, img_scale=(640,640), ratio_range=(1,1), max_cached_images=10, pad_val=(114,114,114)`；默认 flip_ratio=.5、max_iters=15、clip=True、random_pop=True。图像固定0.5/0.5混合，框/类别/ignore拼接，转换与过滤后保留；不是软类别标签。 |
| 模型/权重 | 原始 GroundingDINO Swin-T，embed96、depths[2,2,6,2]、window7；官方 ogc Swin-T 权重 | README 入口为 Swin-B，embed128、depths[2,2,18,2]、window12、另一 cogcoor 权重；直接换入口同时换模型容量与预训练资产，不是增强单变量。 |
| 训练与数据 | 项目设5图5框、batch1、workers0、DefaultSampler(shuffle=True,round_up=False)；U200=40轮，每图40次 anchor；无LR调度，最终checkpoint，关闭训练中验证 | 模板为batch4、workers8、persistent_workers=True、15轮、milestone5/gamma.1、每5轮验证、每轮保存；5图单GPU推导每轮2更新、合计30更新，不等于U200。模板默认 val/test 指向 test.json，不能照搬到本项目选择流程。 |

A 保留 AdamW(lr=1e-4, weight_decay=1e-4)、backbone lr_mult=.1、absolute_pos_embed decay_mult=0、clip_grad=.1。A 的无 scheduler、U200、单图 batch、无 worker、固定 support/dev、禁止 test 选择和最终保存均属于本项目协议；不是官方 COCO 训练默认。配置文件的 `train_cfg.max_epochs=0` 是占位，已有 worker 在运行时写入 U/5=40，实际回执为200更新，不能误报为没有训练。

配置与实现证据：[ETS 数据基类](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/configs/_base_/datasets/CDFSOD_detection_few-shot.py#L115)、[训练设置](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot.py#L222)、[Swin-B 入口](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py)、[HSV](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/mmdet/datasets/transforms/transforms.py#L2907)、[缓存实现](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/mmdet/datasets/transforms/transforms.py#L3465)。上述三种额外增强及 Mosaic 基类与本机官方 MMDetection 3.3.0 对应类的完整 AST 相同；这只是这些类的静态等价，不宣称整套仓库或运行环境等价。[比较证据](TRANSFORM_STATIC_COMPARISON.json)

**缓存与来源曝光不能忽略**

Mosaic 在概率判断前缓存原始图/标签的深拷贝，缓存长度≤4时直接跳过；MixUp 在概率判断前缓存已经过 Mosaic/HSV/Flip 的结果，长度≤1时跳过。两者均是 transform 实例中的 `results_cache`，不是全局均匀重采样器。取缓存索引用的是 NumPy `randint(0,len(cache)-1)`，上界不含，因此排除最新槽位；Mosaic 的3次抽取可重复，也可能抽到相同 image ID 的旧曝光。

MixUp 的缓存不含此前 MixUp 结果，但可能含4图 Mosaic；一次合成最多涉及8个来源出现次数，而不是8张独立图片，本 support 总共仅5个 ID。框裁剪、重复、空框及来源实际可见程度仍随随机过程变化。每个 worker 各自积累缓存；worker数量、batch划分、是否跨 epoch 保留、是否每臂重新置空都会改变实际配方。未来若用 workers0、单个数据集实例跨40轮保留缓存，需写为项目适配，并记录缓存命中来源/框数；不能沿用旧颜色实验“来源曝光、GT、几何和RNG两臂逐步相同”的验收条件。

**固定模板、搜索流程和已选配置是三件事**

固定提交中的数据基类把 `data_root` 和 `metainfo` 示例全部注释了，实际使用处仍引用它们；原样入口缺这两项定义。其 head.num_classes=20，也没有自动变为 fish=1。因此本次给的是可精确填写的差异与算子参数，不声称原 README 命令即刻可跑。

README 示例为8 GPU×50组；`dist_train_muti.sh` 实际重复启动同一配置，未在该脚本枚举参数网格；`sampling4val.py` 从 test 标签抽取验证子集（示例比例.9），`sata_logs.py` 按验证 mAP 选各运行的最佳 epoch。不能把搜索后最好结果叫作“无需额外开发标签的固定基线”，也不能仅凭此入口还原论文完整搜索空间与获胜配置映射。[README](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/README.md#L152)、[启动脚本](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/mmdetection/tools/dist_train_muti.sh)、[验证集抽取](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/sampling4val.py)、[日志选优](https://github.com/jaychempan/ETS/blob/99cfcff8e17844c328e43f1417153b41afb70f9d/sata_logs.py)

还静态核对了三个发布的 `5-shot-dataset*.py`：dataset1 实际5-shot/30轮/几何；dataset2 实际引用 **1_shot.json** /7轮/HSV+几何；dataset3 实际引用 **10_shot.json** /20轮/几何。三者都没有活跃 Mosaic/MixUp，类别也不是 DeepFish。文件名不足以确立 shot 协议，更不能把它们直接当统一 FISH 固定配方。[原配置目录](https://github.com/jaychempan/ETS/tree/99cfcff8e17844c328e43f1417153b41afb70f9d/configs)

**5 张 support 没有相对完整 train 标注删框**

当前 support5.json 与官方5_shot.json字节相同（SHA256 `9978bd0fd09becf9dced032315ecf06c63dab0ee2b7abb32925137d2f8b3c2d4`）。完整 train 为3596图/12434框；仅对这5张逐条比对，image记录和annotation全部字段均一致。

| image ID / 文件 | 完整 train 框数 | shot 框数 | 两者 annotation ID | 遗漏的 train ID |
|---|---:|---:|---|---|
|930 / 7463_F7_f000006.jpg|1|1|2197|无|
|1892 / 9870_Gerres_f000022.jpg|1|1|8431|无|
|1900 / 9870_Gerres_f000031.jpg|1|1|8439|无|
|2244 / 9866_acanthopagrus_palmaris_f000080.jpg|1|1|8913|无|
|2363 / 9866_acanthopagrus_palmaris_f000406.jpg|1|1|9032|无|

没有证据支持“为凑5-shot，从这5张图删掉了其他已标注同类目标”。[机器对照（含原 xywh、area）](SUPPORT_COMPLETENESS.json)

已有13图QA覆盖全部5张 support；本次复看现成的 [support 联系图](../followup-group-provenance-20261031/preview/support_contract_sheet.jpg)，没有添加新图或查看 test。每张的已标框对应可辨认鱼体，未识别出能确定为鱼的额外漏标对象。浑浊、遮挡及缩略联系图限制仍在：这不是视觉完整性证明。若以后确认框外有鱼，那是完整 train 与 shot 共同继承的漏标，而非本次比对发现的 shot 删框。此次未改标签、未把额外标注加入训练。

**下一项单次比较能回答什么**

唯一候选问题：在同一原始 Swin-T、当前5图support/64图dev、同一初始化和优化器、batch1与固定200个 anchor 更新下，预先锁定的 **Mosaic(.6)+HSV(5,30,30)+MixUp(.3)** 组合相对已有随机几何基线，是否改变最终 dev 主 AP？只比较整组固定配方，不搜索强度、不拆多臂、不根据结果追加 seed。可沿用已有种子20261031作为这一新比较的候选种子；目前尚未冻结或运行新实验。

两臂都从原权重和空优化器开始，A保留恒等Color；候选组合臂在同一恒等Color之后插入Mosaic/HSV，在Flip之后插入MixUp。其他参数不变。缓存从空开始、同一arm跨epoch保留；仅使用5张support建立缓存。约束为“相同anchor次数/更新数”，记录并承认额外来源出现次数与算力不同，不声称总曝光匹配。两臂几何抽样配置相同，但新增NumPy抽样、合成尺寸和框数会影响轨迹；这是整套固定增强的总效果，不能拆解为颜色/定位机制。最终checkpoint固定，所有64dev照原口径统一评估，主AP maxDets100；AP50/AP75/尺度AP maxDets1000；AR100/300/1000各自注明。原test仍不用于选择或评估。此适配不称ETS Swin-B全论文复现或新贡献，且dev仍额外使用187个标签。

资源尚未确定，不能由剩余额度反推GO：已有成功CUDA候选是Python3.10.19、Torch2.1.0+cu118、MMCV2.1.0、MMDetection3.3.0的单T4；这里的增强类已存在，不需为读取它们安装第三方fork。新组合的解码/缓存内存、输出尺寸、GT数、更新时间、峰值显存及回取带宽没有实测。旧两臂完整重复检查实测28.75分钟/.55109CU，只能作背景，不能充当新组合成本承诺。

当前历史累计1.8088330816CU、5443.8998GPU秒；若仍适用2.5CU/7200秒边界，算术余量约.69117CU、1756.10秒（29.27分钟），包含全部环境准备、检查、训练、评估、回取和释放。按完整权重保留，两份新checkpoint至少需1,384,351,924字节，现6GiB研究目录余量仅约268MB，还未计日志与预测，因此现存储边界不够；不能删旧结果或采用已证实无效的整张量差分来掩盖。后续必须先确定完整存储空间、可执行比较协议及全成本预算，才考虑任何算力。当前本轮只读工作已完成，没有待运行命令或GPU授权请求。
