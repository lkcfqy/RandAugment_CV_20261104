# Domain-RAG detector-only 静态配置审计

固定仓库：`LiYu0524/Domain-RAG@2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54`（提交时间 2026-02-02T10:09:36Z）。源码归档 SHA-256 `e8cbb3a8534d63f505c2d3dbd5631e3b8ae6744b79fd05261ca95cd2b6fe8c12`。源码只读取，没有执行其配置、安装脚本或训练入口。

解析由 [resolve_configs_static.py](../scripts/resolve_configs_static.py) 完成：用受限 AST 读取常量、dict、列表、变量引用及必要的加法，按配置 dict 深合并/list 替换/_delete_ 规则推导。它没有 import、exec、eval 上游源码，不调用 MMEngine Config，不构造 detector。输出 [static_resolved_configs.json](static_resolved_configs.json) 属于静态推导，未来仍须在固定环境核对运行时 dump。

## 原始 GroundingDINO Swin-T 路径

入口：[grounding_dino_swin-t_finetune_16xb2_1x_coco.py](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py)。

| 项目 | 提交中的实际值 / 静态继承结果 |
| --- | --- |
| 第一数据基类 | `../_base_/datasets/CDFSOD_detection_few-shot.py`，有效 data_root 是 DIOR，非 FISH |
| 另两项基类 | `schedule_1x.py`、`default_runtime.py` |
| 要分析 FISH 的最小数据替换 | 第一基类改为同目录 `CDFSOD_detection_few-shot_FISH_{1,5,10}shot.py`；本次仅在静态解析器内模拟 |
| 权重 | 原始转换权重 `groundingdino_swint_ogc_mmdet-822d7e9d.pth` |
| 模型 | GroundingDINO；900 queries；Swin-T embed_dims=96, depths=[2,2,6,2], heads=[3,6,12,24], window=7 |
| neck | 输入通道 [192,384,768]，输出 256，4 个输出层级 |
| 文本 | bert-base-uncased；BertModel；max text len=256 |
| 原始 bbox_head.num_classes | 80；替换 FISH 数据基类并不会自动改成 1。FISH metainfo 为单类 fish，类别映射和 DN 分支初始化需未来实际模型检查 |
| 原始图像 / annotation | `data/FISH/train/` + `annotations/{k}_shot.json`，只涉及发布 support；这里无生成图、检索或 Flux |
| batch / worker | 每设备 batch_size=4，num_workers=8，persistent_workers=True |
| sampler | DefaultSampler shuffle=True；AspectRatioBatchSampler；没有 RepeatDataset |
| val / test | 两者均为 `data/FISH/test/` 和 `annotations/test.json`；CocoMetric(metric='bbox', classwise=True) |

FISH 数据文件见 [1-shot](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/grounding_dino/CDFSOD_detection_few-shot_FISH_1shot.py)、[5-shot](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/grounding_dino/CDFSOD_detection_few-shot_FISH_5shot.py)、[10-shot](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/grounding_dino/CDFSOD_detection_few-shot_FISH_10shot.py)。这是候选 detector 基线的数据入口分析，不是已证实的作者论文 baseline 配方复现。

## 生效的常规增强

上层 Swin-T 文件第 119–175 行重新给出 train_pipeline，并写入 train_dataloader.dataset.pipeline。因此 FISH 数据基类中的 CachedMosaic(prob=.6)、YOLOXHSVRandomAug、CachedMixUp(prob=.3) **被覆盖**，不能把它们计入此 detector 路径的生效增强。

有效流程：

1. LoadImageFromFile → LoadAnnotations(with_bbox=True)。
2. RandomFlip(prob=0.5，默认水平翻转)。
3. RandomChoice，两条路径且配置没有显式指定分支概率：
   - 直接 RandomChoiceResize：短边候选 480、512、544、576、608、640、672、704、736、768、800，另一边约束 1333，keep_ratio=True。
   - 先在 (400,4200)/(500,4200)/(600,4200) 中随机 resize，随后 RandomCrop(crop_type='absolute_range', crop_size=(384,600), allow_negative_crop=True)，最后同上述 11 个尺度再 resize。
4. PackDetInputs，保留 text/custom_entities 等元数据。

测试流程为 LoadImage → FixScaleResize((800,1333), keep_ratio=True) → LoadAnnotations → PackDetInputs。数据预处理 mean=[123.675,116.28,103.53]，std=[58.395,57.12,57.375]，bgr_to_rgb=True。没有在此有效 pipeline 中发现 RandAugment、额外颜色增强、生成图混合或 CopyPaste。

后续若引入另一种普通增强，必须明确它与以上流程的差异；不能将基类中未生效的增强误报为已有对照。

## 训练循环、优化器和次数

[1x schedule](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/_base_/schedules/schedule_1x.py#L2) 的 train_cfg 为 EpochBasedTrainLoop(max_epochs=12, val_interval=1)。Swin-T 顶层 `max_epochs=30` 仅被其 param_scheduler.end 引用；**不会自动回写 train_cfg.max_epochs**。

实际优化器覆盖基类 SGD，变为 OptimWrapper + AdamW(lr=1e-4, weight_decay=1e-4)；backbone lr_mult=.1，absolute_pos_embed decay_mult=0；clip_grad max_norm=.1。没有冻结 language_model 的设定。没有显式梯度累积或 AMP 包装。

scheduler 整个列表覆盖基类列表：MultiStepLR(begin=0,end=30,by_epoch=True,milestones=[11],gamma=.1)。基类 LinearLR warmup 不再生效。auto_scale_lr 合并为 enable=False, base_batch_size=32，默认不会因单 GPU 自动改 LR。

数据 N=1/5/10、batch=4，按 [MMEngine v0.10.7 DefaultSampler](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/dataset/sampler.py) 的 round_up=True，以及该仓库 AspectRatioBatchSampler 的 drop_last=False 推导；所有 support 均为相同横向尺寸，无 repeat、无梯度累积：

| shot | 单 GPU 每轮 optimizer 更新 | 直接配置 12 轮 | 若显式设置 30 轮 | 4 GPU 每轮更新 | 4 GPU 每轮样本呈现数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 12 | 30 | 1 | 4 |
| 5 | 2 | 24 | 60 | 1 | 8 |
| 10 | 3 | 36 | 90 | 1 | 12 |

完整假设及计算见 [STATIC_UPDATE_COUNTS.json](STATIC_UPDATE_COUNTS.json)。4 GPU 的样本数含 sampler 补齐重复，不等于独立 support 数。上述 30 轮是显式覆盖 train_cfg 的假设，不是提交中的 Swin-T 直接入口行为。也没有证据说明作者期望何种有效更新预算，不能悄悄用 repeat/长迭代补足。

直接 Swin-T 路径每轮仍对 909 张 test 图做 val；12 轮共 10908 张验证图像呈现，会明显影响时间预算。其 default_runtime CheckpointHook(interval=20) 未设 save_best；在 MMEngine 默认 save_last=True 下仍会存最后 checkpoint。即便不自动 save_best，也不能用每轮 test 指标人工挑选方法或轮次。

## 自动化脚本不是本次可执行入口

[auto_modify_swin_t_config.py](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/auto_modify_swin_t_config.py)：

- 默认 DATASETS=['ArTaxOr']、SHOTS=['1']；FISH 在注释列表中。
- 先修改 Swin-T 的首个数据基类，却在临时配置中继承 SWIN_B_CONFIG，train_cfg.max_epochs=30。
- Swin-B 又继承 Swin-T，覆盖 backbone embed_dims=128、depths=[2,2,18,2]、window=12、neck 和另一 Swin-B 权重。
- 执行字符串是 dist_train.sh 临时配置 **4**。脚本有文件修改、os.system 和恢复逻辑。

本次没有运行，也没有修改它。静态报告保留其实际行为，没有将其当作 4 GPU Swin-T 正式基线。

## 必须隔离的 mm_grounding_dino 路径

[另一 FISH 1-shot 配置](https://github.com/LiYu0524/Domain-RAG/blob/2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_fish_1shot.py) 有独立差异：

- 继承 mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py；
- load_from 是 `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`；
- num_classes 改为 1，20 轮、milestone=15，backbone 和 language_model 的 lr_mult=0；
- val 数据仍为 test.json，checkpoint interval=1 且 save_best='auto'。

未下载这份约 1.094 GB 的权重，未混用其冻结策略、结果、更新轮数或 test 选优规则。该差异与自动脚本的 Swin-B 问题也不是同一个问题。

## 后续显式配置仍需决定的项

若另行授权吞吐/效用实验，至少需要一份独立审阅配置，固定 FISH shot 文件、原始 Swin-T 权重、模型类别/DN 初始化处理、有效 pipeline、world_size、batch/累积、显式 train_cfg、scheduler、冻结范围、文本资产路径和随机种子；保存运行时完整配置和 key-load 报告。训练期应关闭 test 驱动的验证与选优，并事先确定最后模型或独立 dev 策略。此次仅给出这些缺口，没有替用户选择新的实验协议。
