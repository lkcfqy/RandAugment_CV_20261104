# 研究记录：固定少样本适配与开发选择（2026-11-01）

本轮完成，尚无新算法或可验证创新结论。DeepFish 出现一次有害选择；Brackish 零次有害选择、一次错失收益。不能合并为“适配普遍失败”，也不把两个数据源、四个相关 episode 当独立重复。

| 数据／模型 | dev AP@100 (%) | 已评估确认子集 AP@100 (%) |
|---|---:|---:|
| DeepFish P0 | 24.2424 | 38.9383 |
| DeepFish A0 | 27.9068 | 36.9446 |
| DeepFish A1 | 27.8027 | 40.4134 |
| Brackish P0 | 18.0475 | 13.5118 |
| Brackish A0 | 15.1784 | 14.5762 |
| Brackish A1 | 11.9410 | 9.6703 |

统一解释规则：每个 episode 独立比较自身 A 与 P0；dev AP 严格更高选 A，否则选 P0，平局选 P0；不在两个 A 中挑赢家。DeepFish 原确认主问题是 A−P0，下表选择是按该规则回算，未伪称存在事前部署选择回执。Brackish 的选择回执确已在确认推理前冻结。

| 数据／episode | dev 选择 | confirm A−P0 (pp) | 所选−P0 (pp) | 描述 |
|---|---|---:|---:|---|
| DeepFish A0 | A0 | -1.9937 | -1.9937 | 有害选择 |
| DeepFish A1 | A1 | +1.4751 | +1.4751 | 无该类失配 |
| Brackish A0 | P0 | +1.0644 | +0.0000 | 错失收益 |
| Brackish A1 | P0 | -3.8415 | +0.0000 | 无该类失配 |

共同实现：官方原始 GroundingDINO Swin-T；A 各5图／5鱼框、seed20261031、U200、普通几何增强和 Color1、最终 checkpoint。P0为同一完整初始化的未适配推理参照；原903键按既有契约补齐908键，训练专用DN随机状态推理未调用。同CPU／FP32、文本 `fish. `、原生top300；主AP maxDets100，其余AP指标1000，AR100/300/1000分开。全部12指标与来源哈希见 [机器可核记录](SYNTHESIS_EVIDENCE.json)。

限制与数据：DeepFish dev64图187框；确认232图576框、29前缀各8图，前缀隔离不等于视频／场景独立，不能当2014图全池AP；历史3张test曾QA，test未用于本次模型效用。Brackish同一相机、日期隔离；支持候选池6日期，实际两套支持图都只来自2019-02-20及03-06，共享3视频但图像不重复。dev288／325GT／213无鱼标注图，confirm120／38GT／83无鱼标注图；38框很小，帧相关、完整性未知。17个dev越界原框保留；AAU/一致COCO映射fish+small_fish，排未标尾，不用YOLO补标。CC BY-SA／COCO MIT字段冲突未澄清，继续限本地。**两份已评估确认子集均已见，后续不得再称新确认。**

成本：DeepFish A0/A1训练进程18.652/18.603分钟，P0 dev参照2.633分钟，确认生命周期27.659分钟（含准备35.047）；这些是分阶段实测，未拼成全研究总时长。Brackish全流程87.690分钟、峰值6.348GiB、400正式更新及1224前向。历史云消耗保守累计上界1.820706996 CU／5481.0703 GPU秒，非账单实扣，用户上限仍2.5 CU；之后CPU流程及本次整理新增CU为0。所有本轮模型进程已退出。

原记录：[DeepFish dev](../unadapted-reference-20261101/REPORT_ZH.md)、[DeepFish确认](../confirmation-sampled-20261101/REPORT_ZH.md)、[Brackish更正文档v2](BRACKISH_REPORT_ZH.v2.md)。v1及其审计manifest不变，见[文档修订](DOCUMENT_REVISION.json)。已关闭的 [M／曝光线](../same-domain-decision-20261101/DECISION_ZH.md)、[固定饱和度线](../cpu-cross-input-diagnostic-20261101/REPORT_ZH.md)只保留链接，不纳入主结论。

---

# FT-FSOD 阶段LR组件：只读可行性

结论：存在有意义的组件对照候选，尚无效用证据或运行GO。官方固定 `StageWiseFreezeHook` 先仅将other组LR置0，中点后将other设为当时head LR；backbone、language_model、head继续更新。没有设置requires_grad=False，不能称冻结backbone或节省backward。AdamW状态仍累计，所有梯度仍参与原clip，不应改成detach、移出优化器或重置动量。[固定官方hook](https://github.com/Intellindust-AI-Lab/FT-FSOD/blob/c68cea00fac282ec526d59bca55f27b3203daf37/mmdet/engine/hooks/stage_lr_hook.py#L238)（提交 `c68cea00fac2`）。

按既有4份一致的优化器回执和896个参数名／形状静态映射：

| 官方组名 | 张量数 | 参数量 | 当前基础LR |
|---|---:|---:|---:|
| backbone | 175 | 27,520,506 | 1e-5 |
| language_model | 197 | 108,891,648 | 1e-5 |
| head（含neck/decoder/text_feat_map/DN） | 242 | 14,296,867 | 1e-4 |
| other | 282 | 22,203,392 | 1e-4 |

other占全部172,912,413参数的12.84%：`encoder.layers.*` 7,693,056；`encoder.text_layers.*` 4,738,560；`encoder.fusion_layers.*` 9,474,048；`level_embed` 1,024；`query_embedding.weight` 230,400；`memory_trans_fc.{weight,bias}` 65,792；`memory_trans_norm.{weight,bias}` 512。不是近乎空操作。官方普通/HED detector均有上述encoder及辅助状态，名称／模块角色可比，但HED、MMGDINO-B与原始Swin-T不等价。[逐参数证据](PARAMETER_GROUP_MAPPING.json)、[官方定义](https://github.com/Intellindust-AI-Lab/FT-FSOD/blob/c68cea00fac282ec526d59bca55f27b3203daf37/mmdet/models/detectors/grounding_dino_HED.py#L70)。

最小候选只能是 **single-stage cosine 对 fixed-midpoint stage＋同一cosine**：两臂从相同原始state独立开始，原数据／seed／U200／batch1／AdamW／clip／增强不变。令更新t=0…199，c(t)=(1+cos(πt/200))/2；基础LR乘同一c(t)。阶段臂仅令other在t<100时为0，自t=100起跟随head；每更新在cosine后施加该掩码，不能让scheduler意外恢复LR。40轮的中点为20轮；最终200步模型，无dev调度／选优／重启。**旧A为恒定LR、无scheduler，不能直接充当cosine对照，需两臂均重跑。**

这是阶段组件适配。官方FISH5-shot配置实际用HED、plateau监控及save_best，AdamW decay0.05、backbone/lang倍率0.2；本项目保持原decay1e-4与实际组LR，既不替换backbone/HED，也不复制额外dev调度，不能称FT-FSOD全复现。[官方配置](https://github.com/Intellindust-AI-Lab/FT-FSOD/blob/c68cea00fac282ec526d59bca55f27b3203daf37/configs_cdfsod/final_configs/grounding_dino_swin-b_finetune_FISH_5shot.py#L136)。

只可提出的问题：在同更新预算和同cosine下，延后上述编码／查询组的参数更新是否改变适配收益？当前已见集合只能做开发比较。因仍算全部梯度，未来时间应按两次完整A训练（约37分钟）再加预检、评估、落盘估计，不能预扣算力节省；未作新吞吐实测。两新完整ckpt至少1,384,495,924字节，现有项目逻辑约11.89GB，新增将超12GiB上限；当前也无自动执行条件，不删旧资产、不扩大预算。
