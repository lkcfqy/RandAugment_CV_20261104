# 候选池、拒绝日志与实际数据覆盖

2026-10-21。下列仓库与小表由只读数据研究者核验；ForNet两个验证JSON由协调者进一步实际下载、解析并保存。没有运行作者模型代码。任何“未找到公开日志”均不等于证明作者从未保存。

| 数据入口与固定版本 | 实际核验内容 | 能做什么／不能做什么 |
|---|---|---|
| [ALIA官方](https://github.com/lisadunlap/ALIA/tree/ce01c941311a6e919a371e31661e83c6d54e9a05) | 完整目录、README、过滤／生成代码与3CSV。Waterbirds captions 4,795行，源标签0=3682/1=1113；iWildCam训练子集6,052行、7标签、15地点、95非空组合；另一caption表127行仅1标签 | 可核源支持，不是增强候选分母。[过滤代码](https://github.com/lisadunlap/ALIA/blob/ce01c941311a6e919a371e31661e83c6d54e9a05/filtering/filter.py#L301)能输出拒绝/保留索引和预测CSV，但实际发布文件未取到。后续匿名API已实读21个dataset最新manifest及2个历史版本、5个prompt/outputs表；候选图片清单已可读，全池两门日志与原运行对齐仍未取得（详见主报告第6节） |
| [DDB官方](https://github.com/ArianYp/DDB/tree/42bcda35b075fc6175d757510bbb12233fa6afdf) | 完整目录、README、generate.py；仅通过者保存图和filename/label/place；分数打印而非完整ledger | 不足以回算拒绝有效性；[代码](https://github.com/ArianYp/DDB/blob/42bcda35b075fc6175d757510bbb12233fa6afdf/generate.py#L132)把place置为原候选0、生成者1，不能当实际背景。需要原池/日志/源映射或未来另获准重建 |
| [ForAug代码](https://github.com/tobna/ForAug/tree/fca8765fe8e5c3f99ee8d5e426b87d06b1c817cb)、[ForNet数据](https://huggingface.co/datasets/TNauen/ForNet/tree/dd6b027a1a01b00fea442276668a54a697eb838b) | 掩码择优代码会删除落选版本；末端背景ratio在剪除前有公开JSON。协调者实际读取两验证JSON，总5,989,250字节，哈希与资格统计见coverage receipt和analysis JSON | 当前可重建末端ratio键的资格两侧，不能恢复早期候选掩码、独立图像有效性或实际背景类别；未取得图像zip，元数据键不等于实际文件验证 |
| [Spawrious官方](https://github.com/aengusl/spawrious/tree/651e90415bfb414a5b6579cb1f53b83e6bb8c165) | 完整目录、README、LICENSE、generate_dataset.py；整批布尔清洗在保存前执行，重试直至prompt配额满足 | 公布图是清洗后，未确认失败批次caption/NSFW/latent/ID ledger。关键词实现切词集合交集，不能直接替代犬种与背景视觉审核。批次联动不能单凭代码推成最终群体失衡 |
| [Hong审计代码](https://github.com/hongrachel/clip-filtering-bias/tree/23a28a23f54e0036101791a0d73db74718ce0e61) | 8个目录文件及6个notebook JSON的静态代码；依赖clip_l14分数、is_kept和本地表 | 未随仓库提供逐行parquet/JSON审计表；没有现成可许可的全池替代品。自然图文与imputed群体也不等同增强对象×背景。未下载人像或作新属性推断 |

## 许可

- ALIA、DDB与Hong仓库本轮未见LICENSE，不据公开访问就推定可再发布数据或代码；增强图及原图许可仍需独立确认。
- ForAug代码MIT、ForNet卡标MIT，不能覆盖ImageNet原图权利；当前保留公开JSON作本地元数据分析，没有图片再发布。
- Spawrious LICENSE写CC0-1.0而README Licensing写CC BY4.0，具体图包条款未确认。不能默默选择更宽松一项。

## 实证门槛

ALIA完整候选池、两个gate输出及独立有效性未到位，群体误拒实证仍NO-GO；ForNet末端metadata分析已GO且完成。若可取得旧artifact和日志，不必先生成；否则未来需先保存生成即刻的全部候选，再评分／人审。训练、评分或生成均未在本轮执行，不因元数据可得就启动GPU。

已知错误与覆盖缺口：DDB原文表2 NotBlond=120,000而表4合计12,000、Waterbird=1,112/1,113及均值阈值定义不一致，未替作者猜修；需要以锁定代码为准并报告差异。本轮数据研究者最初受沙箱DNS限制，获准只读网络后成功；没有未解决自动审批拒绝。旧CodaLab证书、OntoAug403、CUA Codex拒绝不重试。


2026-10-21公开发布补核：图片发布manifest已确认可匿名读取；请勿将此前HTML-only结果解释为未发布图片。`label_predictions.csv`为类别汇总，非逐候选表；固定代码活跃置信度门使用原标签阈值（307–308行），不同于被注释的预测类版本。当前双门审计仍因匹配日志／运行版本缺失而材料阻塞。3个图片HEAD均405，GET未尝试，不能推断下载权限；无待处理的自动审批拒绝。
