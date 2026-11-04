# PairTally × CountGD++：仅资产核验

入口：[资产报告](reports/ASSET_REPORT_ZH.md)。官方数据、模型和tokenizer已下载并核验，94张图/47对及六个示例框身份已固定；共同类名候选待冻结，按钮组文字与示例身份存在冲突。

模型构造、forward、GPU、Colab、训练均为0；旧实验与环境未改。现有transformers版本不符合作者要求，尚不能宣称运行就绪。

- [固定选图](reports/selection_frozen.json)
- [共同类别候选](reports/COMMON_CLASS_CANDIDATES_ZH.md)
- [重要：点坐标是随机占位](reports/point_semantics.json)
- [主权重静态覆盖核验](reports/model_static_verification.json)
- [原生提示静态推导](reports/native_prompt_static_trace.json)
- [版本与许可](reports/source_pins.json)
- [资产哈希](reports/asset_manifest.json)
- [资源细账](reports/resource_ledger.json)

`models/model.safetensors` 是指定CountGD++主权重；`models/bert-base-uncased` 是额外原生构造依赖。`sources/` 保留官方源码/元数据，均未执行。`data/images/` 只含94张固定图；`data/zip_ranges/` 是对应官方ZIP压缩条目。
