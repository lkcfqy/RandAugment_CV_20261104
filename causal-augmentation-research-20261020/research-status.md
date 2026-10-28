# 研究进度与范围

检索截止：2026-10-20。目的：判断因果推断 × 视觉数据增强是否值得投入，最多保留三个可证伪候选，不保证新颖或发表。

已核实的重要先例：2026-07 的《Breaking Spurious Correlations via Generative Randomization and Cross-Variant Self-Supervised Learning》已经组合背景生成随机化、跨背景正样本对及后续 GroupDRO；2026-09 的 SAGE 已研究生成补齐稀有子群及合成验证集。不能把这些组合直接作为新贡献。

当前阶段：**文献可行性探索完成，独立审查五项修正已处理。** 保留两项候选：锁定的配对处理及跨编辑器迁移审计优先，具体过滤规则误拒审计次之。ForAug 是更近直接先例，差异已收窄；没有新颖性或发表保证。最终字节／哈希核对记录见 delivery-verification.json。

未训练、未生成推理、未使用 GPU/A100、未安装软件、未付费、未发布/上传或发送外部消息。既有 PDF/HTML 与 8A 不在修改范围。

用户模型要求：研究和审查请求统一 GPT-6.1-Sol / max。三位子智能体均显式请求这一组合，未获得实际运行model/effort回执。只读Mac默认配置为GPT-6.1-Sol / ultra；默认值不证明某次实际执行，当前协调会话与过去每次实际档位仍无法核验。Codex应用只读界面核验被CUA安全策略明确拒绝，已停止、未绕过。详见 model-audit.md。

主报告：feasibility-report-zh.md；来源覆盖：evidence-catalog.md；审查：independent-review-raw.md、review-disposition.md、review-inputs.json。evidence-initial.json 仅保留初期快照，早期“方法待核”等状态以最终覆盖记录为准，DDB标签变化和ALIA过滤方向已有明确更正。

剩余限制：OntoAug正文403；Places条款全文、IN-9图片授权链及配对manifest、Waterbirds/iWildCam实际CSV和图像完整性未验证，iWildCam类别版本冲突待核。没有等待用户批准的训练或写操作，也没有结果不明动作。
