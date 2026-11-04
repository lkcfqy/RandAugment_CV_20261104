已完成 CPU 官方加载契约核验，未实例化模型。

固定 MMDetection 3.3.0 配置直接引用同一 822d7e9d 原始 Swin-T 权重。with_box_refine=True、as_two_stage=True、6 decoder layers 导出 7 个独立分类分支；contrastive_cfg.log_scale=0.0 在官方 ContrastiveEmbed 构造函数中生成可学习 float32 标量，初始 exp(0)=1。它们并非被冻结不学习的参数。配置关闭 BERT pooling；官方语言模块仅消费 encoder hidden states。Transformers 4.34.0 的 position_ids 为 non-persistent buffer，其值与旧 checkpoint 的 arange(512) 一致。

官方 MMEngine 0.10.7 默认 strict=False。此次修订明确采用“官方 non-strict 加载 + 精确已知差异与全部共同张量校验”，不声称 strict=True。仅允许准确的 8 missing（7 log_scale、1 类 DN embedding）和 3 extra（pooler 两张量、旧 position_ids）。运行前仍验证 7 标量全为 0、可训练、形状正确；所有 900 共同张量逐值相等；任何其他差异停止。

本机安全读取原权重 SHA256 822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3：903 tensors 全为 CPU，所有浮点张量有限，pooler 形状 (768,768)/(768)，旧 position_ids 精确一致。weights_only=True，无 unsafe fallback，无动态 pickle 白名单。

A/B 配置、数据选择、标注、资产/实现 pin 均与首版逐字节一致；U 决策、曝光、轨迹、初始化比较函数 AST 相同。之前持久化与累计预算的 context 校验改动属于基础设施。此次只修正检查器，不改模型、数据、优化器、U 规则或阈值。正式与预检始终独立重新初始化。

证据：reports/CPU_CHECKPOINT_COMPATIBILITY.json、sources/OFFICIAL_COMPATIBILITY_SOURCES.json；后者记录 6 个官方源文件 URL、字节和 SHA。下一次 GPU 仍必须通过完整初始化与两臂配对门。当前正式/预检 optimizer updates 均 0，dev/test 效用观测均 0；累计保守账本 0.3172815439545578 CU / 795.0891099910737 GPU 秒。
