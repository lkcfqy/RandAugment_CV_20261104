# cifar100_view_route_pilot_v1：唯一一次预结果域修订

主线在 Pets 所有真实 LP、校准、效用观测均为 0 时，传达 Pro 审过的资源驱动域修订。注册时间、原协议 hash 和决策记录见 `AMENDMENT.json`。Pets 原协议、资产失败证据及关闭快照保留；本修订仅因原始资产可用性，不使用之前 TTA-CP 的 ResNet20 输出挑域，也不复用其 40k/20k 划分。

除以下明确差异，完整继承 `../PROTOCOL_ZH.md` 的模型资产、八臂梯度定义、优化器、180 秒训练时间目标、校准、最低步数、B5 阶段、J/最终权重和验证、全局选择、六模型 test、配对实用门、恢复、科学定位与停止规则。

- 仅官方 CIFAR-100 原始 32×32 RGB 和 fine label 0..99。只读复用已有 MD5 `eb9058c3a382ffc7106e4002c42a8d85`、SHA256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` 原始归档。不给原始 Python pickle 任何执行能力：用 `pickletools` 的受限惰性 opcode 解释器读数值、字符串、列表/字典和已限定 uint8 ndarray 状态，不调用反序列化 globals、不加 torch globals。图像可保存原分辨率 uint8 数组，不缓存放大图像。
- official_image_id 固定为 `train/00000`..`train/49999`、`test/00000`..`test/09999`，取官方记录顺序；保留原 filename、批文件 SHA、原 CHW record SHA、宽高+RGB SHA。原始数组没有 EXIF 元数据。按原始 record/RGB 重复连通组去重，标签冲突全组隔离；跨官方 split 保 test 字典序最小，排除其 train；train 内保字典序最小。
- 每类 validation=16，共 1600，按 SHA256 UTF-8 `cifar100-route-v1|validation|class_id|official_image_id`。其余官方 train 池各 seed 11/22/33 取16，共1600/seed，键 `cifar100-route-v1|support|seed|class_id|official_image_id`；允许 seed 间重叠。同一 seed 八臂共享 support 和初始化。
- 每类从清理后官方 test 合格组取50，共5000，键 `cifar100-route-v1|test|class_id|official_image_id`；其余 test 不用。各类不足立即停止，不换图或重划。无验证/测试效用后修改。
- LP、validation、test：原 32×32 RGB 直接 PIL bicubic resize 到 224×224，随后 ImageNet FP32 normalize；没有 256 resize/center crop 或 TTA。weak 在原32²上做原 RRC(scale .8..1, ratio .75..4/3)输出224²，再 flip .5；strong 是同 weak 副本加原生 RA(2,M9,bins31,bicubic,fill124,116,104)，不改操作池。
- 官方完整1000类 DeiT strict load 后替换100类head。每 seed 1600support deterministic特征，head Normal(0,.01)/bias0，MPS全批 AdamW lr .003/wd0/200 updates，其他 LP 规则不变。EntAug qi 缓存1600，熵归一化 log100。补充100类 shape/gradient oracle。
- 初始化、数据、weak、RA、routing 的 seed 派生算法和域独立规则保持原协议，仍为 `pets-route-v1|rng-v1|domain|...` 前8 bytes/63bit；这是固定计算随机键，不是数据划分键。只有上述三个划分 namespace 改为 `cifar100-route-v1`，从而保留原 headinit/modelinit 派生 seed。
- 真实新数据校准后才可锁定 N。普通配方 N=floor(180/cr)、至少128；B5 NFT=floor(144/cFT)至少96、Nhead=floor(36/chead)至少16。B2/B3共测同N、C1/C2共测同N；B4 warm/active 均测取保守路径。每实际 path 5 warmup + 3×10 完整链路，cr取块中位数，最大块做完整预算预测。不预设 CIFAR 更快。
- 24runs全部 J及final各1600validation，最终选择后6模型各5000test，全部计入保守预测。不删除臂、减评估、改步数门或补时救资源门。
- 全局B1..B6选B*，C1/C2选C*，跨3seed final val正确总数最大、并列ID；selection锁hash后只算6个test。实用门仍为平均配对Δ≥1pp且三seed都>0，不是显著性、不成立原创。
- **原预算不重置**：2026-10-30 19:14 UTC 开始、2026-10-31 01:14 UTC截止；全研究累计MPS阶段7200s，已有5.0354287500013015s；新增文件峰值3GiB，包含Pets保留产物和本分支。共用 `../mps-ledger.json`。只用Mac MPS FP32，无Colab/付费/新OAuth/公开上传。若完整计划装不下，记resource_unresolved；本次是唯一域修改，出结果后不再换域。

本分支不会声称 CIFAR 结果可证明 Pets、通用弱路由最优或新的梯度路由原则。需要额外披露每类16个验证标签；16-shot仅指梯度标签。
