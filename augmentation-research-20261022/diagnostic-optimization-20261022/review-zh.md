# 独立执行前审查

结论：**当前先 HOLD，修复下面的 RGB 哈希定义并保留失败时的原始梯度后，可有条件运行这一次 8 个重置单步诊断。** 没有读取新结果或执行模型、前向、梯度、GPU。原四格 pilot、原工程门失败和首次严格范数审计失败均保留。

## 执行前事项

1. `diagnostic.py` 当前用 `sha256(im.tobytes())` 对照 `decoded_RGB_sha256`；冻结 pilot 的 `qa.py:65` 实际用 `sha256(str(im.size).encode()+im.tobytes())`。定义不一致，第一张图就会失败。须统一到原定义；不能重算/覆盖原 manifest。此判断来自代码，不读取照片。
2. 在结构/配对/oracle 断言可能失败的位置，先保存已经得到的梯度、更新后 trainable tensors 和失败阶段。当前 A/B 断言在 `step()` 内、head/oracle 断言在 checkpoint 保存前；若失败仅留 traceback，无法无重跑地独立诊断。这是审计证据修复，不改变 8 次预算、容差或结果门。非有限原始张量可用 CPU checkpoint 保留，JSON 只存有限摘要/状态。
3. 冻结前建议加显式无 `results`/`analysis.json` 检查、selected_rows 六字段与训练白名单逐项相等、4×16+2×16/角色/范围/重复簇断言；分析时检查自身锁哈希和 completed 的 protocol SHA，并在 NumPy import 前固定 CPU 线程1。准备脚本当前从固定 manifest 直接构造，执行代码有 protocol/code 哈希门；这些补充可降低手工配置/归档误用，不能用于结果后更换样本。

## 解析与配对

`common.py` 的 Q/V 修正是 `BA`，K 修正恒为零；A 按 slot 共享，B=0，底座冻结、head 同初始化，65,317 个 trainable 参数。令 G=∂L/∂(BA)，则 ∂L/∂A=BᵀG=0，∂L/∂B=GAᵀ。B 非零梯度是可检验条件，并非任意数据/损失的数学保证；若不满足就按预定无效/不完整保留，不能换批。

首个新 AdamW 的偏置校正给 `theta1=(1−eta*wd)*theta0−eta*g/(abs(g)+eps)`。因此 A 数据梯度为0仍可有解耦衰减（FP32 下也可能舍入成无可见变化）；B 的梯度范数不与第一步幅度成比例，eps 附近尤其如此。原 warmup step0 的 adapter LR=3e−4/111、head LR=1e−3/111 正确。每批恢复全部 trainable 值、创建全新 optimizer 并清梯度，可避免沿用动量；共有8个孤立更新。固定四个 train 梯度批和两个互斥 train 探针、同一预生成 tensor 用于两布局，公平性成立。读取旧初始小 checkpoint/manifest 元数据不读取开发准确率或封存像素。

CLS 位于 encoder LayerNorm 之后、head 之前，排除了更新后 head 对该量的直接作用；初始共同 head 仍决定梯度方向。原 ViT 默认 dropout 为0；这个诊断不增加随机增强条件。独立复算将逐项对照原初始 tensors、fresh-step oracle、A零/B非零梯度、merged norms、前后特征及全部 batch/probe 比值。

## 可以解释到哪里

同参数量、同 LR、相同 A/B 初始化保证同配方预算，**不保证相同函数步长**。S 包含位置输入、反向梯度方向和下游敏感性；`S/||ΔW||F` 只是实际方向的诊断，不等于等方向比较、Jacobian 谱或初始化单独因果效应。步长不同也不会自动使原“共同配方下位置×增强交互”估计无效；它限制机制归因。当前 weak-only、首个 warmup step 不能解释 RA 差异、30 epoch 收敛或最优 LR。

两次相同初始探针与 `max(repeat RMS, eps_FP32×initial RMS)` 是预先约定的经验分辨率门，不是严格浮点误差上界。低于10倍门就未分辨；不放大 LR。最新 analyze 已处理 S=0 时比值/log 和零 log-SD，合理。两探针、四固定梯度批不构成 seed 或动物重复，无 p 值/总体区间；必须报告所有8个 case、两探针、布局内变异和原始幅度，而不能只报几何均值。FP64 聚合的是保存的 FP32 状态/特征，不会恢复 FP64 模型前向精度。

若有一致且可分辨的位置步长差异，允许提出一个单独冻结的控制：只用训练内校准、对称规则确定位置 LR，使预先指定的局部函数步长预算匹配，并与共同 LR 原配方并列。它改变估计对象，不等同整个轨迹公平、方向相同或新算法；本轮不执行校准搜索/新训练。如果未分辨或批次方向不一致，结束此诊断，不能据此宣布不存在优化差异。任何结构/oracle 失败按预定停止并保留，禁止放宽阈值救结果。

600秒 cap、CPU1、单串行 MPS、.1 memory fraction 和按同步 host 段休息2倍可接受；启动/加载/CPU审计未被同样节奏覆盖，观测 driver 峰值也不是连续硬峰值承诺。硬件可用性以主线实际回执为准，此静态审查不认证资源已通过。不检查8A内容/全机空闲。

## 解释来源与审计限度

已定向核读原始 [LoRA §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1)、[AdamW 原论文](https://arxiv.org/html/1711.05101v3)及 [PyTorch 2.14 AdamW 算法](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)。上述梯度/首步公式是直接代数推导；没有据此声称原创或有限 ViT 位置机制。

新 verifier 只用 stdlib+NumPy 和 CPU `torch.load(weights_only=True)` 读取小状态，不 import 主线模块，不建模型、不前向、不调用 optimizer。它可独立复算状态/特征数字与实际文件哈希；已保存 tensor 哈希不能独立证明未保存输入 tensor 的像素变换，冻结底座前后 SHA 也只能核对运行回执。相关边界会在结果中单列，不能包装成重放了训练过程。当前只写脚本并做语法检查，未执行新结果复算。
