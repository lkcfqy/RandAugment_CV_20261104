# pets_view_route_pilot_v1 冻结协议

本文件在任何效用观察前记录。来源：本线程父任务完整冻结指令；网页 Pro 审查结论为允许有限效用筛选，未授予原创性结论。开始准备 2026-10-30 19:14:00 UTC，绝对截止 2026-10-31 01:14:00 UTC。历史停止快照已阅读；本次新的明确执行指令仅恢复此独立候选，不修改旧实验。

## 定位和边界

弱图更新 backbone；head 平均 weak 和 strong 的监督；strong 对 backbone 直接梯度截断。这是小样本短迁移经验配方。Deep Augmentation（TMLR 2025，arXiv 2303.14537v5）已有隐藏增强与 stop-gradient；APNetwork（2107.11990）已有轻重增强不同路径；Gradient Routing（2410.04332）已有通用路由。通过实用门也只能支持下一轮直接 prior 比较，不成立新梯度路由原则、原创性或弱路由最优。本轮不增加 reverse、hidden-layer 对照和数据集。

## 资源和资产

- 仅 Mac MPS FP32；关闭 AMP、compile、dropout、stochastic depth、CPU 模型 fallback。允许 CPU PIL 解码、hash、指标、序列化；不通过减 batch、降精度、CPU 模型救 OOM/算子失败。
- 首次准备到报告 6 小时；累计 MPS 阶段最多 7200 秒，含加载、增强、切换、同步、保存、等待、失败重算；新数据/模型缓存及产物峰值不超过 3 GiB，不含已有环境。所有校准和 LP 计账。资源失败为未定，不能作科学阴性。
- 官方 DeiT `deit_tiny_patch16_224-a1311bcf.pth`，timm `deit_tiny_patch16_224.fb_in1k` 非蒸馏。官方 README / 配置 / 许可 / revision / 文件 SHA 固定；`weights_only=True` 安全加载，不增 globals，不 unsafe pickle。1000 类普通完整模型 strict load 成功后明确替换 37 类线性 head。不用 DINO center。
- Oxford-IIIT Pets 官方数据，CC BY-SA 4.0，原图版权保留。存下载来源、MD5、SHA256；下载包核验解压后允许删除本任务压缩包，不删除历史数据。无 commit、push、公开上传或购买。
- 首先预算预测：图片约 0.8 GiB、48 份最终/J 权重约 1.05 GiB、单正在运行模型双恢复检查点约 0.13 GiB、初始化/模型/依赖/日志预留 0.25 GiB。正式前以真实文件尺寸重算；恢复槽仅用于正在运行的 run，完成后由最终/J 永久权重替代，不为所有完成 run 永久保留优化器。

## 数据

官方 trainval/test 边界。为全部官方列名图像记录文件 SHA256 和 EXIF transpose→RGB 的 `struct.pack('>II',width,height)+RGB.tobytes()` SHA256。按像素哈希联合文件哈希连通分组；标签冲突整组隔离；涉及 test 的组只保 test 字典序最小图并排除全部 trainval 副本；trainval 内部留字典序最小。仅 manifest 排除，不删除原图，无近重复挑图。

class_id 用官方 1..37，image_id 为官方不带扩展名字符串。每类 validation 按 SHA256 UTF-8 `pets-route-v1|validation|class_id|image_id` 排序取 16。剩余池按 `pets-route-v1|support|seed|class_id|image_id` 为 seeds 11/22/33 各取 16；不同 seeds 可重叠，各臂共用。各类不足 STOP。test 为清理后官方全部 test。余 trainval 不训练。16-shot 指梯度标签，另有每类 16 验证标签。前期读取 test 仅做 hash，不能称 test 文件完全未读取；测试指标仅最终选择锁定后。

## 初始化与视图

每 seed 共用一次 LP：官方冻结 eval backbone，support 短边 resize256 / center224 deterministic 特征。head Normal(0,.01)、bias0；全批 AdamW lr .003 / wd0 / betas(.9,.999) / eps1e-8，200 updates CE，无 schedule、平滑、早停，在 MPS 上完成。每 seed 八臂初始化同 hash，不继承 LP optimizer。

weak：PIL RGB RRC224 scale(.8,1)、ratio(.75,4/3)、bicubic，再 flip .5。strong 从同一个 weak 复制后 torchvision 原生 RandAugment(num_ops2,magnitude9,bins31,bicubic,fill(124,116,104))，原操作池。ImageNet mean/std FP32，无其他增强。val/test 同 resize256/center224，无 TTA。

随机键在实现中固定为 SHA256 UTF-8 `pets-route-v1|rng-v1|domain|seed|position...` 的前 8 bytes big-endian 并屏蔽到 63 bits；domain 独立包括 data/weak/ra/routing/headinit。视图键含 global update、batch slot、image_id；data 每 epoch 固定 randperm，drop_last=True。弱/RA 均用独立 CPU torch Generator 状态作用域调用 torchvision，路由独立键；省略分支不消费其他键。同 seed 同 update prefix 样本/weak/RA 抽样一致。RA 原生操作及符号共用该 RA 独立作用域，跨臂不受其他 RNG 推进影响。

## 八臂与优化

正式 batch32，每轮 shuffle/drop_last。AdamW (.9,.999),eps1e-8，不 clip；backbone基础lr1e-4、head1e-3；低 LR 仅 backbone×.1；矩阵 wd.01，bias/LayerNorm0。每 run 前 ceil(.05N) linear warmup，然后 cosine 到基础 LR .1。固定版本后不升级。

- B1 weakFT：backbone gw，head hw。
- B2 dualFT：backbone (gw+gs)/2，head (hw+hs)/2。
- B3 dual lowLR：同 B2，backbone LR×.1。
- B4 EntAug-FT：一个自适应图 ge/he，非原文复现。每 support qi 初0，前 ceil(.05N) weak 并更新 cache；之后 M=clip(floor(9qi+.5),0,9)，同 RA2ops。该次单图 forward 后 qi=clip(1-H(softmax z)/log37,0,1)。无额外 clean/teacher/EntLoss、新 cache 重做或跨 run cache；warmup 不做 RA，之后 M0 按原库行为。
- B5 weakFT→headrefit：80%预计时间弱图 FT，20%冻结骨干用双图均值 head loss；head 不重置。第二阶段重新 head AdamW，独立 schedule；第一阶段用其 N schedule。不缓存增强特征。
- B6 randomroute：每 batch Bernoulli(.5)，骨干取 gw 或 gs 系数1，head 均值。
- C1 candidate：骨干 gw 系数1，head 均值；强图 no_grad 特征、head 可学。
- C2 candidate lowLR：同 C1，骨干 LR×.1。

同参数状态算完两视图后只 step 一次。独立 autograd oracle 验证梯度尺度，不能候选变 gw/2 或 2gw；允许 concat64 等价优化。截断仅直接骨干梯度，head 可间接影响后续骨干。

## 校准与预算锁

任何 val/test 效用前，各实际 path 在丢弃副本上 5 warmup + 3 blocks×10 updates；每块同步完整链路。记录 perstep 块中位数为 cr，块最大值做保守总预算。B2/B3 合并同 N，C1/C2 合并同 N；B4 weak 与 activecache 都测且取保守 path。普通 run N=floor(180/cr)，B5 NFT=floor(144/cFT)，Nhead=floor(36/chead)。所有普通 FT 至少128，B5至少96FT+16head，否则 resource-undetermined，不放宽门。正式前实际保守预测 24 runs、LP、校准、val/J、选中6测试、保存全部放进剩余时间与磁盘，保存 `protocol.lock.json` 包括全部 N/schedules，runorder 用 Python Random(20261030) 打乱 8×3 列表并存。

次要共同步 J 为集合 {64,96,128,192,256,384,512} 中不大于所有配方 total N 的最大值；保存 J 与最终权重/val logits+ID，B5 标明 stage。J 不主选模、不做投入门。正式只跑锁定 N，不根据结果补训练。

## 选择、评估和可靠性

24 runs 全完后 final val 跨3 seed平均 Acc。B1..B6 选一个 global B*，C1/C2 一个 global C*，以整数正确数和并列 recipe ID 确定，不能 per-seed 选。`selection.json` 及 SHA 锁定后仅 B*/C* 三 seeds 共6模型开 test。实用继续门：mean paired test Δ≥1pp 且三 seed Δ均>0，不是显著性。未过不换 baseline/LR/seed，过门不自动扩实验。

每 run 永久保存最终/J 权重、val logits+ID、配置/初始化/划分hash、N/完成/实际时间；B5 stage boundary 另存。进行 run 在四分之一、J、stage、final 保存完整 model/optimizer/schedule/RNG/数据增强路由位置/Entcache/累计时间，双槽轮替、SHA 读回。有限恢复最多一次且不重置累计账本；不支持算子/OOM直接资源未定。最终报告状态分别写科学、资源、资产、验证，区分执行失败与科学阴性。精确结果、来源、版本、hash、实际时间、近似预算误差和 limitations 均归档。
