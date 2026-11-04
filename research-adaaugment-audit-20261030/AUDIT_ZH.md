本次有界审计确认：在 AdaAugment 固定公开实现 `d854af15467285ad8f46500d6fc2da5225fb5bd0` 中，当前 adaptive 输入的 reward 归因于缓存中的旧动作，而 actor 更新使用随后采样的新动作的 log-prob。应记录为该固定实现的动作归因问题。论文 Algorithm 1 也具有先计算旧输入 loss、再决定新 magnitude 的关键顺序，因此本结论不是“代码偏离论文”，也不能证明原论文准确率错误。

研究立项维持 **NO-GO**。这次只完成正确性 fixture，没有自动修改算法、启动训练或提出性能贡献。

| 完成项 | 结果 |
|---|---|
| 正式 fixture 检查 | **105/105 通过**；包含 26 个来源哈希、21 个源码行锚点及 58 个语义/数值/干预/资源检查 |
| 正式计算墙钟 | 成功尝试 4.389 秒；首次沙箱失败 0.626 秒；合计 **5.015 秒 < 600 秒** |
| 数据 | 8 个合成 sample_id；5 组主缓存测试共 80 条访问记录；2 组预取探针共 32 条 |
| Monte Carlo | 固定种子 20261030；新/独立旧动作各 262,144 个，共 524,288 个 Beta 标量 |
| 置换 | 7 类案例，145 个穷举索引置换；另含破坏联合动作的拒绝测试 |
| 资源与外部写入 | 目标模型训练 0、optimizer 步 0、GPU 计算 0、Colab **0 CU**、真实数据/权重下载 0、框架安装 0、commit/push/上传 0 |
| Ctrl-A 保留检查 | 309 个文件的大小/mtime 前后一致；≤2 MB 文件额外核对 SHA-256；未写入旧目录，不将此检查冒称大权重逐字节校验 |

所有文件位于当前允许写入的 `research-adaaugment-audit-20261030`。计算用本机已有 Python 3.14.10 / PyTorch 2.13.0 / macOS arm64。代码、协议、完整 JSON、原始日志及失败尝试均保留。

**来源核验与依赖补全**

固定 [GitHub 提交](https://github.com/Jackbrocp/AdaAugment/commit/d854af15467285ad8f46500d6fc2da5225fb5bd0) 的 API committer 时间是 2025-07-20T02:17:32Z；它不能被表述成 2026 年刚发布的新实现。本轮仅审计指定版本。论文为指定的 [arXiv 2405.11467v3](https://arxiv.org/html/2405.11467v3)。

只下载 20 个 Python 文本、2 个 YAML、README、论文 HTML，以及 commit/tree 两份元数据，共 26 份、516,766 字节。没有下载仓库中的日志、pyc、图片、训练数据或权重。23 个仓库文本的 Git blob SHA-1 与固定树逐一匹配；由 tree 条目重建的根树和 7 个子树全部匹配，根树为 `fb45e179cf00202c45a6f72f93d8ec2c8a57bb0d`。API tree 响应的 sha 字段回显了请求的 commit；因此额外重建 Git tree 对象核验，而不把该字段直接当根树哈希。

| 文件 | SHA-256 |
|---|---|
| Dataset.py | `606a593a437f6863f2c7e843891638cd801bc9e2e4f486537766d98d1416eee4` |
| train.py | `b2e4f9af7565b9d68378e2fc948b98d139d4b4a1a8aad595fa31052259128f2a` |
| A2C.py | `95165b5ecc02bdf7ffac77df375e4929f397b6cbba92bf472832814d16f66cf4` |
| augmentation/adaaugment.py | `5c95357c403c51ad67492b9223f4907859cd41c20b925837bbf04cf6de9a1f97` |
| 论文 HTML | `674ef6e6788f27bb921d68ac21c70eadfc65af90098506bddea51c12c16e3487` |

完整 URL、HTTP 状态、抓取时间、长度、SHA-256 和 Git blob 哈希见 `source-receipts.json`；树对象验证见 `source-tree-integrity.json`；逐行映射见 `SOURCE_MAP.md`。

静态核验纠正/补充了以下依赖，避免只盯住三行代码：

- 缓存是 CPU float32 Tensor，原地索引写入；不是普通 Python list。首次 adaptive 访问前，`train_first_epoch` 已采样并写缓存，不能把它简单视为全零起步。
- 官方 loader 硬编码 8 workers、batch=128、shuffle=True；CLI 的 `--num_worker` 没被该构造调用使用。官方未指定 persistent_workers/prefetch/context。当前库默认 persistent=False、prefetch=2；旧版本/操作系统的精确默认环境未被仓库锁定。
- `make_magnitude_transform` 将 PARAMETER_MAX 设为 29；实际 level 是 `min(int(29m)+1,29)`，并不是浮点 m 本身。Rotate、平移、Posterize 还会做各自的整数参数转换；3 个标准操作忽略 magnitude。
- adaptive 操作在每次构造 transform 时重新抽取；full_transform 在 Dataset 初始化时建立并持有其抽到的主操作。none/full/adaptive 的 crop、flip、cutout 等不能自动视为同一随机实现。它们会影响实际 reward，但不会让已经算好的 reward 事后依赖新动作。
- 可选 reward normalization 默认关闭；即便开启，它处理的仍是旧输入的 loss。critic 读取 none/ada feature，后者也在新动作之前产生。源码中未发现隐含的重新生成当前输入、重新计算当前 reward 或延后对应 log-prob 的路径。
- `net.train()` 下三路前向与 BN 状态相关，`no_grad()` 不能当作 `eval()`。所以跨 epoch×class 置换可能改变 minibatch 组成、BN、历史与联合操作结构，不能称为纯因果分解。
- README 只列 Python/PyTorch/torchvision 下界，未完整锁定运行依赖。当前环境中 yaml、tqdm、warmup_scheduler 的模块 spec 不存在；源码还用 NumPy/Pillow。没有安装这些依赖，也没有试跑官方 train.py。这不影响只需已有 PyTorch 的独立 fixture，但不是官方训练环境复现成功的证明。

**时序与真实 worker 语义**

证据链为 [Dataset.py 72–88](https://github.com/Jackbrocp/AdaAugment/blob/d854af15467285ad8f46500d6fc2da5225fb5bd0/Dataset.py#L72-L88) → [train.py 151–181](https://github.com/Jackbrocp/AdaAugment/blob/d854af15467285ad8f46500d6fc2da5225fb5bd0/train.py#L151-L181) → [A2C.py 67–90](https://github.com/Jackbrocp/AdaAugment/blob/d854af15467285ad8f46500d6fc2da5225fb5bd0/A2C.py#L67-L90)。

1. 读取缓存动作，量化并物化输入。
2. 由该输入计算 adaptive loss/feature，以及 none/full loss。
3. actor 采样新动作，写入缓存。
4. 当前 reward 由第 2 步的 loss 得到，agent.update 却使用第 3 步新动作的 log-prob。

fixture 对每条访问保留 sample_id、visit、proposed_action、cached_action、applied_level、reward_origin、logprob_action、缓存版本、worker ID/PID、物化时刻。固定同一物化输入，将新动作分别改成 0.2 与 0.8，80/80 条记录的当前 reward 完全不变。所有记录中的新 log-prob 动作均与产生当前 reward 的缓存动作不同。这里是确定性依赖/归因检查，不是图像分类性能测试。

| 配置 | 第 2 次访问读到的缓存 | Tensor 在启动后是否共享 | 解释 |
|---|---|---|---|
| workers=0 | 上次新动作，8/8 | 否 | 同一主进程 |
| workers=8，spawn，persistent=False | 上次新动作，8/8 | 是 | PyTorch CPU storage 序列化建立共享存储；下一轮也创建新 worker |
| workers=8，fork，persistent=False | 上次新动作，8/8 | 否 | 本轮主进程写入与旧 worker 快照分离；下一轮重建 worker 继承更新值 |
| workers=8，spawn，persistent=True（敏感性） | 上次新动作，8/8 | 是 | 持久 worker 仍共享 Tensor storage |
| workers=8，fork，persistent=True（敏感性） | 初始缓存，8/8 | 否 | 持久的非共享快照不自动同步 |

因此，“多 worker 必然导致缓存永远更新失败”不成立。官方未启用持久 worker，这个默认设置下下一轮读到了更新。Mac 默认是 spawn；额外显式 fork 是缓存语义敏感性检查，不冒称已复现原作者 Linux/CUDA 环境。Python 普通标量 `is_magnitude` 也不等价于共享 Tensor；若擅自启用 persistent worker，warmup 后该标志的传播还需独立考虑。本轮主要缓存测试从 adaptive 模式开始，没有把该标志切换作为动态实验验证。

由于 8 个样本少于官方 batch128，主测试每轮只有一个批，实际供数的是 worker0；保留 8-worker 启动配置，但不宣称覆盖完整训练队列。另设 8-worker、batch1、同一 iterator 内重复 0..7 的合成预取探针，用“已物化读取”队列确认时点，不靠睡眠猜测：

- prefetch=1：先收到 8 条读取确认，再写缓存；前 8 条用旧缓存、后 8 条用新缓存。
- prefetch=2：先收到 16 条读取确认，再写缓存；16 条均用旧缓存，即使父进程 Tensor 已更新。

这证明共享缓存也不能回溯改变已经物化/排队的输入。探针是有重复索引的压力例；官方每 epoch 不放回 shuffle 没有这种同轮重复，它不改变“下一 epoch 可读更新值”的结论。CPU fixture 关闭 pin_memory，未调用 GPU。

**固定 Beta actor 的原始 score 对照**

设固定状态、α=2、β=3、critic=0，a_new 和 a_old 独立同分布于 Beta(2,3)，r(a)=a。动作以 sample() 产生并 detach，reward 无路径导数。只对 log-prob 求 autograd；没有 actor 网络训练或优化器。

同动作正对照的解析值为：

`E[a · ∇_(α,β) log p(a)] = ∇_(α,β)(α/(α+β)) = (0.12, -0.08)`。

独立旧动作负对照为：

`E[a_old · ∇_(α,β) log p(a_new)] = E[a_old] E[∇ log p(a_new)] = (0,0)`。

| 配对 | α 梯度均值 [99% CI] | β 梯度均值 [99% CI] |
|---|---|---|
| 同动作 reward | 0.120190507 [0.119008638, 0.121372375] | -0.080070236 [-0.081389399, -0.078751073] |
| 独立旧动作 reward | -0.000088741 [-0.001556794, 0.001379311] | 0.000012895 [-0.000923872, 0.000949662] |

固定 N=262,144、seed=20261030、float64，CI 为均值 ±2.5758293035489004×样本标准误；这是逐坐标 99% 常态近似 CI，不是全家族联合置信保证。预定绝对容差 0.006、解析值落入区间、正对照方向均通过。直接 autograd 与逐样本解析 score 均值差低于 1e-12。另验证了官方 softplus+1 参数化的链式法则。

源码最小化的是带负号的 sum；这里展示便于解释的 raw ascent score 均值，`results/score-gradient.json` 同时记录负号及 N 倍的原 loss 缩放。负对照的有限样本 sum 仍不为零。**零期望仅属于固定条件、独立旧动作配对的原始 score term**，不推出 Adam 更新为零、不推出策略恒定，也不推出实际准确率无效。真实训练存在变动状态/参数、critic、weight decay、优化器历史；本轮没有测量这些轨迹。

**置换干预合格性检查**

动作以完整 bundle 保存：原始 magnitude、操作类型、量化 level、物理参数、方向、apply 标志、操作顺序，以及 crop/flip/cutout 的受控随机性占位值。完整 bundle 原子移动，所有索引置换前后完整多重集相同。多操作例保留共享强度和联合顺序；独立打乱其中一个操作的错误实现能被拒绝。这里的“参数变化”不等于“像素变化”，没有执行 PIL 或真实图像增强。

可置换覆盖率定义为：有至少一个其他 donor 能改变该 receiver 完整有效动作参数 tuple 的样本比例；不是承诺每个样本都能同时错配。实际参数改变率按本次置换后的有效 tuple 比较；无强度参数的操作不把无效 level 字段计作变化。

| 案例 | 有效可置换覆盖率 | 固定随机置换的原始 m 改变率 | 实际参数改变率 |
|---|---:|---:|---:|
| 单例 | 0% | 0% | 0% |
| 全部同强度 | 0% | 0% | 0% |
| 不同 m，量化后同 level | 0% | 50% | 0% |
| 不同 level，Posterize bits 相同 | 0% | 75% | 0% |
| Identity 忽略强度 | 0% | 100% | 0% |
| 两操作共享强度 | 100% | 100% | 100% |
| 已知好坏配对效用表 | 100% | 100% | 100% |

最后两例各有 4 个不同有效动作：穷举 24 种置换后，平均实际改变率是 75%，全体都改变的概率只有 9/24=37.5%。本次固定随机抽样恰好全变，不代表随机置换等于强制错配。

效用表预设对角=1、非对角=0：已知好配对均值=1，显式循环坏配对=0，所有随机置换均值=0.25。这个有限集合随机置换会保留固定点，不能与上一节独立旧动作的零期望命题混用。完整实际动作多重集、每个索引置换及效用表保存在 `results/permutations.json`。

**边界与结案**

这是一项固定源码的控制流/依赖审计，以及独立合成 fixture 的验证。它没有复现原作者训练环境，没有测量 benchmark accuracy，没有评价修复的收益；也没有把原始论文结果判错。多操作回归案例是测试工具例，不是原 AdaAugment 的新算法。

AdaAugment 本文已有 fixed/random/linear/sine magnitude 和固定 λ 对照（论文 IV 节消融）；Pro 转交的立项判断还指出 InstaAug 的 without-input/class-specific 与 LSADA2026 的方向反转相关工作。本轮未独立扩展检索后两项，不把转交信息冒充本轮新核验。只有一张 shuffle 表不足以立项，跨 epoch×class 置换也不足以单独识别纯样本匹配因果效应。研究 NO-GO 不因 fixture 通过而改变。

首次正式尝试在 torch_shm_manager 启动处被 OS 沙箱以 Operation not permitted 阻止；保留在 `attempts/attempt01-sandbox/`。它完成了 worker0，未执行多 worker、Monte Carlo 或置换计算。同次有 3 个 A2C 行号锚点相差一行，已根据原文修正 68/75/82→69/76/83；原失败记录保留。随后通过工具自动权限审查，用原种子、原样本数、原容差、原协议重试；成功完成，未通过替换缓存类型等方式绕开共享内存语义。总计算预算仍含失败尝试。该阻塞已解除，**当前授权审计无未完成项**。

`protocol-lock.json` 记录成功运行前的脚本、协议、来源清单哈希；`run-receipt.json` 记录外部 watchdog、退出码和总墙钟。进一步算法修改或训练未执行。Ctrl-A 保持关闭，其产物未改动。
