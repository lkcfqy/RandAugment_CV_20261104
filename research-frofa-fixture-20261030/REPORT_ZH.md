# FroFA 相关机制的数学／合成 fixture：对照实现合格，未进入训练

本轮已完成 **919 项断言检查，全部通过，0 失败**。这不是 919 次独立实验。正式计算只有一次尝试，连续墙钟 **1.20084 秒 / 上限 600 秒**，未重试、未改 seed、未调整容差。只使用 Mac 上现有 Python、NumPy 和 Torch CPU；没有图像／权重下载、GPU／Colab、训练、反向传播或 optimizer.step。

**结论限定在对照构造：**在固定合成 X 和 a 下，C 与 G_C、E* 与 G_E 各自匹配完整条件均值与协方差；仿射读出的期望平方损失相等；预定的三次统计量仍能区分这些同矩分布。这个结果不证明 channelwise clipping 的实际分类收益能被二阶矩解释，也不构成新理论或可发表贡献，不触发数据或模型训练。

## 1. 源码核对及复现边界

只读下载了指定提交 `48424b86061fae71e983d96fc803769185575dbe` 的两份源码，均以 `.py.txt` 保存，未执行或导入：

| 文件 | 只读核对 | SHA-256 |
|---|---|---|
| `models/orco_vit/feature_aug.py` | v3 的 139–159 行；按样本/channel 沿 token 求 min/max，保留 `1e-8`，B×1×C 偏移，clamp 赋回，再还原 | `5680796db88811dfb79b6c6291ff4806114118acc31ea62534adc629ad5b22f9` |
| `models/orco_vit/Network_vit.py` | 221–226 行调用 v3，只增强 patches，拼回原 CLS | `a1e8dfbbf6d011ba1522ad3e64dd64cd72ed5daf5246e49df50dd164da2bf584` |

原始链接：[feature_aug.py](https://github.com/learninginvision/FAOrCo-ViT/blob/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/feature_aug.py)、[Network_vit.py](https://github.com/learninginvision/FAOrCo-ViT/blob/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/Network_vit.py)。本地逐行锚点及 Git blob 哈希见 `results/source-map.json`。

[FroFA v2 附录 A2](https://arxiv.org/html/2403.10519v2#A2) 的代码展示中，clip 调用没有赋回返回值。因此这里只对 FAOrCo 指定版本中的 clipping 机制做独立数学实现，**不称为逐位复现 FroFA，也不据此推断原论文实验有错**。

## 2. 条件矩、结构化算法与独立校验

每次均条件于同一个固定样本的 X、a；不混入跨样本变异。对每个 channel：

    l=min_t x_t, h=max_t x_t, d=h-l+1e-8, u_t=(x_t-l)/d
    Delta_t(s)=d[clip(u_t+s,0,1)-u_t], s~Uniform[-a,a].

C 在该 channel 共用 s；E* 改成每个 token/channel 独立 s，其他操作相同，所以完整逐坐标边际相同。不同 channel 相互独立，完整向量的条件协方差是 channel 分块对角矩阵。不存在随机 apply gate。

将 `-u_t`、`1-u_t` 中落在 `[-a,a]` 内的断点排序分段。每段 Δ 是仿射函数，2 点 Gauss–Legendre 能精确积分一阶、二阶乘积及本轮三次统计量。用归一化节点权重构造

    mu=sum_j w_j g_j
    B[:,j]=sqrt(w_j)(g_j-mu)
    G_C=mu+Bz, z~N(0,I),  BB^T=Sigma_C.

E* 的协方差为 `diag(Sigma_C)`，G_E 用精确逐坐标均值、方差和独立高斯。G_C／G_E **没有再次裁剪或归一化**。

小 T 的独立 oracle 直接在每段积分仿射函数的多项式系数，核对 Gauss–Legendre 的 μ、完整 Σ；再用标准基向量逐列核对隐式 B 与稠密 B。这里没有用 MC 估计协方差构造对照，没有以对角近似代替 G_C。

实际结构化实现对排序节点做两个前缀和。令 `p_j=sqrt(w_j)z_j`、`P(k)=sum_{j<k}p_j`、`R(k)=sum_{j<k}q_j p_j`，并预先找到 token 的两个断点下标 L_t、H_t：

    sum_j p_j g_t(q_j)
      = d[-u_t P(L_t)+R(H_t)-R(L_t)+(1-u_t)(P(J)-P(H_t))].

再减 `mu_t P(J)` 即得到 Bz。每 channel 预处理 **O(T log T)**，每次采样和持久数组 **O(T)**；J≤4T+2。所有 channel 合计 O(CT log T) 预处理、O(CT) 采样／存储。196×768 检查没有创建 T² 协方差或 T×J 因子。完整推导见 `DERIVATION.md`，独立实现见 `controls.py`。

## 3. 检查覆盖及结果

所有配置、seed 和容差在正式启动前写入 `protocol.json`，运行器将协议、推导、实现及测试脚本 SHA-256 封存在 `budget.json`。主要恒等式容差为 atol=5e-12、rtol=5e-11；μ／B 按 d、Σ 按 d² 标度化，避免用普通绝对容差掩盖常量通道的 epsilon 级错误。

| 类别 | 断言数 | 结果 |
|---|---:|---|
| 源码哈希与行号锚点 | 14 | 通过 |
| 9 个小形状 × 5 个固定 a，积分、因子、全部隐式列、边际、端点和重复坐标 | 628 | 通过；最大标度化误差 5.274e-16 |
| 无 epsilon 的 u=(0,1) 解析均值／协方差／秩 | 16 | 通过 |
| 无 clipping 的共享平移 rank-one 负控 | 135 | 通过；最大误差 3.331e-16 |
| 常量 epsilon 的非零均值／协方差 | 4 | 通过 |
| Torch float32／float64、CPU、原输入与 CLS 保留 | 60 | 通过 |
| C／E* 完整边际同分布、非对角不同 | 7 | 通过 |
| 预定三次统计量 | 8 | 通过 |
| 多 channel 完整协方差、仿射平方期望 | 8 | 通过 |
| Householder 方向、不变量、退化／对易和无 clip 负控 | 18 | 通过 |
| 固定 20,000 样本 MC 辅助抽查 | 14 | 通过；不用于对照拟合 |
| 一次 1×196×768 四臂形状／内存检查 | 7 | 通过 |
| **合计** | **919** | **0 失败** |

小形状覆盖 T=1、2、4、5、6、8，含端点、重复、常量 0／3、近常量、非对称及正负特征。a 固定为 0、0.1、0.37、1、1.4。源设定中的常量并不意味着理想扰动恒为零：d=1e-8、u=0，正偏移仍能通过 clipping。本轮没有删除 epsilon 或把常量强行归零。

Torch 默认 dtype 实测为 float32。官方 `torch.empty` 没有显式 dtype，原实现的有限精度随机数／运算不等于连续实数 oracle。本轮匹配的对象是定义好的连续 U 分布和升至 float64 的条件矩。输入量化、输出舍入以及偏移在非零常量 float32 特征上可能被舍去，全部记录在 `results/dtype.json`；未覆盖 fp16 缓存。

### 已知平方期望校验

对固定仿射 W、b、目标 y，记 `r=W(vec X+mu)+b-y`：

    E||W(vec X+Delta)+b-y||² = ||r||² + tr(W Sigma W^T).

两 channel 小例的真实分段乘积积分与高斯对称 cubature 给出：

| 配对 | 原扰动期望 | 对应高斯期望 |
|---|---:|---:|
| C / G_C | 13.867813369661668 | 13.867813369661668 |
| E* / G_E | 13.872417369950067 | 13.872417369950067 |

这只是已知二阶矩恒等式的实现校验。C 与 E* 自身因非对角协方差不同，不要求平方期望相同。不能替换为 CE，也不能宣称有限 SGD 轨迹等价。

### 非高斯联合分布仍然有差异

固定 `u=(0,0.2,0.6,1), d=1, a=0.35`，使用中心化三次统计量：

| 统计量 | C | E* | G_C / G_E |
|---|---:|---:|---:|
| `(Delta_0-mu_0)^3` | 0.00133984375 | 0.00133984375 | 0 |
| `(Delta_0-mu_0)(Delta_1-mu_1)(Delta_2-mu_2)` | 0.001880208333333333 | 0 | 0 |

第二行尤其检查了联合约束。数值来自固定例的解析分段积分，MC 不是结论来源。可以说这些分布并不被 μ、Σ 完全决定；不能据此推断此差异有益于学习。

### 方向干预有效，但存在明确的无效情形

固定 x、a、anchor 后，只做一次投影，构造 Q=I−2vvᵀ，验证 Q1=1、Qμ=μ。只旋转 centered Δ，X 不变。

完整均值、每次完整扰动能量、token 均值和协方差谱均保留；**`||QΣQᵀ−Σ||_F / ||Σ||_F = 0.3025171860`**，超过预定 1e-6 门槛，因此该固定例是非平凡方向干预。最大不变量误差 3.331e-16。

T=2 的补空间退化、常量通道的对易协方差、a=0 均被正确标为无效干预。无 clip 共享平移在相同 Q 下保持不变，是通过的负控。Q 并不保证保留逐坐标边际、边界支持或对不同读出均产生同一种行为；没有对 Q 后输出二次 clipping。

## 4. 一次合成尺寸／内存实测

输入仅为 seed=20261032 的合成 float64 数组，B=1、T=196、C=768、a=0.37。四臂 C、E*、G_C、G_E 均生成一次。

| 项目 | 实测 |
|---|---:|
| 全部 channel 预处理（含 E* 的逐坐标矩） | 0.14106 秒 |
| 一次 G_C 采样 | 0.02592 秒 |
| 其余三臂合计 | 0.02607 秒 |
| 持久数值数组 | 10,841,472 字节 |
| 本次 shape 阶段 tracemalloc 峰值 | 17,727,727 字节 |
| 整个 Python／Torch 进程最高 RSS | 247,545,856 字节 |
| 若保存 768 个稠密 196² float64 协方差，仅矩阵就需要 | 236,027,904 字节 |

实际每 channel 节点数 132–258；持久表示全部是 1-D 数组，E* 的逐坐标积分只用每坐标 6 个局部节点。这是一次形状检查，不是重复性能基准或真实特征缓存吞吐量结论。RSS 含框架导入与前面的小测试，不能与 tracemalloc 峰值混称。原始数组、四臂输出摘要和输出哈希均已保存。

## 5. 预算、归档和限制

正式 supervisor 开始：2026-10-30 14:21:01.673989 UTC；结束：14:21:02.874907 UTC。worker 内墙钟约 0.67123 秒；包含启动／退出的正式连续墙钟 1.20084 秒。运行器保存同一持续 deadline，完成后拒绝额外运行。公式整理、源码读取和收尾报告不属于正式 fixture 数值计算。启动前普通沙箱获取 GitHub 原文曾遇到 DNS 限制，随后同一个只读源码下载获自动权限审查批准并完成；正式计算没有权限失败和恢复重试。

环境：macOS arm64；已有 Python 3.14.10、NumPy 2.5.4、Torch 2.13.0；CPU、单线程、禁用 grad。训练／optimizer 步数为 0。没有 commit、push 或上传。历史项目不做编辑；归档只新增指定目录。

主要文件：

- `protocol.json`：预先固定测试、容差、seed 和预算。
- `DERIVATION.md`、`controls.py`：公式和独立实现。
- `fixture.py`、`run_fixture.py`：检查及持续预算运行器。
- `budget.json`：封存源码哈希、一次尝试和起止时间。
- `worker-attempt-01.log`、`results/checks.jsonl`、`results/checks.json`：全部断言原始记录。
- `results/summary.json`、`results/*.json`、`results/shape-*.npy`／`.npz`：结果与合成原始输出。
- `source/`：指定提交的两份只读源码和下载清单。
- `preservation-before.json`、`preservation-after.json`：历史文件元数据、Git HEAD／index 与工作区状态核对。
- `SHA256SUMS`、`archive-verification.json`：归档清单、逐文件 SHA-256 及非破坏复制验证。

正式归档目标：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-frofa-fixture-20261030`。归档是否完成，以 `archive-verification.json` 的实际验证结果为准。

剩余研究问题仍未回答：在真实冻结特征及非线性读出下，clipping 的表现差异是否超出完整条件 μ、Σ；这种差异有无可复现学习收益、是否超出现有理论和近邻工作。本轮不含这些实验，不能把 fixture 的通过包装成研究贡献。**当前动作到数学对照归档为止。**
