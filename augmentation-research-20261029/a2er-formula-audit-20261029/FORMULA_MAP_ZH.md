# A2ER：主文、伪代码与发布代码对照

主文：Taisuke Kobayashi, *Improvements to dark experience replay and reservoir sampling for better balance between consolidation and plasticity*, Frontiers in Artificial Intelligence 9:1649239，2026-02-19，[DOI](https://doi.org/10.3389/frai.2026.1649239)、[PMC 正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12961966/)。补充 PDF 第 1 页给出作者仓库，第 2 页 Algorithm 1 给出训练步骤。固定作者代码提交：[`f1334df1104724bb3b56f1f353857d92b1141565`](https://github.com/prinlab/A2ER_O2S/tree/f1334df1104724bb3b56f1f353857d92b1141565)，2025-09-25。

以下用 D 表示单条 Delta，Q 表示 Delta_Q，w 表示作者 bar-gamma（即回放优先权，不是修正率 gamma）。

| 环节 | 主文／伪代码 | 固定发布代码与差别 |
|---|---|---|
| 残差 | Eq.8 附近：`D = ||h(x)-z||² / 2` | `derpp/derpp.py:154–155` 是向量维度求和／ord，默认 ord=2。不是按输出维度平均的 MSE |
| Q 的更新 | Eq.9：`Q ← (1-b/N_RS) Q + (b/N_RS) Quantile(D,rho)`；Algorithm 1 初始 Q=0，先更新 Q，再算 eta/gamma | `derpp.py:106–110` 用 `D_stable = D.detach() + eps`；分母是**当前实际条目总数**，不是容量；`torch.quantile` 默认 linear 插值。默认 eps=float32 epsilon，分类示例显式 eps=1e-5 |
| eta | Eq.12–13：`U=Q/rho; eta=1-(clip(D,Q,U)-Q)/(U-Q)` | `derpp.py:110–114` 等价分段式用 D_stable，分母至少 float32 tiny，eta 再夹到 [0,1]；因此 Q=0 的主文未定义式不能直接指控代码 |
| gamma | Eq.14–15：`D_corr=eta D+(1-eta)Q; gamma=1-sqrt(D_corr/D)` | `derpp.py:115–116` 算 `ratio=(eta D_stable+(1-eta)Q)/D_stable`，`gamma=1-ratio**(1/ord)`。这些量没有到 h 的梯度 |
| 当步蒸馏损失 | Algorithm 1 line20：`mean((1-gamma)² D)` | `derpp.py:119` 用 **ratio × 原始 D**，没有把 eps 加进损失。gamma 从 D_stable 得到，因此原始 D 修正后不严格等于主文直接混合式 |
| alpha 更新 | Algorithm 1 line21：`-alpha_real × mean(eta × (sg(D)-Q))` | `derpp.py:114` 用 `D_stable-Q`，仍是实数参数空间的线性损失；正值权重通过 **squareplus**，非主文 softplus |
| beta 更新 | Algorithm 1 line10：`-beta_real × (sg(L_RS)-sg(L_FIFO))` | `derpp.py:87–98` 同样在实数空间线性驱动；映射是 **squmoid**，非主文 sigmoid。参数映射对模型损失使用 `.data`，不是再乘映射导数的普通重参数化 SGD |
| 写回 target | Eq.17：`z ← (1-gamma)z + gamma h`；在模型优化之前 | `derpp.py:124–127` 把 h/gamma 脱离图，交给 `ReplayBuffer.update`。原始 D 已计算并保留；不重新用已写回的 target 计算本步 loss |
| 回放权重 | Eq.16：`w ← (1-lambda)w + lambda(1-gamma)`，再以 w/sum(w) 抽样 | `derpp.py:129–133` 在 target 更新之后更新访问过的条目；单 key 等价；多个 key 时使用各 `(1-gamma)` 的**乘积**。`replay_buffer.py:128–131` 是归一化权重**无放回**抽样。未访问条目不更新，batch 纳入概率不能简单写成 b×p |
| 初始化与储存 | Supplement Algorithm 2 line11：z=h(x)、w=1 后进入 RS | `derpp.py:158–176` 从 FIFO 溢出后才调用 der_func 并储存 target；并非必然在首次输入时存 logit。`replay_buffer.py:42–60` 写入 float32 数据 |

## 实际一步次序

1. 用旧优先权抽取 RS 条目。默认 `replay='disjoint'` 一次抽两个 batch 再切分：蒸馏与监督回放不重叠，见 `derpp.py:75–85`、`replay_buffer.py:63–77`。
2. 算 FIFO／监督回放损失，组成 beta_real 更新项；计算蒸馏 batch 的当前 h 和旧 target 对应 D。
3. 用 detached D+eps 更新 Q；用新 Q 算 eta、ratio、gamma；加入 alpha_real 更新项及 ratio×D 的模型损失。
4. 用 detached h/gamma **写回访问过的历史 target**，随后更新访问过的优先权。数学上单 key 的这两次状态写入互不依赖；都使用本步已经算出的 gamma。
5. 返回联合损失，由外层 backward／optimizer 更新模型及实数 alpha/beta 参数。数值审计只做一次向量导数核对，未执行 optimizer.step。
6. 下一步以更新后的优先权采样。block 是平滑降权；与 O2S 跨缓冲区迁移时的 omission 丢弃机制不同。

## 必须拆开的路径

`D → Q → eta → alpha 实数参数更新`

`D,Q,eta → gamma/ratio → 当步蒸馏损失缩放`

`gamma → 历史 target 写回 → 未来残差`

`gamma → priority EMA → 未来样本身份／概率`

另有 `L_RS − L_FIFO → beta 实数参数更新`。因此虽然科学因素可按 alpha/beta 自适应、target correction、replay sampling 三组讨论，直接替换 residual 实际会同时改变多条路径，还改变当步蒸馏强度。不能称为“仅修改 gate”。固定整个权重向量再改分配身份的候选必须另冻协议，不由本审计自动授权。

## 与原 DER++ 的差别

固定 Mammoth [`e75a491c69fd729edeb01431afb753d9157d9a81`](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/derpp.py)：`models/derpp.py:32–58` 用当前增强输入算 logits，储存 `not_aug_inputs` 和已有 logits；未来 `Buffer.get_data(..., transform=...)` 对样本再次变换，见 `utils/buffer.py:526–547`。这种跨视图蒸馏本身是设计选择，不能当 bug。

Mammoth 损失为 `L_current + beta_M L_replay + alpha_M mean_all_elements(error²)`；A2ER 主文为 `(1-beta_A)L_FIFO + beta_A L_RS + alpha_A mean_batch(sum_features(error²)/2)`。忽略 batch／FIFO 差别，仅代数归一化可令 `beta_A=beta_M/(1+beta_M)`、`alpha_A=2 alpha_M/[d(1+beta_M)]`。共同缩放损失还会改变固定学习率的效果。因此不能直接移植相同 alpha/beta 数值后宣称忠实对照。

## 尚未核实的实现范围

- 作者仓库可读取、有核心代码。并不是“没有官方代码”。但是 `model/classification.py:9` 导入的 `derpp.derpp_ablation` 不在该固定完整 Git 树中，且没有发布 MNIST/CIFAR benchmark driver。直接复现完整论文的可执行入口仍不完整。
- 主文图像 benchmark 段落说训练数据采用 Random Erasing；公开核心缓冲区／分类模型没有 augmentation 调用，且上游驱动缺失。因此不能确定 Erasing 在入库前、重复回放时、还是两者均有。不得把对 Mammoth 的代码观察转嫁成 A2ER 的事实。
- 本审计只运行原始核心 AST，排除持久化方法和训练入口，没有安装 pandas 或作者其他依赖、没有补造 `derpp_ablation.py`。
