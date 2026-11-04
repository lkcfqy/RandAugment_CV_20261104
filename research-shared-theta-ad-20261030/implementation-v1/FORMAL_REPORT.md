# Shared-theta AD：完整冻结运行终局

**未达到预定继续门：本候选在本冻结设置下不追加计算，不以 selector 或其他 shuffle 替换 primary。**

全部3个seed ×59个配置已完成，共177组方法结果，51个外部baseline均保留；每seed为64校准项+962测试项，合计3078个查询。正式运行前后冻结 SHA256 未改变：`db58e313f00018241490e099a81af2aec6bf49e9501416a8f95c5190c418037d`。本报告在全部运行完成和独立完整性复核通过后生成。

## 预定继续门

| 条件 | 实际结果 | 冻结要求 | 判定 |
|---|---:|---:|---|
| 等权域 AUROC 相对最高平均 baseline | -5.0335 pp | ≥ +2.00 pp | 未通过 |
| illumination 相对该域最高平均 baseline | -5.7800 pp | ≥ +3.00 pp | 未通过 |
| same 相对 identity/pool9 较强者 | -0.0751 pp | ≥ −1.00 pp | 通过 |
| 三 seed 相对同一全局 eq 最强 baseline | -6.1764 pp / -4.4134 pp / -4.5106 pp | 每个都 > 0 | 未通过 |
| primary − shuffle9001 等权域 AUROC | -0.5778 pp | ≥ +1.00 pp | 未通过 |

全局最高平均等权域 baseline 是 `projection/Sc/r32/identity/post_sq`；illumination 域最高平均 baseline 是 `projection/Sc/r32/identity/post_sq`。两者均在完整51方法集合中计算。oracle 包络仅用作本轮投资继续门，不代表可部署的超参选择、显著性证据或独立泛化证明。

## AUROC（跨seed平均）

| 方法 | same | illumination | 两域等权 |
|---|---:|---:|---:|
| `primary` | 65.92% | 67.77% | 66.85% |
| `identity` | 65.99% | 67.24% | 66.62% |
| `pool9` | 65.86% | 68.23% | 67.04% |
| `robust_color` | 70.54% | 71.64% | 71.09% |
| `projection/Sc/r32/identity/post_sq` | 70.21% | 73.55% | 71.88% |

全部59配置、各seed阈值以及两域 AUROC/TPR/FPR 见 `reports/all-methods-by-seed.csv`；全部方法均值见 `reports/all-method-means.csv`。没有省略任何 baseline。校准阈值使用每方法/seed的第62个升序分数，只有严格大于阈值才判异常；两域共用该阈值，未宣称目标域FPR保证。

## Primary 的三个seed

| seed | same AUROC | illumination AUROC | 等权域 AUROC | 相对同一全局eq baseline |
|---|---:|---:|---:|---:|
| 11 | 68.79% | 65.82% | 67.31% | -6.1764 pp |
| 22 | 64.66% | 66.77% | 65.72% | -4.4134 pp |
| 33 | 64.31% | 70.73% | 67.52% | -4.5106 pp |

## 机制诊断（不代替主门）

| shuffle seed | 平均等权域AUROC | primary − shuffle | 用途 |
|---|---:|---:|---|
| 9001 | 67.42% | -0.5778 pp | 唯一 primary gate |
| 9002 | 67.22% | -0.3696 pp | 仅映射敏感性 |
| 9003 | 67.12% | -0.2704 pp | 仅映射敏感性 |
| 9004 | 67.37% | -0.5196 pp | 仅映射敏感性 |
| 9005 | 67.03% | -0.1854 pp | 仅映射敏感性 |

| selector | same | illumination | 等权域 |
|---|---:|---:|---:|
| `primary` | 65.92% | 67.77% | 66.85% |
| `selector/all256` | 66.16% | 68.29% | 67.23% |
| `selector/min_top13` | 66.30% | 68.39% | 67.34% |

primary 的 identity θ 选择率如下；其余四个shuffle、primary shuffle与两个selector的全部domain×N/A数据见 `reports/identity-selection-rates.csv`。N/A分别为正常/异常。

| 域/标签 | seed11 | seed22 | seed33 |
|---|---:|---:|---:|
| same/N | 13.91% | 10.00% | 18.26% |
| same/A | 7.63% | 3.70% | 9.80% |
| illumination/N | 5.33% | 9.33% | 4.00% |
| illumination/A | 12.63% | 10.61% | 8.59% |

## 完整性与独立核验

- 3seed、59方法、51外部baseline、所有1026查询ID/顺序/标签/域数量及finite检查全部通过。
- **4113**个缓存/状态文件的大小与SHA256全部复核，总负载 **4,293,475,110字节**；包含3078个距离缓存与1026个原始query feature文件。
- 全部查询缓存shape、索引范围和finite检查通过；96个identity率单元由缓存重新计数。
- 预先固定30个query/seed对，独立NumPy复算 **1770** 个分数，最大绝对差 **1.19e-07**，未事后放宽容差。
- 所有阈值和AUROC/TPR/FPR由独立sklearn/NumPy实现复核，最大指标差 **2.22e-16**；五个继续门与终局判定一致。
- 该复核不重新运行模型或NN；完整证据见 `reports/formal-independent-verification.json`。旧4.46.3官方CPU环境及MPS数值检查证据仍保留。

## 成本、释放与限制

正式运行实际 **10.19分钟**；完成时距首次GPU启用 **30.34分钟**（含等待）。绝对截止 **2026-10-31 00:28:54 UTC** 未重置。生成本报告时，包括已存资产、参考环境和全部缓存的实际分配磁盘 **7.035 GiB**，低于8 GiB。完整流程没有新增Colab、没有改数据/科学配置、没有提交或推送；MPS清理已返回，运行进程已退出，退出证据另见 `reports/process-exit-verification.json`。

原始文件全部保留。训练图像的精确重复已按冻结规则合并为520个簇，但官方没有物理叶片实体或采集批次ID；这里仅评价官方图像划分下的采集条件变化，不把图像簇当作独立实体，也不提供伪实体CI。样本精确无重复不能证明实体独立。
