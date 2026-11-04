> 后续状态：完整正式运行及独立核验已完成，终局见 [FORMAL_REPORT.md](FORMAL_REPORT.md)。本文件保留开跑前交接记录。

# 实现与无效用预检交接

本轮完成全部实现、CPU 合成验证、清单冻结及限定的 MPS 吞吐预检。**没有正式测试计分，没有真实 AUROC/TPR 或 θ 胜出分布输出，没有分配 Colab。** 正式入口仍要求父线程确认，且需要独立静态审查。

## 冻结与范围

- 冻结文件：`frozen-manifest.json`，SHA256 `db58e313f00018241490e099a81af2aec6bf49e9501416a8f95c5190c418037d`。
- 包含 15 个实现/检查脚本的 SHA256、模型 revision/SHA、完整 split manifest、59 个方法名、51 个外部 baseline 名、依赖版本以及固定 shuffle 映射。
- 521 个训练路径先合并为 520 个 exactRGB 节点。固定 pHash64/Hamming≤4 + 灰度256/SSIM≥.995 连接分量后仍为 520 簇。原始文件保留；64 个 calibration 簇及 seed11/22/33 各64个参考簇已经冻结，cal 与 refs 全簇隔离；不同 ref seeds 可重叠。
- `split-manifest.json` 保存 canonical、成员、cluster_id、近重复边和跨 split 报告。无物理叶片身份假设，不生成实体独立 CI。
- 九 θ 的顺序、浮点线性光变换、整图 bicubic 224、最终 LayerNorm 后 L2、全部检索、PCA、颜色规范化及 BoW 参数见 `frozen-manifest.json` 的 `protocol`。

## 版本对照是真实官方环境运行

实际实现环境：Python 3.14.10 / torch 2.13.0 / Transformers **4.57.1**。旧版本依赖没有 Python 3.14 的可用 tokenizer wheel，因此另建 `.reference-446`，使用已有系统 Python **3.9.6**、官方 torch **2.5.1**、Transformers **4.46.3** wheel。没有降级或改写实现环境；依赖均来自标准 PyPI，版本和 SHA256 见 `requirements-reference446.lock`、安装报告。

同一份已核验官方权重，以本地 safetensors、strict 内置 Dinov2Model、CPU FP32 eager、eval 运行。禁止 `torch.load`，不执行下载源码、不用 TorchHub 或 remote code。协议 revision `150c8e7bb7cef2d30ec31b13a517af14840ee3f7` 与已取得资产 revision 的权重 SHA256 和 config 一致，未重新下载权重。

在运行两版本输出之前，`reports/encoder-comparison-lock.json` 已锁定三个 224 合成输入和容差，归一化 patch 要求 maxabs≤1e-4、meanabs≤1e-5。实测：

| 检查 | 最大绝对差 | 平均绝对差 |
|---|---:|---:|
| native position interpolation | 0 | 0 |
| 第一层 eager attention | 2.09e-6 | 3.70e-9 |
| 最终 LayerNorm 后输出 | 2.88e-4 | 5.75e-6 |
| CLS 去除后归一化 patch | **4.86e-6** | **1.07e-7** |

所有 shape/finite 检查通过，各版本内部的最终 LayerNorm 核对最大差为0。完整证据及安装源文件哈希见 `reports/encoder-cross-version.json`；两个进程的实际输出 `.npz` 留存。这是锁定输入和所选 FP32 eager 路径的容差验证，不宣称不同后端/模式的普遍逐位一致。先前旧公式对照仅作为补充证据，不替代此独立运行。

## 方法与完整性检查

实现 **59** 个配置，其中 **51** 个外部 baseline：identity/pool9、robust_color、两种 reference 筛选器各 K=1/4/10×selected-view/pool9，以及 36 种 projection 配置。两个 selector 和五个 shuffle 只作机制报告，不进入外部 oracle 包络；primary shuffle 始终9001。正式完成时会按 domain×N/A 输出 identity 选择率，并报告其余四个 shuffle 的映射敏感性，不能替换主门。

11 个 CPU 合成测试通过，覆盖：

- θ identity/浮点变换、shuffle 容量/union 与 pooled 严格不变、全选 identity 时一致；`Spool ≤ Sminθtop13 ≤ Slower204`。
- 缓存与显式 brute-force、tie 顺序；projection 的对称预归一化、新全 bank NN、原空间 winner 后投影、rank 嵌套和中心化拟合。包含投影后真正改变 NN winner 的反例。
- BoW 的5×5网格与 KL 方向、36/12配置数量、robust_color IQR floor、阈值第62顺序统计量、AUROC ties。
- 审查发现的 summarize 缺口已修复：只接受恰好 seeds11/22/33、每 seed 精确59个方法和51个baseline、完整1026条查询、固定 IDs/顺序/路径/pixelhash/标签、64 normal calibration 和指定962 test域计数，所有分数须 finite。
- 缺 seed、缺 baseline、额外方法、额外/重复ID、重复路径、错误标签/域计数、短向量和 NaN 均必须失败；完整合法的合成输入也实际调用 summarize 并通过。未授权 formal 入口必须拒绝。

关键审查入口：`stheta/metrics.py`、`stheta/methods.py`、`formal_runner.py`、`stheta/runtime.py`。测试日志：`reports/synthetic-tests.log`。

## MPS 吞吐与成本

仅使用 seed11 的固定64个 source-normal refs，以及固定 source-normal calibration 图 `train/good/view/IMG_7130.png`。真实 bank 为 `[64,9,256,384]`；robust_color 另跑64个 reference 和1个 query 的新 encoder，未复用原特征。原空间 NN、全部预投影新 NN、后投影、BoW、两种参考选择、五个 shuffle、两个 selector、颜色 baseline 和持久化/哈希均测量。

先做 CPU/MPS 合成数值检查：缓存距离 maxabs 2.38e-7、投影 maxabs 4.77e-7；归一化 encoder patch maxabs 9.16e-6、meanabs 1.91e-7，均通过预定容差。随后吞吐只留时长、缓存哈希和配置数，不留 source-normal 标量分数或 θ 选择统计。

- 首次 GPU：**2026-10-30 22:28:54 UTC**。
- 本次完整预检窗口：**8.83秒**，已释放 MPS，低于5分钟。
- 三次真实 bank 查询，原距离+全部预投影为 0.240/0.124/0.113秒；selector/BoW/后投影为 0.163/0.016/0.016秒。外推使用最大值，保留冷启动开销。
- 按 **1026 query ×3 seeds**、最多 **6024 forwards**，包括 CPU 数据处理、完整 setup、持久化/哈希/磁盘检查，剩余全流程基础估计2631.82秒；再加 **20%**，并计入已耗，测量时总估计 **3166.94秒（52.78分钟）**。
- 包含已有资产、独立参考环境和所有未来缓存的磁盘峰值预测 **7,675,472,678字节（约7.15 GiB）**，低于8 GiB。

详见 `reports/throughput-preflight.json`。这是吞吐与容量估计，不是研究效果。没有缩 test、删 baseline 或改变科学参数。

## 正式入口和剩余事项

`formal_runner.py` 作为 CLI 直接退出；库函数 `run_formal` 默认拒绝。获明确确认后还会校验代码/数据/shuffle冻结哈希、完整预检和磁盘门，并重新计算**自第一次GPU起的绝对剩余预算**，不会重置计时。绝对总截止为 **2026-10-31 00:28:54 UTC**；静态审查、CPU 等待和后续运行都计入该窗口。如果等待导致完整预测超限，会停止，不能通过删对照或重新开时钟绕过。

缓存完成写出/flush/fsync 后留 SHA256，query ID 随缓存登记；所有seed完成前不产出最终效用结论。已有 formal 目录会阻止未经核验的覆盖/复用；当前没有正式输出目录。正常退出或异常会清理MPS，另有绝对wall watchdog。

本轮未提交/推送，未修改其他实验。**实现和预检无当前阻塞；正式计分尚未执行，等待父线程独立静态审查与下一消息确认。** `reports/handoff-verification.json` 给出最新墙钟预算和冻结验证结果。
