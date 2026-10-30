# PyTorch／MPS 环境验收：通过

日期：2026-10-29。**本次完成环境和硬件验收，未启动科研训练；不能据此声称任何候选方法有效或具有原创贡献。**

## 环境

| 项目 | 已验证值 |
|---|---|
| 芯片／CPU | Apple M5 Pro；18 个逻辑 CPU |
| 系统内存 | 64 GiB（68,719,476,736 字节） |
| 系统 | macOS 27.2，arm64 |
| Python | 3.14.10，conda-forge `h0ae1c2c_100_cp314` |
| PyTorch／torchvision | 2.13.0／0.28.0 |
| NumPy／Pillow | 2.5.4／13.0.0 |
| MPS | `is_built=True`；`is_available=True` |
| 回退／精度 | `PYTORCH_ENABLE_MPS_FALLBACK=0`；`PYTORCH_MPS_FAST_MATH=0` |
| 隔离 | 包路径均位于新 venv；系统 Python 仍为 3.9.6；原 base 中仍无 torch |

解释器路径：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104/environment-preparation-20261029/venv/bin/python`。

安装来自 PyPI 的 13 个二进制 wheel，合计 133,821,729 字节；依赖全部固定版本和 SHA-256，安装前已核验文件，实际安装使用本地 wheelhouse。`pip check` 返回 `No broken requirements found.`。磁盘分配快照：venv 765,832 KiB，wheelhouse 130,704 KiB；两者均由本目录 `.gitignore` 排除。

## 数值与执行检查

12 项汇总检查全部通过，详见 [原始 JSON](results/acceptance-001.json)。CPU torchvision NMS 返回预期索引 [0,2]，证实其 CPU 扩展可用。MPS 测试期间禁止静默回退。

小 CNN 含 1,562 个参数，输入形状为 8×3×32×32；固定种子、相同初始权重和输入，使用 float32，另有 CPU float64 对照。模型无 BatchNorm／Dropout，所有前向与反向结果有限。执行后逐项确认权重未变，**optimizer steps = 0**。

| 比较 | logits 最大绝对误差 | loss 绝对误差 | 参数梯度最大绝对误差 | 输入梯度最大绝对误差 |
|---|---:|---:|---:|---:|
| CPU float32 vs CPU float64 | 2.381e-8 | 1.587e-7 | 1.353e-8 | 4.865e-12 |
| MPS float32 vs CPU float64 | 1.747e-8 | 1.587e-7 | 1.577e-8 | 8.037e-12 |
| MPS float32 vs CPU float32 | 2.980e-8 | 0 | 1.490e-8 | 9.095e-12 |

上述误差均通过执行前固定的逐元素绝对／相对阈值，不是要求逐位一致。16×16 精确算术在两端误差均为 0；64×64 矩阵乘对 CPU float64 的最大绝对误差均为 1.163e-7。完整阈值及方法见 [验收协议](ACCEPTANCE_PROTOCOL_ZH.md)。

## 耗时

| 小 CNN 前向＋反向 | CPU float32 | MPS float32 |
|---|---:|---:|
| 准备模型／输入 | 0.347 ms | 2.149 ms |
| 首次执行 | 1.368 ms | 1,654.859 ms |
| 后续第 1 次 | 0.892 ms | 2.277 ms |
| 后续第 2 次 | 0.772 ms | 2.149 ms |
| 后续第 3 次 | 0.655 ms | 1.531 ms |
| 后续 3 次中位数 | 0.772 ms | 2.149 ms |

MPS 计时前后均同步。这个极小模型上 CPU 更快，MPS 首次执行的开销明显；这些短诊断不能预测较大训练任务的速度，也不构成完整性能基准。没有为比较性能增加训练循环。

整个验收脚本墙钟时间 **14.717 秒**（外层启动与收集为 15.378 秒，包含导入及初始化）。安装阶段另计：创建 venv 1.444 秒，解析 1.577 秒，下载 109.225 秒，离线安装 10.427 秒，pip check 0.137 秒。工具等待和文档整理不计作模型运算时间。

## 内存

- 整个验收进程峰值 RSS：**437.875 MiB**（459,145,216 字节）。
- 检查点中观测到的最大 MPS 张量分配：**0.628 MiB**；最大 MPS 驱动分配：**10.891 MiB**。
- Metal 报告的建议最大工作集约 51.840 GiB；这不是当前空闲 GPU 内存、硬性上限或可直接分配的保证。

MPS 数字是离散快照，不能称为全程峰值。Mac 使用统一内存；进程 RSS、张量分配和驱动分配不可相加。没有修改 MPS 内存水位或其他系统设置。

## 可复现与下一步边界

[锁文件](requirements-macos-arm64-py314.lock)、[wheel 清单](sources/wheel-manifest.json)、[解释器来源](environment-origin.json)、[执行前协议哈希](sources/protocol-freeze.json)、命令日志及原始 JSON 均已保存。参照 [README](README.md) 可使用同一解释器与本地 wheelhouse 向新目录重建；未为重复验证而再安装第二份环境。

当前没有执行阻塞。通过只覆盖本协议的少量 float32 算子、模型和输入形状；后续若选定真实研究协议，应针对其算子、精度和资源需求另行做有界试跑。**尚未启动训练、下载大型数据／预训练权重或使用 Colab。** SADA、SRA 等既有否决与封存证据均保留。
