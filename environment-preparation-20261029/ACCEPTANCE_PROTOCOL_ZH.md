# 环境与硬件验收协议

日期：2026-10-29。此协议在执行诊断前固定；用途仅是 PyTorch／torchvision 安装和 MPS 可用性验收。

## 范围

- 使用项目内新建 venv，基于现有 conda-forge CPython 3.14.10 arm64；不改系统 Python 或既有 base 环境。
- 固定 torch 2.13.0、torchvision 0.28.0 及解析出的全部依赖；只使用 PyPI 二进制 wheel，保存 SHA-256 锁文件与安装日志。
- 只生成合成张量；不下载数据集或预训练权重，不启动科研训练、不进行优化器参数更新、不使用 Colab。
- 保留既有 SADA、SRA 和所有其他研究记录，不改变其判定。

## 诊断与事先阈值

| 检查 | 固定范围 | 判定 |
|---|---|---|
| 安装 | pip check、版本对、模块路径、CPU torchvision NMS | 无依赖冲突；包来自新 venv；NMS 输出 [0,2] |
| MPS | is_built / is_available | 两者都为 true；禁止静默 CPU fallback |
| 精确张量运算 | 16×16 可精确表示的 float32 值，乘 2 加 1 | 与 CPU 结果逐元素相等 |
| 矩阵乘 | 固定 CPU 随机数，64×64 float32 | 与 CPU float64 对照：atol=2e-5，rtol=2e-4 |
| 小 CNN 前向 | 8×3×32×32 合成输入，2 个卷积层、池化、线性层 | logits：atol=2e-5，rtol=2e-4；交叉熵 loss：atol=1e-5，rtol=1e-5 |
| 小 CNN 反向 | CPU/MPS 使用相同初始权重、输入、标签；无优化器 | 参数梯度与输入梯度：atol=2e-5，rtol=2e-3；有限值；权重执行前后不变 |

同时检查 CPU float32 对 CPU float64、MPS float32 对 CPU float64、MPS float32 对 CPU float32。判定逐元素满足 `abs(actual-reference) <= atol + rtol*abs(reference)`。保留全部通过／失败结果；不因结果而放宽阈值。

CPU 线程数 4、互操作线程数 1；种子 20261029；关闭 MPS fast math，设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`。模型无 BatchNorm、Dropout，采用 eval 模式但启用 autograd。每个 float32 设备执行 1 次首次前向／反向加 3 次重复；CPU float64 参考只执行 1 次；每次重置梯度，始终无 optimizer.step。

## 耗时与内存的含义

记录模型准备／传输时间与同步后的前向／反向时间；首次运行包含首次算子准备开销，后续 3 次用于确认可重复执行，不用于预测真实任务速度。只报告观测值，不声称已完成吞吐量基准。

记录 Darwin 进程峰值 RSS（字节）、当前 RSS，以及 MPS 当前张量分配、驱动分配、建议工作集大小的若干快照。快照不等于 GPU 峰值；统一内存数值有重叠，不能把 RSS 与 MPS 字节数相加。空闲内存和速度会受其他活动影响。

## 停止条件

任一依赖冲突、MPS 不可用、算子异常或数值阈值失败均记为验收未通过；诊断具体原因后才进行必要的修复与新结果记录。不得以开启 CPU fallback 冒充 MPS 通过。通过也只覆盖这些算子、形状和精度，不能保证后续所有模型、算子或科研结果可复现。
