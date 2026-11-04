# RandAugment 本地准备与 MPS 验收

完成日期：2026-10-30。本次仅创建新开发环境、验证一个小型 MPS 张量运算，并整理课堂工作稿。没有运行历史训练、模型/数据下载、Colab 认证或远端推送。

## 本地项目与保存情况

- 路径：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104`。
- 分支：`main`；HEAD：`8493460a088daab425c52732f7a8d1f9c955ee4f`。
- origin：`https://github.com/lkcfqy/RandAugment_CV_20261104.git`。
- 完整克隆，非 shallow。目标目录创建前不存在；没有覆盖用户旧副本。
- INCLUDED 备份清单的 469 个文件大小和 SHA-256 校验通过；克隆后的 2,066 个 tracked 文件已保存新 SHA-256 清单，准备结束逐一复核均未改变。
- 历史课程材料、停止说明、协议、脚本、旧失败和冻结证据保持原样。所有新材料尚未 commit/push。
- 克隆时临时禁用钩子并跳过 LFS smudge；克隆后已撤销本次新增的本地 `core.hooksPath` 覆盖，不更改全局 Git 配置。

## 解释器与官方依赖

- 新解释器：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104/.venv/bin/python`，CPython 3.14.10，arm64。
- 基于现有 `/Users/lkc/miniforge3/bin/python` 创建标准 venv；未启用全局 site-packages。
- Python 构建 `h0ae1c2c_100_cp314` 与历史记录一致；没有改动系统或 Miniforge base。
- 使用历史 `requirements-macos-arm64-py314.lock`：torch 2.13.0 / torchvision 0.28.0 / NumPy 2.5.4 / Pillow 13.0.0 等 13 个 wheel。pip 为 ensurepip 提供的 26.2.2。
- 13 个 wheel 的文件名、URL、字节数、SHA-256 均与官方 PyPI 元数据及历史台账一致。PyTorch 和 torchvision 均为 `cp314-cp314-macosx_14_0_arm64` wheel，适配当前 Python 与 macOS 27.2。
- 从 `https://pypi.org/simple` 解析，只下载二进制 wheel。全部下载后再核对哈希，然后从本地 wheelhouse 离线、强制哈希安装。`pip check` 通过。
- 安装耗时约 25.47 秒；没有因当前存在更新版本而升级历史锁。

版本与来源证据：`installed-freeze.txt`、`requirements-macos-arm64-py314.lock`、`official-wheel-verification.json`、`pip-install-report.json`、`installation-steps.json` 和 `logs/pip-*.log`。`.venv/` 与本目录 `wheelhouse/` 已被 Git 忽略。

## 本次 MPS 回执

| 项目 | 实测值 |
| --- | --- |
| 芯片 / GPU / 内存 | Apple M5 Pro / 20 核 / 64 GiB |
| macOS | 27.2（26B123），arm64 |
| torch / torchvision | 2.13.0 / 0.28.0 |
| `torch.backends.mps.is_built()` | True |
| `torch.backends.mps.is_available()` | True |
| 输入 | 固定合成 16×16 float32 张量 |
| 运算 | `x @ x.T + 0.25` |
| 实际结果设备 | `mps:0` |
| GPU 同步 | 运算前后均调用 `torch.mps.synchronize()` |
| CPU fallback / fast math | 均关闭 |
| 与 CPU 比较 | allclose=True，atol=1e-5，rtol=1e-5 |
| 最大绝对误差 | 0.0 |
| 有限数检查 | True |
| 同步运算耗时 | 0.181074 秒 |
| 包含导入的进程总耗时 | 8.630 秒（限时 45 秒） |
| 结果 | passed，退出 0 |

`probe_mps.py` 是本次新建的诊断脚本，只导入官方 torch/torchvision；以 `-I -B` 从 `/tmp` 工作目录运行。没有导入项目历史入口，没有反向传播或训练。脚本和执行器拒绝覆盖已有回执。

这只证明当前解释器中的小型前向运算确实可在 MPS 上运行，耗时不是训练性能基准。尚未验证其他模型、算子、反向传播、长时间稳定性或科研收益。MPS 为 PyTorch 内置后端，不另装“pytorch-mps”。

## 剩余限制与课堂准备

备份排除清单的 614 个文件在新克隆中均缺失，包括数据、检查点、部分固定第三方源码及全文；旧运行环境目录也未恢复。锁定依赖和 Python 构建匹配不代表历史实验复现。若日后重建某个实验，仍需按其专属协议核对全部资产与条件。

新的课堂工作稿位于 `../development-preparation-20261030/`：`PRESENTATION_ALIGNMENT_ZH.md` 与 `NEXT_STEPS_ZH.md`。当前为 7 页、8:30 的内容预算，非最终 PPT 或实测演讲时长。研究候选 ActiveAugment 个体匹配未过新颖性/机制门，本次没有开展相关实验。

Colab CLI 独立环境仍为 0.7.5，未被项目安装修改；token 文件仍不存在。OAuth 持久授权尚待用户明确同意，没有运行时分配或计算额度使用。

## 复核入口

- [MPS 原始回执](mps-receipt-001.json)
- [45 秒限时进程回执](mps-process-receipt-001.json)
- [最终保存检查](preservation-after.json)
- [安装前仓库记录](repository-before.json)
- [官方 PyTorch 安装说明](https://pytorch.org/get-started/locally/)
- [官方 torch 2.13.0 元数据](https://pypi.org/pypi/torch/2.13.0/json)
- [官方 torchvision 0.28.0 元数据](https://pypi.org/pypi/torchvision/0.28.0/json)

`prepare_environment.py` 是本次有固定路径和存在性保护的执行记录，目标已存在时会停止；不要把它当作覆盖当前环境的一键升级入口。
