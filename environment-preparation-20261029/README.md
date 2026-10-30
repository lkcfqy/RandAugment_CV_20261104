# RandAugment 本地 PyTorch／MPS 环境

建立日期：2026-10-29。**环境验收通过；仅证明本次有限诊断可运行，不是科研训练或贡献验证。**

## 环境与入口

项目目录：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104`。

解释器：

```text
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/environment-preparation-20261029/venv/bin/python
```

- CPython **3.14.10**，现有 conda-forge `h0ae1c2c_100_cp314` arm64 构建。
- torch **2.13.0**，torchvision **0.28.0**；NumPy 2.5.4、Pillow 13.0.0。
- 13 个应用与依赖 wheel 从 `https://pypi.org/simple` 解析，实际包来自 `files.pythonhosted.org`；只用二进制 wheel。每个文件都通过 SHA-256 验证后离线安装。
- pip 26.2.2 来自该解释器的 ensurepip；没有安装 torchaudio、CUDA、Colab 或其他无关工具。
- 不必另装名为“PyTorch MPS”的软件包；本次安装的 PyTorch Mac arm64 wheel 已提供可用的 MPS 后端。

系统 `/usr/bin/python3` 仍为 3.9.6；原 Miniforge base 未安装 torch。虚拟环境不使用全局 site-packages，但标准库仍依赖记录中的 base Python；不要删除或原地更换该 base 后假设此环境仍完全相同。

## 证据文件

- [验收报告](ACCEPTANCE_REPORT_ZH.md)：硬件、数值误差、耗时、内存和局限。
- [事先固定的验收协议](ACCEPTANCE_PROTOCOL_ZH.md) 与 [验收脚本](acceptance.py)。
- [原始结果](results/acceptance-001.json) 与 [原始执行日志](logs/acceptance-001.log)。
- [完整哈希锁文件](requirements-macos-arm64-py314.lock)、[已安装版本](installed-freeze.txt)、[包来源清单](sources/wheel-manifest.json)。
- [解释器来源](environment-origin.json)、[硬件及隔离检查](hardware-and-isolation.json)、[官方来源说明](SOURCE_NOTES.md)。

## 使用与复查

后续需要 Python 时直接调用上面的绝对路径，不必改变系统默认解释器。以下仅重复有限环境验收；不会开始研究训练。结果文件必须使用新名称，脚本拒绝覆盖旧结果。

```sh
cd /Users/lkc/Downloads/projects/RandAugment_CV_20261104/environment-preparation-20261029
./venv/bin/python acceptance.py --output results/acceptance-002.json
```

`acceptance.py` 显式禁止 CPU fallback，关闭 MPS fast math，只使用固定合成数据。若未来算子不支持，应单独记录和决定处理方式，不能用静默回退宣称 GPU 验收成功。

## 重建

保留的 `wheelhouse/` 与哈希锁文件可以离线重建同一依赖组合。使用 [rebuild_environment.py](rebuild_environment.py)，它先校验记录中的解释器构建和 wheel 哈希，再向**不存在的新目录**安装；不会自动升级 base Python，也不会运行模型。

```sh
cd /Users/lkc/Downloads/projects/RandAugment_CV_20261104/environment-preparation-20261029
./venv/bin/python rebuild_environment.py --target /tmp/randaugment-mps-rebuild-20261029
```

该例中的目标目录若已存在，须另选新目录。若 base Python 被移走，可通过 `--python` 指定相同记录构建；如果只有不同 Python 构建，应建立新的环境与验收记录，不假称完全复现。

本轮未再次创建第二份环境来做重复安装；原环境的离线安装、哈希核验、pip check 和诊断已完成。重建辅助脚本已作语法检查。操作系统、驱动和硬件仍属于运行条件；依赖锁定不保证任何训练都逐位一致。

## 保存边界

已检查项目内及相关祖先目录的 `AGENTS.md` 与项目 `.agents/skills`，未发现适用的额外指令。

本目录的 `.gitignore` 排除 `venv/` 和 `wheelhouse/`，避免把运行时和安装包误纳入版本控制。源码、协议、清单和结果可供审核。没有 commit/push；没有改动既有实验、封存测试或 SADA/SRA 的判定，没有下载训练数据／预训练权重，也没有启动 Colab。
