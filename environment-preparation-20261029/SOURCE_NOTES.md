# 版本选择与官方依据

核验日期：2026-10-29。选用明确配对且有当前 Python ABI 对应 wheel 的稳定版本；没有跟随移动的 `latest` 标记自动升级。

1. [PyTorch 官方 macOS 安装说明](https://pytorch.org/get-started/locally/)说明使用 pip 安装 torch 与 torchvision。该页不同静态区域的版本标记存在不一致，因此没有单凭页面选择器上的数字决定版本。
2. [官方历史版本安装页](https://pytorch.org/get-started/previous-versions/#v2130)明确列出 macOS 的 `torch==2.13.0 torchvision==0.28.0` 配对。
3. [torchvision 官方兼容表](https://github.com/pytorch/vision#installation)核对 torch 2.13／torchvision 0.28 支持 Python 3.10–3.14。当前 README 已通过实际 Git 提交固定保存，见 `sources/additional-source-manifest.json`；发布标签 README 也保存，但其表格尚未加入本次最新一行，不能单独把它当作完整兼容表。
4. PyPI 官方项目元数据进一步验证：torch 2.13.0 的 Python 要求为 `>=3.10`；torchvision 0.28.0 为 `!=3.14.1,>=3.10`，依赖精确的 `torch==2.13.0`。现有解释器为 3.14.10，且两者均有 `cp314-cp314-macosx_14_0_arm64` wheel。
5. MPS 的同步与内存指标含义核对 [PyTorch v2.13.0 的官方 MPS API 源码](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/mps/__init__.py)，本地保存文件与 SHA-256 见 `sources/official-source-manifest.json`。这里只阅读来源，没有执行下载的来源脚本。

包安装与来源不是凭网页示例推断：完整解析结果见 `sources/pip-resolution.json`，实际 wheel URL／哈希见 `sources/wheel-manifest.json`，已安装包元数据见 `sources/pip-inspect.json`，安装命令、返回码和时长见 `logs/pip-install-runs.json`。

官方文档网页的版本重定向曾无法打开，旧 `.rst` 路径返回 404；这些来源访问问题保留在清单中。随后通过官方版本目录确认当前文件为 `docs/source/notes/mps.md`，成功保存为 `sources/pytorch-mps.md`；其中描述 macOS 14.0+ 和将张量／模型移到 MPS 的流程。版本兼容和验收已由官方配对说明、兼容表、PyPI 元数据、实际安装和运行共同核实，不构成当前执行阻塞。

验收协议与脚本在执行前冻结，SHA-256 见 `sources/protocol-freeze.json`。没有依据通过／失败结果改变误差阈值；原始 `acceptance-001.json` 保留全部检查。
