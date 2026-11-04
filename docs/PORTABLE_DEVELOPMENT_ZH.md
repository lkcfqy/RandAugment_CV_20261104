# 换机继续开发

本开发版恢复研究源码、配置、测试、说明和选定实验记录，提供独立的合成开发验证入口。它不是全部历史实验资源的镜像。四个指定的大文件不在此版本中；相应历史完整复核仍受资源缺失约束。

## 最小开发环境与验证

从仓库根目录，在 Python 3.12 或更新版本的独立环境中操作。固定版本的 NumPy 2.5.4 要求 Python 至少为 3.12。本次实际验证为 macOS arm64、Python 3.12.16、NumPy 2.5.4；未声称验证 Windows/Linux 或原模型环境。下面将环境创建在仓库目录外：

```sh
python3 -m venv ../RandAugment-dev-venv
../RandAugment-dev-venv/bin/python -m pip install --isolated --index-url https://pypi.org/simple -r portable_dev/requirements-dev.txt
../RandAugment-dev-venv/bin/python -B -m portable_dev doctor
../RandAugment-dev-venv/bin/python -B -m portable_dev test
```

`doctor` 只检查开发依赖、源文件和若干外部资产是否存在。`test` 只执行明确列出的 CPU 合成测试，每组使用独立子进程，不自动发现整个研究目录的测试，也不读取数据集、构建真实模型、下载资源、训练或启动云任务。

本轮测试覆盖：8 个路径配置测试、4 个 CIFAR 合成持久化/封存门测试、5 个选择数学测试、24 个 CNS 合成身份/统计测试，以及 2 个严格整数比较/完整并列边界自检入口。原实验封存门拒绝、身份/字节篡改拒绝、未知状态和分母逻辑仍被检查。测试成功仅证明这些开发能力，不证明历史模型结果重新复现。

CIFAR 测试只在自建的临时目录创建小数组并清理，采用 `-m unittest` 避开原脚本 `__main__` 写历史 `TEST_RESULTS.json` 的逻辑。不要自行把此命令替换为全目录 discovery：例如 DeiT 的 `test_second_encoder_contract.py` 在导入阶段就访问真实资产并写固定结果。

## 项目路径配置

`portable_dev.paths.project_root(__file__)` 从源文件向上定位当前克隆，避免依赖原电脑目录名。默认不需要环境变量。必要时可明确设置 `RANDAUGMENT_PROJECT_ROOT` 为另一份含 `portable_dev/paths.py` 和 README 的开发克隆；无效目录会报错，不会静默退回旧机器路径。

82 个选定自有/控制脚本中的原项目 `Path(...)` 构造表达式已改为此解析器，其他逻辑保持不变。通过反向替换后的 AST 相等检查；旧、新 SHA-256 和原表达式行号见 `PORTABILITY_DERIVATIVES.json`。这些代码是开发派生版本，不能冒充历史字节原件。

如需直接运行其他已审查的开发模块，请先在仓库根设置 Python 的模块搜索路径，例如 POSIX shell：

```sh
export PYTHONPATH="$PWD"
```

这只使新路径辅助模块可被导入，不构成运行历史实验的许可或资源恢复；不要直接执行冻结的 `run.py`、资源准备器或云入口。

可选的 `RANDAUGMENT_ASSET_ROOT` 供 `portable_dev.paths.asset_path` 和 `doctor` 定位另一个本地资产目录，使用与仓库相同的相对布局。它不会下载、复制或上传文件，也不会自动重接全部历史脚本的资产变量。历史代码已有的来源 SHA 和配置仍需逐项核对。

## 资源与历史记录的边界

四个明确不上传的大文件：

- `augmentation-selection-cifar100-run-20261102/dinov2_small/W.npy`
- `augmentation-selection-cifar100-run-20261102/dinov2_small/support_features.npy`
- `research-frofa-pilot-20261030/cache/features.fp32.npy`
- `research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy`

原数据集、预训练权重、虚拟环境及部分未提供素材也可能不在克隆中。`doctor` 展示这些缺项，开发测试允许不具备这些资产；完整历史分析与训练仍可能拒绝执行。

原 `FIT_SEAL`、`QUERY_SEAL`、身份、manifest、历史截止时间和防覆盖逻辑没有为开发测试而取消。封存源码的旧 SHA 在历史记录中保留；当前源码路径迁移后，旧完整源码校验应如实报告不一致。新研究必须另建运行协议和输出目录，不能覆盖旧结果或沿用已过期的运行预算。

本入口只锁定最小 NumPy 开发依赖。完整模型主题的 Torch、Transformers、CUDA/检测框架依赖仍按各主题来源记录恢复；原 Mac wheel 锁也不等于其他操作系统已经验收。标准库、合成测试与可编辑源码可先恢复，原精度实验复核另行满足资产和环境条件。

## 交付范围说明

原始日志、运行环境、缓存、Cookie/凭据和敏感登录材料未交付。如需恢复日志或访问记录，应先审查具体字段，去除认证信息和敏感登录内容，并保留来源与修改说明。历史 manifest 所列文件若缺失，仍应如实报告缺件；合成开发测试通过不代表它们已恢复。
