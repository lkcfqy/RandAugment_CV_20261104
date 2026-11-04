# HVP-DINO stage 1：资产门阻塞报告

最终状态：**ASSET/VALIDATION-BLOCKED**（原因码 `ASSET_BLOCKED`）。生成时间：2026-10-30T18:13:07.822027+00:00。

指定官方权重安全加载成功，但缺少协议必需的 `dino_loss.center`。本次没有得到真实图像上的熵效应结果，不能判为支持或否定该方向，也没有建立研究贡献。这一关闭仅针对该冻结 pilot；其他探索线由主任务继续协调。

## 已核事实与精确阻塞

- 官方来源为 [facebookresearch/dino 固定提交 README](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/README.md)。README 指向本次 [ViT-S/16 完整 checkpoint](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth)。固定提交 `7c446df5b9f45747937fb0d72314eb9f7b66930a`。
- HTTPS HEAD 与 GET 最终域均为 `dl.fbaipublicfiles.com`，HTTP 200，无跨域跳转。长度与实际文件均为 **352,247,243 bytes**。
- 本地 SHA256：`9aba1fa8e6875e8413b413e8b8d7b843413280b33b972bd313d2ace8c4129943`。这是本地下载后计算并复核的哈希，**不是作者公布哈希**；未把 S3 ETag 当 SHA256。
- 唯一加载调用：`torch.load(path, map_location="cpu", weights_only=True)`。共执行 **1 次**，安全反序列化成功。不是 Namespace allowlist 阻塞。
- 实际顶层键恰为 `early_stopping_epoch`、`student`、`teacher`；访问 `checkpoint["dino_loss"]["center"]` 抛出 `KeyError: 'dino_loss'`。该必需状态不存在，不能补零、估计、拼接其他模型中心或更换加载策略。
- student/teacher 张量有限性与规定 head 形状的代码已先运行到中心检查；完整模型的 `strict=True` 加载尚未执行，不能把本次称为完整模型验证通过。详见 `asset_gate.py` 和 `asset_gate.json` 的实测调用与堆栈。
- [官方 args](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt)已下载核对：原生 teacher temperature=.07、800 epochs、out_dim=65536、use_bn_in_head=false、norm_last_layer=false。README、Apache 2.0 LICENSE、原生 args 原样保留并哈希。

这次操作的结果已知：**加载已执行且成功，必需中心检查已执行且失败**。没有待审批动作，没有不确定的在途 GPU 作业；不是权限审核拒绝造成实验阻塞。未执行不安全加载、globals 扩展或 TorchHub。

## 数据、模型与实验执行边界

按照“先权重门，再数据”的顺序，Caltech101 zip 和图片均**未下载**，因此不存在数据文件哈希或1010图样本清单；`data_provenance.json` 显式记录缺失，未伪造产物。已查 [Caltech 官方记录页](https://data.caltech.edu/records/mzrjq-6wc02)：v1.0、DOI `10.22002/D1.20086`、137.4 MB、页面公布 MD5 `3138e1922a9193bfa496528edbbc45d0`。页面可读 License 栏为空，当前未独立确认 CC BY 4.0；这不等于禁止研究使用。权重门已失败，本轮不继续数据下载或许可证分支。

MPS 未启用，真实图像数 **0/1010**、真实池 **0/2020**；首16图双draw吞吐门、正式前向、真实负控、bootstrap 均未运行。没有训练、backward 或 stage 2/3。没有复制或执行 HVP 代码，也未下载官方 DINO 模型源码来运行。未验证实际梯度一致性。

## 已完成的独立准备与校验

`PROTOCOL.json` 保存冻结范围、种子编码、采样、评分、负控、bootstrap、门槛和预算。候选表按 PCG64(20261031) 从51,480个组合无放回取128个编号并排序，所有条件共用；表中去重 local 子集数为 **127**。表 SHA256：`a1891c81079ee92c32e8944332f5e970cf3baed488f489103ee5178c3c643925`。

`score.py` 为独立数学实现；`test_score.py` 仅用53维合成概率，不使用权重或真实图像，也没有拿合成中心补 checkpoint。两温度下 CE=KL+H 最大绝对误差分别为 **1.55e-15**、**1.33e-15**；独立 Python 标量实现最大差异不超过 **2.66e-15**。合成 local 负控严格改变率0，同分优先与1e-6严格容差用例通过。以上只证明该合成评分实现的相应性质，**不能替代真实模型负控、完整样本或贡献证据**。

`results.csv` 中两温度均为0样本，科学指标留空；没有把“未测量”写成效应为0。`synthetic_tests.csv` 与 `test_score.json` 单独标注合成测试。

## 预算与复核

首次下载至本报告：**187.86 秒**，上限7200秒。显式权重门与合成校验记录 CPU process time 合计 **0.613 秒**、wall time **0.585 秒**；初始解释器版本检查小于1秒，归档哈希计时另见归档收据。MPS **0秒**，没有消费 GPU 额度。权重约336 MiB，所有交付副本与归档会在3 GiB内核验。

软件：Python 3.14.10、PyTorch 2.13.0、NumPy 2.5.4、Pillow 13.0.0；macOS arm64。沿用项目 `.venv`，未安装或更新依赖。项目及各级上级没有发现 AGENTS.md；读取了 validate-data 技能与相关证据质量规范。使用技能不改变冻结协议。

证据见 `asset_download_receipts.json`、`asset_gate.json`、`STATUS.json`、`budget_ledger.json`、`PROTOCOL.json`、`candidate_table.json/csv`、`test_score.json`、`MANIFEST.sha256` 与归档验证收据。没有 commit、push 或公开上传。保留原有实验历史。

## 后续边界

本冻结 pilot 在资产门关闭；缺中心不能用“继续找贡献”的总授权绕过本轮硬约束。只有主任务另行审查可用完整中心的官方资产或重新冻结其他设计，才可开启新实验。本报告不擅自选择替代 checkpoint，不启动后续阶段。
