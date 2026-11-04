# Pets 视图梯度路由有限效用筛选：资产阻塞，科学未定

本次没有完成效用筛选，也没有得到候选有效或无效的结论。官方 DeiT 模型已取得并通过安全加载、完整 strict load 与 MPS 梯度 oracle；Oxford-IIIT Pets 原始数据归档未能取得。真实 LP、吞吐校准和正式运行均未开始，未计算任何真实 validation/test 指标。24 个正式 run 完成数为 **0/24**。

## 状态与解释

| 维度 | 状态 | 可支持的判断 |
|---|---|---|
| 科学 | `UNDETERMINED / NOT_EVALUATED` | 没有精度、配对差值、继续门或原创性结论 |
| 资产 | `BLOCKED` | 模型通过；Pets 原始图像及标注归档未取得 |
| 资源 | `THROUGHPUT_NOT_CALIBRATED` | MPS 可用，但不能判断完整 24 runs 是否满足吞吐和预算门 |
| 模型/梯度验证 | `PASSED` | 实际 DeiT、MPS FP32、合成输入的 7 类计算路径 oracle 通过 |
| CPU 实现检查 | `PASSED_ON_SYNTHETIC_FIXTURES` | 固定划分、重复/冲突隔离、随机键及 schedule 端点检查通过 |
| 真实数据与完整执行验证 | `NOT_RUN` | 去重后的真实划分、LP、优化器训练、恢复和完整选择/测试流程尚未用真实数据端到端验证 |
| 审批 | 无拒绝、无待审批 | 连接超时不能描述为 GPU 授权或权限被拒绝 |

本次只停止这个缺资产的候选执行。整体研究路线由主线继续；不能把本次阻塞记入算法阴性结果。

## 已完成工作

1. 阅读项目 README、停止快照与备份恢复说明；项目及其父目录未发现 `AGENTS.md`，项目内未发现 `SKILL.md`。新的明确执行委托授权在独立新目录进行本轮研究，历史停止记录和旧实验均未修改。
2. 使用已有 Python 3.14.10、Torch 2.13.0、torchvision 0.28.0 环境。必要依赖从官方 PyPI wheel 核对 SHA256 后隔离安装在本目录 `deps/`；没有升级原环境的 Torch/torchvision。timm 固定为 1.0.22。全部实际版本、来源与 hash 见 `dependency-lock.json`、`sources/receipts.json`。
3. 确认本机 Apple M5 Pro 的 MPS 可用。仅在 MPS FP32 执行了模型 oracle；禁用 fallback，无 CPU 模型前向/反向、AMP、compile、dropout 或 stochastic depth。
4. 保存原始冻结协议及实现：数据 manifest、独立 RNG、8 配方梯度路径、LP、校准、预算锁、固定 run order、双恢复槽、最终全局选择及仅 6 模型 test 流程。实现已经通过语法检查及下述局部验证，**完整执行控制器仍未做真实数据端到端验收**。
5. 为可能找到的合法旧缓存保存 `import_archives.py`：只接受尺寸、SHA256、MD5 全部匹配的原始归档，拒绝重打包替代。当前并未导入任何 Pets 数据。

## 模型证据

- 规格：`timm/deit_tiny_patch16_224.fb_in1k`，非蒸馏、ImageNet-1k 预训练。原始 [DeiT README](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/README_deit.md) 与 timm 配置指向同一 [官方权重](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)。模型卡和原仓库许可为 Apache-2.0。
- 原仓库 revision：`7e160fe43f0252d17191b71cbb5826254114ea5b`；[timm 模型仓库 revision](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k/tree/80e968688553f219e4a86f940ed945a23709c16f)：`80e968688553f219e4a86f940ed945a23709c16f`。
- checkpoint 大小：22,917,895 bytes；本地完整 SHA256：`a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`，与官方文件名提供的前缀 `a1311bcf` 匹配。
- 使用 `torch.load(..., map_location='cpu', weights_only=True)` 做安全序列化读取，未增加 globals。先对原 1000 类完整模型 `strict=True` 加载成功，再替换 37 类 head 并将模型移至 MPS；CPU 没有执行模型计算。
- 原始完整参数状态 hash：`91344431276dbbd0bb6c8d190867bef3ca6dde7baa34c73a431ffc8d5e7eeb91`。替换 head 后共有 5,531,557 参数，特征维度 192。详情见 `model-validation.json`。

## 梯度与 CPU 检查

`gradient-oracle.json` 使用实际 DeiT 和两批独立合成输入（每批 2 张），分别求 weak/strong batch-mean CE 的原生 autograd 梯度，再独立组合目标梯度，与正式生产路径比较。没有使用真实训练、验证或测试图像。

| 路径 | backbone 目标 | head 目标 | 结果 |
|---|---|---|---|
| weak / Ent 单图 | gw | hw | 通过 |
| dual（B2/B3 共用图） | (gw+gs)/2 | (hw+hs)/2 | 通过 |
| head-refit | 0 | (hw+hs)/2 | 通过 |
| random 选 weak | gw | (hw+hs)/2 | 通过 |
| random 选 strong | gs | (hw+hs)/2 | 通过 |
| candidate（C1/C2 共用图） | gw | (hw+hs)/2 | 通过 |

候选 backbone 在该 oracle 中与 gw 的最大绝对差及相对 L2 差均为 0；候选 head 最大绝对差为约 `7.004e-7`。所有路径最大绝对差约 `9.611e-7`、最大相对 L2 差约 `8.695e-7`，低于预设的 `5e-5` / `2e-4` 容差。低学习率变体共享同一梯度图；本次没有验证训练收益。实际进行了 9 次 autograd 梯度请求，**优化器 step 为 0**。

CPU fixture 生成 1,374 张 8×8 合成图，仅用于验证规则。检查覆盖跨 trainval/test 重复保留 test、trainval 内重复、文件字节不同但 RGB 相同、标签冲突整组隔离、每类 16 验证 + 16 support、3 seed support 规则、592 样本每轮 18 个完整 batch、独立随机键以及 warmup/cosine 端点。检查通过；这些数目**不是 Pets 实际数据清洗结果**。详情见 `implementation-fixtures.json` 和 `logs/cpu-fixtures.txt`。

## 数据来源与阻塞证据

[Oxford 官方页面](https://robots.ox.ac.uk/~vgg/data/pets/)标明 CC BY-SA 4.0，原图版权归原持有人。官方页面列出 thor 原始下载和 Academic Torrents。原始归档核验目标由 [TFDS 官方 checksum 文件](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/oxford_iiit_pet/checksums.tsv)及安装的 torchvision 0.28.0 数据集定义共同记录；本地快照见 `sources/`。

| 归档 | 官方目标 bytes | 目标 SHA256 | torchvision MD5 |
|---|---:|---|---|
| images.tar.gz | 791,918,971 | `67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d` | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| annotations.tar.gz | 19,173,078 | `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91` | `95a8c909bbe2e81eed6a22bccdf3f68f` |

准确阻塞操作和结果：

- `curl --head --location --max-time 20 --retry 0 https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz`：已执行，退出 28；`Connection timed out after 20010 milliseconds`，未取得 HTTP 响应或归档。
- 对 `https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz` 的 IPv4 HTTPS HEAD：已执行，退出 28；`Connection timed out after 20006 milliseconds`。
- torchvision 引用的 www 下载地址也在 20 秒超时。robots/www 官方网页的两次直接读取分别在 90 秒连接超时，保留原失败日志，没有循环重试。
- Oxford 官方链接的 Academic Torrents 描述元数据已保存；只解析元数据，**未启动 Torrent/P2P**。描述列出的 HTTP web-seed 一个返回 404，另一个超时；HTTPS 升级尝试分别出现证书验证失败或超时。没有关闭证书验证或绕过警告。
- 根据主线允许的一次本机 Chrome 检查，对上述 annotations HTTPS URL 的导航等待超时。此后页面仍为 `about:blank`，常见下载目录未出现归档或 `.crdownload`；未确认下载成功。该检查页已关闭，无保留的下载/审批流程。Chrome 扩展可访问；原生应用界面因 Mac 锁屏不可用，本轮没有要求用户解锁或尝试绕过锁屏。
- 项目完整文件名扫描及 Downloads / `.cache` 顶层未发现原始归档。跨 Documents 的广域只读搜索运行约 9 分钟无结果后终止；这不是“全盘已经证明没有缓存”。主线旁路也没有提供已验证同 hash 的非 Oxford HTTPS 镜像。

详细机器可读结果见 `source-connectivity.json`、`browser-check.json`。操作没有被自动审批拒绝；CLI 失败结果已知，浏览器没有成功导航/下载确认。当前所需的两份可校验归档缺失，因此资产步骤未完成。

## 账本与冻结边界

- 首次准备起点：**2026-10-30 19:14:00 UTC**；原绝对截止：**2026-10-31 01:14:00 UTC**，未重置。
- 已计 MPS 阶段：**5.0354287500013015 秒 / 7200 秒**，包括该独立 oracle 进程的导入、模型加载、计算、同步和收尾。账本 `mps-ledger.json` 没有活动阶段。
- 报告准备时新增本目录文件约 **62.2 MB（0.058 GiB）**；最终精确字节数及已观察峰值在 `resource-ledger.json`。3 GiB 预算没有触发。模型、wheel、依赖、合成 fixture 和全部日志均保留。
- LP 更新：0；真实校准更新：0；正式更新：0；validation/test 预测：0；正式完成：0/24。
- 只有预执行 `PROTOCOL_ZH.md` 与固定 seed 20261030 的 `run-order.json`。**没有 `protocol.lock.json`、没有锁定 N/J、没有 selection.json、没有 test 输出。** 不能声称完整计划已经通过吞吐门。
- 无 Colab、无购买、无 commit、push 或公开上传。所有临时执行进程已退出；本轮自己的异常耗时只读搜索已终止。

## 交接入口

当前资产阻塞报告可直接归档；不需要唤醒用户处理 GPU 授权。若主线在原截止时间内取得**同 hash 的两份原始归档**，先核验导入，再生成真实 manifest，然后通过累计账本监督器运行 `pilot.py preflight`。仅在最小步数、完整 24 runs 时间和磁盘门全部通过并生成协议锁后才运行 `pilot.py formal`。不能通过减少臂、shot、batch、精度或使用重打包 split 恢复。

精确入口与注意事项见 `README_ZH.md`。若原 6 小时窗口过期，现有脚本会拒绝导入/执行；不能静默重置时钟。

科学定位保持冻结：这是小样本短迁移的经验配方筛选。Deep Augmentation、APNetwork 和 Gradient Routing 的既有工作限制了原创性主张；即使未来本轮通过实用继续门，也只值得另行审查下一轮直接 prior 比较。本轮当前尚未评估该门。
