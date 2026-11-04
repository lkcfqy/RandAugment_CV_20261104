# EuroSAT / SigLIP 冻结特征 pilot：现象门未通过，按预注册停止

六个确认 episode 的五臂均完整完成。C 相对 clean 的平均测试准确率差为 **−1.061 pp**，相对 E* 为 **−2.517 pp**；两项均未达到预定的“均值≥1 pp且单侧95%下界>0”。因此本轮结论是 **phenomenon_gate_failed_stop**。完整矩匹配的三个对比也没有通过等效标准或预定的局部不充分标准。

资源预检通过，配方完整执行。本轮未复现继续研究所要求的 clipping 优势；不能据此宣称二阶矩已经解释全部效果，也不能宣称非高斯机制产生新贡献。到此停止，不追加 seed、shot、a、数据集、模型或方向Q。

## 固定实验与实现

- EuroSAT RGB，Zenodo 7711810 v2。27,000张RGB解码SHA256全部唯一，无跨标签或同标签重复；按 `frofa-es-v1` 盐排序，每类固定100 trainpool / 100 validation / 300 test，共5,000张。校准仅用每类trainpool前10张，确认seed1–6分别从剩90张抽10张，未换图。
- SigLIP `google/siglip-base-patch16-224`，revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`；safetensors、trust_remote_code=False、本地离线加载。视觉权重missing/mismatched/error均为0；双塔checkpoint中未使用的文本／logit键不属于视觉权重缺失。
- 官方确定性224预处理，encoder eval冻结，缓存post-layernorm / pre-MAP的196×768 patch。无CLS、无图像增强、无额外特征归一化；FP32缓存全部finite且逐位读回一致。
- fresh MAP顺序：learned query进行12-head cross-attention，取pooled，再 `pooled + MLP(LayerNorm(pooled))`，最后Linear10。MLP768→3072→768、tanh GELU、dropout0；丢弃预训练MAP。初始化遵循锁定的官方库规则，逐episode/arm记录并核对init和batch-order hash。
- C/E*只改变每channel共享或每token独立的uniform偏移；minmax、epsilon、a、clip均相同。GC/GE匹配固定样本完整条件矩，CPU float64构造后cast FP32给MPS，无二次clamp/renorm、无apply gate。Gaussian结构仅为当前100 support构建，episode后释放。
- 每fit 400 updates、batch20每类2例；AdamW、20-step warmup后cosine至0，CE无label smoothing或grad clipping，固定末步。六次校准选择全臂共同 **LR=0.001、WD=0.01**，四个增强臂共同 **a=1.0**。校准权重、200步预检权重均丢弃，未续训。
- 省略可选确认validation。全部30个确认模型于 **15:37:42.437376 UTC** 冻结并保存哈希后，统一一次clean-cache测试；不存在依据测试指标选模型或调参。

## 预检

| 检查 | 实际结果 |
|---|---|
| 官方资产校验 | RGB ZIP MD5、模型SHA256均通过 |
| 20张固定开发图CPU/MPS encoder相对L2 | 最大3.2043e-5，门槛1e-4 |
| 2图×4 channel×两a，完整196-token解析矩检查 | 16组全部通过；最大标度化误差3.2187e-9，门槛1e-5；非MC拟合 |
| FP32缓存 | 5,000×196×768，全部finite、逐位读回通过 |
| 固定资源测速 | 两a×五臂×20 updates=200；权重丢弃 |
| 预检通过时间 | 首次下载后794.196秒（13分14秒），早于45分钟 |
| 余量投影 | 960.084秒，允许上限10,204.353秒 |

预检最慢候选a的更新p90分别为clean0.012495s、C0.017501s、E*0.032030s、GC0.192612s、GE0.027958s；100-support预处理两a分别3.566s、5.139s。投影使用较慢值，包含评估及180秒归档余量。

矩误差对象是连续条件float64算子；同时记录FP32因子舍入的协方差误差。实际训练输出的FP32舍入不被包装为无限精度分布的逐位等效。

## 六个完整配对episode的测试结果

准确率单位为百分比；表中均值只是固定pool/test上的六次训练重复。

| seed | clean | C | E* | GC | GE |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.500 | 75.833 | 78.600 | 76.033 | 75.700 |
| 2 | 79.267 | 78.800 | 74.967 | 79.833 | 81.233 |
| 3 | 76.767 | 77.333 | 81.733 | 79.933 | 76.467 |
| 4 | 81.700 | 77.900 | 83.033 | 79.400 | 80.967 |
| 5 | 78.100 | 74.400 | 80.667 | 74.867 | 77.033 |
| 6 | 78.233 | 80.933 | 81.300 | 81.167 | 81.967 |
| **均值** | **78.594** | **77.533** | **80.050** | **78.539** | **78.894** |

主对比为pp；t统计单位是6个完整配对episode，df=5，没有用3000测试图bootstrap替代训练重复。

| 对比 | 均值pp | 90% CI | Bonferroni家族95% CI |
|---|---:|---|---|
| I=(C−E*)−(GC−GE) | −2.161 | [−5.851, 1.528] | [−8.632, 4.310] |
| dC=GC−C | 1.006 | [0.241, 1.770] | [−0.336, 2.347] |
| dE=GE−E* | −1.156 | [−4.554, 2.243] | [−7.116, 4.805] |

家族95%对三个对比采用各自98.333…%双侧t区间。三个90%区间没有全部落在[−0.5,+0.5]pp内，等效不成立；没有任何家族95%区间完全位于+0.5以上或−0.5以下，局部不充分标准也未达。

| 现象门 | 均值pp | 单侧95%下界pp | 判定 |
|---|---:|---:|---|
| g=C−clean | −1.061 | −3.142 | 未通过 |
| h=C−E* | −2.517 | −5.583 | 未通过 |

## 实际预算及执行记录

首次直接下载：**2026-10-30 15:11:34.718396 UTC**。原预检截止15:56:34.718398 UTC；原总截止19:11:34.718398 UTC，均未重置。核心指标归档15:38:06.532695 UTC，1591.814秒；监督进程正常结束15:38:08.765953 UTC，约**26分34秒**。最终说明与哈希核验亦在原四小时期限内完成。

| 项目 | 实际 | 硬上限 |
|---|---:|---:|
| 下载（含失败准备的metadata） | 957,410,072字节 | 2,147,483,648 |
| 新目录磁盘，worker阶段已记录峰值 | 4,876,689,500字节 | 8,589,934,592 |
| 保守RSS峰值+当时MPS driver的观测最大和 | 7,131,643,904字节 | 25,769,803,776 |
| encoder图像前向 | 5,040=5,000缓存+20CPU+20MPS | 5,500 |
| 完整fit | 36=6校准+30确认 | 36 |
| 更新 | 14,600=14,400完整fit+200预检 | 14,600 |
| 训练特征曝光 | 292,000 | 292,000 |
| 增强views，clean不计 | 211,200 | 已包含在训练曝光内 |
| head评估图像 | 96,200=200预检+6,000val+90,000test | 100,000 |
| Colab CU | 0 | 0 |

各臂训练曝光为clean80,800、C64,800、E*/GC/GE各48,800。进程RSS峰值5,967,200,256字节，MPS driver观测峰值1,456,193,536字节；两项独立峰值之和约7.423GB也低于24GiB。driver是同步检查点的观测值；另设置本进程MPS分配上限10GiB。

CPU解码去重墙钟1.840s、CPU图像预处理3.657s；CPU增强/特征准备墙钟519.122s，确认support预处理30.620s。encoder MPS缓存同步墙钟35.402s，CPU parity0.789s、MPS parity0.645s；head训练MPS同步段200.022s、评估19.454s。训练CPU process time565.048s，评估7.696s。这些分段指标有包含关系，不应相加当总墙钟。MPS fallback显式禁用，CPU图像预处理和增强属于预定路径。

三次训练前工程修复全部保留在 `attempts/attempt01..03` 和 `amendment-01..03.json`：PyPI无效历史版本过滤、资产日志字段名冲突、显式传入视觉塔config。各次修复未改科学协议／数据／种子／容差，未发生训练更新；原下载和墙钟总账累计保留。此后36个完整fit没有失败或重启。

`ledger.json` 的有效fit计数在 `counts.complete_fits=36`；顶层遗留的 `complete_fits=0` 是初始化后未使用的字段，原日志保留不回写。`artifact-hashes.json` 是worker内生成的快照，最后一次completion事件在其后追加；最终文件校验以收尾生成的 `SHA256SUMS`／`completion-audit.json` 为准。

## 资产与证据

模型SHA256：`2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`。

RGB ZIP官方MD5：`f46e308c4d50d4bf32fedad2d3d62f3b`。

固定manifest SHA256：`0b4700a0311be08335ce2a55959dfea1a707c0493bc7fdf788e4b5df0dc48d73`。

FP32缓存SHA256：`ae3a3efd142cbff6a2806976e9e95c4ab1c5a0b0779500ff27730b27f22c6201`。

`protocol.json`、`PROTOCOL_NOTES_ZH.md`、原始与恢复的source-seal保留所有预定规则；`library-source-lock.json`记录实际官方库源码哈希和MAP调用顺序；`dependency-lock.json`与`downloads.json`记录隔离官方PyPI依赖和下载。`licenses/`保存MIT、Apache2及Sentinel法律通知原文。数据/模型仅本地保存，无外传。

`RESULTS.json`是未四舍五入的统计结果；`fits/`包含36次完整训练的逐步耗时／损失／support／初始化与batch-order哈希；`heads/`保存30个最终safetensors；`test/`保存统一测试的准确率和逐图预测；`events.jsonl`、`supervisor.json`、各原始log和`ledger.json`保留完整执行证据。

本轮只适用于该固定EuroSAT pool/test、SigLIP和既定few-shot配方。不称空间OOD，不排除预训练重叠；没有证明任何模型／数据集上的普遍充分性或不足。旧fixture及历史资料保留，无commit、push或上传。**当前无计算阻塞；按科学门停止，不自动延伸。**
