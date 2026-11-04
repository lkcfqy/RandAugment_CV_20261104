# Shared-theta AD：资产核验与交接

本轮资产准备完成。仅取得并核验数据和模型，未运行候选算法、科学效用评估、GPU 或 Colab。完整研究协议仍待独立审查；这里的结果不是科学阴性或阳性。

## 官方来源与取得方式

- 数据来源：[作者 MMR 仓库](https://github.com/zhangzilongc/MMR)，固定 commit `22d1b171687509024e6f5b53b5a5e772d653d32b`。README 指向 [Google Drive 的 AeBAD.zip](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view)，文件 ID 为 `14wkZAFFeudlg0NMFLsiGwS0E593b-lNo`；另有作者提供的天翼云盘镜像，本轮未使用。
- 原始 ZIP 精确体积 **1,616,195,913 字节（1.5052 GiB）**，来源于 Drive 文件元数据并由 HTTP `Content-Range` 总长度复核。全 ZIP 中央目录为 7,415 项，总解压字节数为 1,696,716,623。
- 官方 Google Drive 的公开下载响应实际支持 **HTTP 206 Range**。本轮通过正常的公开大文件下载确认表单取得 URL，读取 ZIP 末尾和中央目录，再按明确目录范围获取所需内容。没有凭证、付费、第三方 repack 或权限绕过，也没有下载完整 ZIP。
- 精确范围保存在 `sources/aebad-range-plan.json`；21 段实际传输 **1,002,725,046 字节**，每段响应范围和 SHA256 留存在 `logs/aebad-acquisition.json`。每个提取成员核对本地文件头、路径、大小及 ZIP CRC32。完整 ZIP SHA256 未取得，不能将分段哈希称为整包哈希。
- 没有发现作者提供的逐图 URL 或实体 manifest。`sources/aebad-zip-manifest.json` 是本地从官方 ZIP 中央目录生成的目录清单，不是作者另行发布的标注表。
- 数据许可是 **CC BY 4.0**，原始 `LICENSE-DATASET` 同时保存在 `sources/mmr-LICENSE-DATASET` 和 `data/LICENSE-DATASET`；MMR 代码 Apache-2.0 原文保存在 `sources/mmr-LICENSE`。作者、论文及修改声明见 `data/README-SOURCE.md`。图像与掩码字节、文件名均未改动。

## 已落地的数据

根路径：`data/AeBAD/AeBAD_S/`。

| 划分 | 域 | 正常 | 异常 |
|---|---|---:|---:|
| train | 所有官方 train/good 子目录 | 521 | 0 |
| test | same | 230 | 459 |
| test | illumination | 75 | 198 |
| 合计 test | 两个预定域 | 305 | 657 |

同时取得 657 张对应异常掩码。共 1,483 张图像、657 张掩码，解压文件总计 **1,051,850,985 字节**。未提取其他测试域、AeBAD-V 或 macOS 附带文件。

所有图像和掩码均为 **3024×3024**。一个容易误读的格式事实：1,483 张图像虽然扩展名为 `.png`，实际字节格式全部是 **JPEG/RGB**；657 张掩码是 PNG。应使用按内容识别格式的解码器，例如 Pillow，保留官方命名。所有文件可完整解码，异常掩码无缺失且尺寸全部匹配。

逐文件原始 SHA256、图像的解码 RGB SHA256、尺寸、格式和元数据保存在 `reports/file-manifest.jsonl`。RGB 哈希定义为 `SHA256(uint32_be(width) || uint32_be(height) || Pillow.convert('RGB').tobytes())`，不缩放、不做 EXIF 方向变换。

## 重复与独立性

**跨 train/test 的原始字节重复组为 0，解码 RGB 重复组为 0。** 测试集内部精确重复组也为 0。

训练集内部发现同一张图片在两个路径出现，原始字节和解码 RGB 均相同：

- `train/good/view/IMG_7138.png`
- `train/good/illumination/IMG_7138.png`

因此官方 521 个训练路径对应 **520 个不同的原始/解码图像内容**。本轮不删除或替换；正式协议应预先说明重复处理方式。完整哈希及路径见 `reports/duplicates.json`。

官方仓库树、完整 ZIP 中央目录及[论文 §3.2](https://arxiv.org/html/2304.02216v2#S3.SS2)没有给出逐图叶片实体 ID 或采集批次 ID 映射。取得的图像仅有 JFIF 基本信息，没有 EXIF 实体、序列号或拍摄时间字段。`IMG_` 文件号、domain 文件夹和无精确重复都不能证明独立物理叶片。近重复、同一实体多角度和跨集实体重合仍未验证，不能臆造实体分组或声称独立实体泛化。

## 模型和安全读取

官方模型：[facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small)。固定 HF revision：`ed25f3a31f01632728cabb09d1542f84ab7b0056`。

- `models/dinov2-small/model.safetensors`：**88,249,960 字节**。
- SHA256：`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`，与预先给定的官方值相同。
- 同 revision 的 `config.json`、`preprocessor_config.json`、模型卡及上游 Apache-2.0 license 均归档；来源与哈希见 `reports/source-artifact-manifest.json`。
- 用本地 Transformers 内置 Dinov2Model、`trust_remote_code=False`、`local_files_only=True`、`use_safetensors=True` 在 CPU 加载。加载期间将 `torch.load` 设为报错，确保未走 pickle 路径；未用 TorchHub，未新增安全 allowlist。
- 参数量 22,056,576；无缺失、意外或形状不匹配权重。用一张合成全零张量检查几何，**没有任何真实图像通过模型**。
- 输入 `[1,3,224,224]` → `last_hidden_state [1,257,384]`；去 CLS 后是 256 个 384 维 patch，重排为 `[1,16,16,384]`，patch size 14。
- 最后训练得到的 LayerNorm 是 `model.layernorm`。`outputs.last_hidden_state[:,1:,:]` 已经过该 norm；与 `model.layernorm(outputs.hidden_states[-1])` 逐元素一致，最大差为 0。不要误用未经过最终 norm 的 `hidden_states[-1]`。

机器可读核验见 `reports/model-cpu-verification.json`，纯 CPU 可重复脚本为 `scripts/verify_model_cpu.py`。

## 预处理约束与环境

后续输入几何明确为 **整图直接缩放到 224×224，不裁剪**。官方 HF processor 默认“短边 256 + 中心裁剪 224”只作来源归档，不能直接沿用。插值、抗锯齿、亮度/对比度公式及顺序、特征 L2 归一化、检索距离与评价决策仍待正式协议预先规定；见 `reports/preprocessing-contract.json`。

本轮先检查现有 `.venv`：torch 2.13.0、torchvision 0.28.0、Pillow 13.0.0、numpy 2.5.4 已有；Transformers/safetensors 缺失。因此在本目录创建 `.asset-venv`，以只读导入路径复用已有科学计算包，只在新环境中安装 PyPI 官方 wheel。固定 Transformers 4.57.1、safetensors 0.8.0，其他依赖版本与 SHA256 见 `requirements-assets.lock`。未修改既有项目环境；`pip check` 通过。依赖冻结与来源保存在 `reports/environment-freeze.txt`、`reports/dependency-wheel-ledger.json`。

运行核验脚本时使用本目录 `.asset-venv/bin/python`，并设置 `PYTHONDONTWRITEBYTECODE=1`。数据已取得，无需重跑下载脚本。

## 成本与交接

截至成本报告：已记账的有效下载负载 **1,115,523,201 字节**；另留 50 MiB 给浏览器页面、包索引和小元数据，保守总计 **1,167,952,001 字节（约 1.09 GiB）**，低于 2 GiB。分段数据取得耗时 141 秒，所有文件解码/哈希审计 17.5 秒，模型 CPU 结构检查 2.6 秒。合成张量前向仅证明可运行，不能据此承诺真实数据或检索吞吐量。

目录保留了原始 ZIP 分段便于复核。记录时实际分配磁盘约 2.24 GiB；连同临时文件和最终报告的保守峰值上界约 **2.34 GiB**，低于 3 GiB。精确数字、统计口径及 wall 时间见 `reports/cost-ledger.json`。本轮 GPU 秒数 **0**，Colab 分配 **0**。

仅作容量算术：每个增强标签的全量 521×256×384 float32 特征库为 **204,865,536 字节（195.375 MiB）**。增强标签数量、分块检索及实际运行时间未确定，也没有分配下一轮算力。

**交接状态：资产与 CPU 读取可用；完整协议、训练重复处理及实体/批次独立性表述待审查。没有权限拒绝、待批准的下载或未明执行结果。其他实验目录未修改，未提交或推送。**
