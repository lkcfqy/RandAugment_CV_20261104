# 固定源码与只读调用链

没有执行以下代码。`sources/` 仅用于证据核对；GitHub 返回内容写为原始 UTF-8 字节，并核对 Git blob SHA-1。

## TRIQA

仓库：[rajeshsureddi/triqa](https://github.com/rajeshsureddi/triqa/tree/6e8a7a1c9c38b9f2dd0e9cce3052d65ac6ceec73)，固定 commit `6e8a7a1c9c38b9f2dd0e9cce3052d65ac6ceec73`，提交日期 2025-09-07T18:08:30Z。递归树 `truncated=false`。

| 位置 | 观察 |
|---|---|
| `triqa_simple.py:55–82` | 加载 content/quality ConvNeXt、scaler、回归器，设为 eval |
| `triqa_simple.py:84–115` | 完整图及半分辨率图分别提特征，再拼接、缩放、回归 |
| `convnext_finetune.py:66–124` | 128 维头与单图 forward；文件名含 finetune 不等于训练入口 |
| `README.md:54` | 作者发布模型的 UT Box 链接；本轮未打开/下载权重 |
| `README.md:90–92` | 声明 MIT；当前完整树未见 LICENSE，GitHub 元数据 license=null；不推断可额外授权 |

完整树没有原生训练 loop、三元组数据生成器或 dataset loader。这只限定当前发布版本，不能断言作者没有训练代码；也不是整个 IQA 研究的阻塞。

## SEA-RAFT

仓库：[princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT/tree/9137517ba24e628442aec097d3afe71d03503b75)，固定 commit `9137517ba24e628442aec097d3afe71d03503b75`，提交日期 2026-03-26T15:55:19Z。递归树 `truncated=false`；BSD-3-Clause LICENSE 已随源码保存。

仅追踪 `chairs` 的默认非 PWC dense 路径；不对所有 sparse/PWC 配置做外推。

| 顺序/位置 | 观察 |
|---|---|
| `config/train/Tartan-C368x496-S.json` | dataset=chairs，4 iterations，var_min=0，var_max=10，use_var=true；这是已发布配置，不是本轮运行 |
| `core/datasets.py:414–419` | 创建 FlyingChairs + FlowAugmentor；没有启用 pwc_aug |
| `core/datasets.py:110–128` | 增强→tensor；dense valid 来自 flow 幅度条件，未加入擦除区域 mask |
| `core/utils/augmentor.py:32–68` | 颜色增强；0.5 概率在 img2 写入均值色矩形；仅返回两张图 |
| `core/utils/augmentor.py:70–117` | 同步缩放/翻转 flow 向量，再同坐标裁切图像与 flow |
| `core/utils/augmentor.py:119–145` | 顺序为颜色→擦除→空间变换，返回 img1,img2,flow |
| `core/raft.py:135–159` | 对各迭代计算混合 Laplace NLL；配置 var_min=0 使小尺度 log_b 固定 0 |
| `core/loss.py:11–24` | valid 与 max_flow=400 掩码、有限数过滤；按迭代 gamma 加权 |
| `train.py:39–57` | loader→model(flow_gt=flow)→sequence_loss(output,flow,valid)→backward→optimizer.step |

“保留光流标签”描述对应运动，不等于声称第二帧仍可直接观测对应点；已遮挡点的上下文预测也是任务的一部分。仅检查标签同步不能得出训练有害结论。

逐文件 Git blob 与 SHA-256 在 `GIT_SOURCE_VERIFICATION.json`。固定版本不能冒称 Pro 对源码做了独立复核。
