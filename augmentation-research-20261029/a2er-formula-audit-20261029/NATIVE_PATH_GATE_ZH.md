# 原生增强路径门结果

2026-10-29 UTC。**已发布分类样例：fixed stored input，fresh-replay 增强方差机制不适用；论文图像 benchmark：增强时点未验证。当前不进入训练门 A／B。** 这是实现前提门，既不是模型性能阴性结果，也不是对所有 A2ER 实现的否定。

## 固定来源

作者补充材料 p.1 指向 `prinlab/A2ER_O2S`；固定提交 `f1334df1104724bb3b56f1f353857d92b1141565`，完整未截断 Git tree 见 `sources/a2er-tree.json`。确实有官方代码。下列路径均相对 `sources/a2er/`，原始 Git blob 校验通过：

| 证据 | 实际路径 |
|---|---|
| `derpp/replay_buffer.py:42–60` | append 将调用方给的输入存成 float32 数组 |
| `derpp/replay_buffer.py:63–77,163–164` | sample 只选 ID、回读数组并转 tensor；没有 transform 参数或新增强 |
| `derpp/derpp.py:75–104` | 抽取回放后直接交给调用方 loss_func／der_func；核心允许不同回调，但没有自带图像增强 |
| `model/classification.py:65–69,75–77,85–88` | 所发布分类回调只 convert→FCNet→Linear；append 存 x；update 使用同一回调 |
| `module/network.py:71–80,109–113` | convert 只 reshape／float32／device；FCNet 确定性前向，无 dropout 或增强 |
| `model/classification.py:9` 与完整 tree | 导入 `derpp.derpp_ablation`，对应文件未发布；不得把完整样例包标成已成功运行 |

论文“Image classification benchmarks compared to the state-of-the-art methods”段落说明训练数据用了默认 Random Erasing，并说明 CIFAR 使用 GroupNorm ResNet18；未给出入库前与回放时的具体位置。现有发布树没有 MNIST/CIFAR benchmark 驱动。不能用 toy 分类样例证明未发布图像实验一定固定视图，也不能将 Mammoth 的 replay transform 自动归于 A2ER。

来源：[主文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12961966/)，[固定作者代码](https://github.com/prinlab/A2ER_O2S/tree/f1334df1104724bb3b56f1f353857d92b1141565)。

## CPU 路径检查

`NATIVE_PATH_GATE_PROTOCOL_ZH.md` 和脚本先冻结，随后执行 `native_path_check.py`。两组 seed=20261029、20261030，各 8 次真实作者 ReplayBuffer.sample；16 个固定合成输入／历史 target，按身份重新对齐。

- 8 项检查全部通过；每次输入与 target 的最大差异均为 0。
- 预定确定性算术输出下，两组均 Davg−Dmean=0、视图方差=0。
- 核心耗时 **0.010121333 秒 CPU**；没有神经网络运行或 optimizer step。
- 随机的是回放 ID 顺序，不是同一样本的新视图。此检查不是完整分类模型执行、不是正式 512-buffer 门 A，也不能提供性能结论。

因此公开样例不能提供本次机制所需的独立视图变化。在固定模型和同一存储输入的数学条件下，Davg 与 Dmean 没有可供重分配的视图方差差别；无需先训练前缀来确认这一点。

## 下一步的确切阻塞

请求的共同前缀及 N/A/B/R/U 短分支**未执行**。不是自动审批拒绝、不是 MPS／数据故障，也不是训练超时；其先决条件“可核实的原生回放增强路径”未满足。原生样例固定输入；完整图像 benchmark 的 Random Erasing 路径缺失。

若以后取得作者图像入口，可核对其增强时点再决定是否运行已提出的科学门。若选择在 Mammoth 等已有 fresh-replay 路径上组合 A2ER 控制器，必须作为明确的新组合与新协议，先解释其科学对象；本轮没有擅自这样扩展。原第六轮的预算和终点仅为条件方案，没有消耗训练预算，也没有变成已验证方法。
