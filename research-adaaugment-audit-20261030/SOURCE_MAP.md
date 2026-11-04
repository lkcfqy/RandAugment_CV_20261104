固定提交：`d854af15467285ad8f46500d6fc2da5225fb5bd0`。下表为人工逐行核对后使用的映射；`results/imports-and-source-anchors.json` 保留全部 20 个 Python 文本的静态 import 清单和关键行断言。所有下载文件只读、解析和计算哈希，从未导入或执行。

| 已检源码位置 | 含义 | 独立 fixture 位置 | 保留与简化 |
|---|---|---|---|
| Dataset.py 52–56、112–116 | CPU float32 Tensor 缓存；初始化 full/none/warmup transform | audit_fixture.py 69–80 | 同类型 Tensor；只用 8 项；初值用固定非零值表示已完成 warmup 后的缓存 |
| train.py 120–131、230–234 | 先用 warmup 输入采样并写缓存，再启用 adaptive 分支 | audit_fixture.py 51–58 | 不执行目标网络；不声称第一次 adaptive 访问使用零缓存 |
| Dataset.py 72–73、117–118 | 按索引原地写 Tensor | audit_fixture.py 83–84 | 使用等价的原地单索引赋值；源码为批量索引赋值，逐样本值语义相同 |
| Dataset.py 75–88、119–127 | 先读 `.item()`，用缓存强度构造并物化 adaptive 输入 | audit_fixture.py 86–100 | 保留读取时点和标量快照；图像替换为 `[sample_id/10, level/29]`，不能复现 PIL 像素行为 |
| organize_transform.py 7–18；augmentation/adaaugment.py 363–366、634–641 | standard、30 strengths；选 1 操作；level = min(int(29m)+1,29) | audit_fixture.py 46–48、264–289 | 精确复写强度量化；操作参数来自已读公式；不调用原始变换代码 |
| train.py 72；confs/resnet18.yaml、resnet50.yaml 第 3 行 | batch=128、shuffle=True、workers=8、pin_memory=True | audit_fixture.py 139–176 | batch/shuffle/workers 保留；CPU fixture 显式关闭 pin_memory，不复现 CUDA/pin-memory 路径；8 样本只形成 1 个小尾批 |
| train.py 151–165 | 物化输入先计算三个 loss、none/ada feature | audit_fixture.py 107–120 | 无模型；固定代数 loss 是缓存 level 的确定性函数，保留时序和依赖关系 |
| train.py 167–169 | 随后采新动作并写缓存 | audit_fixture.py 121–130 | 固定可追踪动作替代 Beta 抽样；与统计部分分开；两次反事实写入后恢复同一新动作 |
| train.py 171–175 | reward 来自已算出的三个 loss；可选运行归一化 | audit_fixture.py 107–113、116–136 | 保留 reward 公式，固定 λ=0.25；默认归一化关闭；没有复制整个归一化器 |
| train.py 177–181 | 更新目标网络后，以新 magnitude 和旧输入 reward 调用 agent | audit_fixture.py 131–136 | 记录归因字段；完全跳过目标训练与所有 optimizer 步 |
| A2C.py 26–37、67–70 | softplus+1 的 Beta 参数；用 `sample()` 采样 | audit_fixture.py 217–225、248–255 | 固定 Beta(2,3)；detach 的 sample；另验证 softplus 链式法则；不构造 actor MLP |
| A2C.py 72–83 | 取传入新 action 的 log_prob；critic TD 优势；负号与 sum | audit_fixture.py 226–247 | critic=0，reward=a；autograd 原始 score 均值；另记录原 loss 的负号与 N 倍缩放 |
| A2C.py 84–90、63–64 | actor/critic 优化步、Adam weight decay | 无对应执行 | 刻意不运行，零均值 score 不能推论这些更新的期望或训练轨迹 |
| augmentation/adaaugment.py 119–124、130–133、185–202、265–272 | Rotate、Posterize、TranslateX、Brightness 的参数转换 | audit_fixture.py 264–289 | 正方向固定；保留操作类型、level、物理参数、顺序、apply 与随机性占位字段 |
| augmentation/adaaugment.py 91–106、606–620 | Identity、AutoContrast、Equalize 不依赖 magnitude | audit_fixture.py 281–282、296–299 | effective tuple 中不把无作用 level 当参数变化 |
| augmentation/adaaugment.py 636 | 官方每次只有一个主增强操作 | audit_fixture.py 302–360 的多操作例 | 多操作共享强度仅是合成干预工具回归测试，绝不声称是官方多操作实现 |

PyTorch 的实际缓存机制另见 `framework-source-evidence.json`。它保存本机已安装版本 2.13.0 的 DataLoader 与 multiprocessing reducer 原文件路径、SHA-256 和带行号片段，包含默认 prefetch=2、persistent=False、迭代器重建、worker 启动及 CPU storage 共享序列化路径。相关推断通过真实安装的 DataLoader + 独立合成 Dataset 验证，而非仅靠文档猜测。

论文核对位置为 [Algorithm 1](https://arxiv.org/html/2405.11467v3#alg1) 的第 6–10 步，以及 III 节 Overview 关于下一 epoch 应用 magnitude 的描述。只称“先算旧输入 loss、再决定新 magnitude”这一关键顺序一致；论文在后续 actor/target 参数更新顺序上与源码并非逐行完全相同。

fixture 没有隐藏执行边：`source_checks` 只做 AST 解析；`train.py` 的顶层配置、数据读取、CUDA 模型创建均不执行。其 imports 没有发现其他会在 set_MAGNITUDE 时回写当前已物化输入或重新算 reward 的依赖。运行依赖仅为已安装 PyTorch、其已有 NumPy 支持和 Python 标准库；未安装任何包。
