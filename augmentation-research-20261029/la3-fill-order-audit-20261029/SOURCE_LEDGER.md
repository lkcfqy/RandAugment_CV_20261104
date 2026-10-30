# 固定来源与既有结论边界

保存的是公开论文与官方发布代码，不包含训练数据、权重或私人材料。文件 SHA-256 以 `sources/papers/manifest.json`、`sources/code-manifest.json`、`sources/policy-download-manifest.json` 为准；完整 PDF 的页码与正文文本由 macOS PDFKit 提取至 `sources/papers/extracted-text.json`。

## 主要被审计对象

1. **LA3: Efficient Label-Aware AutoAugment**，Mingjun Zhao、Shan Lu、Zixuan Wang、Xiaoli Wang、Di Niu。ECCV 2022；这里使用 [arXiv:2304.10310v1，2023-04-20](https://arxiv.org/abs/2304.10310v1)／[固定 PDF](https://arxiv.org/pdf/2304.10310v1)，16 页。第 5 页 §3.1 三操作与随机幅度，第 8–9 页选择过程，第 10 页策略大小与评价采样；原生执行细节以同目录固定代码为准。论文不声称本审计的局部补边结果。
2. 作者仓库 [Simpleple/LA3-Label-Aware-AutoAugment](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/tree/3e593946980912524fb167dad00e83b1b08ec16e)，提交 `3e593946980912524fb167dad00e83b1b08ec16e`，2022-07-16，树 `8f6f02cbbdc3a9970a8cfd4d63c9216ceead62e3`。树响应未截断；只取相关源码与三个小型策略文件。无依赖版本锁定，不能把 2026 环境称为历史环境复现。

| 主张 | 固定源码位置 |
|---|---|
| 策略选择、顺序、独立幅度采样 | [data.py L535–552](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/data.py#L535-L552) |
| 几何默认参数与符号 | [augmentations.py L12–60](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/augmentations.py#L12-L60) |
| Invert、Equalize 等；Cutout 另有颜色 | [augmentations.py L63–147](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/augmentations.py#L63-L147) |
| 幅度范围与线性映射 | [augmentations.py L163–202](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/augmentations.py#L163-L202) |
| 类策略在 CIFAR 后处理之前 | [dataset.py L247–253](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/dataset.py#L247-L253)；CIFAR-10 L635–641 |
| 类策略在 ImageNet crop/resize 前 | [dataset.py L1311–1338](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/dataset.py#L1311-L1338)；[data.py L66–76](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/data.py#L66-L76) |
| 三操作 embedding 求和 | [neural_predictor_model.py L160–170](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/networks/neural_predictor_model.py#L160-L170) |
| 初始条目加 N 项 | [search_stage_2.py L146–163](https://github.com/Simpleple/LA3-Label-Aware-AutoAugment/blob/3e593946980912524fb167dad00e83b1b08ec16e/LA3/search_stage_2.py#L146-L163)，L274–280 |

## 最近前作：已知内容与尚未证明的区别

3. **Adversarial Machine Learning at Scale**，Alexey Kurakin、Ian Goodfellow、Samy Bengio，ICLR 2017；[arXiv:1611.01236v2](https://arxiv.org/abs/1611.01236v2)，[PDF](https://arxiv.org/pdf/1611.01236v2)，17 页，§4.2 在 PDF 第 6–7 页。已讨论依赖真实标签的一步扰动产生 label leaking；因此“标签相关变换会留下捷径”不能笼统当新发现。其 FGSM 机制与 LA3 的补边通道不同，不能替代后者的原生损害证据。
4. **Visual Chirality — Supplemental Material: Commutativity and the Chirality of Imaging Processes**，Zhiqiu Lin、Jin Sun、Abe Davis、Noah Snavely，CVPR 2020；[作者补充 PDF](https://linzhiqiu.github.io/papers/chirality/supp.pdf)，11 页，§4–6（PDF 第 5–10 页）。已有交换残差、随机／Gaussian 载体、裁剪与边界控制，且说明单个非交换例不能自动证明分布不对称。因此仅做交换、载体解码或去边框对照不足以主张新颖性。该 PDF 未提供独立版本号，以下载哈希固定快照。
5. **Towards Order of Processing Operations Detection in JPEG-compressed Images with Convolutional Neural Networks**，Belhassen Bayar、Matthew C. Stamm，Electronic Imaging 2018；[作者 PDF](https://research.coe.drexel.edu/ece/misl/wp-content/uploads/2018/04/BayarStammEI18.pdf)，9 页，摘要／引言与处理链检测方法。已有顺序条件指纹与处理顺序检测；“可以从像素识别顺序”并不是本候选可单独提出的贡献。未把该取证结果解释为 LA3 分类损害。无独立版本号，以哈希固定。
6. **Model Patching: Closing the Subgroup Performance Gap with Data Augmentation**，Karan Goel、Albert Gu、Yixuan Li、Christopher Ré；已核对 [arXiv:2008.06775v1，2020-08-15](https://arxiv.org/abs/2008.06775v1) 摘要。CAMEL 使用类内跨子群增强及一致性／鲁棒目标，限制“去除伪相关的条件增强”泛泛新颖性主张。本轮未重跑或下载其训练材料；该项只用于背景边界，不声称已逐项复现。
7. **Learning Instance-Specific Augmentations by Capturing Local Invariances**，Ning Miao、Tom Rainforth、Emile Mathieu、Yann Dubois、Yee Whye Teh、Adam Foster、Hyunjik Kim；[ICML 2023，PMLR 202:24720–24736](https://proceedings.mlr.press/v202/miao23a.html)。InstaAug 学习输入相关的局部不变性增强。因此输入／类别依赖本身不是反常现象或缺陷，必须区分合理语义适配与所假设的实现通道。这里仅核对官方摘要与元数据，没有新增方法效果检验。

最近前作的“已知”不构成 LA3 此机制无贡献的绝对证明，也不能降低后续因果识别和独立 clean 泛化证据的要求。
