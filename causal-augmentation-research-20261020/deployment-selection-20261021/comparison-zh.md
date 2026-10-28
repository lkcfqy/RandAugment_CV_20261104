# 从真实部署目标出发的两项问题核对

2026-10-21。本轮只比较两个具体问题，没有为凑候选增加第三个，也没有运行新的模型或合成反例。**当前没有达到“相对先例有实质差别、数据真值可得”双重要求的实验候选，故没有冻结或启动训练。** 整体项目未宣布结束。

| 比较项 | A：单因素验证选择增强，部署遇到真实联合变化 | B：换相机后，以无干净标签风险选择噪声增强 |
|---|---|---|
| 真实部署变化 | 已登记物体/纹理在相机观察方向和照明同时改变后仍需识别；不以2D旋转加颜色抖动冒充真实联合拍摄。 | 同一去噪系统换相机、ISO或ISP，真实噪声相关结构改变。 |
| 可检验命题 | 预固定P/Q，P在“只视角变”和“只照明变”的真实标注验证上均优于Q，却在未用于选择的联合条件上劣于Q。 | 固定两配方、同一目标相机原始验证图，比较无GT分数差Δself与成对GT风险差ΔGT是否异号。 |
| 标签/评价真值 | 类别或登记样本ID；需要每张图的物理实例、相机/视角、照明，以及四格真实拍摄对应。 | 同一场景成对noisy/clean参照、相机/ISO/场景ID；clean为估计参照，需审计配准、裁剪及偏差。 |
| 关键对照 | 等标注预算的少量真实联合验证；预固定候选与训练预算；按物理样本聚类，避免将重复视图当独立样本。 | 无插值的坐标置换、匹配像素方差的独立噪声；所有候选在同一native验证分布上以有效J-invariant路径评分；GT仅用于最终审计。 |
| 反证 | 独立物理样本上无稳定且有实际幅度的排序反转，或简单少量联合验证已解释并解决现象；不能拿不显著直接证明永不反转。 | 条件满足时固定候选的self-risk排序与GT一致；噪声相关也未必足以使排序反向。若仅训练增强发生改变，不能据此声称验证等式失效。 |
| 当前处置 | 广义主张已有直接覆盖；Pareto排序反转只是更清楚的域失配诊断，不足以单独立项。数据材料也尚不完整。 | 核心机制已被原始工作覆盖，且一开始的“训练增强自动破坏排序”推断不成立。淘汰，不做小滤波器演示。 |

## A的先例与剩余增量边界

[Demystifying Contrastive Self-Supervised Learning，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/22f791da07b0d8a2504c2537c560001c-Paper.pdf) 已研究自然视角、照明不变性与合成增强的错配；[Local Manifold Augmentation，2022，Table12](https://arxiv.org/html/2211.02798) 已在ALOI/ALOT比较自然因素不变性。[Modularity Trumps Invariance，2023，§3–4](https://arxiv.org/html/2306.09005v1) 直接研究单项腐蚀训练到组合腐蚀测试，并发现单项表示不变性不能预测组合识别能力。[VLA-Gauge，2026](https://arxiv.org/html/2610.16567v1) 也已有单因素诊断、组合扰动及定向增强的闭环，虽然对象为机器人任务。

本轮未见上述已读材料逐项报告“真实照片上RA/TA的单因素Pareto占优反转＋选择后悔值”；但**仅缺少同名实验不是新意**。独立审查认为这仍是普通域泛化/验证分布失配，尚无解释或选择能力的增量。至少需有可提前预测反转的交互条件，或在等标注预算下优于“补少量真实联合验证”的选择信息，才值得将它作为研究候选。当前两者均未建立，不能先训练来装饰窄反例。

资源核对产生了以下实质约束：

- **ALOI不能提供目标联合真值。** [官方采集说明](https://aloi.science.uva.nl/aloi/description.html) 说明换相机时同步转动物体，使呈现的物体侧面不变；3相机×8灯是24种光照配置，不是视角×照明交叉。72视角和12光色另行采集。不能把它们拼成联合真值。
- **smallNORB有完整交叉，但未确认小量读取路径。** [NYU说明](https://cs.nyu.edu/~yann/data/norb-v1.0-small/) 包含50个玩具、5类别、6光照×9仰角×18方位×2相机，train/test各25个互斥物理实例；研究用途、禁出售、需引用。[TFDS](https://www.tensorflow.org/datasets/catalog/smallnorb) 列完整下载250.60MiB。没有为尚无增量的问题下载矩阵大包。
- **ALOT有可查的真实交叉命名，但当前未取得实际小清单。** [里尔大学说明](https://color.univ-lille.fr/datasets/alot) 给出250类、`<class>_<camera><illumination>`，相机c1…c4，照明l1…l5/l8/i；其半分辨率整包为4GB，未下载。类别数不自动等于独立物理样本数；若每类仅一个物理样本，应明确目标是已登记样本识别，不能同时声称闭集分类与物体互斥泛化。
- **Meta-Album是独立发布的预处理版本，许可已可查，因素字段仍未知。** [官方数据页](https://meta-album.github.io/datasets/TEX_ALOT.html) 声明CC BY-NC4.0；[info文件](https://meta-album.github.io/assets/files/info/TEX_ALOT_info.json) 列micro=20类×40图、mini=250类×40图、extended=250类×100图。采用方形裁剪及128×128重采样；它与原始半分辨率图不是同一版本。[格式说明](https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat) 只强制FILE_NAME/CATEGORY/SUPER_CATEGORY，不能据此断言真实发行文件保留或丢失camera/light/实例字段。实际清单、文件名字映射、交叉格覆盖及小包字节数未核实。

## B的先例与前提纠正

[Noise2Self，ICML2019，Proposition1](https://proceedings.mlr.press/v97/batson19a/batson19a.pdf) 已用自监督损失选择参数，条件包括零均值、跨掩码组噪声独立和J-invariant预测；[Noise2Same，2020](https://proceedings.neurips.cc/paper/2020/file/ea6b2efbdd4255a9f1b3bbc6399b58f4-Paper.pdf) 并未消除所有独立性限制。[AP-BSN](https://arxiv.org/html/2203.11799) 已直接研究ISP插值所生相关噪声与blind-spot失效；[TM-BSN，2026](https://arxiv.org/html/2604.04484v1) 及[Next-Scale Prediction，2025](https://arxiv.org/html/2512.21038) 继续处理相关噪声与细节保留的取舍。

**仅在训练侧加入几何/插值增强，不自动破坏正确验证分布上的等式。** 对同一native验证分布、满足条件的固定候选，训练历史本身不改变该恒等式。可能失效的是验证输入/有效预测路径已混合噪声、ISP原本相关，或跨不同变换域直接比较不同噪声常数。这些属于已有保证条件；普通有限样本验证过拟合也不作为新机制。

若定义d=g_a(z)−g_b(z)、n为验证噪声，相关交叉项可写作 `Δself=ΔGT−2E<d,n>`（同一噪声常数抵消）。相关性本身不保证异号。没有真实配方及输出时，跑一个小滤波器反向例只能复现已知条件失效，因此没有运行。

[SIDD官方站](https://abdokamel.github.io/sidd/) 声明原数据/代码MIT；Small有160对及相机/ISO元数据，但原版逐文件入口和体积未核实。GT经配准及稳健聚合估计，不是不可错的物理真值。[2026 SIDD-CC原文](https://arxiv.org/html/2607.11090v1) 已报告原GT流程clipping导致的颜色偏差；其[作者发行目录](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/tree/main/SRGB/train/Data/0017_001_GP_00100_00060_5500_N) 可逐PNG取用，示例noisy/GT约16.4/10.9MB，约27.3MB一对。衍生卡片标`sidd-terms-of-use`并要求引用，不能直接当成独立MIT许可。未下载图像或聚合MAT。

## 本轮实际完成与未完成

完成了两路各8次聚焦检索及独立反驳；主线程继续核对了具体元数据、字段与最近原文。另读取ALOT首页2,701字节HTML，解释了网页工具空文本实际来自frameset；不是图像数据下载。KTH恢复读取仍502，未继续。新出现的不可访问及restricted URL均列入`access-status.json`，未换CLI/SDK/镜像获取同一受阻资源。没有新建账号、对外通信、依赖安装、数据包下载、模型训练、GPU/MPS、ARIS、调度或8A/PPT改动。

**当前未完成项是科学增量及可用材料，不是已完成实验等待包装。** A的实际发行清单尚未核实；B机制已淘汰。没有冻结新的训练方案或宣称新候选验证通过。应继续在现有线程寻找能提供额外选择信息的具体机制；不重启旧亮度/背景、全图擦除或CER外推，也不通过追加程序代替缺失证据。主与审查沿用Sol/max优先请求；工具无实际档位回执，不能核验每次执行。
