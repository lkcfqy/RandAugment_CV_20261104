# 新增强家族只读筛选：联邦、卫星时序、视觉控制

日期：2026-10-30 UTC。起点 00:34:15 UTC。**本轮保留 0 条可提交实验的候选；没有实验失败结果。**
本轮做一手文献、官方仓库元数据及少量源码检查，没有运行模型、图像操作、RDKit 或训练，没有取数据/权重、安装或付费。
MolScribe 已单独关闭，见同级 `ocsr-closeout-20261030/CLOSEOUT_ZH.md`。切片检测仍归 first-cv；这里不开展同一实验。

## 1. 联邦增强：三项泛泛问题均已有直接重叠

| 被筛的问题 | 最强已知证据 | 未解决之处与本轮判断 |
|---|---|---|
| 共享增强会不会不适合异质客户端，需不需要个性化 | FedAvP §4.3 已比较共享与适配后的客户端策略、不同 label-skew 条件 | 尚无区别于“个性化改善异质性”的新预测；不保留 |
| 额外策略信息有没有额外隐私泄漏 | FedAvP §4.4、Appendix A.1 将策略梯度加入重建攻击，Eq.13、Table2 | 这不是普遍隐私保证，也没有排除所有属性推断；但“策略也依赖数据”或增加一个攻击指标不是当前贡献。没有开展攻击或声称新风险 |
| 相同合成样本预算放到哪些客户端/类别更有效 | FedEAS §3.1、§4.1 直接提出 entropy-adaptive allocation，并同预算比较 Uniform、Missing-only | 若没有与该分配机制不同的可检验预测，只做新的分配消融不保留；没有下载生成器 |

一手来源：
- Hong, Yun, Jeon, Kim. [FedAvP: Augment Local Data via Shared Policy in Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/20563b8508ba42e1b688d922e926ee26-Paper-Conference.pdf), NeurIPS 2024，§4.3–4.4、Appendix A.1 pp13–15。
- Lee, Park, Lee. [WHERE to Generate Matters: Budget-Aware Synthetic Augmentation for Label Skewed Federated Learning](https://arxiv.org/html/2607.06616v1), 2026-07-07，§3–4。它用共享的预训练生成器及一次生成缓存；本轮不把这视为免费材料。

固定代码 `alsdml/FedAvP@3f30be3bc561c064c28d164dca8cb5834cfdaef4`：
`fedavp_FastUpdate.py:135–177` 取本地标签直方图并更新客户端，201–202 聚合策略/模型；
`models/Faal_step_fast.py:80–193` 策略评分、增强样本、梯度内积和策略更新，196 起策略聚合。多处直接使用 CUDA。仅凭源码存在不能报价“Mac 免费几分钟”；尚未建立本地可运行的联邦轨迹。FedAvP 的原生适配链可定位，不称作者无代码。

综述 [2405.09839v2](https://arxiv.org/html/2405.09839v2) §V 仅用于定位异质性、通信/计算与隐私问题，不作为新颖性证明。

## 2. 卫星时间序列：三个直接重叠，保留未知边界

| 被筛的问题 | 最近邻已回答什么 | 还未知什么；为何当前不立题 |
|---|---|---|
| 学习 temporal dropout 比率并用多次遮挡估计不确定性 | Miranda, Mena, Dengel 的 MC-TD / MC-ConcTD 已在三个 EO 回归数据上研究此问题；§3.2–3.3 为 Bernoulli / Concrete 时间掩码 | 真实非随机缺失下的全面校准保证不是该文已经证明的；但泛泛“随机缺失不等于真实缺失”缺乏新增可证伪规律 |
| 用连续/季节性缺口替代独立丢帧 | TimeSenCLIP §4.1 明确 Random TSDrop、随机季度遮挡、median pooling 及谱/时间遮挡，§5.4.2 有消融 | 不能提“首次模拟季节缺口”。同缺失数但不同时间结构的严格匹配是否已完整覆盖，本轮未证明；没有据此直接保留“再加一个匹配实验” |
| 增强模拟缺失能否迁移到真实云缺失，并兼顾完整序列 | Joint_FRP §4.1–4.2 将完整/不完整样本分开，比较模拟及真实环境，比较 WS/TD，Appendix A–B 比较不同缺失水平 | 这不是所有真实缺失分布都解决了。当前尚未提出超出既有缺失模式/信息减少解释的具体增量；作者实现未发布只是复现材料缺口，不是科学否定 |

一手来源：
- [An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks, 2504.06915v1](https://arxiv.org/html/2504.06915v1)，2025-04-09；§3、§4.1。正式出处 IDA 2025 pp389–402，由官方 README 交叉核验。
- Wang 等 [A Joint Learning Framework with Feature Reconstruction and Prediction for Incomplete Satellite Image Time Series in Agricultural Semantic Segmentation, 2505.19159v1](https://arxiv.org/html/2505.19159v1)，2025-05-25；§4.1 L168–182、§4.2、Appendix A–B。网页自动生成的 journal 字段不作为已发表期刊证据。
- Jain 等 [TimeSenCLIP, 2508.11919v1](https://arxiv.org/html/2508.11919v1)，2025-08-16；§4.1 L174–195、§5.4.2。本文的具体设置来自一手正文，不采用第三方摘要里不同的混合概率概括。

固定代码和数据门：
- `mmiranda-l/Temporal-Dropout@ed11f677fe0f44e11c753e178e0afa84d28adb4b`，2026-05-13。完整树可读，读取 README 与 `src/models/single/temporal_dropout.py`。该模块对输入时间轴用 Dropout1d，在 training 模式下遮挡；默认取消普通 dropout 的幅值缩放。**没有追完 Concrete 与 MC inference 调用链，不声称复现论文。**
- 树列出 `data/lfmc.nc`，1,839,451 bytes；只看元数据，未下载或读取。README 链接 LFMC、Swiss yield、UCI PM2.5 原始来源。可用真实云 mask、独立地块/年份分组、原始使用条件尚未核实；不能把 1.8MB 文件的存在等同于可做真实缺失研究。
- `wangyuze-csu/Joint_FRP@9b0d168c794463cf953736ca4f1785a56c716c01`，2025-05-23；完整树 truncated=false，仅 README，说明接受后发布 code/data。这是当前发布状态，不是工具访问拒绝。

## 3. 视觉控制：已进入独立家族，当前仍无合格候选

**排除的泛泛问题一：逐帧随机扰动与时序连贯扰动对控制是否不同。**
Nath 等 [Augmentations for Robust and Efficient Imitation Learning in Streamed Video Games, 2607.14200v1](https://arxiv.org/html/2607.14200v1)，2026-07-15 / CoG 2026，已在 §III–IV 定义 frame-wise 标准增强及 chunk-wise 流媒体增强。后者含移动像素块、条带、模糊和递归 ghosting，并有平滑时间调制；§VI-B/D 比较两类增强及组合。仅增加 temporal coherence 不能构成本项目新问题。

**排除的泛泛问题二：把真实延迟当普通图像损伤。**
[Delayed Reinforcement Learning by Imitation](https://proceedings.mlr.press/v162/liotet22a.html)，Liotet, Maran, Bisi, Restelli，ICML 2022，已从无延迟示范学习延迟策略。本轮只读其一手摘要/定位，不宣称完成延迟控制文献审计。观测晚到、动作晚执行和同一时间戳上的视觉损伤不同；发现这种区分本身不是新贡献。

仍未知而**不保留**：真实编码器的内容依赖损伤与动作关键时刻之间，是否有独立于严重度、运动大小、观测年龄和一般分布偏移的增量。本轮定向检索未找到足以证明新颖性或效应的证据；“同边际换联合分配”不会自动隔离机制，不能把一般相关性实验包装成新题。没有为此创建训练协议。

材料门：
- 官方仓库 `microsoft/temporal_streaming_augmentations_for_imitation_learning@2c8d64038f089fb1e8ce70ed58481f0b1b0bbdb0`（2026-07-29），完整树可读；包含增强库、README、演示资源。此固定树未含 PIDM 训练管线、30条/任务示范集或完整 rollout 标注，不能当作已具备论文复现材料。
- 本轮只读 README 与树，未审计增强函数实现，不把论文公式标成代码验证。未下载 demo notebook、视频、Theia 权重或训练数据。
- [视觉 RL 增强综述 2210.04561v4](https://arxiv.org/html/2210.04561v4) §6 用于定位。版本记录为2024-10-21，HTML页内另显示2026-08-24，不能误记为已核实的2025新综述。没有凭综述开放问题直接立题。
- NeurIPS2023 Learning Better with Less 与一些2025–26检索条目仅作定位，没有完成其方法审计，不加入“已经排除所有近邻”的清单。

## 可执行性与继续条件

本轮没有形成够具体的新假设，故训练成本**未估算、未批准、未发生**；没有把旧 CIFAR 检查点或已消费 dev 当成新任务的免费确认材料。可见小源码和小数据元信息并不解决标签、真实条件、许可和独立评价单位。

下一次有候选时，必须同时交付：一个与最近邻不同的预测；现实条件与独立标签；明确可实现的比较和哪些因素无法匹配；观察到什么即说明只是已知现象；包括数据准备及基线在内的成本。不要求新网络，也不以“存在前作”自动关闭尚未问清的具体问题。

本轮止于完成这些有界筛选记录，不宣称 Rand 研究已经结束或领域不存在贡献。沒有待处理权限拒绝；真正未满足的是具体增量和部分材料条件。

