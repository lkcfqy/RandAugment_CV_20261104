# 固定Mammoth DER++基础学习诊断（独立新协议）

2026-10-29 UTC，结果前冻结。目标是核验一套有官方preset依据的基础学习配置；不是旧760步协议续跑、不是A2ER复现或候选贡献验证。只用一个seed=20261029，不根据结果换seed、超参数、checkpoint或延长epoch。绝不操作重置卡。

## 原始出处与实现边界

固定官方版本 `aimagelab/mammoth@e75a491c69fd729edeb01431afb753d9157d9a81`：`models/config/derpp.yaml`、`datasets/configs/seq-cifar100/xder.yaml`、`datasets/seq_cifar100.py`、`models/derpp.py`、`utils/buffer.py`、`backbone/ResNetBlock.py`、`backbone/__init__.py`、`models/utils/continual_model.py`、`utils/training.py`、`utils/args.py`。保存原文件、Git blob与SHA-256。依据是这个固定官方仓库preset；不冒称逐项恢复了2020年原论文所有历史运行条件。

不安装整个Mammoth依赖栈。由已审计AST直接使用官方网络类、backbone初始化/to方法、`Derpp.observe`原函数体和`ReservoirSampling`；本地buffer适配器提供同样的500槽位uniform sampling/逐观察reservoir、原图/标签/历史logit存储和审计metadata。使用真实torchvision变换，独立校验字节/float→PIL roundtrip，保留版本差异。这是按固定配方实现的小子集基线，不是完整框架逐位复现。

## 固定配置

- 模型：CIFAR 3×3 stem的BN ResNet18，基础宽64，四stage各2个BasicBlock，10维head。无预训练。预期11,173,962参数、20个BN。float32/MPS，不启用CPU fallback或fast math。
- SGD，LR=.03、momentum=0、weight_decay=0、nesterov=False（官方args默认）；每个task重建optimizer和scheduler。每task50 epoch，epoch35/45结束后LR乘.2，即epoch1–35=.03、36–45=.006、46–50=.0012。
- 当前batch32，最后不足batch不丢弃；两个独立uniform无放回replay batch各最多32，可彼此交叠。buffer500，从空开始，跨两个task保持；每次训练观察都参与reservoir，包括同一原图在不同epoch的再次出现，允许buffer包含同一source的不同观察。
- 损失：当前CE + .3×全logit元素mean MSE + .5×独立监督回放CE。task1第二个batch起也有回放。
- 严格保留官方observe顺序：当前forward→蒸馏回放forward→监督回放forward→合计backward→SGD step→把未增强原图和**本步更新前当前增强视图的logit**写入reservoir。三个forward不合批，BN状态照此更新。target不永久修正、不做残差降权或alpha/beta自适应。
- 真实torchvision：当前uint8→PIL，replay float32原图→PIL，RandomCrop(32,padding=4,fill=0)、RandomHorizontalFlip(.5)、ToTensor、Normalize(mean=.5071/.4867/.4408,std=.2675/.2565/.2761)。dev仅ToTensor/Normalize。

## 两任务适配与原基准差异

| 项目 | 固定官方seq-cifar100 | 本次明确适配 |
|---|---|---|
| 类别 | CIFAR100全部100类，通常10task×10类 | fine IDs0..9；task1=0..4、task2=5..9，固定自然顺序 |
| head | 100 | 10，class-incremental，不使用task mask |
| 训练/评估数据 | 官方TRAIN训练、benchmark test评估 | 只读已校验TRAIN；旧SHA256分层划分，每类400 fit/100 dev；不读任何test |
| 训练量 | 50 epoch/task | 同样50；每task2000 fit，63step/epoch，共3150+3150=6300step |
| buffer/model/超参 | preset500、宽64 BN ResNet18、α=.3 β=.5 LR=.03 | 保持；不沿用旧512 snapshot buffer或小GN模型 |
| 样本顺序/RNG | 原框架dataloader及变换随机源 | 每epoch预定numpy permutation，torchvision使用固定torch随机流；唯一seed，保存计划及epoch RNG状态 |
| 平台 | 原始论文具体机器与版本未复原 | 现有Mac MPS、Python3.14.10/torch2.13.0/torchvision0.28.0，明确不是硬件逐位复现 |

任务1名称apple/aquarium_fish/baby/bear/beaver；任务2 bed/bee/beetle/bicycle/bottle。每task各2000 fit、500 dev。计划总当前观察200,000；除首batch外两次replay共6299×64=403,136，合计模型图像观察603,136。不选择样本难度，不使用dev选型。

## 执行、资源与记录

已有资源检查：64GiB，95%空闲、swap0、磁盘可用869GiB；未见其他活跃Python训练。单进程顺序执行。首32个真实训练step作为吞吐检查，保留更新；先做无optimizer的合成CPU/MPS数值与BN状态检查。每步记录独立optimizer hook、同步后耗时、各loss、LR、当前/replay source IDs和reservoir接受事件；每epoch记录曝光、BN计数、buffer组成、CPU/MPS内存。

使用外层子进程时钟计完整启动到退出；内部阶段和每step均在`torch.mps.synchronize()`之后计时。数据读取、CPU变换、buffer写入、评估和保存按边界注明。最多90分钟执行；若预测超预算、非有限值、内存压力或其他重Python训练出现而不能安全并行，则在安全边界保存checkpoint并记录未完成，不能把部分运行称6300step。不得杀其他任务进程。

只评估初始化旧类dev、task1结束的旧类dev、task2结束的旧/新类dev和合并10类；保存逐例logit/NLL/错误及每类指标。固定任务终点，不用中途dev挑checkpoint，不自动追加训练。沿用旧操作性基础学习门：task1旧类accuracy≥.50、NLL≤1.5且比初始化降低≥.30；task2旧/新各accuracy≥.35且NLL≤1.8。保留所有具体指标；达到这些弱健康门不等于科研训练充分或贡献成立。即使task1门不通过也按固定50+50完成，除非执行安全/有效性失败。

保存task1和task2终点模型、optimizer/scheduler、BN状态、reservoir/历史targets、随机状态、完整日志和源校验。旧760步结果与门槛不改动；新结果是独立基础诊断。**门A与五分支未获本协议许可，绝不运行。** 不用云GPU、付费、test、重置卡、commit/push。
