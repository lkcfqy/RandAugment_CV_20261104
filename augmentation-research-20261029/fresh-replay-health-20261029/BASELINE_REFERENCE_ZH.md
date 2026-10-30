# 一套固定公开DER++基线，供父任务冻结下一协议

这是未执行的基线提案，不是把旧结果改成成功。来源固定为 `aimagelab/mammoth@e75a491c69fd729edeb01431afb753d9157d9a81`，实际文件、Git blob及SHA-256保存在 `sources/`。采用现成preset，未按本轮开发结果搜索超参数。

| 配置 | 公开值与出处 |
|---|---|
| 数据协议 | seq-cifar100；原公开任务是100类、每task10类。小范围适配需另声明 |
| 模型 | CIFAR ResNet18，四stage各2个BasicBlock，基础宽64，标准BatchNorm；`backbone/ResNetBlock.py:246`默认num_filters=64 |
| buffer | 500；`models/config/derpp.yaml:3` |
| alpha、beta | .3、.5；同文件:4–5 |
| SGD LR | .03；同文件:6 |
| epoch/task | 50；`datasets/configs/seq-cifar100/xder.yaml:5` |
| batch | 32；同文件:2；建议明确replay minibatch也32，避免隐式默认 |
| scheduler | MultiStepLR，epoch35、45乘.2；同文件:1、3–4 |
| optimizer默认 | SGD，momentum=0、weight_decay=0、nesterov=False；`utils/args.py:311–319`。不要凭习惯填成.9或5e-4并冒称官方preset |
| 增强 | padding4随机裁剪、p=.5水平翻转、CIFAR100标准归一化；前轮保存的seq_cifar100.py |
| 回放／target | 原图存储、回放fresh transform、logit MSE加独立监督回放；当前输出入库，标准reservoir成员策略；不加残差降权或永久target correction作为基础健康对照 |

链接：[DER++ preset](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/config/derpp.yaml)、[50 epoch与scheduler](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/datasets/configs/seq-cifar100/xder.yaml)、[优化器默认](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/utils/args.py)、[网络](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/backbone/ResNetBlock.py)、[DER++执行顺序](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/derpp.py)。

建议下一步先建立这套普通DER++的学习基线，再决定是否值得重新进入采样身份问题。若仍使用当前两任务各5类、每类400 fit/100 dev，则是明确的小子集适配：head=10，每task2000 fit；保持50 epoch，每task `ceil(2000/32)*50=3150`步，总**6300步**。它已超过旧1000步前缀上限，必须由父任务另冻范围，当前没有执行。

buffer512代替500、GN代替BN、snapshot target代替逐次入库，都需要在新协议中显式说明，不能称完全按官方preset。本提案首先给出一套可核对的正常原始基线，不同时提出多个可选LR或模型去试结果。若后续科学干预要求GN，应把BN→GN作为明确适配并重新完成基础学习验收；本轮没有私自合并改动。

## 成本估算，非实测承诺

宽64 ResNet18、10维head的解析参数量约**11,173,962**；卷积/线性前向MAC约555,422,720/图，是本次小网络的**20.84倍**。仅按MAC比例缩放小网络实测task1暖态步时.007016秒、task2阶段均步时.016230秒，6300步约**1526秒（25.4分钟）纯训练估计**。

这个线性模型未测真实宽64吞吐，不覆盖BN/GN差异、算子利用率、单独三次forward与合批差异、评估、数据处理和温控，不能作为保证。用于排期可预留**30–90分钟**；下一协议若获准，应先做固定短吞吐验收再决定是否适合Mac预算，不能凭估计直接用Colab。此提案没有启动新训练或下载模型。
