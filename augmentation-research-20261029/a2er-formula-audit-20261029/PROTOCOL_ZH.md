# A2ER 公式与可执行性审计协议（CPU，不训练）

在数值执行前写定。已读主文 Eq.8–17、出版社补充 Algorithm 1–2，并取得作者仓库 `prinlab/A2ER_O2S@f1334df1104724bb3b56f1f353857d92b1141565`。DER++ 对照固定为 `aimagelab/mammoth@e75a491c69fd729edeb01431afb753d9157d9a81`。不能把未取得作者代码当阻塞，也不能把主文的未定义零分母直接指认为实现缺陷。

只运行 NumPy／torch CPU 标量、向量与缓冲区操作，无神经网络训练、优化器步骤、数据输入、权重、MPS 或 Colab。从检查过的原文件 AST 取核心类与三个纯函数；排除持久化／pickle 方法与整个训练入口，不安装作者包。测试使用固定索引选择，以分离计算顺序；不能把它当实际随机回放轨迹。

1. 主文 literal 分支：Delta 为半平方 L2 和；Q 为当前样本分位数的 EMA；计算 eta、gamma、corrected Delta、权重 EMA、target correction。分别保留 Q=0、Delta=0 的 NaN，禁止默加 epsilon 冒充主文。正值格点检验 Eq.14/15/17 数值恒等、边界 eta/gamma 与范围。
2. 官方实现分支：直接执行原 DERPP.__call__ 与 ReplayBuffer.update 等核心方法；eps 采用默认 float32 eps 和分类示例 1e-5 两种；量化差异，不把数值稳定版本说成主文严格相等。记录实际缓冲区长度、quantile-before-gamma、detach、ratio loss 和 target-before-priority 更新。源文件哈希不变。
3. 零残差／零初始 Q、正常初始 Q、常数残差、部分填充缓冲区；检查选中与未选中 entry 的 target 和 priority。保持初始状态选择显式。
4. 统一缩放：正值 Delta/Q 同比缩放 1e-12、1e-6、1、1e6、1e12；主文 eta/gamma 理想尺度不变与固定 eps 版本对照。只讨论该局部门，不能推论 alpha 动态或完整算法尺度不变。
5. 异质敏感性向量：24 个固定低残差与 8 个高残差条目；固定输出与正负交替输出两种构造，各 32 次固定访问。每步先修正 target 再进入下次，不能一边改 target 一边假装残差永不变。记录全部步；不称为增强损害的实证或新发现。
6. float16／float32／float64：对近零、阈值附近、较大幅值执行公式；保留 NaN、Inf、精度差异与原生不支持的 dtype。不给官方代码加 cast/floor 修复。
7. 一次 CPU 标量向量 autograd 核对作者 ratio-detached 损失的导数；没有 optimizer.step。区分该梯度与“先形成常量 corrected target 后重新计算 MSE”的梯度，按作者实际选择解释。
8. 三个科学干预因素必须拆开：alpha/beta 自适应、历史 target correction、replay sampling。另明确 gamma 同时影响当步蒸馏损失、eta 影响 alpha 更新；换 residual 是多路径改动。对 CAMA 的近邻结论不另做未经授权的新颖性宣称。

CPU 数值预算 120 秒；每个预定用例保存参数、状态与结果。无 NaN 用例删除或阳性筛选。完整训练可执行性、Random Erasing 在入库前／回放时的实际位置若未被发布源码覆盖，则标未验证，不填补作者缺失实现。

数据准备独立进行：只取 Toronto 官方二进制 CIFAR-100，官方 MD5 `03b5dce01913d631647c71ecec9e9cb8`，先校验再白名单提取 TRAIN 与类别名称。仅用 TRAIN 行号和标签，按每类 SHA256 固定排序分为 400 fit／100 development，不读取测试成员、测试结果或旧封存测试。
