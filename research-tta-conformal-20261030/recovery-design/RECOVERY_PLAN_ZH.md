# 单次基础设施恢复方案（待主线审定；未分配GPU）

此方案只修改执行可靠性和累计计算审计计数。原 `PROTOCOL_ZH.md`、`protocol-lock.json`、首轮全部失败账本及其哈希保留。科学合理性已获主线转述的Pro认可；本文件不把它等同于新的GPU分配授权。当前只进行了只读诊断与本地设计。

## 1. 故障链：哪些已证实，哪些仍未知

实际命令不是 `colab run`，而是 `colab new --gpu T4` → `upload` → `colab exec --file launch_remote.py --timeout120`。launcher通过 `subprocess.Popen(...,start_new_session=True)`启动独立训练进程，约1.958秒后CLI退出0，仅证明启动提交成功。没有取到训练子进程的退出码。把PID或launcher的exit0当成训练成功是不成立的。

| 阶段 | 实际证据 | 结论 |
|---|---|---|
| 分配 | 17:52:19 UTC开始；成功T4，1.15CU/h | 分配和付费方式真实有效 |
| 启动超时 | exec120秒，外层CLI180秒；实际1.958秒后exit0 | 120秒不是24.37分钟失去会话的原因 |
| 状态查询 | 18:02:39 exec45秒；最终由外层180秒截止，exit124 | 一次客户端/传输读取失败；现有回执不能定位在kernel启动、prelude或payload的哪一层等待 |
| 远端继续执行 | 该超时结束后约643秒，仍有epoch75真实训练日志 | 不能把这次状态超时说成训练立即终止 |
| 会话消失 | 18:16:40 usage=0assignment/0rate；18:17:12文件下载404；18:17:27sessions为空 | 确认远端分配丢失，并非单纯本机进程退出 |
| 本地清理记录 | 18:17:28唯一session_terminated事件reason=pruned | 是清理失效本地记录，不是stop请求的user_requested |
| 最后训练进度 | 18:16:23记录epoch75、23475正式steps | 已知下界；之后确切步数未知，不能填成精确总量 |

原远端110分钟、local115分钟、总120分钟限制都不匹配24.37分钟。已安装官方0.7.5中，`exec`退出时默认关闭客户端channel，`runtime.stop(shutdown_kernel=False)`不主动shutdown；`prune_session`只删除本地记录，不unassign。因此目前没有本任务主动结束VM的证据，也没有提供商终止原因或训练进程退出状态。根因仍未知。

官方 `run` 默认timeout30秒；`--keep`控制结束后是否保留VM；不加keep时，正常或传输失败都可能进入finally清理。这些是代码审计事实，**不能拿未使用的run默认值解释本次故障**。安装README笼统说可取回outputs，但实际run.py没有自动下载任意输出文件的过程。以已安装代码和help为准。

官方FAQ说明VM会受动态空闲/寿命限制影响，没有公布固定24分钟寿命；也没有提供本次终止的原因。因此“后台进程与kernel执行状态脱节”是设计风险，不是已证明的根因。恢复方案改为真正的前台计算调用，不加入伪心跳、绕过限制或反复换账户。[官方FAQ](https://research.google.com/colaboratory/faq.html)

诊断读取了本会话官方history、已归档CLI回执和已安装源码。没有打开凭据文件、sessions token文件或中央colab.log：源码显示中央debug log会写完整HTTP响应体，可能包含token，故不为查原因而扩大秘密读取范围。详见 `FAULT_DIAGNOSIS.json`。

## 2. 可恢复产物与“没有中途checkpoint”的责任

官方CLI的download只读当前VM文件系统；本次没有挂载Drive、没有云对象仓库、没有notebook输出文件、没有自动持久化服务。启动脚本是.py，不触发exec针对.ipynb的 `_output.ipynb` 写回路径。当前本地没有任何.pt或正式logits.npy，只有完整过程日志、协议、来源和已核验的CPU重建数据清单。CLI里未找到已消失VM的恢复/undelete输出接口。不继续尝试内部服务或新授权。

这是我的持久化实现缺口：旧脚本只在80epoch结束时调用一次torch.save，之后才推理、打包、统一下载；checkpoint还只有模型state_dict及元数据，没有optimizer/RNG等恢复状态。“最后epoch作为科学模型”并不禁止中途恢复快照，我不应把二者混在一起。一次后台运行时丢失因此造成75epoch已知计算无法继续使用。

在新分配前，如果出现可独立验证的首轮完整80epoch模型+全量logits，就取消重试。仅有进程/PID、零退出码、文件名或部分模型不满足取消条件。若在指定重试已经启动后才取回首轮完整结果，预定科学分析只用 `attempt02_designated` 的完整轨迹；不按准确率/集合大小/门结果挑选，不合并为独立seed重复。首轮保留为基础设施与计算账本。

## 3. 完全不变的科学内容

沿用原协议SHA256 `28a8f92ca9fdd76cbcebe4998e9ff55f259f5e774a29dca190232f2eb5587f0b` 及全部源文件锁。CIFARResNet20 option-A、seed0、splitseed20261030、重复组原子分配40k/4k/6k/10k、训练归一化、80epoch/313步每epoch/25,040步、SGD/CE/cosine/增强、末epoch模型、十视图和FP32 logits完全不变。U种子、7点温度网格、四臂、全部评分、tuning选择、alpha主敏感性、Dhat、999次paired group bootstrap、同时CI族、A/B及负控门不变。

不得以基础设施故障为由缩epoch、改seed、改推理batch/view order、改参数或删失败臂。新增的checkpoint与验收不读取eval来选模型。保存所有中途快照只是恢复机制，科学模型仍唯一是指定轨迹epoch80。

T4/PyTorch2.11.0+cu130/CUDA13/cuDNN92700/NumPy2.1.3/Python3.13.16等执行环境在分配后复核；恢复时与本次尝试自身的环境清单一致。不同环境不能凭相同seed声称逐位相等。PyTorch官方也不保证跨版本/平台完全可复现。[PyTorch2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html)

## 4. 官方CLI执行与本机持久化流程

采用已验证支持的 `new → usage/status → exec → download → stop → sessions/usage`。`exec`中直接执行实际训练块，不用Popen/daemon/nohup。每块5epochs，共16块。建议train块 `--timeout600`、本机进程上限660秒；全局累计预算watchdog独立存在，绝不以单块timeout重置总预算。一次exit0不足以晋级；必须取得任务自己的完成receipt和本机可读的完整checkpoint。

1. 只新增一次T4分配。分配后先核GPU/实际rate/余额；同原协议，费率>3CU/h或预算不够则释放。无新OAuth、无Drive mount、无新持久凭据。不购买、不改云权限。
2. 下载同一官方数据包，核MD5/SHA；复用固定角色ID与图像hash，逐项对上首轮已恢复的清单。初始状态epoch0先持久化本机。
3. 通过第5节200update以内的恢复/测速门后，销毁测速权重，重置seed0，开始完整80epochs。
4. 每到epoch5、10、…、80，远端CUDA synchronize，保存恢复快照到临时文件，flush/fsync后rename为不可变正式文件；计算SHA256与字节数，最后原子发布receipt。命名含run_id、epoch、global_step和内容hash，禁止覆盖旧快照。
5. 控制端通过官方download取receipt、再取checkpoint到本机 `.part`，核字节数/SHA，反序列化核结构与身份，关闭后重新打开再hash；本机fsync并原子rename，追加本地承诺记录。**只有文件已离开Colab临时盘并读回一致才允许下一训练块**。失败不把“远端已有文件”当持久化完成。
6. 当前正式恢复只承诺epoch边界；不任意跳到worker批次位置。现实现没有DataLoader worker/prefetch，GPU tensor管道不引入worker游标；仍保存epoch/batch_cursor/permutation字段，边界时cursor=0且无正在进行的batch。父线程若要求任意步恢复，须额外实现并验收，不能假称支持。
7. 若调用超时，先通过官方文件接口检查同一块的不可变receipt/checkpoint；完成且hash一致则采纳，正在运行则等待受总预算限制，状态不明不重复训练。read/download可以有一次有界重试；不重放未知是否已完成的训练块。若VM再次消失，本提案不再新分配第二次GPU；本机快照可供主线随后决定，失败开销继续入账。

每个完整checkpoint必须包括：

| 类别 | 必须保存并验证的内容 |
|---|---|
| 身份 | run_id=attempt02_designated、原科学协议SHA、amendment SHA、模型代码SHA、数据包SHA、split与图像manifest SHA、环境版本、模型seed |
| 模型 | 全部parameter与buffer state_dict，包括BN running_mean/var、num_batches_tracked；训练/eval模式 |
| 优化器 | SGD state_dict、momentum buffers、参数组与参数名映射、lr/momentum/wd/nesterov等配置 |
| scheduler | 原实现是逐step手工cosine，不伪造不存在的scheduler对象：保存implementation ID、base_lr=.1、Tmax=25040、next_global_step、current_lr和原公式；恢复后逐项核对 |
| RNG | Python random状态、NumPy全局RNG、独立Generator状态（若运行中存在）、Torch CPU RNG、所有CUDA RNG状态；初始化/加载完成后恢复，不能恢复后又调用随机初始化 |
| 进度 | completed_epoch、next_epoch、global_optimizer_step、正式/测速/累计呈现、batch_cursor、当前epoch排列或明确的边界标记、数据顺序/augmentation计数 |
| 数值环境 | FP32/TF32关闭/cuDNN deterministic与benchmark设置；原来无AMP，不能新增GradScaler |
| 提交信息 | checkpoint字节数与SHA、各tensor/optimizer/RNG规范化内容hash、上一个已承诺checkpoint SHA、远端完成时间、本机下载与读回时间 |

PyTorch官方的恢复checkpoint说明要求模型之外保存optimizer与训练进度；本方案另外补齐随机性、数据身份与本机提交记录。[PyTorch checkpoint教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## 5. 固定列账的短恢复测试与性能门

新增尝试的全部测速/验收训练物理updates **最多200**、呈现最多25,600；不另加隐藏测试：从seed固定的测速模型先训练100步，保存C100并下载到本机读回；继续50步得到参考C150；把本机C100通过官方upload回到本任务VM，重建模型/optimizer，恢复scheduler/RNG/排列游标，重放同样50步得到C150'。共100+50+50=200次真实更新，全部计费入账。比较下一步及整个50步的loss、参数、momentum、scheduler、RNG、下一batch身份/增强随机状态；同环境要求逐位一致，否则不正式训练。这个测试验证序列化/恢复状态完整，正式断点仍固定epoch边界。

完整十视图测速仍512张×10=5,120views，使用测速模型；同时把一个512图logit块下载本机，验证ID/view/dtype/值hash、均值与中心view；测速产物只作验收，不进入正式统计。测试model/optimizer丢弃，正式reset seed0。

**原性能门仍是累计预测<=90 GPU分钟，包含首轮已用24.3703分钟，不重置为第二轮90分钟。** 重新测训练50步段（包含同步/增强/搬运）、完整推理、checkpoint写出+下载+读回、本机/CLI调用等待。预测中分别加总真实GPU计算及控制/IO等待，不能漏算或把每个5epoch调用等待错误乘到每一步。

规划计算：沿用首轮最慢段1.5倍保守值及其完整新run预测60.7247分钟，再预留17个checkpoint各10秒、16个前台块各2秒，总增加202秒，第二轮约64.0913分钟；加首轮24.3703=**88.4616分钟**。这些IO数字只是审批用规划值，未实测；新分配后的实际门必须替换为测得的保守值。约1.54分钟门内余量不算宽，若新环境慢或下载更慢使累计预测>90，就释放并报告资源门失败，不悄悄放松。

原预测240秒取回/释放余量必须覆盖40个推理块传输与释放；用512图块实测估计，超过240秒则向上增加预测。不要将分块传输额外成本漏掉。累计硬上限仍120分钟，计算/释放后备按累计GPU时间（110/115分钟）而非新会话重置；第二次会话最多约90分钟，保留总上限余量。

## 6. logits分块：完成一块即成为本机可恢复产物

正式20,000图像按固定heldout ID顺序分成512图/块，共40块（最后32图）。每块仍用原128图推理batch和同一十view顺序，保持计算分批不变。每块约2.05MB未压缩FP32 logits，直接用官方download；CLI本身没有文件字节级resume参数，不编造该选项。

块文件与manifest必须含/绑定：run_id、epoch80 checkpoint SHA、原协议/amendment/model code SHA、数据与split hash、绝对row start/end、imageID、官方source/index、每行role、viewIDs顺序0..9及具体crop/flip政策hash、classIDs0..99、shape[B,10,100]、dtype float32/endianness、字节数、完整文件SHA与logits原始字节SHA。混合role的块逐行带role，不靠文件名猜划分。

远端以临时文件+fsync+atomic rename发布不可变块，最后发布完成manifest。控制端下载到本机.part，核文件hash并load核dtype、shape、ID、role、view/class顺序，重开读回后承诺。本机已承诺块永不重算；传输中断重下同一不可变块，不重新推理。远端状态未知时不重复前向；只在证实未计算且原调用结束后才执行缺块。

本机拥有epoch80完整checkpoint、40块完整身份校验后，直接在本机按ID拼装FP32 cache、float64均值，做原有中心图/均值/顺序/无损检查，再停止GPU核sessions/usage。所有统计都只读本机已承诺产物，不依赖原runtime仍活着。任一ID缺失/重复/错role/view/dtype/hash均不进入调参、cal或eval。

## 7. 单次恢复需要审定的累计计数

首轮未知尾段按可能完整80epoch记最坏账，不从账本排除。

| 项目 | 首轮已知下界 | 首轮保守上界 | 指定重试完整量 | 两轮拟定上界 |
|---|---:|---:|---:|---:|
| 正式optimizer steps | 23,475 | 25,040 | 25,040 | **50,080** |
| 正式呈现 | 3,000,000 | 3,200,000 | 3,200,000 | **6,400,000** |
| 测速/恢复验收updates | 200 | 200 | <=200 | **<=400** |
| 测速/验收呈现 | 25,600 | 25,600 | <=25,600 | **<=51,200** |
| 全部训练updates | 23,675 | 25,240 | <=25,240 | **<=50,480** |
| 全部训练呈现 | 3,025,600 | 3,225,600 | <=3,225,600 | **<=6,451,200** |
| 全部推理views | 5,120 | 205,120（按冻结代码完整调用量） | <=205,120 | **<=410,240** |

首轮推理尾段没有完整记录；不能未经证实把它记0。因此除了训练审计计数，也明确提请把**累计推理审计上限**修订为410,240。若只采用“首轮仅测速”的已知下界，两轮210,240views虽低于原220,000，但这不是对未知尾段的保守记账。每轮正式科学评估仍只有20k×10view，网格、视图和样本不增加。

已用GPU上界24.3703分钟，原120分钟中剩95.6297分钟；已用约0.47CU，6CU中剩5.53CU。按1.15CU/h和上述新预测，累计约**1.6984CU**。实际费率仍须复核并遵守原<=3CU/h门。首次下载时钟不重置，硬截止是 **2026-10-30 23:53:27 UTC**，涵盖CPU统计和归档；如审定时间较晚，启动前重新评估剩余墙钟并为统计/归档保留至少60分钟，不靠延长deadline救题。

本地新增当前约343.63MiB（包含首轮工作目录+归档重复数据包）。新attempt02使用唯一持久目录，避免复制大快照两份；17个checkpoint按每个<=8MiB规划，共<=136MiB，40个logit块约76.3MiB，拼装cache/均值/metadata约100MiB。即使保留所有旧账本及临时传输余量，总量仍远低于2GiB；实测文件计数器超过上限就阻塞，不静默删除失败历史。

## 8. 审定边界

本方案仅准备**一次新增GPU分配、一次指定seed0重跑**。未经主线明确开放分配，不执行new/run/upload/exec训练。首轮完整结果若在开始前恢复则取消；开始后只用指定重试轨迹，不按表现选。

待审项为：累计正式50,080步/6,400,000呈现，固定额外验收<=400步/51,200呈现，累计推理保守410,240views，以及上述可靠性流程。原累计120GPUmin/6CU/6h/2GiB和90分钟性能门、全部科学门保持不变。再次基础设施失败仍保留进度与开销，不自动第三次分配。此文件不是已通过新GPU恢复验收的声明；新的短验收必须在获批分配后实际执行并留证。
