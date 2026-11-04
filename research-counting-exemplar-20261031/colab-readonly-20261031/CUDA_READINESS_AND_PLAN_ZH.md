# Colab只读资源检查与未执行的CUDA备选

现有官方google-colab-cli 0.7.5只读检查成功：余额40.79 compute units，消耗率0.00 CU/小时，服务器会话0，活跃分配0。现有授权发生1次access-token续期；新OAuth分支在检查进程内被禁止，未读取或输出token。没有新GPU/VM、远端exec、安装、购买或训练。

当前没有存活会话，因而无法检查“当前远端”Python、Torch、Torchvision、CUDA、权重缓存或GPU具体可分配性。未为弥补此信息而创建会话。CLI帮助支持选择T4/L4/G4/H100/A100，这只是参数能力，不保证当前容量或权益。历史2026-10-30独立任务记录过T4、Python3.13.16、torch2.11.0+cu130、CUDA13.0、cuDNN92700，费率1.15 CU/小时；该会话已释放，这些只能作历史参考，不能写成当前运行时或当前报价。

## 依赖可用性

复用此前保存的官方PyPI元数据，确认本地CPU准备阶段新增的12个固定版本存在Linux x86_64/Python3.13可选的pure-Python或abi3/cp313 wheels：transformers4.49.0、tokenizers0.21.4、timm1.0.30、yapf0.40.1、addict2.4.0、termcolor3.3.0、opencv-python-headless5.0.0.93、pycocotools2.0.11及importlib-metadata/platformdirs/tomli/zipp。未重复下载或安装。具体wheel名、官方URL和sha见dependency_candidates.json。此项证明注册表候选存在，不证明一个尚未分配的远端环境依赖已满足。

若以后明确授权，先记录实际Python/Torch/Torchvision/CUDA版本与torchvision compiled ops状态，只补缺失依赖到独立环境，保留已有GPU配套Torch，不盲目重装。若现有Python/ABI不相容或Torch/Torchvision组合无效则报告，不安装非官方包或编译自定义CUDA op。

HF固定源码本身在_C不存在时走官方纯PyTorch deformable attention；该分支支持CUDA tensor，不必开发新attention方法。应确认新隔离环境自然没有MultiScaleDeformableAttention扩展，且_C为None；若环境意外加载扩展，停止复核，不能静默改变量来掩盖路径改变。ROIAlign仍调用配套官方torchvision原生CUDA算子，需实际预检确认，不能凭本地MPS返回即宣布CUDA可运行。

固定CountGD++/BERT safe权重、完整键与共享别名strict检查、源码hash、三正三负提示和原生阈值/预处理均沿用。remote code自动加载、pickle、AMP、TF32低精度加速和算法修改均不属于建议路径；若采用CUDA FP32参考，应在预检前明确记录关闭CUDA TF32的数值设置，不改网络。

## 最小完整三模式CUDA预检提案（未授权、未执行）

只申请一次明确指定型号的运行时，不做型号轮换或自动重试。仍用同一人工1333×800输入和六框：CPU F/C/V各1次，CUDA F热身1次，再CUDA F/C/V各1次，最多7次尝试；外部180秒、RSS8 GiB和CUDA分配8 GiB护栏，失败立即停止。CPU参考必须在同一远端软件环境内重建，不能用本地Mac输出冒充CPU参考。GPU计时synchronize并包含输入输出传输。

加入此次已验证的输入hash、CPU clone、迁移hash/共享关系检查和原生top-k索引记录。保持原最终逐行容差与筛选mask门；近排序反转仍可能使CUDA检查失败，不能因换设备便取消这项门。未取得新授权时，不切换人工图或用排列匹配放行。仅保留工程证据、取回后释放运行时并用sessions/usage确认释放；不会自动开始492queries。

本次诊断已经提供了proposal顺序差异的具体证据，建议先独审该证据，避免把一次新CUDA分配当成必然能解决比较定义的办法。

## 剩余30分钟预算条件

当前总预算余额为1745.084214583秒。设新CUDA预检实耗P、未来环境准备/正式加载及热身S、三模式最大合格端到端延迟L、校验写盘余量R，则须满足：

`54.915785417 + P + S + 2×492×L + R ≤ 1800`。

R预留60秒，是明确规划余量，不是实测上界；P按实际所有CPU参考和CUDA计算wall扣除。若为预检预留完整180秒，则即使S=0，也要求L≤1.52956秒；若S=60秒，要求L≤1.46858秒。当前没有合格CUDA样本，不能声称达到这些条件。环境安装、启动或GPU空闲占用需单独记录；未被明确排除的计算时间保守计入S，不能把CPU参考算作免费。

资产传输按原单独资源账本记账，不伪装成计算耗时，但Colab已分配时传输/等待仍可能消费CU。本轮新增范围仅只读，40.79余额不会触发分配；实际CU率须在任何未来分配后读取，历史1.15/hr不能当当前报价。第一次模型资产复制约1.38GB，加现有约1.53GB取得量仍在原3.5GiB下载上限内；若预检后释放再为正式观察重复取得全部权重，可能超过原上限，须提前安排授权内资产复用或报告约束，不能自行扩大下载预算。

证据目录：colab-readonly-20261031/的preflight.json、sessions.json、usage.json、帮助和带OAuth禁止分支的只读脚本。当前阻塞是没有现存CUDA环境可核实且模型工程数值门未通过；没有待完成的GPU分配或不明状态远端作业。
