本机CPU训练有条件可行：在既有官方环境中完成了一个原support样本的完整forward/backward/AdamW step，未发现强制CUDA训练算子。不启动正式两臂。

固定image_id2244（seed20261031原sampler首项），原M配方未改；CPU4线程/interop1/OpenCV顺序、FP32、确定性算法。模型172,912,413个参数/896参数张量全部可训练，冻结参数0；可训练名字hash与历史A一致。只以weights_only=True加载原官方权重，并严格检查已有官方missing/unexpected契约和共有张量字节；无unsafe fallback。

实测：forward+loss 1.029s，backward 2.628s，optimizer 0.222s；完整训练步 3.974s（含梯度检查和阶段记录），峰值RSS 3.971GiB。896个梯度张量与loss各项均有限，896份optimizer状态步数均为1。成功子进程6.829s退出码0，探针权重/optimizer全部丢弃，未保存checkpoint。

重要边界：该固定样本经原随机裁剪后GT=0，Mosaic/MixUp尚处空缓存预热。因此正GT分支、满缓存、最大尺寸和长期内存未实测，不宣称所有输入分支均已运行验证。静态源码显示deformable attention使用官方CPU的PyTorch grid_sample分支；focal loss有官方PyTorch CPU分支，GIoU/bbox运算为PyTorch，Hungarian匹配使用CPU SciPy。已触发的整条autograd和optimizer路径通过；未修改算子或引入fork。来源及文件hash见STATIC_CPU_EVIDENCE.json。

同CPU重跑A/M各U200的400步，单步线性参考约26.5分钟训练。规划按2–3倍步时约53.0–79.5分钟，加两模型各64dev前向约4.7分钟及启动/完整落盘/hash/日志120秒，总计约60–86分钟。此范围是粗估而非实测保证；不能把3.97秒当稳态或p90。内存实测3.97GiB，计划预留6–8GiB RSS但上界未证实，不查询被拒绝的物理内存信息。两个完整checkpoint约1.384GB，应另计目录余量。

最少控制：A/M均从同一原始资产在同一CPU环境独立初始化并核完整state；同seed20261031、trainable/AdamW/LR、U200、batch1、同anchor顺序；原配方和缓存语义不变。固定最终checkpoint，不按dev/loss选优；全64dev原色和主AP@100沿用既有CPU指标。必须重跑CPU A；不能把CPU训练M与CUDA训练A的差异归因增强。探针状态不能复用。无现成完整MPS backward证据，本次不转MPS。

透明记录：先出现一次optimizer容器类型错误（未取图），再出现一次模型ConfigDict属性访问错误（同图部分forward，未backward）；仅修正官方Config容器调用，科学参数不变，记录在attempt1/attempt2及修复JSON。随后同一预定图完成唯一一个训练step；总计3次模型构造、2次同图取样，完整step仅1。总10分钟截止从初次检查起未重置，未新增样本或继续长跑。

dev/test效用读取、正式两臂训练、安装、GPU/MPS/Colab、上传和新增CU均0。累计GPU账本仍1.8207069960365825CU上界；本次只交CPU可行性证据，正式两臂尚无GO。
