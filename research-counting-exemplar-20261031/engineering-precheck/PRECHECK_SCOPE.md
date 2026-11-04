# 已授权的一次人工工程预检

执行顺序固定为 CPU F、CPU C、CPU V、MPS F热身、MPS F、MPS C、MPS V。最多7次尝试，失败占用一次且立即停止，不自动重试。外部wall上限180秒；RSS和MPS分配各8 GiB护栏，整个预检实耗从后续总1800秒计算预算扣除。正式492queries未获启动授权。

固定输入为1333×800 RGB常色(100,110,120)，两组各3个人工框；F为red toy/blue toy，C为toy/空文字，V为两侧空文字。三模式负文本均为单元素列表，负视觉组始终存在。种子42，正式原生预处理短边800/max1333，预期实际张量[1,3,800,1333]。无真实图像、类别真值或计数。进程审计钩子拒绝打开数据集、标注及图像文件。

CPU完整strict加载沿用已经审查的1042物理tensor+66实际共享别名逻辑，全部1108逻辑键逐项核shape/dtype/有限性及加载后精确值。只读取safe权重，禁用网络、pickle和MPS CPU fallback；不修改官方算法/attention/ROIAlign/阈值。

数值条件在首次forward前固定：官方ContrastiveEmbed在无效token和补齐至256的槽位填-inf（utils.py:265–271）。因此单独报告full_logits_finite；原生padding必须精确为-inf、CPU/MPS填充结构一致，有效logits及全部pred_boxes必须有限，sigmoid后全体分数必须有限。有效原始logits和全部框逐元素abs(MPS−CPU) <= 1e-4 + 1e-3*abs(CPU)；原生confidence>.23且positive>negative所得筛选mask必须一致。任何失败不修改容差。

CPU参考与MPS输出只留内存；MPS热身也核对F参考。MPS计时同步且包住count()的预处理、传入、原生前向、原生后处理和传出。为比较额外复制原始输出到CPU的时间也在计时内，略增加成本，不能称纯kernel延迟。只保存工程差异、形状、耗时、资源、状态与错误，不保存toy预测计数/框/分数。结束后删除模型、清空MPS缓存并退出工作进程。

唯一启动器launch.py独占创建launch_once.json；任何第二次启动拒绝。controls.py外部进程按50ms目标间隔监测，超限SIGKILL；实际停止与采样有调度粒度，记录实耗而不把180作为虚构精确实耗。MPS allocator限额与driver分配监测并用；RSS监测使用ps，工作进程另报ru_maxrss。

控制测试只运行空进程/人工数值张量，不导入或前向学习模型。首次RED确认控制和数值模块未实现；原子telemetry并发测试复现同临时文件竞争，已改为每次独立临时文件。默认沙箱不允许ps，因此测试在已批准的进程检查权限下通过，不绕过内存护栏。全套13项通过后才可启动此预检。
