# 单次人工工程预检：ENGINEERING_BLOCKED

已按授权执行一次预检，**4次forward尝试/4次原生模型返回：CPU F/C/V各1次，MPS F热身1次**。CPU三次通过有效输出有限性及原生token/group检查；MPS热身在预定逐元素数值门上失败，立即停止，剩余3次没有执行，没有自动重试。不是科学阴性，未评估数据增强贡献或真实目标选择能力。

| 项目 | 结果 |
|---|---|
| CPU F/C/V端到端时间 | 6.159160 / 6.139380 / 6.137315秒 |
| MPS F热身至数值拒绝 | 4.916636秒，含比较检查；不是合格的正式计时样本 |
| 有效logits最大绝对差 | 1.5562586784 |
| 有效logits最大差/允许误差 | 281.9673157 |
| 框坐标最大绝对差 | 0.3294818997 |
| 框坐标最大差/允许误差 | 1739.8171387 |
| 原生筛选mask | CPU/MPS一致，但不能覆盖逐元素检查失败 |
| 容差 | abs(MPS−CPU) ≤ 1e−4 + 1e−3×abs(CPU)，未放宽 |
| 外部计算墙钟 | **34.214030792秒**；180秒上限未触发 |
| 本候选总1800秒预算余额 | **1765.785969208秒** |
| RSS峰值 | 3,901,505,536 bytes，约3.63 GiB，低于8 GiB |
| MPS张量分配采样峰值 | 1,803,708,416 bytes，约1.68 GiB |
| MPS driver分配采样峰值 | 3,473,260,544 bytes，约3.23 GiB，低于8 GiB |

原生MPS路径确实执行并返回，未出现缺算子错误；当前结果只证明这张人工输入未通过预定CPU/MPS一致性标准。根因尚未定位，不据此声称所有MPS结果错误，也不更改模型、排序、阈值或比较规则来放行。

固定人工图为1333×800 RGB常色(100,110,120)，两组各3个人工xyxy框；种子42；CPU线程2/interop1。F文字red toy/blue toy，C为toy/空文字，V为两侧空文字；全部保留负视觉组。实际原生预处理张量均为[1,3,800,1333]、FP32，覆盖正式最长边1333。真实PairTally图像、计数和标签读取0，正式query0，训练0，pickle读取0，网络连接0。人工预测计数、框、分数均未落盘；只保留工程统计和人工输入定义。

MPS计时同步并包住原生count()的预处理、传入、forward、筛选和传出；为比较额外复制完整输出到CPU也在计时内。第4次比较在原始logits/框返回后失败，不能把它记成未执行的forward，也不能把未执行的MPS F/C/V三个计时补成0。

在任何forward前，静态审查确认官方ContrastiveEmbed会把无效token和补齐槽位填成-inf。预先记录的规则是：这些槽位必须精确为-inf且CPU/MPS结构一致；有效logits及全部框有限；sigmoid后全部分数有限。CPU三模式均满足，full_logits_finite均为false且如实记录。MPS的validate检查通过后才进入compare，最终失败为数值容差。此项是原生固定填充语义的显式处理，不是观察结果后放宽容差；见首次forward前的PRECHECK_SCOPE.md与prelaunch_code_manifest.json。

严格CPU加载再次核对1042物理tensor、66实际共享别名和1108逻辑键，strict=True，missing/unexpected均空，所有加载后值精确对应checkpoint。网络结构、attention路径、ROIAlign和源码修改0；禁用MPS CPU fallback。运行后再次核验，原始资产交付436文件及CPU准备交付56文件全部hash不变，完整官方HF源码hash不变。

退出前model引用释放、MPS同步和empty_cache完成，MPS current_allocated_memory为0。退出前driver仍有24,838,144 bytes框架分配，未谎报为0；随后工作进程PID20258正常退出，独立检查确认不存在。没有遗留工作进程。监控采样目标间隔50ms，报告采样峰值与macOS ru_maxrss，不声称精确捕捉所有瞬时分配。

13项无模型控制/数值单元测试通过后才启动预检，包括失败尝试扣预算、单次启动锁、实际子进程超时杀停、分配护栏、原子并发telemetry、预测输出禁止落盘、固定padding语义与容差内但mask不同仍拒绝。初始默认沙箱禁止ps，已在批准的进程检查权限下验证；没有自动审批拒绝或未决进程。首次报告生成命令选到Python3.9不支持hashlib.file_digest，已改用既有Python3.14环境完成哈希记录，没有重跑模型。

## 成本口径与后续阻塞

保留规划公式：`34.214031 + 正式启动/加载 + MPS热身 + 2×492×max(t_MPS_F,t_MPS_C,t_MPS_V) + 非零校验/写盘余量`。

- 正式启动参考10.253421秒，依据本次工作进程启动至严格CPU加载完成10.040593秒，加CPU到MPS模型转移0.212828秒；不把CPU参考forward加进这项，因为它们已在预检实耗中计费。
- 校验/写盘预留60秒是明确的规划余量，用于图像hash检查及492条持久记录；不是实测上界。
- MPS热身与三模式合格计时未获得，**不能给出有效的492query MPS总成本估计或放行判断**。
- CPU三模式最大测得6.159160秒，单纯492倍已约3030.31秒，尚未加启动、IO或2倍余量，超过剩余1765.79秒。该算术只用于预算讨论，不能自动转CPU，也不是不同真实图的实测总耗时。

41对/82图/492query的名称和输入metadata已按父线程随后批准的资格政策冻结在../observation-entry/；原94图选择hash仍为c7ba6b400d0ec6123eda4274625c4d5a64111a664f978001e31215b3ad27b6bc，47对全部报告，6NA保留理由，无替换或标注修复。两项投资门已记录，但委派消息未给出每query误差E的完整公式，protocol_inputs_frozen.json将其显式留空，不能自行发明。

当前阻塞为MPS数值门失败以及后续正式入口/误差定义复核。正式492queries未授权、未启动；该工程失败不授权新增诊断forward、另一次预检、CPU替代或科学运行。剩余3次尝试也不会自动使用。

证据：run/engineering.json、run/attempts.json、run/supervisor.json、run/memory.json、run/worker.log；reports/compute_budget_ledger.json、reports/post_run_preservation.json、reports/control_test_suite.log。脚本在scripts/，唯一启动锁launch_once.json保留，不能自动覆盖。
