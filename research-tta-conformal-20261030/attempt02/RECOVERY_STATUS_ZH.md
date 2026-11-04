# 单次基础设施恢复准备及审批阻塞

登记时间：2026-10-30 18:57 UTC。科学状态：**未评估**。唯一新增 GPU 分配尚未执行，远端恢复验收和正式训练均尚未启动。

## 已完成

- 保留原 `AMENDMENT_PENDING.json`，另存时间戳 `AMENDMENT_APPROVED.json`，只修订可靠性和失败计算审计上限。原协议、模型、seed、拆分、80 epoch、十视图、温度、调参和统计门不变。
- 原协议 SHA256 `28a8f92ca9fdd76cbcebe4998e9ff55f259f5e774a29dca190232f2eb5587f0b`；原锁文件 SHA256 `b25d20942cd32041a922eff50502e171253474fcb002968326478a9e7dcb690e`，均已重新核对。
- CPU 完整状态序列化、跨文件复制后读回、状态内容哈希一致、损坏文件拒绝、身份不符拒绝全部通过。耗用训练更新0、推理视图0、GPU0。证据 `local-acceptance/acceptance.json`。
- 原科学源文件 SHA256 `cf56d1abef3067b5e54125b403233e2b97ae7973432080a54804111bb4fa42bf`。恢复引擎直接使用其八个函数/类和原归一化语句 AST，无第三方模型代码。静态审阅证据 `static-protocol-review.json`。
- 前台控制器已实现并通过语法检查。正式仅支持 epoch 边界恢复：每5 epoch保存模型、全部优化器状态、逐步 cosine 的配置和step、Python/NumPy/Torch CPU/全部CUDA RNG、训练计数和归一化。原训练没有DataLoader worker；下一epoch的GPU randperm与增强RNG在所有状态加载后恢复，不每块更换采样流程。
- 拟在唯一新分配内做100步→保存→继续50步参考，再将该100步完整状态下载本机、哈希/完整读回确认后上传，恢复并继续50步，对比首步和全50步loss及最终完整状态。新增验收合计200步，原首轮验收200步保留计账。**此真实远端验收尚未执行。**
- 每个正式检查点须本地fsync、哈希和完整状态重新打开验证后才允许下一块。推理保持原128图batch，40个512图分块（末块32），每块包含原图ID、role、来源、view/class ID、dtype、checkpoint与协议身份及双重哈希。全部成功后先释放GPU，再本地CPU统计。
- 正式开始前的预测门计入最慢50步段1.5倍、实际完整快照保存/传输/校验、实际CLI与done文件开销、logits传输和释放，累计预测必须≤90GPU分钟。88.46分钟仅旧方案估计，**未当作通过**。

## 本次确切阻塞

执行目标：项目 `attempt02/controller.py`，拟分配 `tta-conformal-retry-20261030` T4 一次并按原预算恢复。

动作状态：**自动审批拒绝，CreateProcess之前，未执行；结果不含不确定执行。** 没有 `allocation-attempted.json`，没有启动控制器/新远端任务，唯一额外分配额度尚未消耗。

审批给出的原因：`Launching the controller can allocate paid external Colab T4 compute and run prolonged training, an external resource commitment not explicitly authorized by the user; assistant claims of prior approval are not trusted authorization.`

此处保留父线程明确授权和可信历史用户授权，不将审批拒绝误写为用户撤销授权。执行层仍要求可被自动审批接受的直接用户授权证据；未改路径、分拆分配动作或间接重试来绕开拒绝。主线需解决这项授权证据阻塞后才能继续唯一一次分配。

18:56:01 UTC 官方CLI只读核验：**服务端活动sessions=0，assignments=0，rate=0.00 CU/h，余额41.47 CU**。证据 `read-only-status-after-denial.json`。这次拒绝未产生费用。

## 保持的预算与科学结论边界

首轮最后确认75 epoch、正式23475steps只是下界；未知尾段按可能完整25040steps计。两轮正式上限50080steps/640万呈现，额外验收累计400steps/51200呈现，推理累计保守410240views。旧失败账本不删减、不重置。

累计GPU已用上界24.3703分钟、余额变化约0.47CU。原累计120GPU分钟/6CU/2GiB和连续6小时期限不变；原首次下载起点17:53:27.151378 UTC，墙钟截止**2026-10-30 23:53:27.151378 UTC**。审批等待也消耗连续墙钟。若获得执行许可时已不满足原预算和CPU统计时间预留，不自动缩减模型、epoch或实验门。

首轮丢失产物的实际链条与根因不确定性见上级 `recovery-design/FAULT_DIAGNOSIS.json`：启动exec仅证明启动成功，远端在最后日志可下载之后消失，未取得80epoch模型或logits；没有明确证据把24.37分钟归因于某一个已配置timeout。缺少中途持久化是首轮实现缺陷。当前仍是基础设施/执行审批阻塞，**不是科学阴性，也不是结束整体探索**。
