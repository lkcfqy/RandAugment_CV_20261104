# LoRA × RA：必须修复项静态关闭回执

时间：2026-10-22 14:46 UTC。**结论：原代码审查列出的R1–R5静态必须修复项已关闭。可以继续既定的QA、人工视觉审查与dummy资源验收；这些非静态门通过并锁定最终协议后，才可开始已授权的四格单seed开发pilot。** 本回执不等于硬件通过、数据通过或科学效果成立。

原`protocol-review.md`和`code-review.md`保留。此次仅阅读源码和标准库AST解析，不import/运行被审代码；未启动GPU、训练、下载或备选，也未读取确认图像、旧封存数组或8A材料。未给父线程另发消息。

## 核对版本

| 文件 | SHA256 |
| --- | --- |
| common.py | `d12fd21604331ba032686eda26c93306945596a76e11f03b223740c73e9b1853` |
| train.py | `37c898a2f63f37eff26a2ffe32d29e2fa2feabf2d3d8d60a45c22126805001f0` |
| preflight.py | `74ed15204f8b4f3be5e58b1c9e71bcd203baee86066d42c57ebdc6a7d6876db8` |
| qa.py | `8cc95b9ba85bedac47d368544655d4b0deaa5f3dc497878842382d55062a0644` |
| analyze.py | `0e7156348110419f2a140e4dc09a5d2982ffdc36967e09535e2d1ee5d9f76a87` |
| prepare_protocol.py | `86f0aa0563948fc34a69baf45076c6084b1598ba3055fb5097de4cb529592d75` |

六文件均通过AST语法解析；本回执只覆盖这些字节版本。后续修改应在正式lock前保存新SHA及必要差异回执。

## R1–R5逐项关闭

| 原必须项 | 此版证据 | 静态结论 |
| --- | --- | --- |
| R1：增强不得重置MPS RNG | common的weak/RA CPU fork及qa的mask重放均改为CPU `default_generator.manual_seed`；preflight新增合成探针，检查增强前后CPU RNG恢复及MPS RNG不变。 | 代码缺陷关闭；实际隔离断言是否运行通过仍待preflight回执。模型初始化的全设备seed与增强隔离是不同用途，可保留。 |
| R2：节奏、预算、STOP和中断回执 | preflight每步sleep两倍active，设900s/STOP检查；train统一guard覆盖图片读取、建模前后、每训练/评价batch、checkpoint及完成标记前。异常保存结构化incomplete、progress、elapsed，且automatic_retry=false。 | 关闭。预算可能在一个不可分割的计算/IO段内略超时，但随后停止并禁止完成标记，符合开发资源门；不是瞬时硬杀承诺。 |
| R3：像素前manifest边界 | `validated_rows`检查非空/唯一source ID、partition严格train/dev、原候选身份/partition/class/species、旧177 ID排除、合法37类、非空cluster ID、各集37类齐全、无跨集簇及所有保留簇唯一；train在任何JPEG open前调用，prepare也调用。 | 关闭。最终QA是否按事前规则正确删除重复成员，仍需对实际QA与最终manifest验收。检查旧目录仅枚举文件名，未读取旧图。 |
| R4：协议与实现、未知配置及终点 | common拒绝未知layout/aug；train验固定seed、10/30epochs、batch16、四格各一次、最后epoch必须评价、micro主量、关键LR/WD及65,317参数。锁定全部源码并保存实际optimizer defaults/groups。 | 关闭。当前候选prepare中的配方与common相符；正式protocol产生后仍要核对实际SHA/值和锁定时序。 |
| R5：真实更新与可复算证据 | 每步全部可训练梯度存在且finite；实际update和源曝光计数有断言；逐层有效Q/V BA范数必须>0，保存B范数；head改变、固定8图features finite且改变、冻结底座SHA前后不变；初末小checkpoint、logits、逐step日志及RGB/tensor/order哈希保留。 | 关闭。BA门避免仅因AdamW decay改变A就假称学到表示。实际范数、SHA及metrics仍待结果审计。 |

fused投影的Q/K/V拼接方式未改变：只有Q与V存在低秩增量，中间K矩阵的增量逐项为零。3个block、Q/V各r4加共享37类head仍为65,317参数；未见引入可训练底座或梯度截断的新改动。

## 分析及冻结脚本复核

`analyze.py`现先核对protocol及自身锁SHA、fit-manifest SHA，再要求四格原始输出的IDs/labels与冻结开发序列完全一致，保存prediction也必须等于重新argmax。错误率、稳定NLL、两个G及contrast `[1,-1,-1,1]` 的主I保持正确。

5000次PCG64 seed73102、按37个观察breed独立有放回抽source photo、保持每breed计数、将抽样d之和除以总N，仍是micro权重。若重复簇按规则全删除，这个photo级条件敏感性算法与预先界定的单位相符；它不估计训练seed变异、动物身份独立性或总体CI。macro、NLL为次要，工程门仅I>=.01，不追加CI下界要求。

`prepare_protocol.py`现在要求manifest状态及验证通过、preflight完成状态/RNG隔离通过、正且finite的计时、MPS设备及参数量、视觉审查完成状态、实际权重SHA等于receipt，且results尚不存在。原先预结果资源分支保留：dummy compute+pacing估计×1.25落2小时则统一30ep，否则10或不启动。正式锁包括源码和本关闭回执。这个分支可以指导投入，不保证实际全程墙钟；运行中guard仍是必要的保护。

主量继续固定为source-photo micro：`I=(E_front_weak−E_front_RA)−(E_back_weak−E_back_RA)`。30ep或10ep均只取固定最后epoch，所有已显示中途输出完整保留。共同LR、单seed、1pp工程门和不由符号选LR/seed的边界不变。

## 尚未验收，不能提前声称通过的门

1. **完整数据及实际QA**：1184张train/dev与trimap下载/CRC/尺寸/标签来源校验，实际exact/near候选裁决、EXIF问题处理、最终删除清单/类计数/簇ID及manifest冻结；当前源码存在这些路径不等于数据已经通过。
2. **固定视觉页**：尚未收到，不能提前记“主体或品种线索保持”。mask运输只支持epoch0几何可见前景proxy，不能证明光度、细粒度语义或其他epoch。
3. **dummy实际资源回执**：待核验两布局真实AdamW batch16、10% MPS内存限制、节奏、RNG隔离断言和完成状态；本分工没有运行它。max observed driver是采样观察值，不能称完整瞬时峰值；约1/3节奏也不是实测GPU利用率。
4. **最终lock与开始时序**：正式protocol/lock尚待产生。需确认在科学拟合前落盘且所有源码/数据/证据SHA一致。四格不完整时不计算混合终点I。

ZIP范围的旧审查边界继续保留：只提取/存储/解码选定成员；额外512字节header margin可能触及相邻归档字节，尾部不解析并丢弃，不能保证传输字节绝对排除。当前prepare已纳入该scope、旧ID核查与两份代码审查SHA。无需为审计重取或查看尾部。

本轮没有新的权限拒绝或不确定工具动作。剩余事项是待提供的QA、视觉、dummy及锁定证据；**不是另行扩大训练、调参或访问确认集的授权。** 8A优先、单串行MPS、CPU1线程、10%进程内存和既有Sol/max/Fast请求背景保持；未验证实际后台档位。
