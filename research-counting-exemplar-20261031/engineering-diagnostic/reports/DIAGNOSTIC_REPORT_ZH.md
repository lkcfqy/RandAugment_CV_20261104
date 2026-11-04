# CPU F / MPS F定位结果：原数值门仍失败

本次独立授权的两次尝试全部执行并返回，外部墙钟20.701754625秒，未达到120秒护栏。没有重试、C/V前向、真实图像或正式query。加上首次预检34.214030792秒，累计54.915785417秒，1800秒预算剩1745.084214583秒；累计6次模型尝试（CPU4、MPS2）。GPU张量分配已清零，worker退出，未分配Colab。

## 已排除的检查项

本次CPU/MPS有31项dtype+shape+字节SHA256完全一致：三份归一化图像、padding masks、正负六框、BERT输入token/type/position/attention masks、encoder结构mask/索引、原生视觉token插入后的IDs及text mask。该结论来自真实forward边界的检查，而非只看文字配置。

迁移前后1110个注册张量hash一致（1108个逻辑state项和2个显式非持久BERT缓存），66个共享别名在两设备上均保持实际对象和storage共享关系。原始CPU结果使用clone，全部结果clone和24个CPU中间节点的hash在MPS运行后不变。未发现输入、负视觉组、token、权重或CPU参考被原地修改。

## 定位到的差异

| 被观察位置 | 最大绝对差 | 最大误差/原允许误差 | 结论 |
|---|---:|---:|---|
| BERT last_hidden_state | 4.0531e−6 | 0.01638 | 首个非零差异，在容差内 |
| backbone query level0 | 7.6294e−6 | 0.02776 | 在容差内 |
| backbone query level1 | 0.001742959 | 4.26648 | 首个被观察到的超容差节点 |
| backbone query level2 | 0.001154900 | 1.47572 | 超容差；正负exemplar backbone重复出现相同差异 |
| 正/负ROI输出token | 1.2250e−5 / 1.4782e−5 | 0.11187 / 0.10140 | 在容差内 |
| encoder输出visual/text | 1.4745e−5 / 8.7023e−6 | 0.08780 / 0.02306 | 在容差内 |
| top-k前proposal分数 | 0.0001187325 | 0.01712 | 在容差内 |

在22223个候选中，CPU与MPS选中的900个proposal**集合完全相同，但排名743/744互换**：CPU为17465、21628，MPS为21628、17465。其余898位一致。这是明确观察到的decoder前离散顺序差异。

CPU第900名与最好未选proposal间隔0.0003228188，MPS为0.0003232956；CPU选中序列最小相邻间隔4.76837e−7。两端切分边界都没有精确并列，不能声称已证明“精确tie”。由排序反转和最大score差可推出被交换两项CPU间隔不大于0.0002374649；此次没有单独保存它们的精确score间隔，不能补造数值。已保存完整原生900索引及顺序。

原生embed_init_tgt=True（transformer.py:319–324）把固定顺位的learned target embedding与该顺位proposal配对，因此这两项交换并不意味着最终输出只是简单置换。原始逐行比较事实上涉及不同proposal/target配对，可能把小的上游数值变化放大成较大输出变化；这是有证据支持的解释，尚不是单算子因果证明。Swin中间层也确实存在较小超容差偏差，不能把全部差异归为纯top-k排序问题。

最终原比较与首次结果相同：logits最大绝对差1.5562586784，框坐标差0.3294818997；原生筛选mask一致，仍不满足逐元素容差。未对输出排序、匹配、交换、放宽标准或修改模型来宣称通过。

本轮观察了分阶段节点，没有逐个Swin算子trace。下一步应先独审这份顺序反转和中间层证据，决定工程等价性验证应如何预注册；任何改用另一张人工图或进一步定位forward都需要新指令。本轮不自动切换、不重试，也不据此给研究假设科学阴性结论。

## 精度、资源与原样保留

运行时MPS为Apple M5 Pro/20 cores；float32_matmul_precision=highest，deterministic_algorithms=False；MPS_FAST_MATH和MPS_PREFER_METAL未设置，fallback明确为0。记录了原始设置，没有改精度；UNSET不是每个Metal shader编译精度的证明。CUDA matmul TF32=False、cuDNN TF32=True是读到的CUDA设置，本轮不使用CUDA，不能把这些标志解释成MPS精度开关。

RSS峰值4,080,467,968 bytes，MPS driver采样峰值3,479,076,864 bytes，均低于各8 GiB。退出前MPS current=0，driver仍有24,838,144 bytes框架开销；随后PID21093已退出。19项无模型测试在启动前通过。原资产436文件、CPU准备56文件和首次工程预检20文件全部hash保持不变。

详细证据：run/engineering.json（含31项输入hash、两侧注册张量hash、24节点误差、900索引）、reports/diagnostic_summary.json、reports/compute_budget_ledger.json、reports/prior_artifacts_preserved.json、reports/prelaunch_code_manifest.json及run/supervisor.json。没有toy预测计数、框或完整score向量落盘；只保留明确授权的top-k索引/边界分数及工程统计。
