# 第二次、单独授权的工程定位

本次只允许CPU F和MPS F各一次，共2次forward尝试；失败也计数，无重试。外部120秒，RSS/MPS各8 GiB，全部实耗续记到1800秒总计算预算（此前已用34.214030792秒）。不运行C/V、真实图像或正式query，不分配Colab GPU。

沿用第一次人工常色图1333×800、颜色(100,110,120)、六框、red toy/blue toy、seed42、原生FP32和800/max1333预处理。模型严格safe加载逻辑原样复用。无网络、pickle、MPS CPU fallback。源码和算法不改。

诊断在真实forward边界记录归一化query/positive/negative输入、padding mask、六框、BERT token/type/position/attention mask、原生视觉token插入后的IDs及text mask的dtype+shape+字节SHA256。CPU/MPS逐项一致才继续。CPU输出和被观察中间量都detach().cpu().clone()，结束重核CPU snapshot hash。迁移前后全部注册参数/buffer（包括两个非持久BERT缓存）逐一hash对照；66共享别名检查实际对象身份和storage关系，不能仅凭名称匹配。

只读hooks覆盖BERT、投影文本、三次backbone的各输出层、正负ROI输入与输出、encoder输入/输出、proposal投影和选取前logits/分数。记录最早非零偏差和最早超过原容差的被观察节点，不能把“最早被观察节点”冒称其内部单个算子的根因。CPU中间量仅留内存；MPS按同名stage比较，只保存形状、hash、最大误差和是否过原容差。没有重排或替换计算结果。

对原生torch.topk调用先调用原函数，再只读记录其实际900个proposal索引和顺序、边界分数间隔、并列个数、CPU/MPS集合/顺序差异；返回同一个原始输出。不在最终预测上排序或匹配来放行。embed_init_tgt标志实际读取并报告。

原始最终logits/框的逐行比较和筛选mask检查照第一次1e-4+1e-3*abs(CPU)规则执行；native -inf padding规则不变。最终raw比较即使失败，也保留所有已生成定位证据，随后释放GPU并退出。无toy预测计数/框/分数落盘；topk索引和边界分数是本次明确授权的工程诊断信息。

记录MPS fast-math/Metal相关环境变量、float32 matmul精度、deterministic标志与设备信息，不更改这些设置。静态说明：当前PyTorch Python后端没有可直接回读每个Metal shader实际编译精度的统一接口，因此环境UNSET如实记录，不把未设变量等同于逐算子精度证明。

instrumentation单元测试仅用人工CPU张量，验证clone隔离、hash绑定、原容差、非有限掩码、topk重排及集合变化可被发现。已通过的外部watchdog/单次启动/输出过滤控制原样复用。冻结脚本hash后才启动一次新诊断，不覆盖第一次预检和证据。
