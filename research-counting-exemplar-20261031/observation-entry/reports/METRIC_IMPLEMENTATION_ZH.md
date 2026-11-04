# 按图合并A/B的冻结误差与人工核验

E(image, mode) = (abs(nhat_A-n_A)+abs(nhat_B-n_B))/(n_A+n_B)。分母是该图官方A+B总数，必须>0；不是把两个query各自归一化再平均。每pair取两张固定图E的算术均值。

G在41个eligible pairs上平均min(pairmeanC,pairmeanV)-pairmeanF；G>=0.10，且至少10pair满足F<=0.10、C>0.30、V>0.30，两项同时成立才过数值投资门。min使用GT，是oracle包络，不能部署。全部47pair报告，6NA的误差为null而非0；缺失或重复输入报错，不能静默丢样本。

人工反例验证：1/9真值、2/9预测时E=0.1而非0.5；2/8预测虽然总数不变，E仍0.2。两图C/V一好一坏时，先平均得到各0.5，再取min；不能逐图挑最好。不同pair分别C/V好时，先逐pair min，不能先跨pair平均。41对的分母不会被6NA稀释；阈值边界、缺失A/B、零分母、大于1的误差与计数>=800均已测试。

13项测试仅使用捏造计数，无真实模型预测。82张eligible图的官方A+B均>0仅作输入合法性检查。代码使用Fraction精确比较门槛，JSON float仅用于呈现，并保留分数numerator/denominator。

数值过门不自动证明正确目标选择；触发案例仍须查看预测框排除错计抵消，不能据此删样本、改门重算。随机placeholder点不作定位真值。

protocol_v2_frozen.json补齐完整公式；v1、labels_frozen.json、queries_frozen.json保留原字节。正式模型运行未授权、未执行。证据：scripts/metrics.py、tests/test_metrics.py、reports/metric_counterexample_tests.log、reports/metric_verification.json。
