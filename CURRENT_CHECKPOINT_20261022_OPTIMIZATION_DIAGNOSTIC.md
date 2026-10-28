# 当前检查点：优化混杂单步诊断按有效性规则停止

2026-10-22，本轮只做同初始化、固定训练内批次的单步优化诊断，未新增完整训练或种子搜索。**原定8个重置单步未完成；实际执行5次（前层4次完整，后层第1次更新后配对检查失败），已退出，不自动重启。**

后层batch0与前层batch0的共享头权重梯度最大绝对差2.68221e−7、整体相对L2差约1.00383e−6；142/28416元素超冻结atol2e−8、rtol2e−5。两个布局的初始clean训练内探针特征完全相同，但梯度训练批CE相差约4.77e−7。FP32/autograd执行路径是可能解释，根因未定位，不能说已排除实现问题。

5个保存梯度/更新后状态均符合零B时A数据梯度0、B非零有限及原首步AdamW oracle；这只复现已知代数，不是贡献。前层4个CLS变化RMS约[1.01658,.975381,.954991,1.00052]e−6，只有一侧；**前/后函数尺度比不可用**。完整主分析/完成JSON缺失是预期的不完整状态，不能补造为8格成功。

协议先冻结，SHA `29ec9ab66cfd8ef32ba23f2e3fd7095bd0cccfe75f221269acf3667c518b8f91`；只使用96张已用训练照片，64梯度/32探针互斥。同seed、初始A/B/head、首warmup步；无开发准确率或封存像素读取。实际MPS、CPU1、nice10，16:29:09开始，16:29:21停止；约11.83秒UTC记录跨度，不是完整perf-counter资源计时。原会话退出码1，本轮后台训练为0。

- [结论、原失败与唯一修订控制](augmentation-research-20261022/diagnostic-optimization-20261022/finding-zh.md)
- [冻结计划](augmentation-research-20261022/diagnostic-optimization-20261022/protocol-zh.md)、[机器协议](augmentation-research-20261022/diagnostic-optimization-20261022/protocol.json)、[锁](augmentation-research-20261022/diagnostic-optimization-20261022/protocol-lock.json)
- [诊断代码](augmentation-research-20261022/diagnostic-optimization-20261022/diagnostic.py)、[失败JSON](augmentation-research-20261022/diagnostic-optimization-20261022/diagnostic-failure.json)、[原始日志](augmentation-research-20261022/diagnostic-optimization-20261022/diagnostic.log)
- [失败后CPU检查](augmentation-research-20261022/diagnostic-optimization-20261022/partial-analysis.json)、[后台/退出状态](augmentation-research-20261022/diagnostic-optimization-20261022/background-status.json)
- [独立部分结果审查](augmentation-research-20261022/diagnostic-optimization-20261022/independent-partial-review.md)、[交付核验](augmentation-research-20261022/diagnostic-optimization-20261022/delivery-verification.json)
- [原完整四格pilot检查点](CURRENT_CHECKPOINT_20261022_LORA_PILOT.md)

决定：结束本轮，保留不完整测量；不补批、放大LR、改容差、追加seed或完整训练。唯一修订建议为同一ViT参考图放前/后共六处零B适配器，每批一次共同反向后，从相同初始状态分别屏蔽更新前3/后3处及相同head，比较同训练探针。计算梯度102,181参数、每支更新65,317，需明确不同于原三处图计算预算。仍4梯度批、8隔离单步状态，原Mac串行/CPU1/600秒cap，先测资源；**仅提案未执行**。此局部共同参考不替代原MPS图或30epoch结论，无A100需求，不自动启动。

原pilot I=+0.33956pp、1pp工程门失败及原18项范数审计差异/同定义v2通过历史保持；本次新头梯度失败未被归类成已解决的范数定义问题。旧研究、课堂和冻结资料保全；8A优先且未读取其内容。所有工作留Mac，无ARiS/Colab/付费/上传/发布。Sol/max/Fast为请求设置，实际后端回执未暴露。
