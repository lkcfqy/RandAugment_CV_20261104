# 当前检查点：LoRA位置 × RandAugment开发pilot完成

2026-10-22，本轮四格训练和独立复算完成。现行研究范围已按用户指示扩展至普通数据增强；没有要求强行使用因果术语。项目不是Git仓库，分支不适用。

**主交互I=+0.33956个百分点，未达到预先冻结的1个百分点工程投入门。** Front的RA收益+1.01868个百分点，back为+0.67912个百分点；固定模型、按breed分层的配对照片bootstrap区间[−0.84890,+1.69779]个百分点。区间不含训练随机性，不证明等效或普遍无效；本轮尚无成立的新贡献。

四格均30 epochs、1110次更新、17,730次源照片呈现、65,317个可训练参数，同一seed20261022。训练591张、开发589张，固定epoch30主端点。没有额外seed、学习率搜索、确认集读取或备选训练；592个预留确认候选仍封存，范围传输边界另有回执。

实际设备mps:0、fallback关闭、CPU1线程、串行节奏；14:52:39开始，15:53:57完成，墙钟3679.29秒含334.00秒为8A优先的同PID暂停。7200秒cap未延长或重置。原执行会话退出码0，15:54:28进程关闭回执确认PID49366不再存在。本轮后台训练为0，独立CPU审查结束，无调度任务。额外收尾ps重查被沙箱拒绝并停止，详见报告及closeout-checks.json；没有把拒绝的重查当成功。

独立复算20份开发输出、5000个配对重采样值全部一致。原严格脚本在FP64范数oracle处失败，18项Q/V超原容差事实保留；24个同定义CPU FP32范数精确复现。单独v2定义修订的897项检查通过，没有改变训练或科学统计门。不能写成原严格验证全通过；也没有独立重做ViT前向或训练。

- [结论及局限](augmentation-research-20261022/pilot-20261022/finding-zh.md)
- [冻结中文协议](augmentation-research-20261022/pilot-20261022/protocol-zh.md)、[协议及锁](augmentation-research-20261022/pilot-20261022/protocol-lock.json)
- [训练代码](augmentation-research-20261022/pilot-20261022/train.py)、[共同模型/增强代码](augmentation-research-20261022/pilot-20261022/common.py)、[主分析代码](augmentation-research-20261022/pilot-20261022/analyze.py)
- [原始训练结果](augmentation-research-20261022/pilot-20261022/results/completed.json)、[主结果](augmentation-research-20261022/pilot-20261022/analysis/summary.json)、[训练日志](augmentation-research-20261022/pilot-20261022/training.log)
- [独立审查](augmentation-research-20261022/pilot-20261022/independent/finding-review-zh.md)、[原失败](augmentation-research-20261022/pilot-20261022/independent/result-verification-failed.json)、[同定义v2](augmentation-research-20261022/pilot-20261022/independent/result-verification-v2.json)
- [数据许可](augmentation-research-20261022/pilot-20261022/data/license-receipt.json)、[视觉/重复QA](augmentation-research-20261022/pilot-20261022/qa/source-qa.json)、[范围传输边界](augmentation-research-20261022/pilot-20261022/data/range-transfer-scope.json)
- [真实查新范围](augmentation-research-20261022/pilot-20261022/novelty-boundary-zh.md)、[资源核算](augmentation-research-20261022/pilot-20261022/runtime-accounting.json)、[后台状态](augmentation-research-20261022/pilot-20261022/background-status.json)
- [本轮清单](augmentation-research-20261022/pilot-20261022/delivery-manifest.json)、[本轮核验](augmentation-research-20261022/pilot-20261022/delivery-verification.json)、[旧材料保全](augmentation-research-20261022/pilot-20261022/preservation-final.json)

下一决策：停止本版定向共同配方假设的自动扩跑；不以追加seed、改阈值/主指标或开确认集救结果。日后若有独立科学理由重启，先冻结新协议，公平处理位置相关学习率敏感性、配对seed和全层LoRA/full FT参照；本轮未启动它们。备选只保留提案，不训练。普通RA收益不能单独构成新颖性。

Geiping OpenReview正文仍未取得，不能称精确先例已清；历史受阻来源未绕过。Pets路线已核实且本轮完成，无等待中的训练/下载/许可动作。额外ps检查为已拒绝并停止的可选复查；其他剩余缺口为科学证据范围。

旧smallNORB阴性结果、J*更正解释、被否决代理候选、机制图和课堂原论文/可视化完整保留；不触碰8A。所有数据和结果留Mac；无A100、Colab、ARiS、付费、上传或发布。GPT-6.1-Sol / max / Fast为保留的请求设置；实际后端回执未暴露，未宣称逐轮验证。
