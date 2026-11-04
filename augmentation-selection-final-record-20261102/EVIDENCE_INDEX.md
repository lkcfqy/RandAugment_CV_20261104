# 证据索引与结束状态

本记录汇合5个只读来源档案，验证其367个清单文件的SHA256/字节，96个唯一设置与原结果逐项一致。原档案未覆盖或改写。所有原精度特征、分类头和逐例预测仍在原目录；这里不重复复制大资产。

|本目录文件|内容|
|:---|:---|
|[CONCLUSION_ZH.md](CONCLUSION_ZH.md)|可向用户交代的有限结论、8组核心表、反证及已有工作边界|
|[ALL_96_SETTINGS.csv](ALL_96_SETTINGS.csv) / [MD](ALL_96_SETTINGS.md)|96个设置的全部策略准确率、选择、配对差、regret；CSV含每折整数计数|
|[ALL_8_GROUPS.csv](ALL_8_GROUPS.csv)|8组完整固定策略均值、事后最佳、选择频率、H/C、并列界和阶段统计|
|[COMBINED_RESULTS.json](COMBINED_RESULTS.json)|全部原始整数统计、各组12个配对分布、每一种并列组合及预算描述|
|[ALL_TIE_SENSITIVITY.md](ALL_TIE_SENSITIVITY.md)|并列处理的完整上下界，原规则仍为主结果|
|[SOURCE_INDEX.json](SOURCE_INDEX.json) / [VALIDATION.json](VALIDATION.json)|来源哈希、367文件核验、原始数值一致、整数恒等式检查|
|[LIFECYCLE.json](LIFECYCLE.json)|各任务退出回执及当前无运行/待启动实验的状态|
|aggregate_saved.py / write_final_record.py|仅标准库的复算和文档生成实现；不导入模型库|

原始证据：

|档案|关键入口与用途|完整清单SHA256|
|:---|:---|:---|
|[augmentation-selection-cifar100-run-20261102](../augmentation-selection-cifar100-run-20261102/REPORT_ZH.md)|CIFAR100全部结果：JOINT_QUERY_GATE、两FIT/QUERY_SEAL、720头、全部预测、阶段排序、FINAL_VERIFICATION|`3f1d4a75015791e78ecf0f0697355a3aba0f0e3b31b0d3f3dfbc3a9bc6f0f6ee`|
|[augmentation-selection-deit-20261102](../augmentation-selection-deit-20261102/REPORT_ZH.md)|EuroSAT DeiT：同类完整文件；SECOND_ENCODER_CONTRACT_TEST验证共享视图及必要预处理差异|`eb1088df3f80b7f9fdabb600e166ba163475161586a038f75ca6402923ed7276`|
|[augmentation-selection-pilot-20261102](../augmentation-selection-pilot-20261102/REPORT_ZH.md)|EuroSAT DINO：PROTOCOL、SELECTION_SEAL、HEADS、QUERY_PREDICTIONS、POSTRUN_VERIFICATION、RUN/TASK_RECEIPT|`68c2a6c48374b6064fe2b2784d00970c3dbceb8cc4a29f29a827e6d022b101bc`|
|[augmentation-selection-cifar100-preflight-20261102](../augmentation-selection-cifar100-preflight-20261102/REPORT_ZH.md)|CIFAR100资格/协议/身份/计时、VEST阅读边界；历史“尚未执行”状态由后续完成记录接续|`7198fbf0ec29a38c21dcba60b35128851b3f318303919af10eca2fc5c7a3b8d4`|
|[augmentation-selection-synthesis-20261102](../augmentation-selection-synthesis-20261102/REPORT_ZH.md)|EuroSAT保存预测综合：全部并列组合、错误设置、576折计数、三候选混淆修正、5项合成测试|`980d70e169bc8751459bc9a802e367fba9da7b02f617bbab47cf4474d17980d9`|

阶段和图表可直接追溯到 [EuroSAT全部错误](../augmentation-selection-synthesis-20261102/ALL_WRONG_CHOICES.md)、[EuroSAT完整并列](../augmentation-selection-synthesis-20261102/ALL_TIE_ASSIGNMENTS.md)、[EuroSAT逐例图](../augmentation-selection-synthesis-20261102/episode_differences.png)、[CIFAR全部阶段排序](../augmentation-selection-cifar100-run-20261102/STAGE_RANKINGS.md)、[CIFAR逐例图](../augmentation-selection-cifar100-run-20261102/paired_differences.png)。图表不省略不利设置。

已结束任务及生命周期：

|任务|原任务记录状态|退出依据|当前状态|
|:---|:---|:---|:---|
|augmentation-selection-pilot-20261102|COMPLETE_AND_VERIFIED|[known_task_processes_exited=true](../augmentation-selection-pilot-20261102/TASK_RECEIPT.json)|已结束，无待启动步骤|
|augmentation-selection-deit-20261102|COMPLETE_AND_VERIFIED|[known_task_processes_exited=true](../augmentation-selection-deit-20261102/TASK_RECEIPT.json)|已结束，无待启动步骤|
|augmentation-selection-synthesis-20261102|COMPLETE_SAVED_PREDICTION_ANALYSIS_AND_VERIFIED|[known_task_processes_exited=true](../augmentation-selection-synthesis-20261102/TASK_RECEIPT.json)|已结束，无待启动步骤|
|augmentation-selection-cifar100-preflight-20261102|PREFLIGHT_PASS_NO_FORMAL_EFFECT|[known_benchmark_processes_exited=true](../augmentation-selection-cifar100-preflight-20261102/TASK_RECEIPT.json)|已结束，无待启动步骤|
|augmentation-selection-cifar100-run-20261102|COMPLETE_ALL_FROZEN_SETTINGS|[model_processes_exited=true](../augmentation-selection-cifar100-run-20261102/TASK_RECEIPT.json)|已结束，无待启动步骤|

EuroSAT两次主执行session均有exit_code=0；CIFAR监督进程等待全部4个模型工作进程与分析进程退出，正式执行exit_code=0。预检和只读综合同样有已退出回执。本结论限定于这条研究线的已启动任务，不是整机进程扫描；没有调用此前被拒的ps/sysctl。

旧CROSS_DATASET_PLAN_NOT_EXECUTED.md是当时历史计划，已由随后冻结并完成的全类CIFAR100方案接续；不再是待运行任务。预检档案内的“待集成”字段也只是历史状态。LIFECYCLE.json明确当前running_research_jobs=0、pending_experiment_commands=[]、fourth_round_started=false；没有创建新的调度任务。

成本事实：本线累计1440个真实ridge头、96次选择、195194个正式图视图前向，另有320个预检/计时前向；未更新编码器。保存7912800个查询决策和190080个OOF决策。每个原任务的耗时及资源以其回执为准；CIFAR完整75.56分钟已经包含准备，不再把其预检21分钟重复累加。本线新增CU=0。此次最终整理新增模型、特征、拟合、下载、安装、上传和CU均为0，产物大小/时间记入本目录TASK_RECEIPT。

当前研究线已关闭，无剩余实验计划启动。没有把新结果换进已有RandAugment演讲。结论可供2026-11-02韩国19:00汇报使用；这不是提醒或自动发布任务。
