# 后续最小观察入口与无效用吞吐预检方案（尚未授权执行）

当前阶段仅允许隔离环境、CPU严格加载、人工框/占位向量接口检查。**本文是实现和预算讨论材料，不是启动指令；没有执行任何模型forward或GPU预检。** 按钮组语义仍等独立审查，不在此修改。

## 最小可复用入口

建议下一阶段只增加一个薄入口 `observe_countgdpp.py`，复用本阶段已验证的严格加载函数逻辑，以及固定官方HF源码中的 `count()`，不改网络、训练、提示插入、阈值或attention实现。命名明确为“CountGD++官方app/HF基础推理路径”；无pseudo-exemplar、无adaptive crop，不宣称完整论文复现。

入口的固定输入：

1. `selection_frozen.json` 及其SHA256：94张图、47pair、每图原始A/B各3框，不重新选图。尚未处理的语义问题用单独eligible状态呈现，不能依效果删图。
2. 经独审冻结的 `labels_frozen.json`：每pair的F-A、F-B、C共同类名、来源、修改理由和eligible状态。现在的候选表不能直接冒充已冻结标签。
3. 预先固定的协议文件：误差E的正式定义、pair聚合、门槛、饱和/异常处理、统计口径、设备/预算，以及协议hash。本阶段不新定义科学指标。
4. 原始CountGD++权重SHA256及BERT资源版本、本阶段环境版本、完整官方源码hash。

建议函数边界（此处未实现科学运行）：

- `load_fixed_model_cpu()`：读固定JSON配置，仅将BERT路径改成本地目录；验证实际模块共享关系，展开66别名，完整1,108逻辑键shape/dtype/有限性验证，`load_state_dict(strict=True)`，eval、冻结参数。严禁网络、pickle、trust_remote_code、静默漏键。
- `make_query_manifest(selection, frozen_labels)`：按pair、固定图哈希排名、模式F/C/V、目标A/B的固定顺序生成至多564条；每条保留图片hash、三正/三负框身份、完整caption、文本/框hash。
- `run_one(query)`：只调用官方 `count()`。F为正负细类文字；C为共同正类名且 `negative_texts=['']`；V正文字空且 `negative_texts=['']`。负视觉始终是一组3框，A/B互换全部文本、框和计数身份。RGB、短边800/max1333、FP32、batch1、confidence0.23、原生正分数>负分数。
- `append_result()`：每query原子写JSONL并刷新；记录查询hash、count、normalized boxes、scores、计时、设备、模型/协议hash。重启仅跳过所有身份hash均一致的已完成query，禁止静默替换失败query。
- `summarize_once()`：所有eligible pair完成后按冻结协议计算，不中途挑阈值/类名/图/模型。所有缺失和异常显式列出；预测≥800等饱和情形报告，不自动剔除。

最小自检：图片hash与框范围；F/C/V共用6框；A/B整组互换；C/V保留负视觉token；全局forward计数与成功query数一致；预算超限前停止并保留已完成记录。最终检查需读取真实预测框排除计数错误抵消，不能把随机placeholder点作定位GT。

科学门沿用上游：每pair先平均至多2图的误差；再计算每pair `min(E_C,E_V)`，跨全部eligible pair平均，与F的差≥0.10；至少10pair满足F≤0.10且C、V各>0.30。等计数图仍保留，不因A/B输出相同就断言模型忽略示例。结论只针对固定CountGD++路径，不外推CountEx或全部最新方法。

## 无效用吞吐预检提案

目的仅为得到运行成本和设备兼容性；**不读取PairTally图片、类别真值或计数，不计算任何误差/排名/投资门，也不据预检改变研究设置。**

- 输入：一张人工常色RGB图，原始尺寸1333×800；人工6个合法、不重叠xyxy框，完全固定位置。文本使用toy/red toy/blue toy以及C/V空文本组合。没有真实对象或科研标签。
- 模型：同一个严格加载后的官方模型，原生FP32推理路径。拟选择一个设备（优先先讨论Mac MPS）；不同时启用多个设备，不分配Colab。MPS禁用CPU静默fallback。
- 拟定调用上限：**4次模型forward**，1次F模式热身，F/C/V各1次计时；全部运行同一人工图和框。`count()`返回值只在内存短暂存在后丢弃，不序列化count、boxes、scores，只保存计时、输入尺寸、成功/异常、forward数与内存峰值。
- 拟定硬停止：**墙钟180秒、进程RSS8 GiB、forward不超过4**；由外部watchdog执行硬停止。如果MPS算子不支持或出现非有限输出/设备混用，立即停止并保留错误；不改网络、换attention实现、开启fallback或自动转CPU继续。
- 输出：模型加载秒数、3个实际延迟、进程峰值RSS、MPS实际分配内存峰值（若该设备获准）、源码/环境/配置hash、forward计数。分别用观测最大延迟×剩余query数给出保守总时长估计，并明确3次样本不保证正式运行耗时。
- 放行点：父线程根据兼容性与成本，再明确正式观察预算和设备。**本阶段没有授权这4次forward、MPS或180秒计算，不能自动启动。** 若此提案预算不合适，先修改方案而非先跑再解释。

这份提案不代表MPS已验证。当前CPU加载和token/group辅助函数测试，只能证明权重和接口具备下一步检查条件。
