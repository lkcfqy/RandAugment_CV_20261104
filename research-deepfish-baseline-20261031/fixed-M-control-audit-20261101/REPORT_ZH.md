结论：可以把历史seed20261031 A作为明确标记的历史对照，准备一个固定M候选；配置及数据控制已有逐项证据。但新M的完整初始化、首次更新RNG和实际CUDA环境尚未运行，必须作为未来开始训练前的硬门，不能现在宣称新模型已满足同控制条件。此次没有模型实例化、训练、GPU分配、上传或依赖安装。

**历史控制核查**

| 控制项 | 已核实证据 | 新M未来必须通过的条件 |
|---|---|---|
| 官方原始Swin-T权重 | 本机重新流式SHA256核验，`822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3`；BERT七项资产也全部匹配原pins。没有torch.load或模型构造 | 官方原URL直接下载，严格相同SHA；只用weights_only安全加载 |
| 完整初始state | 历史preflight_A/B、formal_A/B四份初始化回执均为 `d40b615baf208851b234590e9022d83c3577dae444a9b227cf6e4777aa1b4f0c`，tensor字节691880148；包含单类DN embedding及7个官方零初始化log_scale，不能只比较预训练共有键 | 新M预检和正式各自独立初始化后，整个state按名称/dtype/shape/字节重算，必须等于该hash；不同即停 |
| 可训练参数 | 四份回执的有序可训练参数名hash均为 `62f7c4fa11c27bb72c9d472e1885d9a2ff1cd02f74523f7a53ea334770c4fe3a` | 完全相同；不以修改冻结策略绕过 |
| optimizer/LR | 历史空optimizer状态；AdamW lr1e-4、wd1e-4，backbone lr_mult.1，absolute_pos_embed decay_mult0，clip_grad.1；无AMP/累积设定。历史日志80个LR打印值均1e-4；它不代表所有参数组都为1e-4 | optimizer必须为空；按官方最长custom-key优先规则检查每个参数组的LR/decay及AdamW默认值，另保存组明细。历史没有完整组明细，不能伪造为已实测逐组一致 |
| scheduler/U/batch | param_scheduler=[]、auto_scale_lr关闭；正式实际U200、40轮、每图anchor40次；batch1、workers0、sampler shuffle=True/round_up=False | 完全保留；原配置max_epochs=0只是占位，由worker按固定U/5设置40；预检15次另计并丢弃所有状态 |
| 数据及管线 | M配置从实际历史A配置复制，结构递归比较仅 `train_dataloader.dataset.pipeline[0].transforms` 不同；相同5图support和64图dev、ID/标签/元数据未改 | M的额外随机抽样、拼图/混合、裁框和更多来源曝光是策略差异，不要求M/A几何、GT数量或后续NumPy轨迹相同 |
| runtime | 复用已成功的Python3.10.19、Torch2.1.0+cu118、MMCV2.1.0、MMEngine0.10.7、MMDet3.3.0、NumPy1.24.3、OpenCV4.11.0.86；原bootstrap含五个核心wheel SHA和Python来源pins；额外冻结历史全部56项已安装版本与54项安装下载记录（53项有历史SHA，Jinja2记录缺SHA） | 必须再次验证实际安装与官方源码pins。Jinja2版本已锁定，但不能声称其历史下载字节已匹配；额外算子/optimizer源码hash已加入候选包；这些文件并非全部在历史GPU回执中单独哈希，不能声称历史逐文件实测全覆盖 |

[初始化对照](HISTORICAL_INITIALIZATION_AUDIT.json)、[完整配置控制比较](CONFIG_CONTROL_COMPARISON.json)、[资产重新哈希](ASSET_REHASH.json)、[历史LR日志核对](HISTORICAL_LR_LOG_AUDIT.json)。保留的历史A最终checkpoint重新核验为 `5a2a8d4955c0a91641c9e66ac879740077fc4640f06379e850640a46da2268f0`。

**无模型重建不是只对文件名**

用现有官方CPU数据环境，重建原A完整200次数据轨迹，逐项对照历史正式A已有记录：**1588个算子步骤**的操作/分支路径、变换前后尺寸、scale、flip、homography、GT、类别和ignore，以及NumPy/Python绝对RNG hash全部一致；200次sampler顺序、最终输入尺寸、GT也一致。新worker逐次检查anchor ID等于历史顺序，不只检查每图计数。200次Color均逐像素恒等，且通过独立局部RandomState核实恰好消耗一次NumPy uniform抽样。[重建核验](A_REPLAY_VERIFICATION.json)、[完整新轨迹](A_REPLAY_TRACE.json)、[逐项检查](A_REPLAY_CHECKS.json)

Torch CPU方面，历史有40种管线入口状态，包含模型初始化和DataLoader迭代器播种；当前不构造模型，也没有完整历史RNG字节，不能从hash反推绝对状态。只验证了每个数据算子在历史与重建中均不消耗Torch CPU RNG。CUDA没有运行。未来M增加了首个更新前的Torch CPU/CUDA hash硬门，并继续检查整个初始state；这些是待执行门，不是已经通过的结果。

因此，单模型比较的表述应是“同seed、固定anchor更新预算下，整套固定混合策略相对已记录A的探索”。不声称曝光匹配因果效应或ETS论文复现。若未来真实runtime或训练/评估控制改变、初始化门不匹配，不能继续把A称作同控制条件；需要明确修复原因或重新考虑重跑A。这里的成本只覆盖一个新候选，不包含重跑A。

**29.16分钟估计包含什么**

原估计1749.83s/.55897CU包含以下项目，采用历史t_M=1.68071s及历史1.15CU/h作为假设，均非新runtime实测：

| 项目 | 秒 |
|---|---:|
| 环境、官方资产准备和吞吐预检的原保守余量 |554.53|
| 正式训练与dev评估两次重新加载 |40.49|
| 200次更新，含50%分项缓冲 |504.21|
| 64张dev评估，含50%分项缓冲 |39.78|
| 一个完整692175962B checkpoint回取，按历史带宽并乘2 |310.82|
| 日志、校验、保存和收尾杂项 |120.00|
| 外部释放预留 |180.00|

预检定义仍为M的15次更新（5次预热+10次计时）和5张support前向，不读取dev指标；状态全部丢弃，正式重新初始化。554.53s是沿用的旧准备/预检余量，没有因减少为单臂而调低，也不是新M已完成的吞吐验证。

**原估计没有统一25%余量**。它已有训练/评估50%、回取100%的分项缓冲，不能事后把它称作全程统一25%。若在原估计上明确加统一余量，结果如下：

| 透明口径 | 新增时间 | 新增CU（历史费率） | 累计CU |
|---|---:|---:|---:|
| 原分项缓冲，没有统一25% |29.16min|.55897|2.36781|
| 工作小计再加25%，释放180s单列 |35.70min|.68434|2.49317|
| 原总计包括释放再整体加25% |36.45min|.69872|**2.50755**|

两种25%口径均超过当前父级自设的总GPU2小时边界；第二行在用户2.5CU上限内仅余.00683CU，第三行已超过用户上限。不能为了通过而默默切换口径；正式GO前必须明确并冻结定义，然后以实际费率、早期回取带宽、当前累计消耗和M预检时间核完整成本。[成本分解机器文件](COST_MARGIN_AUDIT.json)

用户实际批准的总2.5CU为硬上限；本次没有扩大它。GPU2小时、目录6GiB是父级自设边界，当前仍未修改。历史累计1.8088330816CU和5443.8998GPU秒完整结转。新candidate若不能在最终明确的资源边界内完成，应释放并标资源未定，不减U、不改样本、不看loss/dev选预算。

**本机准备及待传范围**

`payload/`含相同69张公开JPEG、原support5/dev64标签、固定M配置、A参考配置/哈希、官方依赖和资产pins、单M阶段worker及原生命周期/传输辅助脚本；没有自动分配器或自动启动driver。新worker只接受preflight_M/formal_M/evaluate_M，要求明确GO，完整初始化/参数控制匹配，并在完整权重回Mac验证后才允许dev评估。外部权重仍从官方源下载，包内没有模型权重或历史完整checkpoint。

10项纯合同测试和脚本静态编译通过，未导入模型框架；对无GO示例调用worker在框架导入前即拒绝。新candidate模型路径未执行，不能把静态检查包装为GPU运行通过。[测试](NO_MODEL_CHECKS.json)、[相对实际历史worker的改动](CANDIDATE_WORKER_DIFF.patch)、[分阶段可执行方案](payload/RUN_PLAN.md)。准确分类字节、ZIP字节与SHA见 `TRANSFER_INVENTORY.json`；未上传、无test/私人文件/凭据。

当前研究目录已有6,180,710,855字节；6GiB自设上限剩261,740,089字节，尚未计入此次约25.3MB归档。即使物理盘空间充足，也放不下新增约692,175,962字节的完整checkpoint。目录上限需要父级另行明确修订才能容纳后续候选；此次未修改上限、未删旧产物。精确归档后余量见 `ARCHIVE_RECEIPT.json`。

最终待传ZIP为 **11684244字节**，SHA256 `1a0a84b45e90ac216bbe9871302ee18980c5cb371695ec568f5552ba17a18068`；展开94文件/11,929,899字节。类别原图69张11,756,375B，support/dev两份JSON26,641B，7份阶段/环境脚本36,682B，16份配置、pins、归属、计划和清单110,201B。相同69图/2JSON逐字节匹配历史包，ZIP逐条解压校验通过。

独立进程复查 `ps -p 47600 -o pid=,etime=,command=` 被本机沙箱拒绝（operation not permitted，exit127），命令未执行，未重试或换路径。原A重建执行会话已返回exit0；当前纯测试也正常退出。无法补充该次独立进程快照，详见 `ACTION_LIMITATIONS.json`。这不影响已落盘的200条重建核验。
