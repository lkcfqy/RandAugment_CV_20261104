# 历史路线排除台账，2026-10-29

本表汇总项目已登记且本轮可定位的路线及父任务明确关闭项，不声称恢复了未记录的口头想法。只读文字报告/现有状态，不重开旧测试或运行；早期“positive/retained/heldout”由较新结论校正。没有把所有停止原因写成方法无效。

原因码：P＝既有解释/增量不足；I＝干预、支持或测量不可识别；G＝指定投入门未通过；B＝基线/执行不足；D＝数据或材料缺口；C＝只支持代码/数学/描述性结论。多码可并存。

| 路线 | 当前准确结论和停止原因 | 证据（相对仓库根路径） |
|---|---|---|
| 早期 CER/线性因果增强 | C/I：线性结构模型推导不能外推为真实图像分类中的已识别因果效应；保留数学记录 | `causal-augmentation-research-20261020/operation-law-audit-20261021/finding-zh.md` |
| 面积/尺度捷径与过滤训练 | G/I：固定小样本/冻结表征的条件结果，不能声称获得新的因果不变表征或普遍收益；旧测试已消费 | `CURRENT_EVIDENCE_STATUS_20261021.md`；`causal-augmentation-research-20261020/area-pairing-cpu-20261021/measurement-scope-decision-20261021/decision-zh.md` |
| 早期短阶段 positive | B：仅小阶段 loss 符号，训练不足；不是 TrivialAugment 收益证明。后来的健康训练不能倒填该结论 | `CURRENT_EVIDENCE_STATUS_20261021.md`；`causal-augmentation-research-20261020/evidence-quality-closeout-20261021/correction-index.json` |
| Equalize 前景—背景配对 | G/P：已见操作损害，未证实保持边际后的额外配对损害；全图直方图的非局部性已知 | `causal-augmentation-research-20261020/area-pairing-cpu-20261021/equalize-pairing-cpu-20261021/result-zh.md` |
| 区域独立亮度 | G：背景敏感性下降约 3.53%，低于预定 20% 投入门；不是准确率提升，也不是零效应证明 | `causal-augmentation-research-20261020/area-pairing-cpu-20261021/brightness-coupling-training-20261021/result-zh.md` |
| 联合 clipping/饱和诊断 | C/I：合成阈值推导不等于相机/真实标签模型；零错误事件时 AUROC 不可评价，不能叫无效诊断 | `causal-augmentation-research-20261020/joint-clipping-mechanism-20261021/finding-zh.md`；`CURRENT_EVIDENCE_STATUS_20261021.md` |
| batch 共享与独立增强状态 | P/G：API 抽样粒度本身不是 bug；固定状态库的训练交互未达事前门，不是整个 RA 无效 | `causal-augmentation-research-20261020/augmentation-sampling-granularity-20261021/finding-zh.md`；`augmentation-coupling-groups-20261021/finding-zh.md`（同父目录） |
| smallNORB 方位起点/取样 | P/I：坐标重命名不改变相同图像；固定数字窗口换图是取样变化，缺少增强特有新预测 | `causal-augmentation-research-20261020/official-intervention-data-gate-20261021/stage-result-zh.md` |
| Invert 与 BN 混合 | P：分离强增强和干净图 BN 已有直接先例；尚无可区分解释，不开换数据集复演 | 同上 `secondary-mechanism-screen.json` 的文字总结 |
| NORB 立体不同步/填充 | P/G：同步原则及允许非对称增强均有前作；具体填充/内容解释未过门，不扩大 fill 网格 | `causal-augmentation-research-20261020/norb-stereo-mechanism-gate-20261021/stage-result-zh.md` |
| NORB view×lighting 分配 | G：J*≈−0.0000126745 nat，条件区间跨零且远低于投入量级；这是两分配收益差，不能说增强收益为零 | `causal-augmentation-research-20261020/causal-resume-20261022/finding-zh.md`；`CURRENT_CHECKPOINT_20261022_CONTINUED.md` |
| 代理错误位置→定向背景预算 | P/I/D：BARACK/GIC 已有相关错误结构；固定名义编辑数未固定真实反例数；缺少独立标签/材料支持 | `causal-augmentation-research-20261020/causal-next-question-20261022/candidate-and-stopping-decision-zh.md` |
| 编辑器来源捷径 | P/D：ForAug/FFR 已覆盖直接动机；跨编辑器比较尚无区别于既有解释的预测和完整 sham 对照 | `causal-augmentation-research-20261020/causal-mechanism-map-20261022/decision-details-zh.md` |
| 多捷径联合覆盖/不完整支持 | P/I：Whac-A-Mole、T2I balancing、AutoBackSwap 已覆盖支持/再平衡；不能把完整因子网格当自然因果识别 | 同上 |
| 背景编辑造成标签损伤 | P/D：AutoBackSwap/ForAug 已讨论掩码和实例完整性；数据 QA 可有用，泛泛问题不构成新方法/机制 | 同上 |
| 冻结骨干 vs 微调代理 | C：是测量平台校准，不是新算法；历史未执行 A100 提案已被后续路线接续，不是当前待批任务 | `area-pairing-cpu-20261021/measurement-scope-decision-20261021/decision-zh.md`（因果研究目录） |
| LoRA 前后位置×RA | G：交互 +0.33956pp，低于原 1pp 门；单 seed，照片区间不是训练 seed 不确定性；不追加配方救阳性 | `CURRENT_CHECKPOINT_20261022_LORA_PILOT.md` |
| CutMix 后增强面积目标 | P/I：原备选与 OA-CutMix 重叠；未形成可隔离的新预测，没有正式实证通过记录 | `CURRENT_CHECKPOINT_20261022_AUGMENTATION.md`；`augmentation-research-20261022/proposal-zh.md` |
| 控制器温度/选择性置信度 | G/P：固定 checkpoint 主比较只有 5.60% 样本显著移动排名，未过 10% 门；不从次比较挑阳性，也不否定所有自适应增强 | `augmentation-research-20261022/rescreen-20261022/first-screening-zh.md` |
| 连续插值 vs 合成变换 | P：fastai 已明确合并几何操作减少细节损失；额外模糊的演示不新。近邻访问限制不等于文献空白 | 同上及 `FOLLOWUP_CLARIFICATION_20261022.md` |
| RA 整数位移与 ViT patch 相位 | I/P：子过程 gcd 事实不证明含 RRC/旋转等的完整管线缺覆盖；位移敏感性已有先例 | 同上 |
| 同一噪声标签同时进损失与增强控制器 | P/I：宽泛双输入/置信度增强主张关闭；四格草案未证新颖性，BN/Dropout/动作路径须匹配，未获训练准入 | `augmentation-research-20261022/rescreen-20261022/label-noise-dual-input-audit-20261022/AUDIT_ZH.md` |
| 错误持续性、较小视图响应 | C：已用 dev 上有窄描述与反例；敏感性变小不保证正确、语义保持或校准变好。不能按错误分组选图后推总体机制 | `augmentation-research-20261022/ra-error-persistence-20261023/REPORT_ZH.md`；`ra-dev-paired-cpu-20261023/REPORT_ZH.md`（同研究目录） |
| 删算子/补 Identity 的 N/M 组成 | C/P：抽样律不同的精确计算正确；有效深度、替代解释有先例，不能当新归因方法 | `augmentation-research-20261022/nm-composition-screen-20261023/REPORT_ZH.md`；`nm-ablation-prior-gate-20261023/EVIDENCE_GATE_ZH.md` |
| Posterize B/Z/R 三臂 | G：B/Z 552、R 550（共589 dev），未得到预定实质异号；保留不确定性，不称等价或算子无效 | `augmentation-research-20261022/posterize-three-arm-mps-20261024/RESULT_ZH.md` |
| RUA 单峰搜索假设 | P/D：泛泛非单峰/噪声质疑关闭；缺同版本 raw 分数、搜索轨迹及复制记录，未观察真实搜索损害 | `augmentation-research-20261022/rua-unimodality-readonly-20261023/decision.json` |
| SRA shape 分支 | C：特定 B=C 人造边界下的实现影响已验证；未证明实际训练配置损害，代码问题不能自动作研究贡献 | `augmentation-research-20261022/sra-shape-impact-20261023/REPORT_ZH.md` |
| SRA 匹配实例—动作 | P/I/D：仅添 shuffle 消融缺不同预测；完整动作/RNG/BN 路径未识别；PBA/MADAug/InstaAug 是强近邻 | `augmentation-research-20261029/sra-pairing-gate-20261029/GATE_ZH.md` |
| SADA KL 代理 | C/P：数学上能逆序，但 soft-target 替换是明示启发式；已知预测漂移/监督梯度不等价，不是隐藏 bug 或经验伤害 | `augmentation-research-20261029/sada-proxy-audit-20261029/VERDICT_ZH.md` |
| LA3 Invert/几何补边顺序 | I/G/P：窄像素机制成立；CIFAR 无条件顺序差，ImageNet 相关条目质量仅0.006931%；全图严重度匹配失败、同类双方向无原生支持 | `augmentation-research-20261029/la3-fill-order-audit-20261029/REPORT_ZH.md` |
| Data Echoing / Revamper 缓存三臂 | 继承父任务第五轮 Pro 的 NO-GO；本轮未找到独立完整评审正文，不虚构其数值或具体实验失败原因；不重开 | 父任务授权/终止消息；本地 `a2er-formula-audit-20261029/REPORT_ZH.md` 也记载保持关闭 |
| A2ER 原生图像增强归因 | I/D：官方样例固定输入；未发布图像入口不能被补写成作者实验。官方仓库存在，不写成“无代码” | `augmentation-research-20261029/a2er-formula-audit-20261029/NATIVE_PATH_GATE_ZH.md` |
| Fresh-replay 760步小前缀 | B：新任务基础学习不足；后续健康检查未发现确定实现 bug。不是残差机制失败 | `augmentation-research-20261029/fresh-replay-reference-pilot-20261029/REPORT_ZH.md`；`fresh-replay-health-20261029/REPORT_ZH.md` |
| 独立 Mammoth 基础学习诊断 | 基础学习通过，保留为平台能力；不使旧760步协议通过，不构成新机制结果 | `augmentation-research-20261029/mammoth-derpp-baseline-20261029/REPORT_ZH.md` |
| 健康模型上冻结 Gate A | G：两组匹配 TV≈0.00482/0.00612，低于.05；身份差一致性亦未过门。仅关闭指定参考组合；已知方差恒等式不是贡献 | `augmentation-research-20261029/frozen-derpp-gate-a-20261029/REPORT_ZH.md` |

父任务额外明确禁止重开“短预算”作为机制主线；上表中训练不足的条目只能按 B 类解释。不能通过换时长把旧停止记录改成科学成功。

共同证据纪律：旧 Pet 64/48/16 等探索池、589 dev、smallNORB 已用对象不得改名独立确认；照片/主体是独立性单位，视图、seed、背景和检查点不是新照片。封存数据保持封存。数值门是投入决定，不是等价性检验。真正的实证贡献可以是新的边界/失败规律，但需超出已知解释并有适当独立验证；不要求为了贡献发明新网络。
