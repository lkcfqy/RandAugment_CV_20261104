本次唯一正式 MILL 观测实验执行成功，预注册结论为 **NO_UPGRADE**。六场景校准的合成输入没有通过 held-out 真实照明的统计匹配门，因此本轮不能保留“输入统计已匹配而冻结 CIDNet 任务压力仍失配”的测量主张。

这是固定候选在固定 panel 上出现的校准/覆盖缺口。资源和验证通过；无需用资源失败解释结果，也不能据此推广为所有合成照明方法不可能有效。没有更改模型、参数定义、阈值、场景、灯档、种子或十轮迭代；没有补跑挑选结果。

| 方法/对照 | 任务响应 PSNR MAE (dB) | 色差响应 MAE (DeltaE76) | 四档等权相对排序一致率 |
|---|---:|---:|---:|
| A | 4.553469 | 5.039616 | 0.698896 |
| B | 3.085669 | 3.658415 | 0.650962 |
| C_anchor | 4.604558 | 6.718905 | 0.604202 |
| C_meanD | 2.722360 | 4.180375 | 0.500000 |

所有方法的四档真实响应排序集合均充分：L1/L3/L5/L7分别27/26/26/20对，每档覆盖8场景。A、B、C_anchor均未达到CLOSE要求（PSNR MAE<=0.2 dB、色差MAE<=1、四档排序充分且等权一致率>=0.8）。C_meanD保留为常数对照，排序为0.5；其PSNR MAE低于B，不能把B的复杂度本身称为贡献。

| 输入统计门 | 阈值 | A通过单元数/32 | B通过单元数/32 |
|---|---:|---:|---:|
| Emean | 2/255 | 15 | 27 |
| Evar | 0.15 | 0 | 0 |
| EKS | 0.05 | 0 | 1 |
| EPSD | 0.15 | 8 | 7 |
| Ecross | 0.05 | 0 | 0 |
| 五门同时通过 | 每个scene×level独立验收 | 0 | 0 |

32个真实单元均可识别，无STAT_UNIDENTIFIABLE。B每一单元都未通过Evar及Ecross；B通过全部五门的单元为0，所以四档B-stat-pass端点配对均为0，Lbad=[]、不同场景误差见证=[]。continue所需的“至少两档各>=8对、覆盖>=6场景、排序<=0.6，且同Lbad内至少四场景出现>=1dB误差”没有成立。没有用全体单元的低排序替代这个门。B测试源落入空训练箱的比例全部为0。

相对D排序，B的L1/L3/L5/L7一致率为1.000000/0.615385/0.538462/0.450000；C_anchor为0.962963/0.653846/0.500000/0.300000。二者使用完全相同的单元和配对集合，包括为空的B-stat-pass子集。D与C_anchor都用test GT评分信息，仅是诊断，不是无GT部署预测器。

绝对R对合成S的排序使用独立的|R_s-R_t|>=0.5 dB集合：L1/L3/L5/L7分别20/26/27/26对，均覆盖8场景。四档等权一致率A=0.669088，B=0.659829；这些值不进入continue。逐对记录保留，不与D配对混用。可选bootstrap CI未计算，不影响预注册gate。

资源与执行证据：

- 固定代码聚合SHA256：`e7b5ddfaf0beef296612e90138fd37d93337066d520576a6e9a9ba638806f6c7`；18个文件逐一复核未变。
- 正式326次forward完整：train real30 + test real40 + A128 + B128；另有人工预检4次，GT从未forward。
- 先完成8组参数与256张最终Q8图的哈希冻结，再开放test Y/GT评分；原生成manifest保留test_Y_opened=false、GT_opened=false、scientific_forward_count=0的阶段边界。
- 正式会话实测203.988462166999秒，向上取整扣204秒；原1800秒总预算已扣234秒，剩1566秒。没有第二次正式attempt。
- worker峰值RSS=1514995712字节（1444.812MiB）；正式完成时资产目录分配860323840字节（820.469MiB），低于2GiB。
- worker与调度进程成功退出，MPS上下文随进程释放；ledger active=null，lock不存在，stderr为空。

完整复核材料：

- `run-state/formal_result.json`：全部326条逐图指标/输入hash、32单元A/B统计、响应矩阵、所有排序配对、C_anchor同集合比较、完整decision。
- `run-state/formal_launch_output.json`：上述结果加最终父进程扣账事件、总耗时及剩余预算。
- `run-state/generated/generation_manifest.json`：256图与8参数的文件/元数据哈希；对应NPZ无pickle，数值数组自身hash在配套JSON中。
- `reports/cells_32.csv`：全部32单元的真实与预测D、各方法误差、A/B五门及Ecor/Ecoh、可识别性、pass和空训练箱比例。
- `reports/forward_records_326.csv`：全部326次正式输入身份及两种原始标量指标；四draw没有先平均图片。
- `reports/ranking_sets_32.csv`与`reports/ranking_pairs_594.csv`：各配对集合覆盖及594条逐对评分，明确relative_D/absolute_R/B_statpass_endpoints。
- `reports/B_fallback_pixels_128.json`：B全部128 draw的空训练箱fallback比例。
- `reports/FORMAL_REPORT.json`：机械可读结论、每项CLOSE/continue布尔门、资源与证据SHA256；`run-state/compute_ledger.json`为持久总账本。

待办仅为对这份完成结果的独立复核。本记录不启动下一轮拟合、调参或实验。
