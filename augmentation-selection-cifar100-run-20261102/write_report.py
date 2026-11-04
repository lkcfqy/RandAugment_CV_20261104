"""Render the complete measured report from saved, verified results."""
from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
r=json.loads((P/'RESULTS.json').read_text());run=json.loads((P/'RUN_RECEIPT.json').read_text());v=json.loads((P/'FINAL_VERIFICATION.json').read_text());lim=json.loads((P/'LIMITS.json').read_text())
keys=[f'{m}:B{b}' for m in ['dinov2_small','deit_tiny'] for b in [8,16]]
labels=['DINOv2-S / 8','DINOv2-S / 16','DeiT-T / 8','DeiT-T / 16']
md=['# CIFAR-100 全类增强选择复核：完整结果','','已完成唯一冻结方案：2 编码器 × 12 支持集 × 2 标签预算 × 3 策略 × 5 拟合（四折及全标签）=720 个 ridge 头；48 次 CV 选择共同封存后才开始查询。没有删策略、删类、改正则、改并列规则或增加第四轮。','',
'结论是有限的：四组均出现错选成本 C 大于事后选择空间 H，但 CIFAR-100 的 H 仅 0.0033–0.0484 pp，三组 CV 优于预先固定弱增强、四组均优于固定 RA。不能据此概括成 CV 普遍无效。DeiT/B8 的阶段排序描述和 DeiT 的预算描述未复现；这些反证完整保留。当前仍不足以称为可靠论文贡献。','',
'S=确定性标准预处理，W=普通弱增强，R=W 后 RandAugment。准确率是共享 9,990 张查询图上的 micro accuracy，再对全部 12 个支持集取均值；差值单位 pp。B 是每类标签数。','',
'| 编码器 / B | 固定 S | 固定 W | 固定 R | 原 CV 规则 | CV−W | CV−R | 选择 S/W/R |','|:---|---:|---:|---:|---:|---:|---:|:---|']
for k,label in zip(keys,labels):
 s=r['summaries'][k];f=s['fixed_mean'];c=s['selected_counts'];d=s['paired_distributions']
 md.append(f"|{label}|{f['clean']:.4f}|{f['weak']:.4f}|{f['ra']:.4f}|{s['selected_mean']:.4f}|{d['selected_minus_weak_pp']['mean']:+.4f}|{d['selected_minus_ra_pp']['mean']:+.4f}|{c['clean']}/{c['weak']}/{c['ra']}|")
md+=['', '每组 12 个成对差值全部列于 [ALL_EPISODES.md](ALL_EPISODES.md) 并画于 [paired_differences.png](paired_differences.png)，没有只报均值或挑例。CV−W 的正/零/负次数依次为 3/5/4、7/4/1、4/7/1、7/4/1；CV−R 为 10/2/0、12/0/0、12/0/0、12/0/0。','',
'H = 每支持集事后最佳策略的平均准确率 − 最佳固定策略的平均准确率；C = 每支持集事后最佳 − 原 CV 所选。二者都依赖查询标签，只用于诊断。尤其 C−H 恰等于最佳固定均值减 CV 均值；这不是新定理，也不是已证明的机制。','',
'| 编码器 / B | H | C | 全部 CV 并列候选的 C 下界 / 上界 | C>H / Cmin>H | 错选数 | 同一对策略在折查询已差 / 错选数 |','|:---|---:|---:|:---|:---|---:|:---|']
for k,label in zip(keys,labels):
 s=r['summaries'][k];lo,hi=s['tie_bounds']['mean_regret_bounds']
 md.append(f"|{label}|{s['H_pp']:.4f}|{s['C_pp']:.4f}|{lo:.4f} / {hi:.4f}|{s['C_gt_H']} / {s['Cmin_gt_H']}|{s['wrong_full_count']}/12|{s['same_pair_fold_inferior_wrong_count']}/{s['wrong_full_count']}|")
md+=['',
'本次 48 次 CV 选择均无最高分并列，每组完整并列组合数量都是 1，因此上下界等于原结果；未发明新的 tie rule。[ALL_CV_TIES.md](ALL_CV_TIES.md) 及 RESULTS.json 保存候选集合。严格不等式使用整数正确数之和判断，避免舍入将等号变成“成立”。','',
'阶段规则是看原 CV 所选与最终全标签最佳之间的同一对策略；不是把任意第三策略的差异称作验证噪声。DINO/B8 是 7/9，DINO/B16 是 2/3，DeiT/B16 是 4/4；DeiT/B8 只有 3/7，未达到事前定义的多数条件，其余 4 个错选在折查询时反而更高。三个阶段全部 48 套排序和正确数在 [STAGE_RANKINGS.md](STAGE_RANKINGS.md)，每折原始正确数在 [ALL_FOLD_COUNTS.csv](ALL_FOLD_COUNTS.csv)。这是有限验证判别与重训练后排序变化的描述，不能纯因果分解。','',
'预算子描述同样有限：DINO 的 H 从 0.0484 降至 0.0434 pp；DeiT 从 0.0033 升至 0.0142 pp，故“更大预算时选择空间下降”在 DeiT 上被反证。C 随预算下降并不能挽救这个不同命题。','',
'与 EuroSAT 的既有结果相比，C>H（包括完整并列候选下界）在本次四组继续出现，但固定最佳策略由增强侧转向标准预处理；CIFAR-100 的策略间平均差异并不为零，按支持集切换策略相对一个固定策略的额外空间却很小。有限经验陈述可写为：在这两个数据集、两个冻结编码器和此三个固定策略下，四折 CV 的支持集特异选择没有兑现超过事后最佳常数策略的收益。不能把事后最佳常数当作事先可部署的已知选择；也不能从这些相关支持集推普适定律、独立显著性或新算法价值。','',
'数据、实现与解释范围：','',
'- 仅既有 CIFAR-100 全部 100 个 fine 类。历史资格规则排除 34 个冲突标签条目及 9 个重复拷贝，保留 49,967 train / 9,990 test；本轮没有按模型效用再排样。每类 hash 固定 100 个候选，12 套支持集 B=8/16 嵌套，支持并集 8,751 张；support/query 的 ID 和 RGB hash 均不交叠。四折内同原图全部视图在同一折。',
'- 标签总量是 800/1,600；每折验证 200/400，明显多于 EuroSAT 的 20/40。图像域、类别数、总标签/验证量及分辨率均不同，不能将结果变化归为纯域效应。两个数据集均已有本地历史研究，不能叫全新确认数据。共享查询和相交支持集不构成 12 次独立试验；没有伪独立 p 值或置信区间。',
'- DINOv2-Small 与 DeiT-Tiny 使用原本地安全加载权重，冻结 FP32 编码器；最后归一化 CLS（384/192 维）逐视图 L2 归一化。标准 resize 依模型官方配置为 256/248，中心裁剪 224；不能假称两模型完全相同预处理。',
'- W：RRC 224、scale=[.5,1]、ratio=[.75,4/3]、bicubic、flip=.5；R：同一弱视图后 torchvision RandAugment N=2/M=9/31 bins、bilinear、fill=[124,116,104]。每原图四视图，S 重复同一确定性特征；ImageNet mean/std。视图参数由预固定 ID/盐值决定，未按效果调强度。',
'- Ridge 为 FP64、lambda=.001、每原图同总权重（每视图1/4）、截距不惩罚。CV 按验证正确总数选择，并列固定 S>W>R；全部策略均以全标签拟合并报告。模型完整 state hash 在每次读取前后相同。',
'',
'冻结 [PROTOCOL.json](../augmentation-selection-cifar100-preflight-20261102/PROTOCOL.json) 和 [IDENTITIES.json](../augmentation-selection-cifar100-preflight-20261102/IDENTITIES.json) 原样保留。既有 EuroSAT 综合档案见 [原综合报告](../augmentation-selection-synthesis-20261102/REPORT_ZH.md)。[VEST](https://arxiv.org/html/2506.04713v4) 使用检索 OOD 与 ID 评价作 CLIP 微调 checkpoint/layer 选择；验证稀缺与模型选择不是首次问题。本实验没有实现 VEST，也不构成其公平性能比较，详见 [相关工作边界](../augmentation-selection-cifar100-preflight-20261102/RELATED_WORK.md)。','',
'验证与成本：','',
f"- 4 项正式入口集成测试通过：缺第二编码器则拒绝查询、720 头完整 fixture、身份/文件篡改拒绝、100 类紧凑无损保存及 argmax 一致。预检另验证嵌套/折划分、精确并列界及两编码器 64 个弱/RA 视图一致。最终从原精度特征和头复算全部 7,192,800 个查询决策及 172,800 个 OOF 决策，逐个一致；全部 720 指标行、576 折记录和 48 选择完整。",
f"- 正式前向 177,498 图视图，720 个解析头拟合；预检另有 192 个支持图视图前向和 12 次合成 ridge，不含真实查询效用。最高模型工作进程 RSS {v['peak_worker_rss_bytes']/2**30:.3f} GiB（非整机总内存），4 个工作进程累计实测 {v['worker_wall_seconds_sum']/60:.3f} 分钟。",
f"- 从 05:36:42 UTC 准备起连续计时，核心计算于 06:40:30 UTC 完成，累计 {run['cumulative_wall_seconds']/60:.3f} 分钟，包含准备、等待和集成；正式监督流程约 {(datetime.datetime.fromisoformat(run['completed_utc']).timestamp()-lim['start_epoch'])/60:.3f} 分钟。报告和归档继续计入同一个 90 分钟上限（07:06:42 UTC），最终实际计时/字节见 TASK_RECEIPT.json 与归档回执。未重置预算。",
'- 使用既有 CPU 环境，Torch/数值线程4、interop1；无 MPS/GPU/Colab、新 CU、下载、安装或上传。权重和原始图像没有复制进本档案。所有已启动模型子进程由监督进程等待退出，执行 exit_code=0；没有调用此前被拒的 ps/sysctl。',
'- 原始 FP32 特征、FP64 头、全部 uint8 类别预测及所有 OOF 身份保存。未存约 5.75 GB 浮点分数立方体；同一冻结 256-query 分块代码可由特征/头精确复算，非量化或候选删减。新增文件总量含预检必须 <500 MiB；最终校验见 [FINAL_VERIFICATION.json](FINAL_VERIFICATION.json)。',
'- 工程记录：查询前仅修正分析严格比较为整数计数并固定分析线程，保留旧源及前后 hash；编码器/拟合代码和科学定义未变。归档验证脚本首次把旧 manifest 的列表误作字典，随后按既有 schema 修正并通过，不涉及模型或结果；另有一次函数工具语法错误发生在任何命令执行前。详见 ENGINEERING_FINALIZATION.json。',
'',
'本轮结束：没有未完成设置，没有效用驱动重试或第四轮。成果是可复算经验记录和清晰的命题边界，目前不是可靠论文贡献；演讲文件未改。']
(P/'REPORT_ZH.md').write_text('\n'.join(md)+'\n')
(P/'PREDICTION_SCHEMA.md').write_text('''# Saved prediction and identity schema

The immutable `../augmentation-selection-cifar100-preflight-20261102/IDENTITIES.json` supplies ordered support_union (8,751) and query (9,990) records with original IDs, source indices, RGB SHA256 and fine labels. The original images and model weights are referenced, not copied. Protocol SHA256: `25c62a9ac48f42b619e85158d377d33da1e263f083b732bc370c1cd17298631d`; identity SHA256: `a79508bafe2ddeedccf07096cc977f8a15aa8940304f7b6f7ef2fbf7010339ae`.

For each encoder:
- `HEAD_IDENTITIES.json`: all 360 rows in canonical episode/B/policy/fold order; fold=-1 is the full-label refit, folds0..3 are CV training heads. Columns0..99 in W/b and classIDs refer to original CIFAR fine labels.
- `support_features.npy`: float32 [8751,9,D], view0 standard;1..4 weak;5..8 weak+RA. Standard policy uses view0 repeated4 times, with the same per-original objective weight.
- `query_features.npy`: float32 [9990,1,D], standard input only.
- `W.npy`, `b.npy`: float64 [360,D,100] /[360,100], D=384 (DINO) or192 (DeiT).
- `QUERY_PREDICTIONS.npy`: uint8 [360,9990], every head/query argmax. Full float64 scores reconstruct as features.astype(float64) @ W[h] + b[h] in the original 256-query blocks; argmax chooses the smallest class index on score ties. All reconstructed argmax values have been verified.
- `CV_PREDICTIONS.npz`:86,400 rows; head_index uint16, support_union_index uint32, fine label uint8, predicted class uint8. Every held-out original is predicted once per fold head/policy/episode/B. CV uses the original standard view, not augmented validation views.
- `FIT_SEAL.json`: hash-pins all heads/support features/CV decisions and24 original selections. Both such seals must precede `JOINT_QUERY_GATE.json` (720 heads/48choices), which precedes both query evaluations.
- `QUERY_SEAL.json`: all exact query feature/prediction hashes and the common gate hash.

`ALL_HEAD_METRICS.csv` has720 complete query metrics; `ALL_FOLD_COUNTS.csv` has576 rows of actual validation/query counts; `RESULTS.json` contains all48 settings, full candidate/tie sets and four complete summaries. `STAGE_RANKINGS.md` exposes complete stage orderings, including equalities. No missing setting is imputed and no query model was refit after selection.
''')
(P/'ENGINEERING_FINALIZATION.json').write_text(json.dumps({'resolved':True,'scientific_changes_after_query':0,'new_model_runs_or_fits':0,'events':[{'action':'functions.exec orchestration for writing verifier','reason':'JavaScript SyntaxError missing closing parenthesis','execution':'not executed; same intended local command submitted with syntax corrected','permission_denial':False},{'action':'verify_and_index.py first execution','reason':'expected old manifest files as dictionary, actual schema list of path/bytes/sha256','execution':'stopped before numerical verification; changed reader to list schema, then all checks passed','permission_denial':False}],'query_before_selection':False,'budget_reset':False},indent=2)+'\n')
print('wrote complete report/schema/engineering note')
