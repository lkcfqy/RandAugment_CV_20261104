# 独立审查原文

来源：`/root/independent_review` 的完整完成报告。创建请求 `gpt-6.1-sol / max`；未获得实际 resolved model／effort。独立上下文、同模型家族、provisional。下列行号引用针对 `review-inputs.json` 中冻结初稿，不是修正后行号。

---

总体 **PROCEED WITH CAUTION（provisional）**。候选 A 通过早期查新门槛，但差异比初稿显示的更薄；候选 B 可按 **PROCEED** 推进测量问题。没有核实到某篇已包含完整拟议结果，因此没有依据给出 ABANDON。

**最近先例与剩余差异。** [ForAug v4](https://arxiv.org/abs/2503.09399v4) 更新于 **2026-07-01**，早于截止日。它已比较原／同类／全类背景、两种修补器、边缘平滑及掩码扰动，评估自然 OOD，还报告了分类型背景过滤留存。相关内容见[正文 §3–4、附录 G/L/M](https://arxiv.org/html/2503.09399v4)。

因此，A 可保留的一句话差异是：**在同一前景的配对语义保持编辑与背景去相关编辑之间，预先锁定处理对比，再检验该对比及训练收益能否跨预留编辑器、背景支持和未编辑自然测试复现。** 我所读的 ForAug 部分未呈现这个锁定后的完整检验；这不是证明其他文献没有做过。单独增加背景替换、平滑、换修补器或自然 OOD，已经不足以支撑贡献。

建议修正以下五处：

1. **更新直接先例与来源归属。** [报告第55行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/feasibility-report-zh.md:55) 和[目录第58行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/evidence-catalog.md:58) 应将 ForAug 提升为实读直接先例，并收窄上述差异。父线程新提供的 AGA 核查也应采纳：将“提出 SIC”改为“使用既有 SIC”；该项属于父线程原文核验，我没有独立重读。

2. **分开固定模型诊断与训练收益。** [报告第57行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/feasibility-report-zh.md:57) 的固定分类器只能测量编辑流程造成的预测变化；它不能直接估计用这些图像重新训练后的自然 OOD 收益。应预先分别定义两个量：固定模型的配对损失差，以及不同增强训练方案所得模型的自然测试风险差。后者需在训练方案／训练运行层面比较，不能把每张编辑图当成独立训练实验。

3. **把关键交叉设计落实成可执行定义。** [报告第67行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/feasibility-report-zh.md:67) 尚未明确：E2 是用于测试 E1 训练模型的输入敏感性，还是锁定方案后重做完整训练对比；二者支持不同结论。“背景支持留出”也须规定按背景文件、来源还是语义类别留出。此外，应明确目标背景分配如何打破 `Y×B` 关联；总是反转原背景并不自动去相关。自然 OOD 的具体任务仍未落地：Waterbirds 本身半合成，而 iWildCam 独立复验需要自己的训练与预算。

4. **修正 B 的“上界”措辞并明确审计组别。** [报告第89行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/feasibility-report-zh.md:89) 的“人工有效性诊断上界”应改成“使用额外人工信息的 oracle 诊断对照”；人工有效筛选不保证下游风险的数学上界。`r_g` 是有效样本留存率，误拒率为 `1-r_g`；组别应明确采用实际编辑结果中的目标标签与背景，而非直接沿用源图组标签。其独立有效性分母仍区别于已有剪除数量／组成报告。

5. **完善统计支持与预算口径。** [报告第75行](/Users/lkc/Documents/Codex/2026-10-21/task/causal-augmentation-research-20261020/feasibility-report-zh.md:75) 的 `80×2×2×2=640`、第124行的 `6×3=18`、第128行的 `4795×2=9590` 均无算术错误，但不是完整实验预算。应预先指定主要对比和每组有效独立主体数；若要外推到新的背景总体，还需处理背景复用造成的相关性，仅按前景聚类未必充分。跨编辑器完整训练、独立自然任务、失败重试和超参数选择均可能增加成本。640 次编辑适合发现明显失败，不能据此承诺细小群体差异的统计功效。

报告的主要因果边界是正确的：随机流程效应不等于纯背景语义效应；sham 不保证伪影相同；迁移成功不证明真实 `do(B)` 或个体反事实可识别。数据部分也明确承认 CSV、配对 manifest、图片检查及部分许可链未完成，没有把入口可读写成数据就绪。这些应保留；本次文献审查没有提供实验有效性证据。

实际阅读范围：三份冻结文件全文；ForAug v4 的摘要与版本记录、§3.1–4.3、附录 A、G、L、M，以及返回的相关表格文字。未逐幅检查图像、运行代码或复算实验数值；其余旧邻居主要依据冻结目录，没有在本次逐篇重新验证。

**OntoAug 阻碍：** [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_OntoAug_Rethinking_Generative_Data_Augmentation_via_Ontology_Guidance_CVPR_2026_paper.pdf)、对应 HTML 和会议索引的直接读取均返回 **403 Forbidden**。官方来源的搜索索引可确认 CVPR2026 条目，并显示前景掩码与多样背景提示的片段；这不足以核验其控制实验、精确公开日期或排除完整设计重合。状态为访问被拒绝，未取得正文，无待执行或结果不明动作；已停止追加尝试。

模型审计：本任务明确要求 `gpt-6.1-sol / max`；我的可见回执未提供 resolved model／effort，不能据提示词或自述证明实际运行设置。本次是同模型家族、独立上下文的 provisional 审查。未修改文件或设置，未读取会话存储或秘密，未安装、训练、生成推理、使用 GPU/A100、付费、上传或访问 8A。
