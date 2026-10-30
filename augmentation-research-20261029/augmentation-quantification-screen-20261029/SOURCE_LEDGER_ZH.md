# 来源与核实级别

2026-10-29 UTC。标记：F＝核对所列正文/公式/表格（并非逐字通读全篇）；C＝固定官方源码核对；A＝仅官方摘要或原始页面定位；D＝发现线索，未核实正文；X＝访问未成功。检索结果的日期不自动等于论文发布日期。

## 当前评审问题的关键来源

1. **A Unified View of Label Shift Estimation**，Saurabh Garg、Yifan Wu、Sivaraman Balakrishnan、Zachary C. Lipton。NeurIPS 2020；arXiv **2003.07554v3，2020-10-16**。F：§4.1 Eq(1–2)、Lemma1、§4.2–4.3 与有限样本讨论。\
   https://arxiv.org/html/2003.07554v3 \
   正式论文：https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf \
   当前有限搜索没有确认作者可执行仓库；不写成“没有代码”。读取早期未带 v 的 URL 返回的版本头为 v3，快照保存。

2. **Label Shift Quantification with Robustness Guarantees via Distribution Feature Matching**，Bastien Dussap、Gilles Blanchard、Badr-Eddine Chérief-Abdellatif。arXiv **2306.04376v2，2023-07-02**；作者 ECML2023 poster 可定位。F：§2 问题P、§3 Theorem3.1/条件、§4.1 RFF 与源端准则、§4.2 数据。\
   https://arxiv.org/html/2306.04376v2 \
   当前有限搜索未确认具体官方代码仓库，正文公式足以定义独立参考实现，但未写或执行该实现。不能把既有 RFF/条件数准则当新方法。

3. **Quantification via Gaussian Latent Space Representations**，Olaya Pérez-Mon、Juan José del Coz、Pablo González。审阅固定 **arXiv 2501.13638v1，2025-01-23**，F：§3.1–3.5、§4.1。HTML 内另有正文日期，不用它替换 arXiv 版本日期。期刊2026记录在早期搜索中定位，本报告不把未复核的期刊正文与 v1 混为同一版本。\
   https://arxiv.org/html/2501.13638v1 \
   官方链接由论文给出：https://github.com/AICGijon/gmnet \
   C：commit **f7ac591dfd63c755c00d6fa440ab563268d97896**，2026-01-15T10:29:44Z；tree **70cfbd19f0f8d6a4a32d58c9cff2ad3157f49783**，truncated=false。\
   README blob **579e012d5bb3760f2575a35858955ef9ac5815b0**；feature_extractor blob **dde29f8240ce61b4df090662beae30e7543917f5**；train_cifar10 blob **75675b6b5af8769e35ac985a76616499ee1d7695**。固定 tree 没有 LICENSE/COPYING 项；不是授权拒绝，只是再使用许可未确认。未下载模型、CSV、实验包。CIFAR 入口存在不意味着复现成功或其结果已独立验证。

4. **Better Aggregation in Test-Time Augmentation**，Divya Shanmugam、Davis Blalock、Guha Balakrishnan、John Guttag。**2011.11156v2，2021-10-11**。F：§3.3–3.4、聚合方法/实验设置相关片段。\
   https://arxiv.org/html/2011.11156v2 \
   此文用于排除普通 TTA 聚合新颖性；不是“已经回答类别比例估计”证据。

## 本轮排除表的一手来源

- F **MetaAug: Meta-Data Augmentation for Post-Training Quantization**，ECCV2024，§3、Appendix A.6/A.7。https://arxiv.org/html/2407.14726 ；https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03914.pdf
- F **Data-Augmented Quantization-Aware Knowledge Distillation**，Justin Kur、Kaiqi Zhao，**2509.03850v1，2025-09-04**，§3、Fig1、Table6(PDF p8)。https://arxiv.org/html/2509.03850v1 ；https://arxiv.org/pdf/2509.03850v1
- F **SQuaT: Self-Supervised Knowledge Distillation via Student-Aware Quantized Teacher Features**，Lee/Jo/Chung/Kim，AISTATS2026 PMLR300:5023–5031；另有2608.10709v1。https://proceedings.mlr.press/v300/lee26f.html ；https://arxiv.org/html/2608.10709v1 。X：论文链接的 https://github.com/lcdbsa522/SQuaT 提交API返回404；未绕过，未用同名他人项目替代。
- C **EWGS** official https://github.com/cvlab-yonsei/EWGS ，commit **56c654cb893d53563eb352dd591d1450c34bdd15**。读取 README 和 CIFAR10/custom_modules.py、train_quant.py；后两者快照及 blob SHA 已保存。它没有本轮验证的 DQAKD 全流水线，不能混称。
- F **When Dynamic Data Selection Meets Data Augmentation: Achieving Enhanced Training Acceleration**，2505.03809v1，§3。https://arxiv.org/html/2505.03809v1
- A **Data-Efficient Augmentation for Training Neural Networks**，2210.08363。https://arxiv.org/abs/2210.08363
- F **Group-invariant Coresets for Data-efficient Active Learning**，2607.01089v1。https://arxiv.org/html/2607.01089v1 （投稿状态不等于正式接收）
- F **Data Augmentation Improves Machine Unlearning**，Andreza M. C. Falcao、Filipe R. Cordeiro，2508.18502v1，§III–IV。https://arxiv.org/html/2508.18502v1
- F **Auditing Approximate Machine Unlearning for Differentially Private Models**，Gu/He/Chen，2508.18671v1。https://arxiv.org/html/2508.18671v1 。作者匿名仓库为定位线索，未取得执行验证。
- X **Towards Efficient Machine Unlearning with Data Augmentation Guided Loss-Increasing (GLI)**，CVPRW2024 official PDF返回403。https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Choi_Towards_Efficient_Machine_Unlearning_with_Data_Augmentation_Guided_Loss-Increasing_GLI_CVPRW_2024_paper.pdf 。未绕过，未据此断言其未覆盖候选。
- F **Data-Centric Long-Tailed Image Recognition**，2311.01744v1，§I–II。https://arxiv.org/html/2311.01744v1
- F **Kill Two Birds with One Stone: Rethinking Data Augmentation for Deep Long-Tailed Learning**，DODA，ICLR2024，§1–2。https://proceedings.iclr.cc/paper_files/paper/2024/file/54d2d38a56a74387d5916ee40e462295-Paper-Conference.pdf
- A/X **CUDA: Curriculum of Data Augmentation for Long-Tailed Recognition**，ICLR2023 / 2302.05499。https://arxiv.org/abs/2302.05499 。摘要可读；尝试的 HTML404，OpenReview遇浏览器挑战，未绕过。
- A **Deep learning-based ecological analysis of camera trap images is impacted by training data quality and size**，2408.14348。https://arxiv.org/abs/2408.14348
- A **Essential tools but overlooked bias: AI and citizen science classification affect camera trap data**，2025 DOI10.1111/2041-210X.70132。https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70132
- A **Lessons and Open Questions from a Unified Study of Camera-Trap Species Recognition Over Time**，2603.20509v1。https://arxiv.org/html/2603.20509v1 。未作为全部方法已核实的证据。
- F/D **Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation**，2603.08364v1，§4.3粒度设置已定位；不声称完成所有结果/代码核查。https://arxiv.org/html/2603.08364
- F **You May Be Running the Wrong Inception Crop**，2610.04147v1，§3算法1/2。https://arxiv.org/html/2610.04147v1
- F **Test-time Augmentation Improves Efficiency in Conformal Prediction**，CVPR2025 /2505.22764v1，方法/独立校准设置片段。https://arxiv.org/html/2505.22764v1
- D Ctrl-A（2603.21819v1）、更多 conformal/quantification 条目仅作检索定位；没有在本报告中给出未读公式或“全覆盖”结论。

## 源码、数据、访问边界

`sources/` 只保存本轮所读的小型公开源码文本及来源 JSON，供行号/哈希核验。没有 clone 新研究仓库或运行作者脚本。原始 `raw/` 工具输出可能包含二手搜索条目，只有上面明确标注的原始来源被用来支持结论。

本地资料只读到 CIFAR 数据准备报告及健康训练文字报告。引用 `a2er-formula-audit-20261029/DATA_README_ZH.md` 中已经验证的 TRAIN 获取情况；本轮不解析图像/test，不生成新划分。旧数据/模型可用，不等于当前假设有独立证据。

上述403/404/挑战是具体公开资源访问结果，未尝试规避；不构成本轮候选总体权限阻塞。没有自动审批拒绝、未决登录或秘密凭据输出。

