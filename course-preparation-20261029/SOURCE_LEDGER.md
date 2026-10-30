# 来源、版本与核验记录

核验日期：2026-10-29。相对路径以本备课目录为起点；所有 PDF 页码采用阅读器的 **1-based 文件页序号**。RandAugment 此处引用页的文件页码与页面印刷页码一致。AA、TA 使用本地作者／开放版本的文件页序号，不替换成出版社卷内页码。

## 核心来源

| ID | 本地文件／身份 | 本次用于什么 |
|---|---|---|
| RA | [RandAugment_NeurIPS_2020.pdf](../RandAugment_NeurIPS_2020.pdf)，12 页；Cubuk、Zoph、Shlens、Le；NeurIPS 2020 | 所有主论文动机、算法、图和结果数值 |
| AA | [AutoAugment_CVPR_2019.pdf](../AutoAugment_CVPR_2019.pdf)，14 页；Cubuk、Zoph、Mané、Vasudevan、Le；首页标 arXiv:1805.09501v3，2019-04-11；CVPR 2019 论文作者版本 | 策略搜索定义、概率／强度区别、作者关系 |
| TA | [TrivialAugment_ICCV_2021.pdf](../TrivialAugment_ICCV_2021.pdf)，13 页；Samuel G. Müller、Frank Hutter；ICCV 2021 对应完整论文 | 单操作与随机强度定义、比较方法与边界 |
| DEMO-1 | [randaugment-animation.html](../randaugment-animation.html) | N/M 与顺序的教学演示候选 |
| DEMO-2 | [augmentation-comparison.html](../augmentation-comparison.html) | AA/RA/TA 规则对比的教学演示候选 |

RA 官方页面与 PDF 已核对，标题、作者、会议信息及 84.7% 摘要结果相符。论文文件中首页亦标明 NeurIPS 2020。AA、TA 的实质核验来自项目内已有完整 PDF；既往 CVF 直连受到 HTTP 403 限制，本轮没有绕过该限制。下面列出官方论文身份入口和开放作者版本入口，不将本地文件声称为已验证与某远端 URL 逐字节相同。

- RA：[NeurIPS 2020 官方论文页](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html)；[会刊 PDF](https://papers.nips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)。
- AA：[CVPR 2019 官方论文页](https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html)；[arXiv v3 身份页](https://arxiv.org/abs/1805.09501v3)。
- TA：[ICCV 2021 官方论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.html)；[arXiv 身份页](https://arxiv.org/abs/2103.10158)。本地 TA 版本以以下 SHA-256 固定，不单凭未带版本号的 URL 推断其精确 arXiv 修订号。

## 逐页证据映射

| 提纲页／笔记 | 来源位置 | 核对的内容与限制 |
|---|---|---|
| 1 动机 | RA §1，p.1–2；AA §2，p.2 | 人工增强、自动选择策略；不声称任意增强保留标签 |
| 2 代理任务 | RA Figure 1、§3，p.3–4 | CIFAR-10、WRN、N=1、20 个初始化；模型／数据规模与最佳 M 关系；不推广为普遍定律 |
| 3 N/M | RA §4，p.5–6；Figure 3，p.6 | 操作池 14 种、均匀有放回抽样、N 次顺序应用、共享 M；p.5 行文枚举不完整时以 Figure 3 全表为准 |
| 3 强度范围／安排 | RA §3 脚注 4，p.4；§4，p.5–6 | 基础刻度与实验扩展范围不同；原文讨论四种 M 安排，不能只概括为所有实现恒定 M |
| 4 视觉例子 | RA Figure 2，p.5 | N=2，ShearX → AutoContrast；M=9、17、28 对应文中不同模型设置；不是同模型的强度越大越好对照 |
| 5 AutoAugment | AA §3，p.2–3；Figure 1–2，p.3 | 搜索控制器、验证奖励、5 个子策略的搜索空间、每个子策略 2 操作、概率与幅度区别 |
| 5 TrivialAugment | TA §3、Algorithm 1，p.3–4；Table 8，p.10 | 1 个均匀操作和均匀强度 {0,…,30}；操作池和映射仍是设定 |
| 6 主结果 | RA Table 3、表注及 ImageNet 段落，p.7 | 原表核对后节选；单位 Top-1 accuracy (%)；3 个模型×3 个方法共 9 个数值 |
| 7 局限／备答 | RA Table 4 及正文，p.8；§6，p.9 | COCO 上 RA 未全面超过 AA；AA 有额外边界框操作；操作池依赖任务，仍需验证 |
| 成本备答 | RA Table 1，p.2；TA §4.1，p.4 | 候选空间、实际搜索成本、最终训练成本不同；跨论文比较需要控制管线 |

## 主结果抄录检查

来自 **RA Table 3，PDF p.7**。已同时检查文本提取和页面图像。提纲保留以下三列；原表另外含 Fast AA 列。

| 模型 | Baseline (%) | AA (%) | RA (%) | RA−Baseline（百分点） | RA−AA（百分点） |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 76.3 | 77.6 | 77.6 | 1.3 | 0.0 |
| EfficientNet-B5 | 83.1 | 83.6 | 83.7 | 0.6 | 0.1 |
| EfficientNet-B7 | 83.7 | 84.3 | 84.7 | 1.0 | 0.4 |

**版本锁定**：NeurIPS 2020 的 B7 是 84.7%。不能把其他 RandAugment 版本出现的 85.0% 替换进此表。数字之差只是描述性比较，不能据此自行增加显著性、误差条或统一复现次数。

**表注来源边界**：ResNet-50 的 Baseline/AA 来自原文 [3]，Fast AA 来自 [17]；EfficientNet 的 Baseline/AA 来自 [34]。Table 2 关于“本文报告结果平均 10 次”的说明不覆盖 Table 3 的所有结果。

COCO 口头备答来自 **RA Table 4，PDF p.8**：ResNet-101 的 Baseline/AA/RA 为 38.8/40.4/40.1；ResNet-200 为 39.9/42.1/41.9；指标为 mAP。相邻正文说明 AA 使用 RA 未包含的边界框专用操作，不能把它作为只改变一个因素的对照。

## HTML 的可用范围

| 素材 | 核对结果 | 后续制作时须保留的说明 |
|---|---|---|
| DEMO-1 | 绘制猫形象；7 种教学操作；N=1–4、M=0–10；顺序组合、有放回抽样；调整 M 时固定所抽操作／符号；M=0 不关闭 AutoContrast | 教学简化，不是原论文完整操作池或官方像素实现；本次未现场运行 |
| DEMO-2 | AA 手写子策略示例；RA N=2、M=5 的固定规则例子；TA 单操作与 31 级强度；含超范围平移的失败示意 | AA 样例不是论文搜索结果；比较的是规则，不是模型精度；超范围平移不代表方法默认行为 |

DEMO-2 的照片出处：[Cat August 2010-4.jpg](https://commons.wikimedia.org/wiki/File:Cat_August_2010-4.jpg)，作者 Alvesgaspar，[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。HTML 嵌入 960×587 缩略图，并用于缩放和变换；若将派生图片导出到幻灯片，需保留作者、出处、许可及修改说明。本轮未重新下载或导出这张照片。

## 文件指纹

SHA-256 均为 2026-10-29 对项目内现有文件重新计算；用于固定本次核对对象。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| RandAugment_NeurIPS_2020.pdf | 1283252 | `00796fa0e1edf667567fbdd43cfca7644f8176cbd2bb55dc33d448ea83e73fa7` |
| AutoAugment_CVPR_2019.pdf | 1138440 | `63e1b703ff7bd955ebcae0c3086506f2a092ec5b03f46dcfbd2c658107366c56` |
| TrivialAugment_ICCV_2021.pdf | 807021 | `a12072beb2bb8aa6724c7e0c8a8b81f1ba90cd07b7fbef3bf45db109921f2d7f` |
| randaugment-animation.html | 18594 | `2c8d58279b4b81cb14d219d8ad580404fa2537a25064e7f137acc07df17cd1af` |
| augmentation-comparison.html | 207907 | `8cada22ff8ff88c1c92ae12886737338dc6e9d784b0c773993a0dc0ef8a752bd` |

## 完成与未完成

已完成文本提取、对应条目核对、RA p.3/5/6/7/8 图像检查、核心结果算术与 7 页时间预算检查。没有制作 PPT、没有生成新性能图或实验结果，没有执行 HTML、没有测量实际讲述时长。最终语言、分工、视觉排版及演示适配仍需在制作阶段确定。
