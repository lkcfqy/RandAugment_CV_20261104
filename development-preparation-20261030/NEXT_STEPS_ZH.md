# RandAugment 本地准备与后续安排

更新：2026-10-30。当前目标是 11 月 4 日的 5–8 页、每组最多 10 分钟课堂报告，讲清 RandAugment，并保留寻找研究想法的空间。课程原话“报告可选”仍有歧义；本文件不替用户解释。

## 本轮已经完成

| 工作 | 实际结果与证据 |
| --- | --- |
| 本地克隆 | `/Users/lkc/Downloads/projects/RandAugment_CV_20261104`；完整 Git 历史，分支 `main`，HEAD `8493460a088daab425c52732f7a8d1f9c955ee4f`；remote 为 `https://github.com/lkcfqy/RandAugment_CV_20261104.git` |
| 历史材料核对 | 克隆前目标目录不存在，未覆盖旧副本；备份 INCLUDED 清单的 469 个文件 SHA-256/大小通过；另保存了 2,066 个 tracked 文件的内容指纹 |
| 项目 Python | 新建独立 `.venv`，解释器 `/Users/lkc/Downloads/projects/RandAugment_CV_20261104/.venv/bin/python`；Python 3.14.10 arm64；Miniforge base 构建与旧环境记录一致 |
| 官方依赖 | 沿用旧哈希锁：torch 2.13.0、torchvision 0.28.0、NumPy 2.5.4、Pillow 13.0.0 等 13 个应用/依赖 wheel；逐一与官方 PyPI 文件名、URL、大小和 SHA-256 对应，下载后验哈希并离线安装；`pip check` 通过 |
| MPS 实测 | `is_built=True`，`is_available=True`；16×16 float32 合成张量在 `mps:0` 上运算并同步；与 CPU 最大绝对误差 0，allclose 通过；禁用 CPU fallback 和 MPS fast math；进程约 8.63 秒，45 秒限时内完成 |
| Colab CLI | 官方 0.7.5 已安装于 `/Users/lkc/.local/share/google-colab-cli`，入口 `/Users/lkc/.local/bin/colab`；version/help 与依赖检查通过，未认证 |
| 备课 | 本地 RA PDF p.3–7 整页图像/文本和 Table 3 数字已核对；两 HTML 已静态检查；本目录保存新的 [内容对齐稿](PRESENTATION_ALIGNMENT_ZH.md) |

环境回执、官方 wheel 来源、锁文件和版本清单见 [本轮环境准备目录](../environment-preparation-20261030-mps/PREPARATION_REPORT_ZH.md)。MPS 原始结果见 [mps-receipt-001.json](../environment-preparation-20261030-mps/mps-receipt-001.json)。这次仅证明一个小型前向张量运算可用，不能推断整套模型、所有算子、反向传播、训练速度或科研收益均已验证。

已有 [course-preparation-20261029](../course-preparation-20261029/README.md)、历史实验脚本、停止说明、冻结结果和来源台账保持原样。新文件尚未 commit/push。没有恢复停止实验、导入旧训练入口、下载模型/数据、开始训练或制作最终 PPT。

## 当前还缺什么

备份 [EXCLUDED_LOCAL_ASSETS.json](../backup-20261030/EXCLUDED_LOCAL_ASSETS.json) 列出的 614 个排除文件，在这个新克隆中均未发现。包括历史数据/视图缓存、模型或优化器检查点、部分第三方源码/论文全文和其他下载资产；这符合仓库保存范围，不是已上传证据丢失。AA 与 TA 的旧全文 PDF 也不在本次克隆中。

基础 Python 构建与 13 个锁定 wheel 相符，仍不能称历史实验已经复现：旧环境目录、数据、权重和部分固定源码快照没有恢复，历史脚本还可能依赖原路径及其他实验特定条件。本次不据环境通过就执行旧脚本。只有确有后续需要时，才逐项按来源/许可/哈希恢复缺失资产并核对实验专属要求。

## 课程准备接下来按这个顺序推进

1. **确认展示约束**：演讲语言、组内分工、Q&A 是否占用 10 分钟；保留“报告可选”的原话，必要时由用户向课程方确认其含义。
2. **按 7 页视觉稿制作**：采用对齐稿中的 45/70/85/80/80/95/55 秒结构。第 3 页分清外层验证选 N/M 和内层随机采样，第 6 页保留同模型结果及来源边界。这是后续工作，当前没有 PPT 文件。
3. **验证实际展示**：选定静态图或短演示；浏览器验证两个 HTML 的控件、字号、投影适配，准备静态替代。比较 M 时固定操作、顺序和随机方向，保留教学简化说明及素材署名。
4. **实际排练计时**：8:30 目前是预算；完成一次全程计时后再调整。若接近上限，先删现场演示和额外旁支，保留算法、主结果及局限。问答是否需占用余量取决于确认结果。

## 研究推进的当前决定

本轮会话给出的最新 Pro 审查结果为：ActiveAugment 逐样本匹配候选**不立项**。InstaAug 已有全局/类别/实例对照，尚缺能够跨方法预测收益的新增机制。因此不执行该方向实验，也不把“不立项”写成“已实验证明无效”。最新审查来自主线程提供的结论，本轮未自行运行另一轮 Pro 或训练。

后续寻找想法时，先用最接近的原论文与代码说明已有工作做到了哪一步，再提出可以被反驳的机制预测和低成本证据要求。只有区别于既有方法且机制明确的候选，才值得另行形成协议和运行决策；当前没有获批的新实验。课堂末页可讨论开放问题，不能宣布已批准研究或新成果。

## Colab 的下一步需要用户授权

安装不等于登录。实际 0.7.5 默认认证方式是 `oauth2`。其 OAuth 会申请 openid、profile/email、Cloud Platform、Colab、drive.file 等范围，并把可刷新的凭据写入 `~/.config/colab-cli/token.json`，不使用 Keychain。[版本源码](https://github.com/googlecolab/google-colab-cli/blob/v0.7.5/src/colab_cli/auth.py)

主线程须先向用户说明这一持久授权并得到明确同意，再让用户完成 Google 登录和同意流程。当前 token 文件未创建；没有账号查询、运行时分配或 Colab 计算额度使用。不要把 `colab auth` 当作 CLI 登录步骤，官方文档中的该命令指 VM 内 GCP 认证。即使完成登录，是否使用云端计算仍应按后续具体任务决定。
