# 本轮操作、边界与未决项

2026-10-31；追加阶段。读取项目及上级指令路径：没有找到适用 AGENTS.md，当前项目及写入工作区均无 `.agents/skills`。沿用已读的 verification-before-completion 流程；本轮另读取 PDF 技能用于论文核查。没有安装技能、依赖或创建子代理。

## 完成

- 读取原始 DeepFish 官方固定源码/说明及 Saleh 2020 原论文；静态核验 habitat selector，记录视频/站点身份未知边界。
- 复用已有 FISH 归档和库存，读取 train 与 1/5/10-shot 元数据，计算分组余额；固定一个 5-shot dev 清单。没有修改任何官方 split 或新增框。
- 对所选 69 图 / 192 框作数值检查；从归档仅提取 5 support + 8 dev 代表帧，核字节哈希、尺寸并看标注叠图。归档流遍历时跳过 test 成员，未解码/展示 test。
- 固定官方 MMDetection v3.3.0 commit；静态比较关键 detector 文件、Color RNG 行为、COCO 坐标转换、依赖范围及将来的 weights-only 强制加载路径。仅保存源码文本，没有执行上游代码。
- 写两臂方案、未执行配置模板与未来输入计量；没有生成训练 runner、训练 dev 标注文件、模型输出或效用指标。

## 公开读取和失败动作

| 动作 / 目标 | 结果、执行状态与处置 |
| --- | --- |
| GET `https://researchonline.jcu.edu.au/64493/1/64493.pdf`，上限 8 MiB | 已执行，HTTP 200，下载 1,585,745 字节，SHA-256 `0b6b74d28d9462b2e21982f6498e780d29e530d1aa8f1fcac539ce9f45bb473e` |
| GET 原始数据目录 `https://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/`，上限 1 MiB / 40 秒 | 已发起请求，`URLError: <urlopen error timed out>`，没有得到归档目录文件。**网络超时，不是审批拒绝**；一次后停止 |
| GET `https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html`，上限 1 MiB | 已执行，HTTP 200，4,160 字节；只取得 wheel 索引，没有下载 wheel |
| 浏览器检索读取 `https://www.mdpi.com/2673-9496/6/4/45` | 服务返回 HTTP 429，没有取得正文；停止该线索，没有用它证明检测框来源 |
| 其他官方论文站点/DOI、MMCV 文档和首次 PDF 截图工具请求 | 部分返回 unavailable/500 或 content-type 错误；以 JCU 论文文本、已固定官方源码、官方 wheel 索引完成独立证据核查。未声称已做 PDF 页面视觉排版核查 |
| GitHub `mmdet/datasets/transforms/augmentations.py` 读取 | 路径不存在；查看已在本机的目录后读取正确 `colorspace.py`。非授权拒绝，不涉及执行代码 |

详细直接下载日志为 `PUBLIC_FETCH_LOG.json`。本轮直接 HTTP 成功 payload 为 1,589,905 字节；另有小型官方代码/元数据响应。上一轮记录的下载 payload 为 1,484,698,643 字节；加本轮和元数据仍显著低于 2 GiB。没有重新下载两个大资产。最终目录占用在核验报告中检查，低于 4 GiB。

## 未做 / 仍未知

- 原始 7.6 GB 归档内部 CSV、前缀到视频/GPS/相机部署映射：未读，部分门户请求超时；不补猜对应关系。
- 检测框制作、点/mask 派生规则、质量复核、独立派生许可：在有界公开核查中未找到，保留未知；未联系第三方。
- CUDA 运行环境、传递依赖 resolver、安装、import、模型构建、完整 checkpoint/model 匹配、前反向吞吐、峰值显存和效用：均未执行；没有相关环境运行或上传授权，也未尝试执行这些动作。
- 官方 test：本轮封存；前一轮纯资产 QA 曾看过 3 张，不将它描述为从未接触。没有 test 推理、AP 或模型选优。
- 之前计数实验与其上传许可：未重启、未复用成新上传授权；没有凭据操作、GitHub push、云资源或付费动作。

本轮没有自动审批拒绝；实际网络失败已按动作和目标记录。没有待决的工具执行或不明下载结果。
