# 真实干预图像资源核对：CausalChamber

核对日期：2026-10-21。只读公开原文、官方数据说明及下载器代码；未下载图像包、未安装软件、未运行视觉模型。此项用于判断下一步是否能取得独立的物理干预证据，不是已完成的新实验。

## 已确认与不能声称的内容

- [Gamella、Peters、Bühlmann，Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-024-00964-x) 的光学腔体允许控制 LED、偏振器及相机参数，并实际拍摄图像。它提供物理干预资料，不等于任意像素编辑都已识别出同一干预。
- [官方数据仓库](https://github.com/juangamella/causal-chamber) 的数据许可为 CC BY 4.0，软件为 MIT。下载和后续衍生结果需保留作者与来源归属。
- [Kostin 等，2026，Appendix C.1.2](https://arxiv.org/html/2606.12680v1) 已使用 `lt_color_regression_v1`：从冻结 CLIP ViT-B/32 特征预测 `(R+G+B)/3`，比较 21 个 ERM/IRM/groupDRO 线性读出及目标域适应。仅换 ResNet、增加 TA 或重做该域迁移比较不作为新贡献。

## 数据字段与最小设计限制

[回归数据说明](https://raw.githubusercontent.com/juangamella/causal-chamber/main/datasets/lt_color_regression_v1/README.md)、[变量表](https://raw.githubusercontent.com/juangamella/causal-chamber/main/datasets/lt_color_regression_v1/variables.csv) 和[生成协议](https://raw.githubusercontent.com/juangamella/causal-chamber/main/datasets/lt_color_regression_v1/generators/color_regression.py) 显示：

- 有 `red/green/blue`、`pol_1/pol_2`、`aperture/iso/shutter_speed`、`image_file/image_timestamp` 等字段。10 个环境的协议各采集 10,000 次，图像分辨率为 100 与 500 像素。
- `brightness_value` 是相机 EXIF APEX 字段，不能误当 LED 强度标签；`intervention=1` 表示 SET 后第一次测量，不能单独当干预目标编号。
- 不同环境连续消耗 RNG 序列，不能把每个 CSV 的第 i 行默认配成相同 RGB 的反事实对。应以实际控制量和采样时序匹配，重复拍摄还需按控制设置/采集块划分，不能把近邻帧当独立样本。
- README 中 `shutter_speed_0.001` 的说明与生成代码的 1/1000 秒有差异，RGB 范围端点也需以实际 CSV 核实，不能先据说明锁定分组。

[相机测试数据说明](https://raw.githubusercontent.com/juangamella/causal-chamber/main/datasets/lt_camera_test_v1/README.md) 的 `palette` 协议含 14 种 LED 颜色 × 8 种相机设置，共 112 个控制组合，提供 200、500、2000 像素图像。112 个组合不等于已确认的独立图像总数或完整包体积。

这些图像可用于评价实际物理域的预测风险，不能将“图像→LED 设置”的反向预测误报成对图像做 do(X) 的因果效应估计。相机曝光与 LED 强度可能产生近似不可区分图像；在控制量及可识别性未核实前，不能把错误一概归因于编码器或增强。

## 取用状态与真实阻碍

[官方 SDK 源码](https://raw.githubusercontent.com/juangamella/causal-chamber-package/main/causalchamber/datasets/main.py) 是先下载整个 ZIP、核验并解压，然后才选择实验或图像尺寸。当前已读实现没有逐图远程取用功能。README 的 ZIP 地址仍为 `TEMPLATE`，回归包列出的 MD5 为 `8e2742277e7ad978301eb830148e1d40`，这不能代替真实下载地址和字节数。

未确认独立的小尺寸包或已发布 CLIP 特征包，不能据此断言作者没有发布。NMI 原文的数据可得性说明仍指向官方仓库及 SDK；其 [Zenodo 代码档案](https://doi.org/10.5281/zenodo.14050466) 不是已确认的小数据包。

本轮只读访问的具体失败如下，均无下载/安装/训练副作用，没有待执行或不确定结果：

| 动作及目标 | 工具返回 | 后续 |
|---|---|---|
| 读取 `https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/directory.yaml` | URL is not accessible via this tool | 已停止；不换 curl/SDK/镜像取同一受阻目录 |
| 读取 `https://api.github.com/repos/juangamella/causal-chamber-package/contents` | inaccessible via tool | 已停止，不重试该接口 |
| 打开 `https://causalchamber.org` | 不可重试的重定向访问失败 | 已停止，不换入口绕过 |
| 读取独立 Zenodo 档案 `https://zenodo.org/records/14050466` 的元数据 | URL is not accessible via this tool | 已停止，不换 API 获取同一档案 |

这四项是网页读取不可用，不是用户权限不足的已证实结论，也不是自动审批拒绝。当前不能承诺经批准即可访问。没有下载未知大小的整包。

**可行性结论：** 数据的物理含义、主要字段和许可可以核查；最小图像＋实际 CSV/配对清单的可取用地址和确切体积仍未确认。该资源暂不足以启动预算明确的小验证。即使获得数据，也须先提出超出已有物理域图像回归的具体增量，不能把资源获取本身当研究贡献。

## 对当前研究的作用

本轮理论结果保留为边界观察；真实图像证据仍为空。没有为取得阳性结果而重开亮度配平、背景区域、Equalize 或过滤分支，没有占用 MPS/A100 或改动 8A。整体研究未被宣布结束，但本目录不声称已建立会话结束后的自动续接。
