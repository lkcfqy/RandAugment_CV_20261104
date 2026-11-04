# CNS-Bench 固定范围重分析准备

状态：**真实预测包未取得；没有实际模型效应或候选结论。** 官方数据页的正常浏览器访问遇到 Anubis 挑战后已停止，未走代理/隐藏接口。只有官方 GitHub 标注和明确标成 `SYNTHETIC_NONEMPIRICAL` 的测试输入已读取。

## 文件

- `PROTOCOL_ZH.md`、`protocol.json`、`protocol-lock.json`：看本地预测结果之前锁定的两个模型、两个 shift、D_C/D_E、5pp 资源门与不确定性规则。
- `metadata-audit.json`：真实标注结构、完整轨迹数量与同源证据缺口；不是预测结果。
- `cns_core.py`：六点轨迹、首错/恢复/末段错误、唯一 key、模型 outer key 合并。
- `cns_statistics.py`：共同首点正确资格、固定类别支持 K、联合 source bootstrap、类别 cluster 敏感性与保守停止门。
- `cns_assets.py`：只读本地 ZIP/标注验收、实际本地基图文件哈希核验；不下载资产。
- `run_reanalysis.py`：CSV/JSON/PNG/SVG 输出；真实数据与合成测试入口隔离。
- `synthetic-validation-001/`：**合成、非实证**端到端验证，图上也明确标注；不得拿来支持 CNS 的任何科学结论。
- `raw/`：被 git 忽略的官方元数据。真实预测包日后也应放这里，不提交。

## 已核元数据

官方固定提交 `3c09a993007b5844ccd799f8c523a00985aa31ad` 的标注文件包含版本 1.0、CC0 1.0、192,168 条图片记录。选定两 shift 共 30,228 条记录：cartoon_style 2,155 条完整轨迹、97 类；heavy_snow 2,883 条完整轨迹、96 类。两侧共同 `(class,seed)` 为 1,608 个，涉及 91 类。

这是完整性和键交集，不是确认同源，更不是模型正确率。元数据只给分开的 shift 图片路径，无基图 SHA-256。没有基图证据时，runner 会将任何未来分析标为 key-matched exploratory，并将严格配对资源判断标为未定，不按负结果关闭候选。原始预测包的 checksum、文件 schema、类别映射和 scale 仍待实际验收。

## 运行

在本目录中，使用项目已有环境：

```sh
../.venv/bin/python -m unittest -v test_cns_core.py test_cns_statistics.py test_cns_assets.py
../.venv/bin/python run_reanalysis.py --synthetic --output synthetic-validation-002
```

真实预测包通过正常官方下载并记录来源/许可后，准备一个 JSON 回执，至少含 `source_url`、本地文件实测 `sha256`、核验的 `license`；应额外保留发布方 checksum、发布日期/版本及其核验状态，不能猜写。本脚本只检查必填项和本地 hash，不能替代来源审查。

```sh
../.venv/bin/python run_reanalysis.py \
  --predictions-zip raw/results.zip \
  --annotations raw/annotations_cns.json \
  --asset-receipt prediction-source-receipt.json \
  --output empirical-reanalysis-001
```

缺省没有基图身份证据时，只输出探索性结果，严格同源主分析不可用。可选 `--identity-manifest` 接收 CSV：`shift,gt_class,seed,local_path,sha256`。`local_path` 只能指向 manifest 同目录内的已取得 baseline 文件；脚本实际计算文件 hash、核对 scale-zero 文件名和两侧字节相同，不相信只填在表里的 hash。首轮要求全部共同完整 key 有证据，拒绝重复源对应多 seed/变体的情形；不自动下载任何图片。

真实运行固定 4,000 次 bootstrap；`--synthetic-replicates` 仅控制合成验证，不改变真实方案。输出目录必须不存在，防止覆盖历史结果。输入缺失模型、重复 key、未知字段值、元数据路径映射冲突等均停止，不据结果换模型或静默修正 scale。

## 解释限制

首错 F 与末段连续错误起点 T 是不同描述量；T 不是永久失败。首次错误后的恢复按 F 分层，末点首错没有恢复机会。全对轨迹始终在比例分母内。D_C 和 D_E 分开计算，并在同一次 source 重采样内重算，不能当因果中介。类别 cluster 是另一推断目标。

所有范围仅是两个模型和发布后过滤的两 shift；匹配/首点正确条件化可能选择样本，scale 不等于可比物理强度。ImageNet-P 已有翻转研究，本任务尚无新贡献宣称。未训练、未推理模型、未请求 GPU、未复跑历史实验、未 commit/push。
