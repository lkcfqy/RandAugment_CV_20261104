本目录是已有本地 ViT 权重的冻结模型诊断，已完成并按预定规则关闭，不训练。阅读 [中文报告](REPORT_ZH.md) 和 [机器可读结果](SUMMARY.json)。演讲和旧研究记录未改。

文件入口：

- PROTOCOL.json：在首次模型构建/输出前保存的科学协议及全部 32 图 ID、SHA、矩形和噪声种子。
- ASSETS.json：原始本地资产来源与 SHA；没有下载新权重。
- RUN_FROZEN.json：运行前代码 SHA、库版本、源码 SHA；run_diagnostic.py 防止同目录无声重跑。
- TEST_RESULTS.json / ANALYSIS_TEST_RESULTS.json：10 项无模型数值和统计规则测试。
- RUN_RECEIPT.json / raw/：256 个唯一身份，786 次完整前向，64 份逐图 NPZ/JSON、模型组件及完整 state 回执。
- ALL_CONDITIONS.csv / ALL_PAIRED_DIFFERENCES.csv：全部条件和全部配对差异，没有筛掉负面样本。
- SUMMARY.json：由所有持久化数据重新计算的统计与冻结决策门。
- ACTION_LOG.json：来源可达性和被拒动作记录；没有绕过最终论文站点的拒绝。
- FINAL_RECEIPT.json / MANIFEST.sha256：最终资源与归档回执、文件完整性。

复算使用已经安装的环境，命令只读取保存张量和原有图像，不构建模型：

```sh
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-shared-theta-ad-20261030/.asset-venv/bin/python -B analyze.py
```

全部脚本/路径以本地研究复核为用途，不上传或发布。原始图像和大模型权重不包含在本目录。预先冻结的条件不可因结果修改；若未来另开研究，需要新目录及新协议。
