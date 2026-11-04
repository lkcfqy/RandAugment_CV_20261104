# 已完成首轮的归档

先读 [FIRST_PASS_ZH.md](FIRST_PASS_ZH.md)。真实结果在 empirical-keymatched-001；test-results*.txt 全部是 synthetic nonempirical 测试回执。原协议未修改，旧脚本保存在 source-before-release-schema-adapter。

数据未放入此归档：项目 research-cns-reanalysis-20261030/raw/results.zip 和 annotations_cns.json 由既有 .gitignore 保护。工作区原件完整保留。

本候选到此停止追加。以下命令仅为复核记录，本次归档不再执行分析：

```sh
../../.venv/bin/python -B run_reanalysis.py --predictions-zip ../raw/results.zip --annotations ../raw/annotations_cns.json --asset-receipt prediction-source-receipt.json --allow-png-jpg-release-mapping --output ../empirical-keymatched-new-review
```

输出目录必须新建；脚本拒绝覆盖已有结果。归档中 protocol-lock.json 验证两个原协议文件。
