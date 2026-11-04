AdaAugment 固定版本的有界 CPU 正确性审计已完成。先读 `AUDIT_ZH.md`，再按需要核对 `SOURCE_MAP.md` 和 `results/`。研究立项结论仍为 NO-GO。

| 产物 | 用途 |
|---|---|
| AUDIT_ZH.md | 中文结论、结果表、解析预期、局限与阻塞经过 |
| SOURCE_MAP.md | 已检源码逐行映射到独立实现 |
| protocol.json / protocol-lock.json | 预设样本数、种子、容差、600 秒上限；运行前锁定哈希 |
| audit_fixture.py | CPU 合成访问、真实 DataLoader worker、Beta autograd、置换不变量 |
| run_bounded.py | 外部墙钟 watchdog、旧 Ctrl-A 目录保留检查 |
| formal-run.log / run-receipt.json | 成功运行日志、退出码和包含失败尝试的总时间 |
| results/summary.json / checks.json | 105 个检查的完整结果 |
| results/loader-traces.json / prefetch.json | 逐样本归因与预取时点记录 |
| results/score-gradient.json | 原始梯度、解析预期、99% CI、sum 缩放及样本哈希 |
| results/permutations.json / permutation-summary.csv | 完整动作多重集、所有置换和效用表 |
| source-receipts.json / source-tree-integrity.json | 固定提交、文本 URL、Git blob/tree 与 SHA-256 来源验证 |
| framework-source-evidence.json / dependency-inventory.json | 本机框架实现与只读依赖清单 |
| attempts/attempt01-sandbox/ | 首次失败输出与原始脚本；不覆盖、不删除 |

本次实际命令是已有虚拟环境中的 `python -B run_bounded.py`，执行路径与环境覆盖见 `run-receipt.json`。原目录有协议锁和结果，脚本会拒绝覆盖，不应直接反复重跑。若需要独立复算，将 sources、source-receipts.json、protocol.json、audit_fixture.py、run_bounded.py 复制到新的获准可写空目录，再使用已安装 PyTorch 的 Python 运行 watchdog；无需获取真实数据、权重或安装包。workers=8 的 spawn 路径需要系统允许本机 PyTorch 共享内存管理器。

仅核对产物完整性使用 `python3 -B verify_artifacts.py`，不执行实验或下载代码。该验证脚本生成 `verification.json`；`artifact-manifest.json` 提供全目录文件 SHA-256（清单本身及 verification.json 除外）。
