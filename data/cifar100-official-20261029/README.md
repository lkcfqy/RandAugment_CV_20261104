# CIFAR-100 官方 TRAIN 数据准备

状态：`ready_train_only`，完成于 2026-10-29 18:32:02 UTC。路径为项目 `data/cifar100-official-20261029/`。

- [官方发布页](https://cave.cs.toronto.edu/kriz/cifar.html)、[官方二进制归档](https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz)。仅使用 Toronto 官方主机，没有镜像或新增协议。
- 归档 168,513,733 字节；官方 MD5 **03b5dce01913d631647c71ecec9e9cb8** 已验证。
- SHA-256：`58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec`。
- 只提取 `train-only/train.bin`、fine/coarse label names。test 只保留在原始归档；未提取、解析或评测其 payload，未访问其他封存测试。归档的所有 member header 都经过路径、类型、大小和重复检查；没有使用 pickle。

训练文件是 `uint8[50000,3074]`：第 0 列 coarse label，第 1 列 fine label，后续 RGB planar 数据可 reshape 为 `[3,32,32]`。本次只读取标签以划分，没有读取图像做模型计算。

开发划分已完成：`splits/fit_indices.npy` 40,000 条、`splits/development_indices.npy` 10,000 条。每个 fine class 分别 400／100；断言确认没有重叠且并集恰为全部 50,000 TRAIN 行。规则为每类按 `SHA256(seed|class|official_row_index)` 排序，前 100 为开发集，其余为 fit；seed=`rand-cifar100-train-development-v1-20261029`。文件哈希均见 `preparation.json`。

实际运行分段耗时：初次官方访问与慢速前缀传输 370.592862 秒（保留 26 MiB 后主动中断）；四路官方 HTTP Range 续传 404.370910 秒、拼接校验 0.263076 秒；首次安全检查因官方额外的 `fine_label_names.txt~` 备份文件停止，未提取任何 payload。核实该普通文件后仍不提取它；最终重新校验／安全提取／划分／哈希合计 1.310120 秒，其中提取 0.980569 秒、划分 0.046668 秒。前后墙钟间隔还含诊断与等待，不将其当计算耗时。

`preparation-initial-transfer.json` 和 `preparation-range-transfer-and-member-stop.json` 保留中断／停止原始记录；`preparation.json` 为最终成功记录。初始脚本和实际恢复脚本均保留。新目的地可用 `prepare_cifar100_v2.py --root /absolute/new/directory` 重现官方获取与同一划分；脚本拒绝覆盖已有目录，已补充识别但不提取官方备份元数据。不要在当前数据目录重跑下载。

没有训练、模型权重、test 指标、Colab、提交或推送。数据可用不代表科学门已通过。
