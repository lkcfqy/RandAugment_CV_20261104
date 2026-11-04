# DeepFish 追加核查与两臂探索准备

已完成本机有限核查；**没有模型运行、GPU、依赖安装、环境分配或上传**。上一轮资产目录与官方数据保持不变。

- **分组**：46 个完整文件名前缀对应 20 个官方代码使用的 habitat 编号。完整前缀是否为唯一视频/站点仍无映射证据。排除 1/5/10-shot 所在前缀后，train 分别还剩 45/43/39 组，12,369/12,122/11,645 框。
- **dev 已定**：仅针对官方 5-shot，一个固定 seed；按固定 hash 和帧号规则选 8 组 64 图、187 个现有框。称“前缀组隔离”，其中 habitat 7463 仍与 support 共享；187 个 dev 框单独计入额外标签预算。
- **框契约**：核 69 图 192 框的几何信息，13 图 37 框实际解码/人工抽查；support 框对准鱼体。两个 dev 框越界 0.5 px，少数极小目标身份不确定，均未改标。检测框制作/派生链与独立许可说明仍未知。
- **计划已备**：唯一官方 MMDetection v3.3.0 原始 Swin-T；相同初始化、几何、顺序与更新数，A 固定 Color=1.0，B=0.8。吞吐预检后才冻结一个短程 U；固定最终 checkpoint，官方 test 封存。

主要交付：

1. [分组语义、余额与公开证据](reports/GROUP_PROVENANCE_ZH.md)
2. [dev 清单与框契约](reports/DEV_AND_BOX_CONTRACT_ZH.md)，[可复算 JSON](reports/DEV_SELECTION.json)，[CSV](reports/DEV_SELECTION.csv)
3. [最小两臂计划、依赖、吞吐与上传范围](reports/MINIMAL_TWO_ARM_PLAN_ZH.md)
4. [静态配置模板](templates/two_arm_config.py.in)、[候选依赖](templates/runtime_candidate.txt)
5. [操作记录和未决项](reports/OPERATION_LOG_ZH.md)、[最终核验](reports/FINAL_VERIFICATION.json)

未来所选图像/标签共 11,783,016 字节；连原始 detector 权重与已有 BERT 候选共 1,144,854,702 字节，另加最多 1 MiB 配置/清单/小脚本。当前没有建立完整运行包；只实际提取了 13 张 QA 原图，其余仍在归档内。旧计数实验的上传授权不适用于本实验。

`DELIVERY_MANIFEST.json` 只覆盖本追加目录；上一轮目录的清单仍是上一轮交付的历史凭据。`scripts/` 是本次本机核查脚本，配置模板没有执行。未知来源项与未做的 CUDA 验证已明确保留，不宣称复现、创新或效用。
