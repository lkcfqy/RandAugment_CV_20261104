# 冻结阶段执行说明

协议和抽样清单在真实训练和效果测量前锁定，见 protocol-lock.json。全部新产物只在本目录；无 Colab、在线四臂、test 评估、commit 或上传。

源版本为 ControlAugment e77a4c19a3db893c120fc18b774fa290e826592b，早于 2026-10-26 v2；不是逐字复现 v2。MIT 许可证保留。源 CLI 返回值实际为 8 项而调用方接 7 项（以本次固定源码为准）。独立入口未运行第三方 CLI。

首次完整 MPS 反向计算因源 channels-last 布局发生 stride/view 兼容错误。仅切换 NCHW 连续布局，模型、优化器与算法未改。失败与成功回执均保留。CPU/MPS 训练梯度相对 L2 差须小于 1%，再进入真实训练。具体证据见 compatibility-amendment-layout.json、benchmark-attempt01.json、benchmark.json。

训练曝光数以 46,000 原图为单位；每个实际 epoch 有原图和固定镜像共 92,000 张，两个 seed 各 10 个实际 epoch，共 1,840,000 张、40 次曝光。SGD 每个实际 epoch 重建，momentum reset 保留。20 实际 epoch 的余弦表分母为 21，与固定源码一致。epoch10 是 horizon20 的中途暂停：在该边界完成第二次控制更新，保存将用于 epoch11 的状态，但不训练 epoch11。

所有图像变换仍为官方 CPU Pillow 路径；模型前后向仅 MPS，PYTORCH_ENABLE_MPS_FALLBACK=0。这不是 GPU 算子不支持时的隐式 CPU 回退。固定源码 Hue(0) 的 HSV 转换和 Solarize(0) 的阈值 255 可能改变字节，适配器保留，不强行当作精确 identity。几何零强度 identity 已单独验证。

冻结诊断中 S 的 full 图像是逐操作实际合成。E/F 的 full 是不中断连续仿射块的最终读出；prefix 是前四操作状态的 32×32 读出。若末端操作延长已有仿射块，full 保留该块的画布/矩阵状态，不能解释为先把 prefix 裁掉再对那张裁后图重做末端操作。相应条件 ROR 是同一渲染干预下 full 与 prefix 读出的准确率比，包含这个渲染状态差异。非零差异本身不反驳 Ctrl-A。

主资源门预先固定为原 ASD、N5、同参数的 Δ_err=error(S)−error(F)。两个 seed 各点估计至少 +0.02 且各配对 95% CI 下界大于 0 才可请求主线程复核。F 变差仅记反证，不取绝对值。即使过门，本任务也不启动在线分叉。

强度匹配固定 11 点网格加最多 4 次二分细化。目标为 N2 S 的冻结平均 augmented CE，不能称在线 κ 匹配。audit 验收要求相对 CE 差点估计和整段 95% CI 均在 ±5% 内，失败则不声称排除总体难度。两 seed 分开；图像 cluster bootstrap 保留每张图全部 15 视图，不代表训练 seed 不确定性。

计算从第一次完整模型测速开始累计 8 小时墙钟，独立 supervisor 在期限前终止阶段。账本逐 batch 预扣曝光/推理，进程失败可能保守计入未完成的 batch。预算用尽保留最新完整 epoch checkpoint，不缩网络、加样本或追显著。
