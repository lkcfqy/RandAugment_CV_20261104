# v2日志修复补记（同一seed／科学配方）

2026-10-29 UTC，v1基线实际完成2个optimizer step后，被新增的损失重构assert停止；保存为原 `run-001/`，未删除或改写。它不是有效完成的基础训练，更不是学习失败。

确定原因：官方 `Derpp.observe` 的 `loss=self.loss(...)` 随后使用 `loss += loss_mse` 和 `loss += loss_ce`。本地logger把该CE的 `.detach()` 存入列表，但detach共享storage，记录随in-place累加变成了总loss；因此二次重构多加了回放项，assert报错。源算法、实际梯度与optimizer未因logger改变，错误位于日志引用与断言。

CPU最小复现见 `LOGGER_ALIAS_REPRO.json`：CE=2、MSE贡献=.24、回放CE贡献=.6；detach引用变成2.84，错误重构3.68；detach.clone仍为2，正确重构2.84。

修复仅将CE/MSE日志改为 `detach().clone()`，独立保存scalar；另把输出和冻结文件切到v2新名称。网络、α/β、optimizer/LR、buffer、变换、数据计划、seed和旧操作门全部不改。`baseline.py`、原freeze、run-001、process.log保持原样；新增 `baseline_v2.py`、`supervise_v2.py`、`protocol-freeze-v2.json`，输出 `run-002/`。

为取得完整、可核对的6300步日志，v2从相同seed初始化重新执行固定50+50epoch。v1的2步中断执行单独记账，不并入有效6300步，也不把重启称新seed或比较结果挑选。先冻结此修复再运行；没有候选门A/五支或其他配方。
