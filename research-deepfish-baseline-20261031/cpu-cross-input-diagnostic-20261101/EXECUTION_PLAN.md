本次已获准完成固定的512次本机CPU交叉输入诊断，不再等待额外GO。原有8次测速不重复，环境不重装。

1. 固定既有2seed×2训练臂×2输入条件×64dev的512唯一身份及模型、数据、环境、入口代码SHA。保留原COCO标注，包括4195/4660 bbox/area；不读取官方test。
2. 先完成无模型测试：重复/缺失/错身份拒绝、原子落盘和故障恢复、跨进程回读、磁盘与时间门、人工COCO指标映射、官方bbox转换与入口静态核查。
3. 四个独立CPU进程各安全严格加载一个完整checkpoint，按固定顺序完成原图64次、0.8输入64次。Torch4/interop1，OpenCV顺序，FP32/eval/inference_mode。Color位于cv2 uint8/BGR解码后和原resize前。无训练、预热或新增seed。
4. 每一图像条件的完整top300预测写入独立JSON，fsync、原子发布并回读核SHA；不保存checkpoint或整幅输入张量。缺失、重复、非有限值、时间或磁盘中断都标incomplete，不选取子集计算效用。
5. 512唯一记录和8条件完整后才统一计算全部指标。主AP使用maxDets100；AP50/AP75/尺度AP使用1000；AR分别为100/300/1000，尺度AR1000。另启无模型进程从持久化预测复算，要求全部stats及完整COCO数组精确一致。
6. 报告两seed全部四格AP及I=[B-A]输入0.8−[B-A]输入1.0，保留所有结果，不改主指标，不宣称机制或真实域泛化。确认所有CPU子进程退出后归档。

正式运行墙钟上限90分钟，包含加载、512前向、落盘、全部评估与复算；48.13分钟仅估计。研究目录仍6GiB，新增产物100MiB以内并保留80MiB余量；触及边界安全停止，不删旧资产、不减图或换模型。GPU/MPS/Colab/上传/CU新增均为0。

两项pipcheck元数据警告保留；Ninja旧WHEEL_sha256实际为换行规范化后的文本hash，原始文件hash另列于DOCUMENTATION_CORRECTIONS.json，不修改环境。旧准备报告保留为历史记录，当前指标映射以RUN_PROTOCOL.json为准。

测试通过后的唯一正式命令：在本目录执行 ../cpu-cross-input-preflight-20261101/.venv/bin/python -B formal_cpu.py --phase run 。执行器拒绝已有RUN_STARTED的重复启动；若发生工程错误，应先核实持久化状态。
