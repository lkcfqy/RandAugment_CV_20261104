# 本轮结果：工具就绪，真实预测仍受阻

协议于 2026-10-30 10:34:37 UTC 锁定。两个模型固定为 ResNet-18 与 ViT-B AugReg，两个 shift 固定为 cartoon_style 与 heavy_snow，D_C/D_E 分开，5pp 资源门未改变。

30 个合成测试通过；合成端到端执行也通过，生成 CSV、JSON、PNG、SVG。所有合成产物标为 SYNTHETIC / NONEMPIRICAL，不是 CNS 的实证结果。项目原有 PyTorch、torchvision、NumPy 和 Pillow 版本保持；为统计图添加官方 Matplotlib 3.10.7，pip check 通过。

官方 GitHub 标注已核：版本 1.0、CC0 1.0，40,705,750 字节，Git blob 与官方树一致。cartoon_style 为 2,155 条完整六点轨迹；heavy_snow 为 2,883 条。共同 class/seed 键为 1,608 个、91 类。数量只描述发布元数据，不是预测效果。

唯一当前预测数据取得阻塞：官方 Edmond 页面正常浏览器访问遇到 Anubis 挑战，已停止。没有预测 ZIP、没有其实际 schema/checksum 验收，故没有真实效应表、模型结果图或继续/关闭候选的科学判断。未通过代理或隐藏 API 重试。

另有明确身份限制：元数据不含共同基图哈希，class/seed 相同不能证明同源。拿到预测后仍可做标明限制的键匹配探索；严格同源结论需基图身份核验，缺失时必须记未定，不能当负结果。

没有训练、模型推理、GPU 分配、图片/权重下载、复跑旧实验、commit 或 push。所有历史文件保留。详见 README_ZH.md、metadata-audit.json、test-results.txt 与 source-and-access-evidence.json。
