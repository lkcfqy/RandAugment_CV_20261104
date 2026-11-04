# 本次操作状态与异常记录

完成：读取项目 README 和 2026-10-30 停止/恢复说明；本次明确有限检查授权作为继续依据。对仓库及上级路径未找到 AGENTS.md，仓库和用户目录所查 .agents/skills 也未发现对应说明。没有覆盖、续跑或修改其他研究目录。

完成：固定两份官方 GitHub 快照；匿名下载公开 FISH 归档和唯一指定 Swin-T 权重；安全扫描归档，哈希所有 JPEG，读取 COCO JSON，解码六张样例；CPU weights_only 张量检查；静态配置审计。脚本和报告均置于新研究目录。

未做：原始 7.6 GB DeepFish 下载、另一路径 1.094 GB MM-Grounding-DINO 权重下载、模型构造、前向/反向、训练、AP/测试效用评估、GPU/MPS/Colab、安装依赖、unsafe pickle fallback、动态白名单、改标签/造框、建 dev、Flux/检索/生成、上传、推送、建凭据、付费资源。

操作异常及精确状态：

| 操作 / 目标 | 结果与原因 | 是否执行 / 后续 |
| --- | --- | --- |
| sandbox 内 curl HEAD 指定 download.openmmlab.com 权重 URL | DNS 无法解析，curl exit 6 | 未取得响应；后续在本次下载授权内通过正常 require_escalated 审核执行匿名官方下载，没有自动审批拒绝 |
| 官方权重 HEAD，max-filesize=32768 | curl exit 63，因为响应 Content-Length=691901857 大于该上限 | 头部实际已保存，正文没有下载；正式受限 GET 后完整核对大小与摘要，不将 HEAD 的非零状态当成功下载 |
| 对自写 get_assets.py / fetch_public.py 使用 py_compile | 文件系统 sandbox 拒绝写系统 Python 字节码缓存，PermissionError: Operation not permitted | 确切失败目标为 /Users/lkc/Library/Caches/com.apple.python/Users/lkc/Documents/Codex/2026-10-31/task-8/research-deepfish-baseline-20261031/scripts/get_assets.cpython-39.pyc.4315115568；未成功写此文件，未重试该缓存写入。后续在内存用 ast.parse 核语法，不依赖 py_compile 缓存 |
| web 工具读取部分 HF/Drive/Nature 页面 | 工具报 Internal Error/不可访问 | 保存了可公开取得的作者 HF README、Drive 目录 HTML；Nature 正文未取得，未以不存在的正文为依据 |
| web 读取 GitHub issue 搜索页 | 工具不支持该 URL | 使用已有 GitHub 连接器的只读 issue 搜索；DeepFish 和 Domain-RAG fish 搜索没有给出来源解释。未发 issue 或联系作者 |

没有未决的自动安全审批拒绝、没有执行结果不确定的下载/训练动作，也没有隐藏重试。系统缓存写入失败不影响资产事实交付；尚未解决的是来源/协议说明和未来运行时验证，并非 GPU 已启动待处理。

交付位置：研究目录先在本任务可写区准备，2026-10-31 按授权以不覆盖方式移动到 Downloads/projects/RandAugment_CV_20261104/research-deepfish-baseline-20261031。目录迁移没有复制大资产，旧研究目录未修改。
