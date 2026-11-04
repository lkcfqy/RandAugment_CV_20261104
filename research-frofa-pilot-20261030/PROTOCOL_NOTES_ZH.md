# 真实冻结特征 pilot：启动前工程细节固定

完整科学与资源协议见 protocol.json；本目录独立于原数学 fixture。source-seal.json 在启动前固定协议与全部 Python 实现的 SHA-256。首次直接 HTTP 请求（包括依赖 metadata）前写入持久连续时钟，4 小时总期限、45 分钟预检期限均不重置。浏览器只读官方资料和本地代码准备发生在该时钟之前。

- 评估计数固定 96,200：6,000 校准 validation + 90,000 最终 test + 200 预检测速。省略可选确认 validation。
- 校准 support 是每类 trainpool 前 10；确认 support 仅从剩余 90 抽样。每类 10 support、每 batch 2，每 epoch 恰有 5 batch；每 fit 80 epoch / 400 updates。
- 支持集、初始化和 batch order 在同一 episode 的五臂相同；增强 RNG 单独的 SeedSequence，不影响 sampler。
- MAP 使用隔离安装的官方 Transformers 4.57.1 类：attention(query,patch,patch)，随后 pooled + MLP(LayerNorm(pooled))，最后 Linear10。没有 query residual、额外输出归一化或预训练 MAP 权重。
- MAP 从 CPU fresh 初始化，应用同版本官方 SiglipPreTrainedModel._init_weights；分类器 normal std=768^-0.5，bias=0。初始化与 batch order 全量 hash 保存并逐臂核对。
- Weight decay 的工程解释固定为参数 ndim>=2，包括 learned probe；bias 和 LayerNorm 一维参数不 decay。AdamW foreach=False/fused=False，β(.9,.999)、eps1e-8。第 20 update 达峰，第 400 update LR=0。
- 明确 eager encoder attention；官方确定性 SiglipImageProcessor，224 bicubic、1/255 和 .5/.5。训练只用 post-layernorm/pre-MAP patch。SigLIP 无 CLS。
- 仅接受官方 PyPI 兼容 wheels，不执行远程仓库代码。HF trust_remote_code=False、use_safetensors=True、local_files_only=True。依赖放 deps/，不改共享 .venv。
- 控制的数学运算由固定 FP32 缓存提升到 float64，最终增强特征 cast FP32 给 MPS；常量仍保留 epsilon。GC/GE 不二次 clamp。精确矩对象是连续条件分布，不能声称浮点舍入后的无限精度等效。
- Gaussian 仅为当前 100 support 构建。每图按 channel 填充零权重 GL 节点，前缀和实现同一个已验证因子；无 5000 图预建、无大尺寸 T² 协方差。仅指定真实 2 图×4 channel 的预检允许稠密196-token解析 oracle。
- 预检恰为两 a×五臂×20 updates=200。每组前 10 步热身，后 10 步 p90 计时；每臂采用两 a 中较慢值。投影另含六次最慢 support 构建、96,000 eval 图以及180秒归档余量，需低于剩余墙钟75%。不因资源不足减少配方。
- MPS fallback 显式为0。CPU parity、图像预处理、精确增强属于预定 CPU 工作，不称 GPU fallback。保守内存检查使用已达峰的 process RSS + 当前 MPS driver allocation；每次 GPU batch同步和更新计时，并另记录 CPU process time。
- 30个确认模型完整冻结并保存哈希后，才统一一次测试。任何资源或实现失败只归档已完成项，不以缺失配对宣称等效，不自动追加方向Q。

官方资产核验来源：

- https://zenodo.org/records/7711810 ：RGB ZIP MD5 f46e308c4d50d4bf32fedad2d3d62f3b，v2。
- https://huggingface.co/google/siglip-base-patch16-224/blob/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed/model.safetensors ：SHA256 2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8。
- https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/siglip/modeling_siglip.py ：MAP、encoder特征位置及初始化顺序。
- https://github.com/phelber/EuroSAT/blob/master/LICENSE ：MIT 原文；另保存 Copernicus Sentinel Legal Notice 和模型 Apache-2.0 原文。EuroSAT 源于 Copernicus Sentinel-2；本地 pilot 不发布资产，不推定具体影像采集年份。

不称空间OOD、不排除预训练重叠。首轮统计单位是固定pool/test下的六个完整配对 episode，非3000测试图 bootstrap。
