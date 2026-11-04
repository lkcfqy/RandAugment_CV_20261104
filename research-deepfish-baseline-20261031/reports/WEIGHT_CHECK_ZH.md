# 原始 GroundingDINO Swin-T 权重检查

下载的是 Domain-RAG 原始 grounding_dino 路径指定的 [OpenMMLab 转换权重](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)。

| 项目 | 实测 |
| --- | --- |
| 字节数 | 691901857，与指定大小相同 |
| SHA-256 | `822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3` |
| MD5 | `aa587e4dde3f2de265f33593bc9de6c4`，与官方 ETag 和指定值相同 |
| 已安装解释器 | Python 3.14.10，来自已有 cpu-readiness 虚拟环境 |
| 已安装 PyTorch | 2.13.0 |
| 读取 | torch.load(..., map_location='cpu', weights_only=True) |
| 对象 | OrderedDict，903 个 state tensor，没有非张量 state 项 |
| dtype | 890 个 float32 张量、13 个 int64 张量 |
| 有限值 | 所有 float 张量有限；整数字段逐项记录 |
| 总元素 | 173532066，包含 buffer，不能等同于独立可训练参数个数 |

关键形状：

| 键 | 形状 |
| --- | --- |
| backbone.patch_embed.projection.weight | [96,3,4,4] |
| neck.convs.0.conv.weight | [256,192,1,1] |
| query_embedding.weight | [900,256] |
| language_model.language_backbone.body.model.embeddings.word_embeddings.weight | [30522,768] |

backbone stage 的 block 索引对应 [2,2,6,2]，与静态 Swin-T 配置一致。[WEIGHT_CHECK.json](WEIGHT_CHECK.json) 记录全部张量键、dtype、shape、device 和 finite 结果；[检查脚本](../scripts/inspect_checkpoint.py) 可审阅。

边界：没有实例化 detector、执行前向/反向、验证完整 load_state_dict 的 missing/unexpected keys，也没有据此声称权重与未来运行时完全兼容。没有 unsafe pickle fallback、动态 safe_globals/白名单或依赖安装；没有调用 GPU。初次官方 HEAD 因 curl 的 max-filesize 对 Content-Length 生效而返回 63，但头已取得，正式 GET 的大小、MD5、SHA-256 均在上述本地检查中独立核对。

检索本项目与本机下载/cache 范围时未发现同名 FISH 或指定 Swin-T 权重，因此只下载一份。已有 bert-base-uncased 目录可作为后续复用候选：7 文件共 441169829 字节，路径和哈希见 [EXISTING_BERT_ASSETS.json](EXISTING_BERT_ASSETS.json)；此次只哈希，没有复制或加载。它与此 checkpoint 的文本构造路径兼容性尚未运行验证。
