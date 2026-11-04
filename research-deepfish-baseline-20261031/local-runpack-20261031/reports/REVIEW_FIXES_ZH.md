# 一次独立静态审查及修复

审查结论为 3 个 Important、0 个 Critical；在一次修复中全部处理，未再扩展审查。审查前后均未调用模型或 Colab。

| 发现 | 修复 | 非模型证据 |
|---|---|---|
| JSON 丢失 tuple 导致官方 resize 构造及文本类别哈希失败 | 读取配置后恢复 classes、scale、scales、crop_size，保留其他列表 | 新纯 Python 回归先失败后通过，且 worker import 不导入 torch/mmdet |
| 封包器可能为等数量但变更的标签重新签发清单 | 固定 canonical annotation SHA、选中 ID/文件成员关系以及 4195/4660 的原 bbox/area | 改 area 或 dev 身份的人工副本必须被拒绝；原记录独立比对见 ANNOTATION_PROVENANCE_CHECK.json |
| 独立 expiry 脚本原可不带授权 context 释放 | 仅接收同一 context，先验证授权与从实际分配起算的截止时间，再查 Colab 模块/写状态/释放 | 未授权时四类副作用均未发生；超出预算的 deadline 被拒绝 |

另已补每次更新后的 Python/NumPy/Torch CPU/CUDA 随机状态，覆盖最后一次更新；将含义不清的 files_without_checksum_metadata 改为 manifest_payload_file_count。

修复回归结果：新增 5 项先出现预期失败，再全套 37 项通过。颜色测试只用人工图像及隔离官方纯方法；官方 MMCV/Runner、完整依赖和 CUDA 集成未执行。最初关于 torch 2.0.0 不支持 TORCH_FORCE_WEIGHTS_ONLY_LOAD 的担忧经源码复核已撤回，没有据此放宽加载限制。

官方类型契约来源：[MMCV 2.1 processing](https://raw.githubusercontent.com/open-mmlab/mmcv/v2.1.0/mmcv/transforms/processing.py)、[MMDetection 3.3 transforms](https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/mmdet/datasets/transforms/transforms.py)、[CocoDataset](https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/mmdet/datasets/coco.py)、[GroundingDINO](https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/mmdet/models/detectors/grounding_dino.py)。安全加载：[PyTorch 2.0 serialization](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.0/torch/serialization.py)。
