# 两条官方调用链及坐标核对

## A. Metric3D：NYU canonical-label → resize/crop → depth-normal supervision

固定提交：eb5b6fac0dc155e4e52f576e304fbf11655ff339，仓库 YvanYin/Metric3D。

| 环节 | 定位 | 已核对的接口 |
|---|---|---|
| 选择配置 | [vit.raft5.giant2.nyu.py L19–50](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/configs/RAFTDecoder/vit.raft5.giant2.nyu.py#L19-L50)；[L89–120](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/configs/RAFTDecoder/vit.raft5.giant2.nyu.py#L89-L120) | NYU 数据；LabelScaleCononical → RandomResize → RandomCrop → mask/flip/appearance；DeNoConsistencyLoss 的 loss_fn=CEL。基础 nyu.py 的 ResizeCanonical 管线被该配置覆盖，不能误当本链。 |
| 数据装载与输出 | [nyu_dataset.py L42–87](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/datasets/nyu_dataset.py#L42-L87) | RGB、depth、normal、intrinsics 同时进入 Compose；返回 intrinsic、normal 与累计 label_scale_factor，字段名 scale。 |
| canonical label | [transform.py L228–277](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/utils/transform.py#L228-L277) | a=f_c/f_x；fx/fy 乘 a，主点不变；depth 乘 a；记录 scale=a。 |
| 随机 resize | [transform.py L779–854](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/utils/transform.py#L779-L854) | 图像坐标按 r 放大、主点乘 r，focal 保持 canonical；depth 空间重采样并除 r；累计 scale=a/r；法向仅空间重采样。 |
| crop | [transform.py L699–763](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/utils/transform.py#L699-L763) | 图像/标签共享区域；主点随 crop/pad 平移。padding 与无效标签不用于以下理想推导。 |
| 数据传入 loss | [__base_model__.py L25–39](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/__base_model__.py#L25-L39)；[L147–174](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/__base_model__.py#L147-L174) | data 并入输出，kwargs 传至 criterion，包含 intrinsic 与 scale。 |
| loss dispatch | [NormalBranchLoss.py L365–394](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/NormalBranchLoss.py#L365-L394)；[L453–465](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/NormalBranchLoss.py#L453-L465) | CEL 路由 forward_S；其读取样本 scale 并交给 Depth2Normal。配置中的 scale=2 是非 normal-GT 数据的权重，不能与样本几何 scale 混淆。 |
| 反投影恢复 | [depth_to_normal.py L211–234](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/depth_to_normal.py#L211-L234)；[L260–288](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/depth_to_normal.py#L260-L288) | 先 inv(K)×pixel×depth，随后仅 Z 除以样本 scale，再求物理法向。 |

**独立代数核对（不属于数值实验或新定理）。**

原始有效像素的物理坐标为：
X=(u−cx)D/fx，Y=(v−cy)D/fy，Z=D。

忽略取整/插值，令 a=f_c/fx，r 为正等比例 resize。上述代码链给出
u′=ru、v′=rv，K′=(a fx,a fy,r cx,r cy)，D′=(a/r)D，s=a/r。
于是用 K′、D′ 反投影先得到 (X,Y,sZ)；Depth2Normal 再令 Z←Z/s，得到 (X,Y,Z)。
后续共同 crop 平移同时抵消像素与主点的平移；统一的 depth 归一化只给三维点增加共同常数尺度，不改变理想法向方向。

所以“canonical depth 的数值变了而 normal 标签未旋转”本身不是监督矛盾。此核对限定在 canonical/resize/crop 子链，排除离散边界、无效点与报告所述水平翻转疑点；不声称全管线逐像素精确或全部 bug-free。

## B. UniDepthV2：ETH3D 双副本 ContextCrop → camera update → output SelfDistill

固定提交：8d8cfe4c7ee15297099983607febf0d4f32eb3d6，仓库 lpiccinelli-eth/UniDepth。

| 环节 | 定位 | 已核对的接口 |
|---|---|---|
| 选择配置 | [config_v2_vits14.json L26–39](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/configs/config_v2_vits14.json#L26-L39)；[L67–108](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/configs/config_v2_vits14.json#L67-L108) | num_copies=2、num_frames=1、SelfDistill、random_scale=2；训练列表含 ETH3D 与 Waymo，本轮只追 ETH3D。 |
| 相机与深度来源 | [eth3d.py L13–18](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/eth3d.py#L13-L18)；[L48–70](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/eth3d.py#L48-L70)；[image_dataset.py L95–129](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/image_dataset.py#L95-L129) | 从元数据 K 创建 Pinhole，原始 depth 除以数据集 depth_scale。未读取训练 HDF5 或任何测试数据。 |
| 两个视图 | [base_dataset.py L273–277](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/base_dataset.py#L273-L277)；[image_dataset.py L132–149](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/image_dataset.py#L132-L149) | 复制同一原样本；每份重置 ContextCrop 的 ctx 并变换。未运行 dataloader，因此不声称已动态验证整个 batch 配对顺序。 |
| 几何同步 | [transforms.py L1246–1329](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/pipelines/transforms.py#L1246-L1329)；[camera.py L78–120](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/utils/camera.py#L78-L120) | crop 改 cx/cy；resize 同时改 focal 与主点；图像/GT/mask 使用同一裁剪区域。GT 仅空间重采样，不附加物理 depth 倍率。 |
| 翻转 | [transforms.py L137–174](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/pipelines/transforms.py#L137-L174)；[camera.py L100–110](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/utils/camera.py#L100-L110) | 图像/GT/mask 同翻转，camera 依方向改对应主点，记录 flip；本轮不将其扩展为所有位姿路径的认证。 |
| 量的定义 | [unidepthv2.py L361–377](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/models/unidepthv2/unidepthv2.py#L361-L377)；[camera.py L255–273](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/utils/camera.py#L255-L273) | rays 与 radius 相乘给 points；对外 depth 是 points 的 Z 分量；Pinhole 重建也以 Z-depth 为输入。 |
| 施加位置 | [unidepthv2.py L168–207](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/models/unidepthv2/unidepthv2.py#L168-L207) | 深度监督用 depth；camera loss 用 rays；invariance 用输出 depth、变换后 K、depth mask、flip。未声称所有训练时相机提示都是 GT；decoder 的预测相机与输入监督相机须区分。 |
| 对齐与损失 | [distill.py L27–130](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/ops/losses/distill.py#L27-L130) | 按 paired K 计算 zoom/translation，撤销相对翻转，pad/crop 后交集 mask；双向 stop-gradient；该配置对 masked squared error 开方。 |

**正确同步之后的剩余问题边界。**

在理想 crop/等比 resize 的保留像素上，u′=Au、K′=AK，所以 K′⁻¹u′=K⁻¹u；以同一相机中心和坐标轴定义的 Z 标签不需因图像 resize 改数值。这保证几何标签兼容，**不保证两个有限上下文的模型预测必相同，也不证明加一致性损失一定有益**。

UniDepth 已把一致性与相机条件化联系起来；本轮没有找到一个既区别于上下文丢失、又不依赖内参错误/离散重采样、并且有明确新预测的后续机制。不能用“增强前后预测不一样”充当贡献证据。

DAC 只读论文，未审第三条代码链。其 range 与 Z 的区别、ERP/FoV/pitch/多分辨率机制见报告列出的原文位置。

