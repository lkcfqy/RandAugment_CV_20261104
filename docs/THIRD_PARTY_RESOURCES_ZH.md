# 第三方源码、数据与换机恢复

本候选保留自有实现、经过逐项来源核对的必要外部源码及其许可。环境副本、缓存、指定的四个大文件，以及许可尚不清楚的外部全文、源码和混合数据不随本次提交。源项目中的原件仍保留。克隆可以继续阅读和开发；原实验重放仍需恢复相应资源，不能将现有历史结果当作本候选已重新运行的结果。

本文记录固定来源和恢复边界，不执行下载或安装。文件哈希用于核对版本，不代表取得分发许可。自有实现、上游代码、模型、数据和论文各有边界；论文公开可读、“学术用途”和代码根目录的 MIT 许可证均不能自动授予相邻图片、模板或全文无限制再分发权。

## 保留的外部源码与通知

以下路径相对仓库根。保留文件自己的版权头、完整许可证、NOTICE 和修改说明；同目录中的图片、数据、日志或压缩包不因此一并获准。最终候选还适用独立的隐私、凭据和日志过滤。

| 本地位置/对象 | 固定来源 | 许可与边界 |
|---|---|---|
| `mawa-feasibility-20261102/sources/` 中已核对源码 | [augmentation_invariance](https://github.com/keenan-eikenberry/augmentation_invariance/tree/14eac6091c74a85fe298a065f9796f8056da636c) | MIT；保留 `LICENSE.txt` 中原版权原文。 |
| `syntheticgen-feasibility-20261102/sources/SyntheticGen/` | [SyntheticGen 固定提交](https://github.com/Buddhi19/SyntheticGen/tree/ca724b2693bbf07725606dbaf4024602bcc0f17a) | MIT；不扩展到相邻 DAFormer 或素材。 |
| 同目录下的 DAFormer `uda_dataset.py.txt`、README | [DAFormer 固定提交](https://github.com/lhoyer/DAFormer/tree/e6f6e041e7477b642106c134c1715fc934ccd210) | 明确 Apache-2.0 的文件；随附 DAFormer LICENSE 与 LICENSES.md。后者列出的 NVIDIA/DACS 等例外不能忽略。 |
| `research-counting-exemplar-20261031/sources/countgdpp-github/`、`countgdpp-hf/` 及已选 payload 源码副本 | [CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/2ecbc4331d6730d017f52a4a97bbcbae7d5d714a) | 根 MIT，并保留 GroundingDINO/DETR 等组件的 Apache 版权头。 |
| `research-counting-exemplar-20261031/sources/pairtally/` 的代码/说明 | [PairTally](https://github.com/bbvisual/PairTally_Benchmark/tree/4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1) | 代码 Apache-2.0；数据另为 CC BY-NC 4.0。旧记录将数据写成 CC BY 4.0 不准确，应以固定 `dataset/README.md` 的数据条款为准。 |
| `research-deepfish-baseline-20261031/sources/CDFSOD-benchmark/` 及可适用的嵌套 MMDetection 源码 | [CDFSOD](https://github.com/lovelyqian/CDFSOD-benchmark/tree/929413d7d61caf72808a8c27cce19bf51c2973dd)、Domain-RAG 固定提交 `2f66fdcd4a92a0fe1d41aeec48f21f68cd709f54` 内的 MMDetection | 按各自完整 LICENSE 保留；不是 FISH 数据、GroundingDINO 权重或 Domain-RAG 全目录的分发授权。 |
| `research-deepfish-baseline-20261031/ets-fixed-baseline-audit-20261101/official_ets/` | [ETS](https://github.com/jaychempan/ETS/tree/99cfcff8e17844c328e43f1417153b41afb70f9d) | 保留原 LICENSE 与原代码通知。 |
| `research-mill-proxy-20261030/vendor/HVI-CIDNet/` | [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet/tree/eb43d7d91e9a336c66856824ff9e4603ae41f408) | MIT，保留作者版权。MILL 数据卡不作为 MILL 研究仓库全部源码的许可。 |
| `research-nsegment-audit-20261030/run-20261030T174528Z/sources/` 中明确选定的源码 | [NSegment](https://github.com/unique-chan/NSegment/tree/5c66795b38ffd351f2d3363bc4eb3e5c3b580fac) | 根 MIT、嵌套 MMSegmentation Apache-2.0 和源文件自身继承通知分别保留；附完整 MMSegmentation LICENSE。OpenCV 引用保留自身版权条件和完整 LICENSE。 |
| `research-shared-theta-ad-20261030/sources/mmr-*` 的代码/配置 | [MMR](https://github.com/zhangzilongc/MMR/tree/22d1b171687509024e6f5b53b5a5e772d653d32b) | 源码 Apache-2.0；数据 CC BY 4.0，分别保留 `mmr-LICENSE`、`mmr-LICENSE-DATASET`。 |
| 同目录的 Transformers DINOv2 引用文件 | [v4.46.3](https://raw.githubusercontent.com/huggingface/transformers/v4.46.3/src/transformers/models/dinov2/modeling_dinov2.py)、[v4.57.1](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/dinov2/modeling_dinov2.py) | 文件明确 Apache-2.0，保留 Meta/Hugging Face 版权头及完整 `transformers-APACHE-2.0.txt`。forward 摘录已核对为 v4.57.1 原文子串，应连同完整有署名源文件保留。 |
| `research-shared-theta-ad-20261030/models/dinov2-small/` 的配置/模型卡 | [固定模型版本](https://huggingface.co/facebook/dinov2-small/tree/150c8e7bb7cef2d30ec31b13a517af14840ee3f7) | 模型卡声明 Apache-2.0，并保留 `LICENSE-UPSTREAM`；这里只说明小型配置/通知。 |
| `research-hvp-entropy-20261030/run-20261030T180959Z/sources/` | [DINO](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a)、[官方 args](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt) | Apache-2.0 的源码说明、完整 LICENSE 和小型参数文件。 |
| `research-view-route-20261030/sources/deit-*` | [DeiT](https://github.com/facebookresearch/deit/tree/7e160fe43f0252d17191b71cbb5826254114ea5b) | Apache-2.0；保留完整 `deit-LICENSE`。 |
| 同目录 `timm-README.md`、`timm-config.json` | [固定 timm 模型卡](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k/tree/80e968688553f219e4a86f940ed945a23709c16f) | 模型卡 Apache-2.0；另带 `timm-APACHE-2.0.txt`，不提交整个安装环境。 |
| 同目录 `torchvision-0.28.0-oxford_iiit_pet.py` | [TorchVision v0.28.0](https://github.com/pytorch/vision/blob/v0.28.0/torchvision/datasets/oxford_iiit_pet.py) | 与同版本安装包对应文件字节一致，SHA-256 `7042f77ee56af19f63c5c0defbfc24d7ccd53b5ae1b9268a865d78e2e7b6e54a`；保留同一包的 BSD-3-Clause `torchvision-0.28.0-LICENSE`。 |

ControlAugment 的完整 MIT 通知位于 `research-ctrla-mechanism-20261030/THIRD_PARTY_LICENSE.txt`。需要恢复上游树时使用 [固定提交](https://github.com/jbcdfm/ControlAugment/tree/e77a4c19a3db893c120fc18b774fa290e826592b)；[固定 tar.gz](https://codeload.github.com/jbcdfm/ControlAugment/tar.gz/e77a4c19a3db893c120fc18b774fa290e826592b) 的记录 SHA-256 为 `8cb10e2f9a459fa2637d2b1e8426c9d3a710eda0a267d5edb6664bcb6bfc4ae7`。本轮没有把原先忽略的整个源树自动加入候选。

## 外部数据和模型恢复

先取得来源和用途所需的许可，再核对固定 revision、文件长度及完整 SHA-256，最后按原 manifest 恢复目录布局。压缩包必须检查路径穿越、重复路径、符号链接及解压体积。只下载了部分 ZIP ranges 的记录不能宣称验证了整个 ZIP。公开获取 URL 也不表示本仓库可以重新分发。

| 对象与目标位置 | 官方来源/版本和核对 | 缺失影响及处理 |
|---|---|---|
| PairTally 94 张选图，`research-counting-exemplar-20261031/data/images/`，以及 payload science 图片/标签 | [固定作者数据说明](https://github.com/bbvisual/PairTally_Benchmark/blob/4a6dbcf916a75090f0c505ace9d3ee8ad5960cb1/dataset/README.md) 中的官方获取入口；原记录只有分段获取，完整归档 SHA 未验证 | 代码 Apache 不覆盖数据；数据 CC BY-NC 4.0，保留作者署名并落实非商业条件。当前不随候选提供图片/标签混合包；计数实验需要恢复原选图和冻结标签，不能用新造标签替代。 |
| CountGD++，`research-counting-exemplar-20261031/models/` | [HF 固定 revision](https://huggingface.co/nikigoli/CountGDPlusPlus/tree/06bd674ecd2d18ea55ec0dd9183fa24603dfa7e1)，937,560,192 B；SHA-256 `94eabe4e2595dac1e04cc1703641458b13c1c624ce10072ba57dd96b625a41f5` | 权重与其配置缺失时不能执行原推理；从固定模型仓取得并保留适用通知。 |
| BERT，`research-counting-exemplar-20261031/models/bert-base-uncased/` | [HF 固定 revision](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)，权重440,449,768 B；SHA-256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` | Apache-2.0；除权重外还需原 tokenizer/config，不能混用新版本。 |
| FISH 原图与检测框，DeepFish 各 `payload/data/` 和原混合包 | [固定 CD-ViTO 数据入口说明](https://github.com/lovelyqian/CDFSOD-benchmark/blob/929413d7d61caf72808a8c27cce19bf51c2973dd/README.md)、[DeepFish 作者页](https://alzayats.github.io/DeepFish/)；原包761,948,029 B，记录 SHA-256 `714ed9b731efab05ddfb4bd3b73127b67ebcb2a3a5f7efa07f66f314b9ad6094`（本地实测，未找到发布方独立 checksum） | 原 DeepFish 图片 CC BY 4.0 不自动覆盖 CD-ViTO 派生框。混合归档未见独立许可通知；本候选排除混合数据和重复 payload，原检测/训练需另恢复4505图与对应检测标注。 |
| GroundingDINO，`research-deepfish-baseline-20261031/assets/` | [OpenMMLab 官方权重](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)，691,901,857 B；SHA-256 `822d7e9db9ce6ff2119b72dc6e78606a1b0a2c307234798adf0cab50f1b424e3` | 已有来源/摘要，不把代码 Apache 直接当作该权重完整分发证明。候选不提供该二进制；原推理需按来源恢复。 |
| Brackish v4 视频、标注、解码帧和拼图 | [作者 AAU 页面](https://vap.aau.dk/the-brackish-dataset/)、[作者 Kaggle 数据集](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset)，固定 version4；原包 SHA-256 `95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da` | 作者标注 CC BY-SA 4.0；COCO 的 MIT 字段来自转换脚本常量，不能替代作者条款。当前排除 `original_v4/`、assets、frames、overlays、pilot data 和混合包。恢复后保留署名、修改标记和适用 ShareAlike；QA 帧未证明与作者导出帧逐像素相同。 |
| MILL 图像，`research-mill-proxy-20261030/data/MILLs-DSLR/` | [作者固定数据版本](https://huggingface.co/datasets/davidserra9/MILL/tree/761a8e0320fa8e818e5929e0cba269015cd60e06)，MILLs-DSLR.zip 204,658,447 B；SHA-256 `cf14eaf7ac287a30822fd626cba76b255f39e416b95bcc089e48c7aa91503112` | 作者数据卡声明 MIT；保留该来源通知。原实验需要固定 panel 的84张原图；依据既有 panel/archive manifests 选取，不能根据图像内容重选。 |
| HVI 模型，`research-mill-proxy-20261030/models/model.safetensors` | [作者固定模型版本](https://huggingface.co/Fediory/HVI-CIDNet-Generalization/tree/51481ef2546f870060c43eb6d6525399f5b3d2d3)，7,920,332 B；SHA-256 `2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894` | 模型卡 MIT；需与原架构、配置和通知一起恢复。过去的严格加载记录不代表本候选已重新执行 forward。 |
| AeBAD-S 图像/掩码，`research-shared-theta-ad-20261030/data/` | [MMR 固定作者说明](https://github.com/zhangzilongc/MMR/tree/22d1b171687509024e6f5b53b5a5e772d653d32b)，保留的 `README-SOURCE.md` 和 `LICENSE-DATASET` | 数据 CC BY 4.0；恢复时保留作者、来源和修改说明。原协议需要521 train、962 test和657 masks；小型来源文档不等于完整数据已经随克隆提供。 |
| Oxford-IIIT Pets | [官方数据页](https://www.robots.ox.ac.uk/~vgg/data/pets/)；按原 manifest 和官方校验记录恢复 | TorchVision 加载器 BSD、DeiT Apache 都不等于照片许可。网页全文快照不随候选保留；照片及标注须由官方来源单独取得。 |

DINO HVP 的原完整 checkpoint 来源为 [官方固定文件](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth)（352,247,243 B）。DeiT 原权重为 [官方文件](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)（22,917,895 B；SHA-256 `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`）。使用各实验保留的固定资产清单核对；不要仅因为文件名含摘要前缀而跳过完整摘要验证。

## 暂不分发的引用源码和全文

AdaAugment 原引用树未找到与这些文件绑定的完整许可证；不能仅凭仓库公开或 README 提及 TrivialAugment 推导授权。原 `audit_fixture.py` 会实际读取、摘要校验并 AST 检查这些文件。省略后，该原引用审计不完整；不得通过删除断言、改历史摘要或用随机 fixture 声称恢复通过。

FAOrCo-ViT 两份引用代码同样尚无本地绑定的许可/许可头证据。它们是来源快照，并不是自有 fixture 实现。下表给出原取得位置与预期摘要；恢复应在自用资源目录进行，并单独确认许可。论文 HTML/PDF/全文也不由相邻源码许可证自动覆盖；通过论文链接阅读即可，不将未确认再分发权的全文打包。

| 目标相对路径 | 固定获取来源 | SHA-256 |
|---|---|---|
| `research-adaaugment-audit-20261030/sources/A2C.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/A2C.py) | `95165b5ecc02bdf7ffac77df375e4929f397b6cbba92bf472832814d16f66cf4` |
| `research-adaaugment-audit-20261030/sources/Dataset.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Dataset.py) | `606a593a437f6863f2c7e843891638cd801bc9e2e4f486537766d98d1416eee4` |
| `research-adaaugment-audit-20261030/sources/Network/AlexNet.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/AlexNet.py) | `9a1243c49dd0012b12fa6ea39a0e823b25961703d33a75fa2a00b981ae477c47` |
| `research-adaaugment-audit-20261030/sources/Network/ResNet.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/ResNet.py) | `0fa715eadaedded512f0090ad01eaf4e2013dd0d6210cb84b68611e9083aa955` |
| `research-adaaugment-audit-20261030/sources/Network/VGG.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/VGG.py) | `039f7aab8fa2a9d538017ebbbf53d4d88a0e0b2d8b681702766a03ae4f79ef90` |
| `research-adaaugment-audit-20261030/sources/Network/Wide_Resnet.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/Wide_Resnet.py) | `da973693a5fb383dc3ba381752e9abfdc9f77e5e1bb90bff0d541bcfd387858c` |
| `research-adaaugment-audit-20261030/sources/Network/__init__.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/__init__.py) | `c6aa295b7580b7ed2fe295f7035e5641b67bd1b7474e8e874664543f852b4e66` |
| `research-adaaugment-audit-20261030/sources/Network/pyramidnet.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/pyramidnet.py) | `8f39c14a0785f3af18b77e13dd5df558628784244723fdb56b49ae282959e487` |
| `research-adaaugment-audit-20261030/sources/Network/shake_pyramidnet.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/shake_pyramidnet.py) | `0bc3f00aec8364ec7a098703c3dd52c0cd8df8d171576e42afcfbb294dcb82ae` |
| `research-adaaugment-audit-20261030/sources/Network/shakedrop.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/Network/shakedrop.py) | `4ca227b517b46476d6f3de891c7ed74594e498c3a01ed6d7ee29c42513b26568` |
| `research-adaaugment-audit-20261030/sources/README.md` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/README.md) | `2fbc1bace67fd33798054965d39a2250c486212d7cb77d1e4c5971b6f614608a` |
| `research-adaaugment-audit-20261030/sources/augmentation/__init__.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/__init__.py) | `7e381f1db43a759c017615dfa221ee28bab74f5e37036bc73335f52f31781fdc` |
| `research-adaaugment-audit-20261030/sources/augmentation/adaaugment.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/adaaugment.py) | `5c95357c403c51ad67492b9223f4907859cd41c20b925837bbf04cf6de9a1f97` |
| `research-adaaugment-audit-20261030/sources/augmentation/autoaugment.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/autoaugment.py) | `756743cd9ea0f0f836ea328ac3c1b8529ba5cdc8b8cff3a001cf6cd4cb039860` |
| `research-adaaugment-audit-20261030/sources/augmentation/cutout.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/cutout.py) | `1662ffbd465fdafb1f48d9782b963b1ef7b01bb72d05a2355c3bd93b2dc71c1f` |
| `research-adaaugment-audit-20261030/sources/augmentation/ops.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/ops.py) | `bfc8a7a2ac17fea2d849e484b95f98387131bda9b04f1a99969fd7bff33ead3a` |
| `research-adaaugment-audit-20261030/sources/augmentation/trivialaugment.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/augmentation/trivialaugment.py) | `9170d4fab2589ee1599b3e0231c3606a09480de7515a128b905df5e7b797e6c6` |
| `research-adaaugment-audit-20261030/sources/config.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/config.py) | `214b46bae75fee2dfa732b69f8b31d718dc9a082f6fcb117815ede75d7935278` |
| `research-adaaugment-audit-20261030/sources/confs/resnet18.yaml` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/confs/resnet18.yaml) | `d096f2773124cada241fb0d47060a8e98bd42b561a3443e9cefdf2b6d91e44a7` |
| `research-adaaugment-audit-20261030/sources/confs/resnet50.yaml` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/confs/resnet50.yaml) | `f3fe8860008922a3e3ee2a81bfed0255d27eb1bc445f713000530082ee042cf7` |
| `research-adaaugment-audit-20261030/sources/normalization.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/normalization.py) | `b602ad2148b02f014a95a8043bf1c00f60bae1b949a72a945321360c90aa203c` |
| `research-adaaugment-audit-20261030/sources/organize_transform.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/organize_transform.py) | `46373b6108a119e5744a7a0b403c551a15409497fd5bee00c95be2a2f8f6cce0` |
| `research-adaaugment-audit-20261030/sources/train.py` | [固定原文件](https://raw.githubusercontent.com/Jackbrocp/AdaAugment/d854af15467285ad8f46500d6fc2da5225fb5bd0/train.py) | `b2e4f9af7565b9d68378e2fc948b98d139d4b4a1a8aad595fa31052259128f2a` |
| `research-frofa-fixture-20261030/source/Network_vit.py.txt` | [固定原文件](https://raw.githubusercontent.com/learninginvision/FAOrCo-ViT/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/Network_vit.py) | `a1e8dfbbf6d011ba1522ad3e64dd64cd72ed5daf5246e49df50dd164da2bf584` |
| `research-frofa-fixture-20261030/source/feature_aug.py.txt` | [固定原文件](https://raw.githubusercontent.com/learninginvision/FAOrCo-ViT/48424b86061fae71e983d96fc803769185575dbe/models/orco_vit/feature_aug.py) | `5680796db88811dfb79b6c6291ff4806114118acc31ea62534adc629ad5b22f9` |

## 环境、四个大件与验证边界

`.venv`、`.python`、`deps`、wheels、HF缓存等是运行环境或缓存，不随候选复制。根据各实验的依赖锁、版本及 wheel 摘要创建新环境；macOS arm64、CPU 与 CUDA wheel 不能直接互换。部分旧记录只有已装版本而非完整跨平台锁，故不能承诺任意机器安装后立即精确复现。不要为恢复环境直接执行来源不明的归档安装脚本。

四个明确省略的文件为：

- `research-frofa-pilot-20261030/cache/features.fp32.npy`
- `research-view-route-20261030/cifar100-amendment-v1/data/original_rgb_uint8.npy`
- `augmentation-selection-cifar100-run-20261102/dinov2_small/support_features.npy`
- `augmentation-selection-cifar100-run-20261102/dinov2_small/W.npy`

详见主恢复文档的备份和摘要记录。CIFAR RGB 文件可以从官方 CIFAR-100 Python 原包按固定 train 后 test 顺序、CHW→HWC 和原 NumPy 保存约定重建；原生成器会拒绝已有 manifest，因此要在独立空目录进行数据恢复，避免改写历史 split/manifest。其他特征、拟合矩阵和历史产物需要原备份或原协议计算，不能填入零数组或随机 fixture 代替。本轮没有执行这些重建、下载数据/权重或运行模型。

历史报告仅描述当时运行。去掉外部源码、环境或数据会使相应的来源摘要检查、资产校验、模型加载或实验重放无法进行；这与不依赖它们的开发检查是不同范围。最终候选应分别记录实际运行的检查与尚缺资源的检查，不能改断言或历史结果以消除这些缺口。
