# DeepFish 两臂最小运行包：本机准备完成

本目录是新增交付，历史研究文件未覆盖。范围只有固定官方 5-shot、seed 20261031、A/B 两臂；没有安装依赖、上传、分配 GPU、构建或运行模型。

- [待上传 ZIP](deepfish_5shot_two_arm.zip)：**11,672,127 字节 / 11.131 MiB**，上限 13 MiB；88 个包成员。
- ZIP SHA-256：`0fa965babadfaa83b9e7a83aadd81470fc76b566c118e7883ade3b0042ab2263`。
- [封包回执与隐私清单](reports/PACKAGE_RECEIPT.json)、[完整 SHA 清单](payload/SHA256SUMS)、[执行计划及自检](IMPLEMENTATION_PLAN.md)。
- [准备状态与剩余条件](reports/LOCAL_PREPARATION_ZH.md)、[独立审查修复](reports/REVIEW_FIXES_ZH.md)、[37 项本机测试记录](reports/LOCAL_TEST_RESULTS.txt)、[测试环境与边界](reports/LOCAL_TEST_SUMMARY.json)。
- [标签原记录比对](reports/ANNOTATION_PROVENANCE_CHECK.json)。

包中恰为 69 张公开图像、support/dev JSON、两份显式配置、driver/worker 与必要契约/落盘代码、候选依赖和官方资产 SHA；没有 test、模型权重、完整仓库、旧实验、个人路径或凭据。69 张 JPEG 的原字节 SHA 全部核实，无 EXIF/comment/XMP；dev 保留 annotation 4195/4660 原 bbox/area。封包器强制核验冻结标签摘要及选中身份后才生成文件清单。

A/B 均调用 Color(prob=1, level=0)，factor 分别为 1.0/0.8；cv2 uint8 BGR。人工图像已验证 A 逐像素恒等、两臂随机数消费相同。源图顺序、官方几何、GT 以及更新前后随机状态可配对核对。每臂预检 15 更新并丢弃全部状态；正式从原权重和空优化器重建。U=5E，U 仅按预检计时从 {25,50,100,200} 冻结，每张 support 恰 E 次；固定最终 checkpoint，无 save_best，先完成两臂配对才评估 64 图 dev。

未来上传/模型运行授权仍待答复。总 2 CU / 2 GPU 小时是拟议上限，包含环境准备至落盘释放；兼容 CUDA、实时费率、吞吐、持久存储和独立 owned-session 释放核对尚未验证，当前资源状态为 **UNDETERMINED**。未证明预算可行或增强有效。若最小候选 U 也无法适配，则报告资源未定，不缩样本。

授权及外部条件落实后，在 Colab 解包目录的唯一待执行命令：

```sh
python -B driver.py --run-context /content/deepfish-run-context.json
```

**现在不执行。** 活动 context 必须在包外依据真实授权、分配时刻、费率上界、持久存储及释放保障填写，且绑定本次 MANIFEST SHA `ec69120a9f62fd13d6e0ee3a774771442f29c11f2812036bda0f7cc7b1a65776`。包内 context 示例明确关闭授权。
