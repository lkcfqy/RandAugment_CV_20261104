# 资产、固定源码与待解事实

所有状态截至 2026-10-30。没有模型/数据下载、模型导入、推理、训练或主动提交申请。

## 当前证据

- TRIQA 固定 `rajeshsureddi/triqa@6e8a7a1c9c38b9f2dd0e9cce3052d65ac6ceec73`；上轮保存的推理源码与 Git blob 已核验。本轮复用，不冒称取得无复合训练的对照权重。官方 `triqa_simple.py:55–80` 加载两支与 KonIQ scaler/SVR；84–115 行全/半图推理。Box 权重页本工具返回不可访问；未下载，准确字节数/哈希仍未知。
- ARNIQA 固定 [`miccunifi/ARNIQA@66d16eb0ff1e1655872d32c0c233614a3922aaad`](https://github.com/miccunifi/ARNIQA/tree/66d16eb0ff1e1655872d32c0c233614a3922aaad)，2026-06-18T05:10:10Z。`hubconf.py` 提供 koniq10k 头，`single_image_inference.py` 为五裁剪双尺度；`models/resnet.py` 载入 ImageNet V1 后再由 hub 覆盖权重。仅查发布元数据：ARNIQA.pth 112,155,914 字节，regressor_koniq10k.pth 19,560 字节；release 无 digest，未下载。Apache-2.0 LICENSE 随源码保存。WACV 2024；本文读取 arXiv:2310.14918v2。
- 2stepQA 固定 [`utlive/2stepQA@e591dea197ca7aef4c518d41c423bd8589384bb4`](https://github.com/utlive/2stepQA/tree/e591dea197ca7aef4c518d41c423bd8589384bb4)，2021-01-17T01:59:50Z。`Functions/two_stepQA.m` 加载经典 NIQE 参数并乘 MS-SSIM。MATLAB 源码，未执行；不能称本机已有可运行 MATLAB 环境。README 中许可原文随证据保存。

## 人类数据

当前官方入口：[LIVE WCmp / Colorado LIVE](https://www.colorado.edu/lab/live/live-wild-compressed-picture-quality-database)。页面要求 access request form 后提供链接。旧 [UT 页面](https://live.ece.utexas.edu/research/twostep/index.html) 要求表单后发密码。两者均未提交；未向作者发送消息，未通过镜像绕过。

原始研究：[Yu et al., Predicting the Quality of Images Compressed after Distortion in Two Steps, TIP 2019](https://live.ece.utexas.edu/publications/2019/xiangxu2019tip.pdf)，§IV，论文 PDF 第 6–8 页。核实 80 源内容及四个版本、评分流程与 23 名保留评分者；评分表是否包含原始受试者/session 字段仍未核实。论文用的是较强 JPEG 档位，是否有足够真实改善事件必须观测，而非推断。

强前作：[Athar & Wang, Degraded Reference IQA, v1](https://arxiv.org/html/2110.14899v1)，§III-B/Fig.10、§IV-B/Eq.13、§V。有噪声后压缩非线性与六参数校准，基本反例已知。其 Waterloo Exp-II 标签为 SQB，不能冒充人类评分。

更早直接相关：Corchs, Gasparini, Schettini, “Noisy images-JPEG compressed: subjective and objective image quality evaluation”, SPIE 9016 (2014), DOI 10.1117/12.2039273。出版社检索摘要指出单一失真优化指标预测困难；本轮正文访问超时，官方机构条目定位未成功，不声称已读其全部实验。该未读项是最终新颖性审查仍须补齐的强近邻，不因未取得而当不存在。

## 不能跨过的解释界限

1. 这次明确允许把适用边界作为潜在贡献，故保留一个可观测问题；没有据“前作也谈复合失真”直接否决。
2. 没有等训练对照，不能把冻结 TRIQA 与 ARNIQA 的差异归因于有序损失；原文 TRIQA Table 3 的收益与本次假说可以同时成立。
3. 公共发布链接不等于资产已取得、字段完整或 Mac 上已验收。Gate 0 可用性需要官方评分表；Gate 1 还需权重与前处理核验。
4. 当前结论是“值得 Pro 决定是否做评分表观测”，不是创新通过，也不是模型失败。无须为了读文献继续投入无限搜索。
