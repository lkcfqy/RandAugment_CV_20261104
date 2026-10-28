# Oxford-IIIT Pet 素材署名、来源与修改

数据集作者：Omkar M. Parkhi、Andrea Vedaldi、Andrew Zisserman、C. V. Jawahar。作品：*The Oxford-IIIT Pet Dataset*；论文 *Cats and Dogs*，CVPR 2012。

- [官方数据页面](https://www.robots.ox.ac.uk/~vgg/data/pets/)当前声明 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，版权仍归原图所有者。官网源文本与读取时间、哈希见 `official-license-page.html.txt` 和 `license-receipt.json`。
- 本轮直接读取原始JPEG和PNG的[公开镜像](https://huggingface.co/datasets/antokun/The-Oxford-IIIT-Pet-Dataset-With-Annotations/tree/003aeb4bc80d584ef307dc24e8b38d26cf9f9e43)，revision `003aeb4bc80d584ef307dc24e8b38d26cf9f9e43`。仅用ZIP范围请求抽取清单选定的成员，没有取得完整ZIP。
- 原始文件名、物种与品种标签、原官方trainval/test划分保存在 `archive-text-1.txt`、`archive-text-2.txt`、`archive-text-3.txt`；本轮实际照片清单在 `pixel-photo-plan.json`。镜像上传者不是这些照片的原摄影者，不能将其替代为摄影署名。
- 官网提供的信息未逐张列摄影作者，本项目保留原始文件名、数据集署名和来源链接，不暗示作者或原摄影者认可实验。
- 原归档 `annotations/README` 的旧声明（研究用途、尊重来源网站条款）原样保存在 `archive-text-0.txt`。当前官网CC许可与旧文字均记录，本轮只作本地非商业科研，不上传或发布数据。
- 修改包括：按trimap类别1提取前景、从类别2区域取背景矩形、缩放、裁剪、alpha边缘处理、合成、固定灰色细边带、拼接接触表。`qa/*.png` 和 `mechanism-transfer-contact-sheet.png` 为这些照片的衍生图，依CC BY-SA 4.0提供并注明上述修改。
- 类别3为未分类边界，未当作确定前景或确定背景；这并不保证人工标注本身完全正确。两个目标标签是猫/狗，背景来源C只是原宠物类别，不是场景真值。

没有把代码仓库许可解释为ForNet/ImageNet原图授权，也没有访问它们的受限图片。本小样本仅用于机制迁移先导。
