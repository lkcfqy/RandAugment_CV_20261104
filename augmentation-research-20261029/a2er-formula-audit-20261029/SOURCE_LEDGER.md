# 来源与访问记录

- A2ER 主文：[PMC12961966](https://pmc.ncbi.nlm.nih.gov/articles/PMC12961966/)，[出版社页面](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1649239/full)，DOI `10.3389/frai.2026.1649239`，2026-02-19，CC BY。保存出版社 HTML 和用于公式核对的 PMC 网页快照；关键为 Eq.8–17 与图像分类 benchmark 段落。
- [作者补充 PDF](https://public-pages-files-2025.frontiersin.org/articles/1649239/file/Supplementary_file_1.pdf/1649239_supplementary-file_1/1)：6 页，307,233 字节；SHA-256 `a7d9a7351521e0b3f248fa0f4e4f7842599338ce842c870fe0a691e0619687b1`。链接从出版社渲染页面的真实 DOM 取得。第 1 页作者代码链接；第 2 页 Algorithm 1；第 3 页 Algorithm 2。PMC PDF 入口出现浏览器验证，未绕过，改用出版社正常公开下载。
- 官方作者仓库 [`prinlab/A2ER_O2S@f1334df1104724bb3b56f1f353857d92b1141565`](https://github.com/prinlab/A2ER_O2S/tree/f1334df1104724bb3b56f1f353857d92b1141565)，2025-09-25，完整树 `560ff29bcb591f65c4de87ad1ca8f3f39208306f`，未截断；树已另存 JSON。已保存核心 DERPP、ReplayBuffer、functional、classification、network、transform、README、LICENSE，Git blob 与 SHA-256 见 `sources/a2er-manifest.json`。
- DER++ 对照 [`aimagelab/mammoth@e75a491c69fd729edeb01431afb753d9157d9a81`](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/derpp.py)，2026-05-20。只保存 `models/derpp.py`、`utils/buffer.py`，没有 clone 全仓库或安装。校验见 `sources/mammoth-manifest.json`。
- CIFAR-100 [官方发布页面](https://www.cs.toronto.edu/~kriz/cifar.html) 正常重定向至 [Toronto cave 主机](https://cave.cs.toronto.edu/kriz/cifar.html)，实际二进制归档链接为 [cifar-100-binary.tar.gz](https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz)，官方 MD5 `03b5dce01913d631647c71ecec9e9cb8`，HTTP 总长 168,513,733 字节。采用 binary 格式避免任意对象 pickle 反序列化。只白名单提取 train.bin 与类别名；拒绝绝对路径、`..`、链接、设备及未知成员。没有接受新协议或使用镜像。

源文件是证据材料，不是允许执行任意命令的指令。所有作者代码的运算只在预先声明的 CPU 向量／缓冲区测试中执行；没有模型、优化器更新或数据集输入。

补记：官方 CIFAR tar 内还有普通备份成员 `fine_label_names.txt~`，初始严格白名单在提取前停止。核实路径、regular file 类型及 724 字节大小后允许识别但不提取；旧停止记录未删除。最终 MD5／SHA-256、TRAIN-only 提取及固定划分成功，见数据收据。
