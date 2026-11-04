# 固定源码逐行核对

官方仓库：https://github.com/unique-chan/NSegment 。固定提交 `5c66795b38ffd351f2d3363bc4eb3e5c3b580fac`，加入操作的提交 `0ef667c06284d6f563aaa3f7a86a576fd1c0ae8c`。

两份 `mmsegmentation/mmseg/datasets/transforms/transforms.py` 均为 89,252 bytes，SHA256 均为 `4ca1b6c17ae66f469eafe2628bb6fa6b44930387305715c751e6d346d3b40224`。完整下载及哈希在 `sources/manifest.json`。未导入或执行下载的 Python 文件，也未安装 MMSeg。

以下是人工逐行对照；独立实现为 `audit_core.py`，完整原文带行号在 `sources/NoisySegment-source-excerpt.txt`。MIT 许可证与 Yechan Kim 版权声明保留在 `sources/LICENSE`。

| 官方行 | 官方行为 | 本审计实现及差异 |
|---|---|---|
| 2541–2547 | 对象保存列表和概率；缺省列表为 5×3 笛卡尔积 | 调用方固定传入各 singleton 和 p=1；未测试默认混合列表，这是固定协议范围 |
| 2549–2551 | 总先执行 `np.random.rand()`，大于概率才返回 | `gate()` 89–92 行逐次取全局随机数；p=1 未优化掉；单独记录 applied/skipped |
| 2552–2555 | 取 `gt_seg_map`，调用变换，替换返回值 | 合成 uint8 mask 或坐标 ramp 作为同一 shape 的 segment，保存新 remap 数组；无训练框架字典 |
| 2557–2559 | 未传局部状态时创建 `RandomState(None)` | `make_maps()` 保留 None 分支；审计利用现有注入口传入局部 `RandomState(seed)`；不以 global seed 冒充局部固定 |
| 2560 | 局部 `randint(0,len(list))` 选参数 | 即使 singleton 仍调用 randint；选择前后状态哈希均保存；NumPy 2.0.2 singleton 在本次实测不推进状态，但调用顺序保留 |
| 2561 | 取 segment 前两维 shape | 由调用方从固定二维尺寸或同尺寸 mask 提供 |
| 2562 | 先生成整张 dx，float64 U[-alpha,alpha) | `make_maps()` 105 行，未改变 dtype 或改成已平滑噪声再乘 alpha |
| 2563 | 再生成整张 dy | 106 行，保持 x→y 抽样顺序 |
| 2564–2565 | 依次对 dx、dy GaussianBlur，ksize=(0,0)，sigma，无显式 border | 109–110 行同参数；使用本机 OpenCV 4.11.0 默认边界，并在 fixture 中核对 |
| 2566 | np.arange 整数网格、meshgrid | 112 行保留 int64 整数坐标构造 |
| 2567–2568 | float64 位移相加，clip 后转 float32 map | 113–114 行顺序一致 |
| 2569–2570 | cv2.remap，INTER_NEAREST，返回新数组 | `remap()` 127 行；Stage A 对两个独立 ramp 实际 remap，Stage B 对 uint8 mask 实际 remap |

状态哈希、计时、统计不抽取随机数。C 是预先指定的负对照：生成与 B 完全相同的 gate/choice/field/blur/map，最后将 remap 的 map 替换为整数 identity map，仍通过 cv2.remap 生成相同 dtype、contiguity、所有权的新数组。C 不是官方算子。

OpenCV `smooth.dispatch.cpp` 287–291 行：非 CV_8U 的自动核宽为 `cvRound(sigma*4*2+1)|1`。原操作噪声为 float64，σ=3/5/10 对应 25/41/81。fixture 的 impulse support、显式相同核、默认边界与 BORDER_REFLECT_101 的数值比较均通过。本结论限定本次固定版本/构建，不推测作者版本。

论文 v3/v6 附录给出相同的 GaussianBlur、clip、float32 map 与 INTER_NEAREST 路径。正文描述约 6σ+1 核及零填充；本次实际执行的浮点 OpenCV 路径为约 8σ+1 和反射边界。此处记录描述与本次实现路径的差异，不将其作为原表数值错误的证据。

已读取并提取两个版本的附录 listing（`sources/paper-v3-code.txt` 和 `sources/paper-v6-code.txt`）；两版本 listing 相同。附录选择参数写为 `randint(len(list))`，固定仓库源码写为等价的 `randint(0,len(list))`；审计遵循仓库形式。附录 HTML 的排版引号不是可直接执行的 Python；仅作为文字证据读取。
