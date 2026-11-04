本机 CPU A/M 配对实验已完整完成。两臂均 seed=20261031、U=200、同一5-shot支持集和64张dev。主比较为CPU训练并CPU评估的M减A。

总运行 40.06 分钟；最高进程峰值RSS 6.878 GiB；新增CU、GPU、MPS、上传、安装均为0。

|指标|A (%)|M (%)|M−A (pp)|maxDets|
|---|---:|---:|---:|---:|
|AP|27.906767|27.490711|-0.416056|100|
|AP50|46.889653|47.974282|+1.084629|1000|
|AP75|27.674652|25.570949|-2.103703|1000|
|AP_small|0.008291|0.121223|+0.112932|1000|
|AP_medium|18.526105|19.290739|+0.764634|1000|
|AP_large|43.350131|40.698236|-2.651895|1000|
|AR100|49.090909|48.556150|-0.534759|100|
|AR300|55.240642|52.887701|-2.352941|300|
|AR1000|55.240642|52.887701|-2.352941|1000|

COCO未定义的尺度指标原值−1（百分数表示−100）保留，不能解释为负的准确率。主AP使用maxDets100；AP50/AP75/尺度AP使用1000，AR100/300/1000分别使用对应上限。所有指标在两臂训练与完整权重安全回读成功后统一计算，并从128条持久化预测独立复算一致。

|实际训练数据使用|A|M|
|---|---:|---:|
|anchor_count|200|200|
|selected_occurrences|200|659|
|visible_occurrences|200|649|
|GT_presentations|175|573|
|empty_GT_inputs|25|13|
|GT_removed|25|81|
|Mosaic_applied|0|109|
|MixUp_applied|0|51|
|same_image_repeated_inputs|0|110|
|selected_same_cached_origin_repeated_inputs|0|21|
|visible_same_cached_origin_repeated_inputs|0|20|
|input_pixels_mean|627256.78|516940.75|

|算子调用次数（不等于概率门生效次数）|A|M|
|---|---:|---:|
|CachedMixUp|0|200|
|CachedMosaic|0|200|
|Color|200|200|
|LoadAnnotations|200|200|
|LoadImageFromFile|200|200|
|PackDetInputs|200|200|
|RandomChoiceResize|294|297|
|RandomCrop|94|97|
|RandomFlip|200|200|
|YOLOXHSVRandomAug|0|200|

同一完整初始state哈希、初始CPU/Python/NumPy随机状态、896个可训练参数张量、AdamW分组、LR、batch和200个anchor顺序全部一致；每张support作为anchor恰好40次。M额外随机算子消费和更多source曝光属于固定策略差异。每臂200次真实数据轨迹与既有冻结数据干跑记录像素/GT/几何一致。

模型与数据：原始官方GroundingDINO Swin-T，FP32 CPU，Torch4线程/inter-op1/OpenCV顺序；不改科学配方和标签。两完整模型、全部轨迹、128条预测、COCO完整precision/recall/score数组、协议和实现hash均保留。

解释范围：这是同一episode/seed、固定更新预算的整套策略探索；不是曝光匹配的纯增强因果效应，不是ETS论文复现，不证明新贡献或跨场景泛化。历史CUDA训练结果仅背景，未用于本次主差值。官方test未用于训练、选择或本次模型评估。

预检选择事前固定为训练轨迹A第2次、M第43/44次，覆盖正GT、大图和满缓存双混合；预检模型/优化器随子进程退出丢弃。正式两臂均独立从原始官方权重初始化。无best-checkpoint或dev选优。

A完整模型：`formal_A/final_model.pth`，692247962字节，SHA256 `08a9c1070b9a34d37b938421ac282b3ddc8131cf0f11eff727dcf19511dffe6b`。
A训练更新累计1092.125秒；训练进程1119.108秒；64张dev前向142.136秒。

M完整模型：`formal_M/final_model.pth`，692247962字节，SHA256 `7af932b6e9c127b7767a5d65306ae89d90bb8af5c55d77bb4d03e886c35c0ba1`。
M训练更新累计916.896秒；训练进程945.123秒；64张dev前向141.940秒。

基础设施修订1：首个dev前向之前，补齐旧预测清单中缺少的几何核对引用。记录全部64张固定原图的既有CPU几何hash，未使用效果选择；训练代码/配置、模型推理函数、COCO计算函数均未改变。旧/新代码hash及5项无模型回归证据保存在`infrastructure-revision1/`。

统计口径：selected_occurrences包含被裁掉至空区域的来源，visible_occurrences仅计最终非空区域。重复同缓存来源分别按被选来源和可见来源统计，因此M为21和20次输入，二者不混用。
