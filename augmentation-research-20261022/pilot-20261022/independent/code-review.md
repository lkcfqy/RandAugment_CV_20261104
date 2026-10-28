# 四格 LoRA × RA trainer 独立静态审查

**版本说明：下方首轮审查保留为历史快照；当前状态以本文末尾“更新源码复审”中的逐项关闭/剩余项为准。**

2026-10-22 UTC。**当前结论：核心 Q/V-LoRA 与错误率计算在静态上成立；下列必须修复项关闭、最终 QA/协议冻结及实际 dummy 资源门通过前，不应开始科学拟合。** 本轮仅读取源码和本机已安装 PyTorch 的相关实现，用标准库 AST 解析检查语法；没有 import 或执行被审代码，没有启动 GPU、训练、推理或下载。没有给父线程另发消息。

## 审查快照

目录：`/Users/lkc/Documents/Codex/2026-10-22/task-2/lora-ra-pilot-20261022`。

| 源码 | SHA256 | 语法静态解析 |
| --- | --- | --- |
| common.py | `c4b0f7cee8ebd5b468b13c8d3e89a9f8876c09459a7b706c7231f10890629470` | 通过 |
| train.py | `2ada412d464353dc20e5655f692466b9e5c3c241992089dbc0d6a5c3c24a08b2` | 通过 |
| preflight.py | `a59b5b310297a8091638e3755c983ce1f7eb8b26fbcc2eac2a8194d794987dac` | 通过 |
| qa.py | `b691a0fe79a4f2c4c6df1f4a0dfff4bc1ecba5a64191dfac7a3e82f97f04a2d5` | 通过 |

这是所列版本的审查，不是任何后续同名代码的自动回执。正式协议尚未提供，本审查不能替代最终源码/协议逐字段一致性核验。

## 已通过的静态检查

1. **fused QKV 的顺序及 K 零增量成立。** `common.py:59–61` 返回 `w + cat(Bq@Aq, zeros_like(Bq@Aq), Bv@Av)`。本机安装的 `torch/nn/functional.py` 的 `_in_projection_packed` 明确按Q、K、V沿dim0存放权重。因此中间768行增量为零；不是将低秩更新意外混入K。基础in_proj_bias也被冻结。此结论基于代码结构，不是一次训练实测。
2. **冻结次序正确。** `build_model` 先严格加载1000类预训练权重，再替换37类head、冻结已有全部参数，然后注册带默认可训练Parameter的Q/V parametrization，最后只开放heads。基础投影、patch embedding、class token、位置嵌入、LayerNorm和bias均保持冻结。optimizer只收集requires_grad参数，未看到对底座参数的更新路径。
3. **参数量正确。** 每block四个低秩矩阵共 `4×768×4=12,288` 参数，3个block为36,864；head为 `768×37+37=28,453`，总计65,317，且代码有运行断言。实际名单仍须由运行回执保存核对。
4. **训练梯度没有被显式截断。** 训练调用 `model(x)` 位于正常autograd上下文，无detach或no_grad包裹。前层LoRA的梯度可经过后续冻结层回传。B初始零导致第一步A的任务梯度为零是正常现象；不能要求每个参数在第一步就有非零梯度。eval的no_grad只覆盖评价函数。
5. **初始处理在设计上可配对。** 四格加载同一base，head用同一seed，A按adapter slot而非层号初始化，B全零；前后布局按slot可有相同初始A。`train.py:57–60` 对全开发集初始logits进行一致性断言。它不是底座哈希检查的替代，但可排除明显初始预测差异。
6. **图像轨迹有合理独立键。** weak/RA分别由seed、epoch、source ID形成键，batch顺序用独立NumPy RNG；RA前先调用相同weak处理。每epoch存有真实增强RGB序列摘要，结束时核对前后布局对应增强一致。RGB后同一个确定性tensor/normalize函数没有布局分支，所以当前代码路径可保持normalized输入一致；如以后更换tensor后处理，应直接哈希实际输入tensor。
7. **AdamW及基本计步正确。** adapter LR3e−4、head LR1e−3、weight decay .01为当前硬编码共同配方，没有按结果调参。当前实现是直接batch16更新，**没有microbatch/梯度累积**。每batch先设LR、zero_grad、平均CE、backward、step。尾batch的epoch loss按样本数加权，dev的错误率为source-photo micro均值。不能把这段实现描述成microbatch2/logical16。
8. **原始开发输出足以重新计算主要metrics。** eval保存FP32 logits、labels、source_ids、predictions和固定前8图的features；NLL用自然对数和稳定平移。argmax按NumPy默认取首个最大值。`error`是主micro，`balanced_error`可作为次要macro；不会因为字段名不同自动冲突。正式汇总必须明确取`error`。

## 科学拟合前必须修复

### R1：增强函数的CPU RNG隔离实际会重置MPS RNG

`common.py:32–45` 在 `torch.random.fork_rng(devices=[])` 内调用 `torch.manual_seed(...)`。本机torch的 `random.py:49–83` 实现表明manual_seed会调用 `torch.mps.manual_seed`；而devices=[]的fork只保存/恢复CPU状态，不保存MPS状态。因此目前不能声称模型设备RNG与增强互不影响。

建议只对CPU default generator设seed，例如在已有CPU fork作用域中使用 `torch.random.default_generator.manual_seed(...)`，或临时设置CPU RNG state，避免全设备manual_seed。预注册模型初始化的全设备seed是另一回事，不必混为一谈。

本机ViT默认dropout及attention_dropout均0，所以没有证据说此问题已导致当前数值偏差；本轮也尚无科学结果。尽管如此，拟合前应修复并保存CPU RNG恢复、MPS RNG未被增强改变的隔离回执。任何需要设备状态的检查由主线在已授权的单串行资源检查中执行，本独立审查不运行。

### R2：dummy preflight未执行1/3节奏；评价阶段没有资源停止门

`preflight.py:26–31` 连续执行5个完整AdamW更新但没有调用pace，违反本轮约max1/3活跃计算节奏约束。应记录原始active耗时后进行同样的节奏等待；墙钟和active分开保存。不因它是资源校准便取消8A优先约束。

`train.py:85–86` 只在训练batch后检查墙钟与STOP。初始/中途/最终evaluate、四格之间以及模型构建前没有检查；末次评价也可能越过cap后仍写`all_four_cells_completed`。应统一guard，在每个训练/评价batch及阶段边界检查STOP和总墙钟，完成标记前再检查。preflight也应有明确短cap和停止检查。

建议异常时保存结构化`incomplete`回执（原因、最后完成cell/epoch/update、elapsed、已有文件），同时保留曲线与原始输出。不得将不完整四格拼成最终I或自动重训。

### R3：读取像素前显式阻止确认集与非法manifest

`train.py:43–47` 对manifest全部rows读像素，只在计训练/开发rows时过滤partition，随后完成回执硬编码`reserved_confirmation_read=False`。如果manifest意外包含reserved行，该声明会不真实。

在任何图片open前断言partition集合严格为train/development，source IDs唯一，ID与class label映射合法，各集非空且37类齐全；核对与锁定candidate/最终split的ID关系，且不存在旧177 ID。不能仅凭manifest自带标签决定某保留ID已经变成development。最终manifest需含QA预定duplicate cluster ID，并验证train/dev不共享已知簇。随后才可根据实际检查记录“未读确认集”，而非硬编码。

`qa.py`会输出exact groups、dHash候选和EXIF issues，但目前没有自动完成manual adjudication/最终分割。主线需在fit-manifest与QA decision中关闭跨split重复及非默认方向；不能把“已列出候选”当作“近重复已解决”。原图/trimap的方向必须一致处理，不能只转一边。

### R4：关键运行设置必须与最终协议一致并提前验收

当前train读取seed、epochs、batch、cell_order、evaluation_epochs等，但增强、optimizer、LoRA rank/位置/scaling仍由common硬编码；最终protocol若写另一个LR或pool，哈希锁本身不会发现语义冲突。

在开始模型构建/任务拟合前验证：四格集合恰好一次、无未知layout/aug；正整数batch/epoch；final epoch在evaluation_epochs且无未来epoch；单seed；micro endpoint；实际optimizer组参数和关键augmentation设置与协议一致。可以改成统一读取冻结配置，也可以将硬编码明确写入协议并作断言。至少保存optimizer.defaults、实际param-group LR/weight_decay、torch/torchvision/Pillow版本及实际trainable名单。

`build_model`现将所有非front值当back，`augmented_pil`现将所有非ra值当weak；应拒绝未知值，不等完成后找不到输出目录才发现。`train.py:94/107`也依赖最终epoch被评价，需提前断言，而非末尾KeyError。

### R5：能力证据不能仅靠任一adapter参数改变

`train.py:100–101` 接受任一非head参数改变；但AdamW的weight decay会改变A，即使任务未学得有效BA更新。因此这条断言不足以证明LoRA的有效表示增量已学到；全局gradient_absmax也可能只来自head。

拟合过程中应保存每个注入block的Q/V参数梯度存在性/有限性（A初始零任务梯度允许）、Bq/Bv更新和有效 `Bq@Aq`、`Bv@Av` 范数，并单列head变化。完成后要求实际有效Q/V增量与固定特征变化可核验，而不是仅A因weight decay变化。保留初始及最终小型trainable state（或可完全复现且哈希的初始化），加冻结底座前后内容哈希/等价逐参数不变回执。静态上底座冻结合理，但现有产物尚没有直接不变证据。

原始logits应在写metrics前检查全部finite，source ID和label在四格完全一致。每格保存实际已完成update数，不仅用`epochs*steps_per_epoch`推算；记录至少逐逻辑step的LR/loss/step或足以完整复核的更新表。每个已向用户显示的dev epoch都保存原始输出，当前设计可以保留。

## 可接受、但必须清楚报告的限制与小改进

- **单LR可作配方pilot。** 当前3e−4/1e−3是共同固定配方，不是各布局调优后比较。后续学习率或seed不能由I的符号挑选。pilot小I/反向不能直接否定所有充分优化后的交互。
- **30epochs可在结果前统一决定。** 目前无任务拟合，若dummy测得完整矩阵加评价/IO/节奏可落预算，则统一30epochs是允许的预结果设计修订；须在lock前写明。592训练、batch16时每epoch37次更新，30epochs每格1110次、四格4440次，源照片呈现71,040次。主终点固定最后30epoch，不能四格分别取最佳epoch。这个修订不使30epochs自动成为收敛证明。
- **preflight是实际照片+人工标签。** 它不是合成输入benchmark；它没有读真实breed目标拟合或报告任务准确率，且丢弃模型，因此可作为资源检查。应如实记录这一点。所读代码只decode592张训练图，train会保留1184张；RSS推断不能把前者等同后者。预测应按最终QA的N和batch计算，留出初始及周期评价、CPU增强/加载、序列化和节奏等待的时间。当前估计明示排除这些，不能直接据它承诺完整2小时。
- **LR尾部的轻微非零可接受。** 当前warmup/cosine按step0…T−1定义，末步略大于零，warmup后有一个重复峰值；这是可锁定的具体schedule，不是严重数学错误。不要在拟合后为了结果变更它。无梯度裁剪也可接受，但不要报告采用了未实现的clip。
- **约10%不是已测峰值。** `set_per_process_memory_fraction(.1)`限制的是MPS推荐工作集比例，不能直接说已证明占物理64GB的10%。`MPS_driver_bytes_at_finish`只是结束时占用，不是峰值；建议每步/评价采样并保存最大观察值，称max observed。memory cap的实际OOM仍属能力失败，不能自动放宽。batch16能否在限制内运行待dummy实际验证，本静态审查不能给资源通过回执。
- **节奏比例是调度约束。** pace在测量段后等待2倍active，有利于保守降节奏，但不等于实际GPU利用率实测33%。CPU预处理、上传、模型构建与系统竞争应另报，不做利用率保证。长阶段必须可中断。
- **dHash是候选筛查。** distance≤4不保证排除裁切、镜像、不同拍摄的同一动物或预训练重叠。视觉review37张预定train示例不等于1184张品种标签逐一认证；应明确source label保留、有限visibility抽样。`all_dimensions_and_label_values_valid`最好改名为trimap values/dimensions，避免被读成语义标签已全部验证。
- **视觉版面尚待独立看图。** qa代码当前生成4页class visibility及可能的重复候选页；父线程提到的两页固定视觉审查尚未提供，不可提前记为通过。它们只用于预先定义的可辨识性/重复裁决，不可用模型置信度筛出有利图。
- **评价函数与完整forward等价需静态保持。** 当前features为process_input→加class token→encoder→取CLS，随后heads，符合torchvision ViT forward。保存前8图初末features有用，但它只是一小组实例，不是一般机制证据。

## 最终统计及独立复算接口

主量继续锁为source-photo micro错误率：

`I=(E_front_weak−E_front_RA)−(E_back_weak−E_back_RA)`。

macro/balanced_error、NLL与特征变化均为次要指标。若QA没删图，每类仍16张，micro与macro数值一致；否则不因二者有利程度更换主量。单seed的1个百分点只作开发投入门；配对photo/cluster区间（若锁定使用）不能估计训练随机性。

当前NPZ的logits/labels/source_ids能完成独立错误率、NLL、两个G、I及逐图d_i复算；还需最终manifest的cluster IDs、模型输出的固定类别顺序、final epoch/实际updates、protocol与checkpoint SHA和锁时间。区间若使用，算法/seed/次数/单位/分位方法须在结果前锁定并保存全部重复。后续独立程序只用CPU1线程读取允许的开发输出，不重训、不另开GPU，也不读取确认像素或旧封存数组。

**剩余阻塞不是权限拒绝：代码按上述项待修、正式protocol/最终manifest和QA/实际资源回执待提供。** 没有遭遇新的自动审批拒绝、403或验证页；没有尝试历史受阻目标。原约束与Sol/max/Fast请求背景保持，实际后端档位未验证。

## 更新源码复审：包括 analyze.py 与 prepare_protocol.py

在未开始科学拟合、无任务结果的同一轮内，已复读更新后的train/preflight/qa、`qa-intent.json`，并读实际文件名为 **`analyze.py`** 的分析程序与`prepare_protocol.py`。先前请求中的`analysis.py`不存在；本轮通过本地文件列表定位到analyze.py，没有访问受限目标或重试拒绝。仍仅静态解析，未执行任何被审程序。

| 最新复读输入 | SHA256 |
| --- | --- |
| common.py（未改） | `c4b0f7cee8ebd5b468b13c8d3e89a9f8876c09459a7b706c7231f10890629470` |
| train.py | `6478d312612ca2cd734ab39b66e4cb3b3e78306f8be036d92dde30125182d004` |
| preflight.py | `4942be12f5b0804b032686ee4d3bc6a375fe7589adcf62e792b60d0a9a064301` |
| qa.py | `2b161c3030cadf4fb045eabf7c530f21869bddfcfd7770f35f2b0393cbc09d8d` |
| analyze.py | `3e2305dbcd3c8fd4e37b5e1442bd46642fe329385712435968b875a2c42fae49` |
| prepare_protocol.py | `0ea19628b12531d024f8d6d7f0c472c6686dc9e18d40a39989e7d5aee1f6df26` |
| qa-intent.json | `022b3ded9c20e9658b147c8e53408e4d0afc54e8b2193e156dab53b34ed740f0` |
| old-id-exclusion-check.json | `0d65f65dd31ef6eeb528a93dd1265933ba6e2f378c3b101c81b85fdb5883c728` |

### 已关闭或部分关闭的意见

- **R2中preflight节奏已关闭。** 新版每步记录active后sleep两倍active，并保存max observed driver bytes，符合拟定调度约束；短cap/停止检查仍可补，最关键的train/evaluate guard尚未补。
- **R5的冻结、初始化和日志证据大部已关闭。** 新版保存冻结参数前后SHA并assert不变、同slot/head初始张量hash、初始小checkpoint、实际逐step LR/loss/样本数及全部trainable梯度存在/finite断言。它还保存真实RGB、normalized FP32 tensor和source-order的逐epoch哈希，证据比首轮更完整。head变化单独断言也已补。剩下的是有效BA增量，而非A decay的证据；最终CPU审计也可直接从初末小checkpoint计算每层的Bq/Bv及BA，若保存材料完整，不必为此增加训练矩阵。
- **旧ID核对已有回执。** 已读取仅文件名检查的回执：177旧ID、1776候选与之零重叠、无确认像素、旧ID列表SHA。它关闭“只有一句排除声明”的资料缺口；本独立分工没有重新遍历旧图片目录或读取旧图。训练前仍需将回执/列表关系锁入最终manifest或协议证据，并限制manifest只能来自候选train/dev ID。
- **QA范围清楚。** `qa-intent.json`在像素QA/拟合前固定：exact/dHash候选按保守连通簇处理，确认或不确定簇的全体train/dev成员都删除、不替补；不是根据模型表现选留。也明确几何可见前景只是proxy，光度/细粒度线索仍可能损伤。若所有已知重复/不确定簇确已移除，后续photo bootstrap而非cluster bootstrap是可接受的固定池描述；动物身份仍不保证。

### analyze.py：主量与5000次分层配对bootstrap正确

四列顺序为front-weak/front-ra/back-weak/back-ra，contrast `[1,-1,-1,1]`；`d=e@contrast`、I=mean(d)、两个G和NLL次量的符号均一致。各breed内固定原观察数量、有放回抽样，逐类累加样本d之和/总N；**这仍是micro权重**，不是将37类的均值等权平均。即使删除重复后每类N不同也与预定micro主量相符。

`default_rng(seed73102)`、5000次、按class0…36依次生成draws、linear方法的2.5/97.5分位是可完全独立重现的算法。bootstrap结果数组完整保存。文案明确“固定四模型及固定breed计数下的source-photo抽样敏感性”，没有将CI用作工程门，`engineering_gate_passed`仅I>=.01；单seed、额外seed0以及两种RA是否各自正收益均有报告。静态统计实现通过，不需要换指标或新增检验。

拟合前还应增加轻量边界断言：四格IDs/labels不仅彼此一致，也与**锁定fit-manifest开发部分**完全一致；IDs唯一、label为合法0…36整数、每breed非空、logits形状与预期N一致、已有prediction与argmax一致（如要验收保存的prediction字段）。当前代码只做四格之间的对齐；独立审计会另外核对manifest。分析程序最好同时验证锁中的自身SHA；至少最终独立审计必须验证被执行分析版本确为预结果锁版本。

### prepare_protocol.py：预结果资源分支可接受，不能当实际完成保证

脚本已把唯一问题、micro主量、固定共同LR、最后epoch端点、无LR/seed搜索、1pp工程门、5000次bootstrap及不含seed不确定性的边界写入。当前其hardcoded optimizer/augmentation与common相符；因此首轮R4不是“已发现LR语义不一致”，而是要求最终冻结版本持续保持一致并提前拒绝非法配置。

根据人工标签资源测量在30/10/不启动之间统一选择，是事前已明示的资源分支，不依赖任务I；**可接受**。`predicted×1.25`中的25%是预算裕量假设，preflight排除了评价/CPU增强/IO；这个分支不证明实际总墙钟必小于2小时，仍需R2的运行中guard。采用30或10都必须四格一致，最后epoch固定。不会因为30比旧草案10长便构成事后改写：前提是正式锁与实际拟合开始时序确实如此，最终将独立核对。

锁定脚本会将源码、QA、资源回执和本审查的SHA存入协议，具有可核验性。应在生成lock前同时验收：final manifest的ID/partitions/label/group约束、QA人工裁决完成、preflight为success且所有耗时/内存值finite、源权重文件实际SHA与receipt一致。脚本当前主要凭文件存在及status字段读取；主线手动验收也可补证据，但不能把“文件存在”当作内容已通过。

### 更新QA实现的专项边界

新版对每张训练图追踪epoch0/固定seed的RA操作，同步NEAREST运输离散trimap，并断言观察器不改变真实RGB；对于已定义的“几何可见前景proxy”是合理实现。它不是bilinear像素信息的精确归因，也不应改称品种语义保持证书。`weak_original_foreground_retained_fraction`是原尺度crop内的mask1比例；`RA_to_weak_foreground_pixel_count_ratio`是重采样后像素计数比，尺度/离散化会影响它，两者不能混同为同一种语义量。只覆盖epoch0，不能推断其余29个epoch均无损伤。

qa新增的本地source metadata与filename breed-prefix校验有助于确定类序及标签来源一致性；它仍不验证每张照片的人类真实品种。人工视觉页及最终删除决定尚未提供，所以这里不提前判为通过。`qa.py:11`复制了R1的全设备manual_seed问题，修R1时应一起改为CPU-only seed；不需要启动新训练。

### 当前仍待关闭的最小清单

1. **R1：common与qa增强/重放作用域改为CPU-only RNG设seed**，不重置MPS RNG；当前common尚未改。
2. **R2：train的evaluate及阶段边界加入统一墙钟/STOP检查**，完成标记前再验cap；preflight已paced，不再要求重复补同一项。
3. **R3：像素open前对最终manifest做明确train/dev限定、候选ID与旧ID排除关系、唯一ID、标签与重复组验收**，并由实际断言支持“未读确认集”。
4. **R4：非法layout/aug与终点评价计划提前拒绝；冻结协议/实际关键设置一致。** 当前候选prepare内容与common一致，但最终protocol尚未产生。
5. **R5：最终有效Q/V BA与头/特征更新的能力证据可核验。** 冻结SHA、初始化、小checkpoint、逐step记录和tensor哈希已补，不再将它们列为缺失。BA可以由独立CPU审计从保存state直接复算；若未通过能力门，不能写科学成功。

最后两个非代码门仍是正式QA视觉裁决及dummy实际资源回执。其余原始静态优点和限制保持。没有实际任务训练结果，也没有进行独立数值复算，故目前不是最终GO/结果回执。

### ZIP range传输边界补充核对

已读本地`data/range-transfer-scope.json`，SHA256为`cbbb3dbd51b85dd5274b0dbebe7ba8ab0002f6ecf9bf683b96074170aee89ac2`；`prepare_protocol.py`只针对该边界的最新表述已查看，当前SHA为`f3f8866cc4eb1df2ad526800206446df85978bbddf90a95e539e8b071e448b73`（替代上表该脚本的旧摘要）。未重取网络范围，也未检查丢弃的尾部。

回执明确：仅1184个train/dev JPEG及对应trimap被作为成员提取、CRC校验、存储和解码；每次range为未知ZIP local extra field预留最多512字节，临时buffer可能触及相邻归档字节，超出部分不解析并丢弃。因此**只能声称未按保留成员名请求、未提取/存储/解码/评分保留成员，不能声称传输层绝无任意保留成员字节**。这种未解析的传输尾部不构成已查看确认图片或以其结果进行科学选择；当前没有理由据此宣称科学封存失效。

候选表原本公开包含reserved ID及class metadata，且用于事前分层分配；不能进一步扩写成“保留标签元数据从未获知”。准确范围是保留图片/输出没有进入拟合、指标或结果选择。建议把该scope回执及旧ID检查摘要一并加入protocol evidence SHA，保留传输边界原文；不为补证据去重新请求或解码任何尾部。此项是透明范围修订，不是新的权限拒绝或要求解封。
