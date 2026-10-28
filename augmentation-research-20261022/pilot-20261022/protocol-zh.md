# 四格开发 pilot：冻结协议导读

本文件是已冻结 `protocol.json` 的中文导读，不是新版本预注册。协议锁于2026-10-22 14:51:37 UTC；SHA256 `9386fc378f876cf12c68ef5d7248f073d7313433d158350b4143942ac1b52a3f`。科学进程始于14:52:39 UTC。正式参数、RNG和代码字节以 `protocol.json`、`protocol-lock.json` 为准。

| 位置 | 弱增强 | 弱增强后增加RA |
|---|---|---|
| 前层 | blocks0–2，Q/V各r4 | 同布局、同初始化、同数据序列 |
| 后层 | blocks9–11，Q/V各r4 | 同布局、同初始化、同数据序列 |

每格可训练65,317参数（LoRA36,864、head28,453）；共同ViT-B/16 ImageNet1K_V1，RGB224、patch16、196patch+CLS，基础参数全部冻结。Q/V缩放1、B零初始化、A按slot配对，K增量严格为零。它是实际表示训练，不是冻结特征读出。

每格30epochs、batch16、37更新/epoch，单个事前种子20261022。AdamW adapter LR3e-4、head LR1e-3、WD.01，10%warmup后cosine，不搜索LR。共4,440更新、70,920次源照片呈现。两布局逐epoch的真实RGB、normalized输入tensor和source次序哈希必须一致。固定顺序front/weak→back/RA→back/weak→front/RA。

数据为原trainval中的新选Oxford-IIIT Pet37类；旧177个已用JPEG ID排除。SHA/CRC/尺寸/标注来源核验后，两对同capture全4个成员删除，不补抽：train591、development589。开发Bombay14张、keeshond15张，其余各16；训练keeshond15，其余各16。确认候选592成员不提取/解码/评分，旧final不解封。ZIP临时range尾部边界见 `data/range-transfer-scope.json`，不能宣称传输字节绝对排除。

弱增强为RRC scale[.8,1]、ratio[.9,1.1]、水平翻转.5；RA为torchvision14操作池N2、M9/31、bilinear、fill[124,116,104]。开发固定resize256/center-crop224及ImageNet归一化。37个预选真实图已查看；3个全背景trimap的照片仍有清晰宠物，保留分类标签并记代理缺失。几何可见性不能证明品种语义完整，强光度操作的线索损伤是解释边界，不按模型分数删图。

唯一主量为干净开发source-photo micro错误率的交互：

`G_front=E_front_weak−E_front_RA`，`G_back=E_back_weak−E_back_RA`，`I=G_front−G_back`。

取固定第30epoch，不取开发最优epoch。四格绝对错误数/率、两个G、I和基线位置差全部报告；macro、NLL、每类表与曲线为次要。1个百分点是既有提案的工程投入尺度，约6个净错误计数单位，不是显著性/power/等效证据。两组RA都损伤时，正I也只能说明损伤程度不同。

区间采用固定37个breed计数内的配对source-photo bootstrap，PCG64 seed73102、5000次、linear的2.5/97.5分位，保存全部重复。只描述四个已训练模型对开发照片抽样的敏感性，不含训练seed变异；未知动物身份/预训练重合未排除。单seed达门只能形成开发信号，不确认稳定性或新贡献。

每个train/eval batch及阶段边界检查STOP与2小时墙钟上限；OOM、非finite、冻结参数改变、缺梯度/有效BA增量、像素不配对或未完成端点，均记测量失败，不能拼合不等长结果。训练CE下降及参数/固定特征改变支持能力检查，不自动等同充分收敛。小/反向I不自动否定所有位置×增强现象；不补seed、单格延长或改指标找阳性。

实际假标签计时：front0.27769s/back0.16259s每步、MPS最高观察driver3.748GB，CPU1、nice10、无fallback；30ep compute+pacing加25%约61分钟。单MPS、内存份额.1、每步等待两倍活跃段，不声称GPU利用率实测1/3。8A优先，无A100/Colab/ARiS/付费/上传/发布。全层r1、全层r4、full FT及备选均未运行，不能据四格宣称优于强基线。

近邻证据沿用主目录：Surgical FT已有位置×移位，Hint-Aug已有细粒度PEFT增强损伤，DynamicAug已有LoRA动态增强，CoCa已有LP/LoRA增强差异；精确四格在已读部分未见不构成穷尽查新。旧阴性与课堂材料不改。请求Sol/max及现有Fast配置保留，实际后端档位回执未暴露。
