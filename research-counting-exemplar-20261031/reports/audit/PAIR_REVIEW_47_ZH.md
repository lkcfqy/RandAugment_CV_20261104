# 47 对逐项名称与身份审查（效果前建议；不覆盖原冻结表）

A/B 始终为官方 pos/neg；表内计数严格保留。`可用候选`表示来源和目视相容，不表示已独立逐物体重数。NA 不按模型效果决定、不换图、不补框。

| pair | C 建议 | F A / F B 建议 | 两图官方 A/B 计数（按 selection 顺序） | 状态 | 审查说明 |
|---|---|---|---|---|---|
| FOO_INTRA_BEA1_BEA2 | bean | black bean / soy bean | 231/230；268/290 | 可用候选 | 深色豆与浅色豆的六框分组相容；品种名称沿用作者，未做品种鉴定。 |
| FOO_INTRA_CFC1_CFC2 | coffee candy | brown coffee candy / black coffee candy | 27/33；94/96 | 可用候选 | 两图A均棕色包装咖啡糖、B均黑色包装咖啡糖。 |
| FOO_INTRA_CHI1_CHI2 | chili | long chili / short chili | 42/23；80/38 | 可用候选 | 两图A细长辣椒、B短粗辣椒。 |
| FOO_INTRA_LIM1_LIM2 | citrus fruit | lime / calamansi | 37/94；37/94 | 可用候选 | A较大青柠、B较小柑橘的外观与作者lime/calamansi分组相容；citrus fruit是共享物体类型，未独立鉴定品种。 |
| FOO_INTRA_PAS1_PAS2 | pasta | spiral pasta / penne pasta | 185/189；108/126 | 可用候选 | A螺旋、B管状意面；部分框有遮挡，未替换框。 |
| FOO_INTRA_PEP1_PEP2 | peppercorn | black peppercorn / white peppercorn | 186/183；74/61 | 可用候选 | A深色、B浅色胡椒粒；暗光下颜色偏移沿用作者语义。 |
| FOO_INTRA_PNT1_PNT2 | peanut | peanut with skin / peanut without skin | 183/165；408/361 | 可用候选 | A带红皮、B去皮浅色花生。 |
| FOO_INTRA_SED1_SED2 | seed | pumpkin seed / sunflower seed | 411/377；214/190 | NOT_APPLICABLE_VISUAL_CONFLICT | 首图A1–A3为黑色条纹狭长种子、B1–B3浅色扁种子，和第二图及作者pumpkin/sunflower方向相反；不交换框或计数。 |
| FOO_INTRA_TOM1_TOM2 | tomato | normal tomato / baby tomato | 10/28；19/58 | 可用候选 | A较大圆番茄、B较小椭圆番茄。 |
| FUN_INTRA_CHK1_CHK2 | checker piece | black checker piece / white checker piece | 50/55；40/50 | 可用候选 | A深色、B浅色圆形棋子；checker piece为作者命名。 |
| FUN_INTRA_CHS1_CHS2 | chess piece | black chess piece / white chess piece | 65/50；70/60 | 可用候选 | A黑色、B浅色国际象棋子。 |
| FUN_INTRA_CSC1_CSC2 | chinese playing card | chinese playing card without a red circular mark / chinese playing card with a red circular mark | 59/58；31/31 | 可用候选 | 同类中国条形纸牌；区别是红色圆形标记而非牌面完全没有红色。建议共同名chinese playing card，并显式保留这一定义。 |
| FUN_INTRA_DIC1_DIC2 | dice | green dice / white dice | 22/20；15/9 | 可用候选 | A绿色、B浅色骰子；dice是共享对象名。 |
| FUN_INTRA_LEG1_LEG2 | lego piece | green lego piece / light pink lego piece | 50/49；30/28 | 可用候选 | A绿色、B浅粉色圆形玩具拼搭片；lego piece沿用作者，不声称独立验证品牌。 |
| FUN_INTRA_MAH1_MAH2 | mahjong tile | bamboo mahjong tile / character mahjong tile | 34/35；28/27 | 可用候选 | A条子（含幺鸡）、B万字麻将牌。 |
| FUN_INTRA_MAR1_MAR2 | marble | big marble / small marble | 17/26；39/42 | 可用候选 | 大小玻璃弹珠；两图相对尺寸和分组相容，裁剪放大后的尺寸不可用于比较。 |
| FUN_INTRA_PKC1_PKC2 | poker chip | blue poker chip / white poker chip | 22/35；26/54 | 可用候选 | A蓝色、B白色扑克筹码。 |
| FUN_INTRA_PLC1_PLC2 | playing card | red playing card / black playing card | 38/37；49/48 | NOT_APPLICABLE_VISUAL_CONFLICT | 首图B3明确框到红心Q/相邻红心3，不能代表black playing card；第二图六框方向相容。保留原框，整pair建议NA。 |
| FUN_INTRA_PUZ1_PUZ2 | puzzle piece | edge puzzle piece / center puzzle piece | 31/61；31/61 | 可用候选 | A边缘拼图片、B内部拼图片；复核放大上下文后能看到A的直边，非首次缩略图疑似错误。 |
| HOU_INTRA_BAT1_BAT2 | battery | small AAA battery / big AA battery | 14/12；20/17 | 可用候选 | A细AAA、B粗AA电池；尺寸类别沿用作者并与图像相容。 |
| HOU_INTRA_BBT1_BBT2 | button | button with 4 holes / button with 2 holes | 40/41；31/33 | 可用候选 | A四孔、B两孔纽扣，两个图六框均可辨。 |
| HOU_INTRA_BOT1_BOT2 | bottle cap | beer bottle cap / plastic bottle cap | 80/50；70/40 | 可用候选 | A金属锯齿瓶盖、B塑料瓶盖。 |
| HOU_INTRA_COI1_COI2 | coin | 5 Australian cents coin / 10 Australian cents coin | 70/25；60/20 | NOT_APPLICABLE_IDENTITY_UNRESOLVED | 共享coin明确；多框只能看到模糊头像/暗面，不能逐一确认5分/10分。不是判定作者标错；严格逐框身份准入建议NA。 |
| HOU_INTRA_CTB1_CTB2 | cotton bud | wooden cotton bud / plastic cotton bud | 96/73；67/32 | NOT_APPLICABLE_IDENTITY_UNRESOLVED | 共享cotton bud明确；首图A1白色杆及多框重叠导致木杆/塑料杆分配不能独立确认；不能凭色调猜材料，建议NA。 |
| HOU_INTRA_HCP1_HCP2 | hair clip | black hair clip / brown hair clip | 18/13；32/24 | 可用候选 | 是抓夹/发夹，不是电动hair clipper。候选black/brown hair clip修正名词且保留A/B。 |
| HOU_INTRA_PIL1_PIL2 | pill | white pill / yellow pill | 160/118；186/149 | 可用候选 | A白色、B黄色药片；不推断药物成分。 |
| HOU_INTRA_TPK1_TPK2 | not-applicable | straight plastic toothpick / dental floss pick | 88/75；118/123 | NOT_APPLICABLE_COMMON_CLASS | A直牙签、B牙线棒；dental pick是否真是二者共同对象类型含歧义，不能以oral hygiene tool功能大类掩盖。建议C为NA、整pair不进三模式主比较。 |
| HOU_INTRA_ULT1_ULT2 | plastic utensil | plastic spoon / plastic fork | 48/45；34/27 | 可用候选 | A塑料勺、B塑料叉，部分倾斜或遮挡；复核上下文后未确认方向冲突。plastic utensil是明确共享对象类别。 |
| OFF_INTRA_CRS1_CRS2 | craft stick | red or orange craft stick / blue or purple craft stick | 13/18；25/27 | 可用候选 | A红/橙、B蓝/紫工艺木棒，背景其他颜色不纳入这两目标。 |
| OFF_INTRA_HST1_HST2 | heart sticker | big heart sticker / small heart sticker | 39/55；54/80 | 可用候选 | 大小心形贴饰，保留作者heart sticker名称。 |
| OFF_INTRA_PEN1_PEN2 | pen | pen with cap / pen without cap | 15/13；25/25 | 可用候选 | A有笔帽、B露出笔尖的笔；框可能仅覆盖识别部位，未改为全物体框。 |
| OFF_INTRA_PPC1_PPC2 | paper clip | colored paper clip / silver paper clip | 437/563；382/439 | 可用候选 | A彩色、B银色回形针；局部有邻物遮挡，未改框。 |
| OFF_INTRA_PPN1_PPN2 | push pin | normal push pin / round push pin | 37/52；75/94 | 可用候选 | A柱状握柄、B圆头图钉；候选normal/round沿用作者。 |
| OFF_INTRA_RHS1_RHS2 | rhinestone | round rhinestone / star rhinestone | 72/92；120/112 | 可用候选 | A圆形、B星形水钻贴饰。 |
| OFF_INTRA_RUB1_RUB2 | rubber band | yellow rubber band / blue rubber band | 164/144；205/181 | 可用候选 | A浅黄、B蓝色橡皮筋；重叠和白平衡使个别圈不清楚，整体六框与分组相容。 |
| OFF_INTRA_SFP1_SFP2 | safety pin | big safety pin / small safety pin | 48/47；30/35 | 可用候选 | 大小安全别针；原图相对尺寸支持分组，裁剪图各自放大不表示等大。 |
| OFF_INTRA_STN1_STN2 | sticky note | dark green sticky note / light green sticky note | 32/28；36/35 | 可用候选 | A更浓绿色、B浅绿色便签；dark/light指作者两种颜色，不按单个像素亮度重标。 |
| OTR_INTRA_BEA1_BEA2 | bead | blue or purple bead / orange or pink bead | 155/130；155/130 | 可用候选 | A蓝/紫、B橙/粉珠子；加bead名词，不能按旧文件名green/pink改回。 |
| OTR_INTRA_BOL1_BOL2 | bolt | hex head bolt / mushroom head bolt | 30/55；45/70 | 可用候选 | A六角头、B圆顶头螺栓。 |
| OTR_INTRA_BUT1_BUT2 | button | clear button / opaque white or beige button | 150/240；130/220 | NOT_APPLICABLE_VISUAL_CONFLICT | 两图A全部透明、B全部不透明浅色，与Beige/Clear方向相反；来源链无本地换位，但计数的真实类别绑定未独立解决，建议NA。 |
| OTR_INTRA_IKE1_IKE2 | bag clip | green bag clip / red bag clip | 120/120；80/90 | 可用候选 | A绿、B红塑料封口夹；共同名bag clip较ikea clip更明确是物体类型，F同步去品牌命名。 |
| OTR_INTRA_NAI1_NAI2 | nail | common nail / concrete nail | 200/30；200/30 | 可用候选 | A细长普通钉、B较粗混凝土钉，与作者类别相容。 |
| OTR_INTRA_NUT1_NUT2 | nut | hex nut / square nut | 40/65；30/55 | 可用候选 | A六角、B方形螺母。 |
| OTR_INTRA_PEG1_PEG2 | clothes peg | grey clothes peg / white clothes peg | 168/168；155/145 | 可用候选 | A灰、B白晾衣夹；共同名clothes peg比泛peg明确，F同步补充clothes。 |
| OTR_INTRA_SCR1_SCR2 | screw | long silver concrete screw / short bronze screw | 20/15；48/48 | 可用候选 | A长银、B短铜色螺钉；材质/用途只沿用作者，不独立验证合金与用途。 |
| OTR_INTRA_STO1_STO2 | decorative stone | red decorative stone / yellow decorative stone | 6/8；12/12 | 可用候选 | 红/黄装饰石（透明玻璃质外观）；共同名decorative stone，yellowstone拆为yellow stone，不指地名。 |
| OTR_INTRA_WAS1_WAS2 | washer | metal washer / nylon washer | 100/100；90/85 | 可用候选 | A金属、B白色尼龙垫圈；材质沿用作者，外观相容。 |
