# TTA-conformal 独立机制诊断：训练前协议

本候选只做一次 seed=0 的 CIFARResNet20 条件诊断，不追加模型、seed、视图或温度。不将四格分解、普通 rank 改善或温度平滑本身称为创新；不声称语义因果。不 commit/push，不上传历史项目。原论文已有这些普通机制。

## 数据、模型与预算

公开 CIFAR-100 Python 包只在本人 Colab 从官方 `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` 下载；预期 MD5 `eb9058c3a382ffc7106e4002c42a8d85`。来源与许可说明保存于 SOURCES.md，不将数据或论文代码标 MIT。独立编写 CIFAR ResNet20：初始 3x3/16，三阶段各3个两卷积残差块，通道16/32/64，阶段首块 stride2，option-A 空间抽样加零补通道捷径，全局均值和100类线性层。初始化 Kaiming normal conv、BN1/0、linear normal(.01)/0；没有加载第三方 checkpoint 或可执行项目代码。

split seed=20261030。对全60k的原始3072-byte RGB内容做 SHA256，精确重复组原子分配，不看标签；同时包含官方 test 的组只能 heldout。先按官方 train 随机ID顺序首次出现的组、在40k容量内分配训练组；再按剩余20k随机ID顺序的组依次填2k pseudocal、2k validation、6k cal、10k eval。如果任何容量无法精确满足，停止于训练前，保留诊断，不静默拆组。此为标签盲的分组随机划分；有重复时不是独立图像均匀抽样。逐图保存来源ID、内容哈希、角色，公布重复/来源计数和角色间交叉检查。近重复未检测，作为局限。

训练归一化均值/标准差只由40k训练图像估计。训练80epochs、batch128、每epoch313步（最后64），总25040步/3,200,000次呈现。普通CE，SGD lr=.1、momentum=.9、wd=.0005，逐step cosine 至0，不Nesterov。pad4零填充、随机crop32、概率.5水平flip，之后归一化。最终epoch checkpoint，不选best。全FP32，不AMP、不TF32，固定种子；GPU算子启用 cuDNN deterministic，无跨硬件位级可重现保证。

性能门：200次训练updates（25,600呈现，全部来自train，测试weights随后销毁，formal重置seed0）；512个heldout输入、每图10views，共5120测速views，覆盖保存/读回。正式20k*10=200,000 views，总205,120<=220,000。测速含CPU→GPU、同步、增强、训练和完整十视图链路；记录前50及后各50步耗时，使用其中最大每步耗时乘1.5保守预测。已用分配时长+剩余训练/推理保守预测+240秒取回余量必须<=90分钟；否则资源停止，不能缩任务。GPU累计<=120分钟且<=6CU，从首次数据下载到归档<=6h，新增缓存/产物<=2GiB。CLI没有免费档开关，不猜参数；仅请求T4标准档，按实际分配/usage验证，可用免费则0CU，否则限现有CU。收费>3CU/h或账户<6CU则立即释放。结束必须sessions/usage核验。Mac只CPU统计归档，不占MPS。

## 输出、评分与调参

十view固定顺序：pad4后 (0,0),(0,8),(8,0),(8,8),(4,4) crop32，再对应五个水平flip。中心view4等于原图，baseline p=softmax(z4/T)，q=softmax(mean_view(logits)/T)。FP32 logits以(imageID,viewID,classID)缓存并逐块读回比特一致、顺序、均值验收；后续均值/softmax/评分用float64。T仅{.5,.75,1,1.5,2,3,5}。h_ab使用a来源的完整排序与b来源的完整有序概率谱，a,b=0指p、=1指q。混合输出不读标签。原输出并列以稳定classID排序；移植时保存donor排序，不再重排序破坏并列。每臂每T每score独立校准。

所有样本U从seed20261031一次生成，按固定图像ID存储，四臂/温度/评分共用；bootstrap连同样本重采样U。允许空集，不补top1。集合为候选score<=阈值。APS=sum_before+U*v_rank；RAPS按本诊断约定为APS+lambda*max(rank-kreg,0)（正则项不随机，不能声称逐字复现原论文的所有随机化实现）。SAPS为 Huang et al. Eq10：rank1 U*pmax，否则pmax+lambda*(rank-2+U)。rankonly=rank-1+U。k=ceil((n+1)(1-alpha))，用第k次序统计量不插值；k>n为inf。记录零概率、累计分数并列、阈值饱和、空/满集合比例。

调参只用2k pseudocal与2k validation、原输出h00、T=1、alpha=.1。RAPS网格lambda{.001,.01,.1}xkreg{1,5,10}，SAPS lambda{.02,.05,.1,.2,.4,.6}，各选一套并在所有四臂/T/alpha共享。可行约束validation覆盖在[.89,.91]；可行者按M最小、Dhat最小、|coverage-.9|最小、lambda小、kreg小顺序确定。无可行者按距[.89,.91]区间距离最小，再上述顺序选，明确标记调参覆盖不可行，B不能GO。不是论文声称的唯一调参方案，是预先锁定的本诊断方案。

温度选择也仅tuning。对每score，在已选参数下计算所有p/q温度。候选可行要求覆盖[.89,.91]且Dhat<=该score原输出T1的Dhat+.001；按M、Dhat、覆盖距.9、|log T|、T升序选各一个T_p*与T_q*。若无可行点则B不可判GO。所有7个p温度均保留为B比较对象，包括可比性不确定者，甚至明确不可比者：这是保守的更强要求，不靠eval删比较点，也不将其称为穷尽连续温度族的前沿。若任何tuning类n<2，D不可估、B不可判GO。

## 统计与决策（训练前固定）

alpha=.1为唯一主，.05只描述性敏感性且不能救题。M为平均标签数。Rp=M00-M10，Rq=M01-M11，J=Rq-Rp。D=100类均值(c_c-.9)^2；Dhat=均值[(c_hat-.9)^2-c_hat(1-c_hat)/(n_c-1)]，不截负值。任何类n<2均标不可估，任何依赖D的门不升级。逐类Wilson95%区间只描述，不是100个同时保证；每类约100例不能证明1pp精度。

只做999次paired cluster bootstrap，seed20261032，cal与eval独立抽各自精确重复组，组内所有样本共同进入；同一重采样跨四臂/温度/评分。每次重新计算各臂阈值；训练模型固定，tuning选择固定，不能把bootstrap当重复训练。重复组存在时样本数可变，k随该次cal n重算；没有重复时等于普通独立样本bootstrap。若任一重采样类n<2，不删该次、不补抽；D置信区间及B标不可估。

预定检验族：APS alpha.1所有7温度的四臂M、coverage、Dhat，7组Rp/Rq/J；所有49个M00(Tp)-M11(Tq)、Dhat11(Tq)-Dhat00(Tp)、coverage11(Tq)-coverage00(Tp)；7组恢复原rank的优势M00(T_p*)-M01(Tq)。另含RAPS/SAPS/rankonly各自tuning所选q温度的四臂M/coverage/Dhat及Rp/Rq/J。这些构成一个固定维度族，不依结果删点；T_p*来自独立tuning。所有有效维度用bootstrap标准差尺度的 centered max-|t|临界值，95%同时区间=点估计±c95*sd；零sd且bootstrap全等者区间为点，异常零sd则不可估。另用同法c90给局部等效描述。此为近似bootstrap同时区间，不是有限样本精确推断。

A：APS T1 alpha.1 Rp与Rq符号相反，95%同时区间中一项完整>.25另一项完整<-.25。还须所有四臂边际覆盖95%同时下限>=.89，无明显累计/阈值数值饱和；若四臂差异的绝对量有>=50%来自按绝对单样本贡献排序的top1% eval样本，不升级，报告异常集中。该门只支持该模型/数据/协议的rank-spectral条件交互。

B：仅使用tuning选择的APS T_q*，所有7个原输出温度比较都必须size差的95%同时下限>.25，Dhat_q-Dhat_p同时上限<=.001，原输出与q边际覆盖同时下限>=.89，coverage_q-coverage_p同时下限>=-.01；不在eval做配对挑赢家。恢复原rank后相对于tuning选择的T_p*，优势M00(T_p*)-M01(T_q*)的同时上限<=.25，且衰减Rq(T_q*)同时下限>.25。后两个条件保守解释“恢复原排序后实质优势消失且至少减少.25”，避免仅凭点估计说消失。

SAPS/rankonly是预定反证：若APS B成立，但在原输出SAPS或rankonly的tuning选择点已出现与APS-q同样或更小M且覆盖/D可比，则不升级“超出rank或MSP已知机制”的贡献；如比较不确定也不能升级，报告仍需独立证据。RAPS仅稳健性。普通rank改善、小交互、宽区间均关闭本候选；90%同时区间落在[-.25,.25]可称局部等效。任一资源/实现/数据门失败报告为未完成或资源停止，不充当科学零结果。每个结果包含所有失败分支。

## 来源

- TTA-conformal CVPR2025: https://arxiv.org/html/2505.22764v1
- TS4CP/ConfTS: https://arxiv.org/abs/2402.05806
- SAPS公式Eq10及论文调参描述: https://arxiv.org/html/2310.06430v2
- CIFAR数据/引用: https://www.cs.toronto.edu/~kriz/cifar.html
- ResNet架构论文: https://arxiv.org/abs/1512.03385

这些论文用于事实/公式核实，未下载或执行其无许可实现。协议中的更强B门和调参约束均是本诊断预注册设计，不冒充原论文算法细节。
