# 冻结基础检查点上的门 A：独立探索协议

2026-10-29 UTC，所有网络输出与评分之前冻结。本轮使用刚完成的宽64 BN ResNet18 DER++ 基础训练第6300步、task2最终检查点，不能改选中途检查点。旧760步试验仍是基础学习不足，绝不改写。本轮不是作者A2ER复现；已知的方差恒等式、跨视图残差或种子匹配都不是创新。

## 固定输入和边界

检查点为相邻 `mammoth-derpp-baseline-20261029/run-002/task2-final.pt`。保存整个文件SHA-256，以及每个模型/BN tensor和buffer原图、label、历史logit、TRAIN row、arrival的哈希。只使用该检查点全部500个槽位，不挑类别、样本、难度或唯一source；重复source保持为不同历史观察，同时报告source聚合结果。

模型 `eval()`、所有参数requires_grad=False、inference_mode；不创建optimizer，不更新模型、BN、buffer、历史targets、alpha/beta。测试集和dev均不读取；clean forward只针对buffer，供预定分层和解释分析使用。运行前后检查模型/BN及全部buffer tensor逐位一致，原检查点文件哈希不变。所有新文件在本目录，旧记录不写。

固定唯一base seed=20261029。8+8组使用独立NumPy PCG64 seed `base+800000+100*group+view`，group=0/1，view=0..7；每组全部500槽位各8张普通视图。裁剪/翻转参数先生成存档并哈希：padding4零补边，crop offset0..8均匀，flip Bernoulli(.5)，32×32和原CIFAR100归一化。严格复用真实torchvision PIL操作；不使用RA、Erasing或增强强度搜索。raw float→PIL字节往返检查。clean视图仅归一化。使用现有Mac MPS float32推理，CPU float64分析；不安装、不使用云GPU或付费。

## 缺少Q/权重的明确适配：冻结模型的影子校准

新Mammoth基线只有uniform replay，没有残差Q或N权重；不能把均匀权重重排得到的零TV当机制失败，亦不能冒造原生降权训练轨迹。因此结果前固定一个**辅助、无学习的160 batch影子回放校准**，只生成参考权重多重集。这个模型固定的校准不是原训练轨迹，也不是A2ER原方法。

初始500个w=1、Q=0，160批、每批32。采样为当前w的Plackett–Luce无放回（Gumbel top-k），每批一组独立温和增强。采样uniform随机数(seed base+700000，160×500)与增强参数(seed base+700001，160×32×3)均事前存档。所有batch的槽位、logit、残差、Q前后、w前后和访问次数完整保存，未访问槽位保持w不变。**仅这段影子流允许更新局部w/Q；它不写检查点或真实buffer**。

沿已审计的公式：D=.5*mean_k((h-z)^2)，d=D+1e-6，Q←(1-b/500)Q+(b/500)*median_batch(d)，eta=clip(1-max(d-Q,0)/max(Q,float32_tiny),0,1)，ratio=(eta*d+(1-eta)*Q)/d，w_i←.5w_i+.5sqrt(ratio)。原A2ER用half-SUM且eps=1e-5，本轮10维half-MEAN的eps同步除10；这使ratio、gamma、w保持原单位转换等价。只保留参考采样模块，不加作者的target correction、当步gamma损失缩放或alpha/beta自适应。

160批结束冻结得到N_shadow与Q；后续全buffer评分绝不更新Q。评分两组与校准随机源独立。归一化p=w/sum(w)是基础权重分布，**不是无放回batch边际纳入概率**。影子校准额外5120张推理如实计入成本，不能称免费权重或训练效果。

## 两组评分与固定强度的身份重分配

对每个槽位i的历史z_i与8个h_iv：Davg_i=mean_v[.5mean_k((h_iv-z_i)^2)]；Dmean_i=.5mean_k((mean_v h_iv-z_i)^2)；V_i=.5mean_vk((h_iv-mean_v h_iv)^2)。逐条验证Davg-Dmean=V，float64容差 `1e-10*(1+max(Davg))`；保存全部原始输出、三个量、误差和排名。

预定分层 true class × 观察年龄四段 × clean真类概率是否≥.5。年龄=`200000-1-arrival_index`，边界为50000、100000、150000；不得按结果改分层。N_shadow的完整权重多重集在每层内固定，A按Davg从高到低配从低到高权重，B按Dmean同法；tie按TRAIN row、arrival、slot打破。单例原样保留。校验每层及全局权重多重集、各层概率质量、entropy、ESS一致。可交换层size≥2须覆盖≥80%槽位；不足记录不可识别。保存class-only层和无分层TV仅作混杂描述，不能用它们救主分层失败。

主投入门固定：**两组TV(A,B)均≥.05，跨组Davg和Dmean Spearman均≥.80，跨组(pB-pA)余弦≥.50，加上≥80%可交换覆盖及全部有效性检查**。零向量相似性定义0。与旧门一致；本次检查点/原生buffer/辅助权重来源不同，所以独立解释。记录A-B within-stratum严格排序反转率、top20%低权重身份重叠、槽位及unique-source聚合TV。未通过即关闭本参考机制短分支路线，不改seed、强度、checkpoint或权重校准长度。

## 预定干扰因素诊断

1. 全局logit平移：每个h和z分别去掉类别维均值，再重算Davg/Dmean/TV/稳定性；验证raw Davg=中心化Davg+.5mean_v((mean_k h-mean_k z)^2)。报告平移解释的残差与视图方差比例。
2. 范数：中心化h/z分别归一成L2单位向量（norm<1e-12记零并记录数量），重算同一评分与身份TV；这是不同度量的敏感性诊断，不替代主分析。报告径向范数差、clean/历史/平均logit范数、视图范数SD与V、权重差的Spearman。
3. 固定class/age/confidence层后，使用预定OLS解释(pB-pA)：层one-hot，clean logit范数、历史范数、平均logit范数、视图范数SD、clean真类概率、entropy、logit均值、年龄。报告in-sample R²而非因果解释或泛化；另列只有层项的R²。两组同表，任何控制不因结果删改。

raw门通过仍不自动说明差异超越置信度/平移/范数解释，更不能说明clean性能受损。控制结果如实影响后续协议能否提出有意义的问题。

## 输出与停止规则

原始完整8+8 logit/增强参数、clean输出、逐槽位CSV、影子校准记录、各分层统计、脚本测试、状态不变证明、同步计时与内存、中文结论。CPU合成测试包含恒等式、零残差、half-sum/mean单位转换、匹配多重集；不把合成反例当真实效果。

若主门不通过，仅交付诊断和NO-GO，不运行任何分支。若通过，只拟定N_shadow/A/B/R/U五分支协议草案，保留来源限制与强uniform对照，仍不执行训练。无论结果如何，不动旧测试、旧门槛、旧负结果，不推送、不操作重置卡。预期推理不到5分钟，执行硬限10分钟；超时/数值/来源错误记执行不足，不判机制失败。
