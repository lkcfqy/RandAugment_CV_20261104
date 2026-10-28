# 独立部分审计：按原失败门结束

**最终判定：诊断不完整，停止；不是关于优化混杂或增强机制的科学阴性。** 原冻结 full verifier 未通过，其失败和原始脚本保留。另行编写的 CPU 部分审计完成了可得数字的复算，但不将缺失 case、探针或 completed.json 补成完成，也不放宽任何阈值。

## 实际可核对的范围

冻结诊断 protocol SHA256 为 `29ec9ab66cfd8ef32ba23f2e3fd7095bd0cccfe75f221269acf3667c518b8f91`。保存的冻结时间16:28:51、开始16:29:09、失败16:29:21 UTC顺序一致。新代码及原 common 的锁哈希一致；输入为原训练白名单中预定96张，未读开发准确率、封存或测试资料。运行回执为 CPU1、MPS、nice10；本次独立审计只在 CPU 读取保存的张量和数组，未建模、前向、反向、调用 optimizer 或访问 GPU。

原 `verify_independent.py`（SHA `0f50586e5a96c4c3fa7878b9183140af18f8f089644bce9634ac0837f7a0960e`）仅运行一次，完成2746项检查后因缺少 `results/completed.json` 退出1；`independent-verification.json`、stdout/stderr、execution receipt原样保留。它没有通过完整审计。

单独的 `verify_partial_independent.py` 读取4个 front 完成收据、5组原始梯度/更新后状态（front0–3、back0），未读取主线 post-failure `partial-analysis.json`。938项事件中，artifact consistency 失败0项，保留 protocol gate 失败1项；脚本退出2明确表示部分复算完成、诊断仍不完整。

## 保存状态与失败原因

两布局的初始 trainable tensors 按slot逐项相同、各65,317参数，并与原pilot相应初始小checkpoint完全相同。所有初始B为0。两布局的32张训练探针初始CLS逐元素完全相同，两次重复前向差异RMS均为0。

5个可得单步中，共30个A数据梯度张量精确为0，30个B梯度张量有限且非零。全部70项参数从已保存梯度按首个新AdamW公式复算，均满足原 `atol=2e-10, rtol=2e-6`；最大绝对误差约2.525e-9，判据包含相对项。原warmup LR仍是adapter3e-4/111、head1e-3/111，WD=.01、eps1e-8。back0的oracle是本次CPU从原始状态新增的审计结果；原runner在head门已停止，没有产生该case的完成收据。

back0与front0的head梯度配对，仍用原 `atol=2e-8, rtol=2e-5`，以front作参考：

| 张量 | 最大绝对差 | 相对L2差 | 超阈值元素 | 原门 |
|---|---:|---:|---:|---|
| head weight | 2.6822090149e-7 | 1.0038273122e-6 | 142/28416 | 失败 |
| head bias | 7.4505805969e-9 | 7.6158718921e-8 | 0/37 | 通过 |

保存的训练批0 CE为front3.7156782150268555、back3.7156777381896973，差back−front=−4.76837158203125e-7。这里核对的是已保存损失和梯度，未重新计算损失；没有梯度批的中间激活或logits，不能定位差异发生位置。初始clean探针相等不证明梯度批的训练前向/反向数值也相等。

FP32归约/可微图或内核路径差异是合理待查解释，**尚未被这些材料确证**，不能称已定位MPS错误。PyTorch官方说明数学等价的浮点计算不保证位级一致，但它没有解释本次具体差异。[PyTorch 2.14 Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。原停止门仍然失败，不因总体相对误差小而改判。

## 仅front可得的函数变化

CPU FP64聚合保存的FP32特征，四个front case跨32张探针的CLS变化RMS为：

| 梯度批 | pooled S |
|---|---:|
| 0 | 1.016576595971e-6 |
| 1 | 9.753811939849e-7 |
| 2 | 9.549913392539e-7 |
| 3 | 1.000524101734e-6 |

两探针的floor分别为7.227612286442309e-8、7.284614866875927e-8。八个front probe测量均超过预定10×floor，S/floor约12.06–15.33；这是经验分辨门，通过幅度有限，不是统计显著性或严格浮点误差界。front pooled S均值9.868683077358e-7，CV=.0275473，log S sample SD=.0275886，仅描述四固定训练批。

back没有更新后探针，无法计算任何完整front/back S比值、几何均值、方向一致性或位置尺度效应。back的冻结底座结束回执也缺失，只有front前后SHA相等收据。不得根据5个参数步长、相等初始探针或front稳定性替代缺失比较。A零梯度/B非零和AdamW首步均符合已知LoRA代数，不是新贡献。[LoRA §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1)，[AdamW算法](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)。

## 唯一可提的未来控制：共同参考计算图

若用户以后仍选择研究这个归因问题，建议单独冻结一个共同参考图的单步控制：在同一个原始ViT图内同时放置前/后共六处零B适配器，A继续按slot共享；每个预定训练批只做一次共同损失/反向，保存全部六处B梯度及唯一head梯度，再从该共同初始状态对称屏蔽更新，只分别更新前3或后3和相同head，比较同一训练探针。这样把初始head梯度设为同一保存量，不需事后放宽本轮阈值；这是参考图下的另一个估计对象，不能用来证明原图差异仅来自内核。

这个控制沿用LoRA/AdamW已知参数化，不称新算法。可预定仍是4批×16、两批探针×16：4次共同反向、8个隔离的masked单步状态、2次共同初始探针组和8次post-step探针组，资源cap不超过原600秒。共同图会为六处适配器计算梯度，必须如实区分“计算梯度参数102,181”和“每分支实际更新65,317”，不能声称计算图完全复制原pilot；也不自动等同整段训练或匹配函数方向。需新的代码、精度门和协议审查；**本轮不运行，也不自动追加这一控制。**

本轮到此停止。原四格pilot的工程门失败、原严格范数审计失败、其后定义诊断/复算均保留；另外对原pilot独立审计历史23个相关文件按原delivery manifest逐SHA核验一致。该核验不冒称重新校验全部旧payload。JSON输出仅将已缓存NumPy scalar转换成Python scalar，不改变算术或判据。新旧项目冻结代码均未编辑。
