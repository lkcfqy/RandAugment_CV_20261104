# NSegment 有限 CPU 审计固定协议

开始 2026-10-30 17:45:28 UTC；总截止 18:45:28 UTC。扫描最多30分钟；trace最多10分钟；RSS小于2 GiB；GPU0。固定两尺度、128 seeds、四参数，逐seed处理。完整机器协议见protocol.json。

保留global gate/局部choice/field采样顺序，通过transform现有random_state参数固定局部场。通过实际remap的X/Y坐标ramp恢复donor；不用CV32S或float32大图flatID。校验identity、±.49/.51、半像素和相邻float32、边角clip及uint8字节坐标。最多1次fixture修正，保留错误原因，不改参数/追加seed。

每seed记录变化率/全场identity、分量preclip Linf/RMS、float32map实际最大分量偏移、donor最大整数位移、按实际kernelradius划分的内区/边带/四角rate、clip/tie/near-tie。记录版本/build、自动kernel、边界；alpha1不是任意seed恒等定理，相关像素不能作为独立Bernoulli构造CI。

StageB仅在目标1,10有限扫描恒等且正控通过后执行，最多16个合成样本，局部状态记录回放。A无操作；B原操作；C相同消耗与identity末步，匹配dtype、contiguity、新数组。比较global RNG、后续crop/flip、样本顺序、最终image/mask、first divergence。此pipeline不是原TableI配置；不得把工程机制升级为训练贡献。

缺少原config/log/checkpoint身份，止于公开实现审计。不安装MMSeg、不下载数据/权重、不执行远程第三方代码、不训练、不占MPS/Colab、不commit/push/上传；保留历史。
