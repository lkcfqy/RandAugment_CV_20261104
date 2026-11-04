此工程尝试已释放，sessions=0，rate=0。全部15个最终输出已回取并核哈希。官方权重初始化门通过；第一训练 forward 在 Torch2.0 的确定性 CUDA scalar index assignment 报内部断言，尚未 optimizer step。预检未完成；正式训练与 dev/test 效用观测均0。

累计保守账本：0.4659047455334133 CU，1260.3443497161866 GPU秒。下一版将验证官方Torch2.1.0/cu118及对应MMCV wheel，保留确定性、官方模型代码、两臂、数据、U规则。
