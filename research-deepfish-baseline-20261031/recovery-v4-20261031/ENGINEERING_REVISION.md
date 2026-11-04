当前修订仅解决运行库兼容问题，科学效用观测仍为0。

上一尝试已确认官方权重900共同张量匹配及7个log_scale零初始化；在第一次forward计算GT权重时，Torch2.0确定性CUDA索引赋值 bbox_weights[pos_inds]=1.0 触发4 vs 1内部断言。与官方issue105819的人工矩阵错误一致。MMEngine在deterministic=True时开启torch.use_deterministic_algorithms(True)。

静态核对官方2.1.0 Indexing.cu已增加sliceSize尾维扩展；2.0.0缺少该分支。选择官方Torch2.1.0+cu118、torchvision0.16.0+cu118，并从OpenMMLab的cu118/torch2.1.0索引取得MMCV2.1.0。mmdet3.3.0、mmengine0.10.7、所有固定模型配置和原始权重不变。该变更是环境兼容修订，不能与失败版本混记为同一次成功运行。

CPU阶段使用原有Python3.10.19、官方来源安装流程、8MiB无敏感标记回取与watchdog；无模型实例、无权重、无数据图片。新增人工900x4矩阵的空/单/多索引标量赋值与compiled NMS门，始终保持确定性；下一次GPU在下载模型之前再执行同一CUDA门。不得通过关闭确定性或修改上游检测源码规避错误。

安全加载维持weights_only=True及TORCH_FORCE_WEIGHTS_ONLY_LOAD=1；静态确认Torch2.1.0支持此强制变量。无unsafe fallback、动态pickle白名单或trust_remote_code。

已完成43项本机非模型测试。当前CPU检查尚未完成，GPU尚未分配；依赖SHA与实际版本将由CPU回执确认后才放行。预算从0.4659047455334133 CU、1260.3443497161866 GPU秒结转；CPU费用也计入2CU。

来源和SHA：sources/TORCH_COMPATIBILITY_SOURCES.json；推理与门：reports/TORCH_COMPATIBILITY.json。
