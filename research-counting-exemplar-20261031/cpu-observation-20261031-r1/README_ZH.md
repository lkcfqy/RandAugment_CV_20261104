CPU后端修订和六次重复性验证完成，正式492查询尚未开始，等待一次差异复核及匹配freeze的GO。

当前有效入口：`/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-observation-20261031-r1`

Freeze：`649a171fb5bd1d659aa43675e072024ad6c5aa7fba75b0efefa70d481517a79b`。20个本地冻结文件、180个已有资产引用；不重新复制模型/数据、不下载安装。7个核心helper字节未改，metrics、strict模型加载、原生token/框契约和科学输入均与已审版本一致。

F/C/V各两次，顺序F1/C1/V1/F2/C2/V2，六次全部进入并返回。两轮原始输出逐元素完全相同：活跃logit与框坐标最大误差均0，原容差 `1e-4 + 1e-3*abs(first CPU)` 未改；14项输入/token/框hash、原生mask及top-k顺序一致，参考clone未变。前后完整注册状态与共享别名hash一致。严格加载1042物理/1108逻辑键和66共享别名通过。负视觉组和V原生token结构由每次真实forward hooks检查。

| 模式 | 第一次完整query | 第二次完整query |
|---|---:|---:|
| F | 6.487069秒 | 6.531374秒 |
| C | 6.525064秒 | 6.527449秒 |
| V | 6.511489秒 | 6.538978秒 |

固定纹理fixture仍为原SHA `311764e9a1f62545d97d904d0ecd377049a19645a87c478a1af68d8b7a15e87b`，输入1×3×800×1333、2/1线程、FP32，模型seed42。未比较CUDA、未改top-k排序或算子，GPU操作、网络连接、pickle、真实图读取、真实query、训练均0。重复性通过只说明该固定CPU配置可重复，不代表比CUDA正确，也不是科学贡献证据。

初始化有一次已保留的实现故障：初版GPU保护把同一个函数对象绑定到多个API，TorchDynamo注册冲突，加载前终止，root forward为0。修订为每API独立closure，独立导入检查通过、GPU操作0。旧初版目录和freeze未改，新r1目录独立claim；未重试任何已执行的模型forward。

原历史545.684807秒，加初版失败2.484607416秒、guard导入检查0.898681334秒及本次外部预检52.099948500秒，累计**601.168044261秒**。本次整个人工验证episode共55.483237250秒，低于180秒。worker与caffeinate均已退出，8个索引证据文件哈希已验证。RSS峰值约3.724GiB，低于8GiB。

唯一有效总hardlimit为**7200秒（累计120分钟）**；90分钟是已被最终资源决定替代的历史草案。原2倍工程成本门保留，不采用1.5倍。正式准入采用完成预检后的累计账本重新计算：

`601.168044261 + 30 + 2×492×6.538978083001 + 60 = 7125.522477934秒 ≤ 7200秒`。

原门通过，余量74.478秒。正式单遍外推3217.177秒（约53.62分钟），含历史、30秒重新加载余量和60秒收尾的名义累计约65.14分钟。实际执行会重新严格加载，并以实际已耗替换加载余量后再核原2倍门；整段到期或缺任一query只能incomplete，不能缩样本或自动重跑。

新入口仅将已审worker派生为CPU分支：删Colab资源与GPU迁移/比较，保留正式492循环、原图和六框身份、原标签GT绑定、.23/800max1333、全47pair含6NA、完整预测框、每query原子落盘及独立Fraction指标复算。launcher采用已用过的Mac ps RSS探测，统一累计deadline涵盖导入/加载/验证/前向/本机收尾。仅一个进程绑定caffeinate，无永久电源设置更改。模型及metric代码未改，未使用adaptive crop/pseudo-exemplars。

[主要差异](reports/CPU_BACKEND_REVIEW.diff) · [初始化修复差异](reports/INITIALIZATION_GUARD_FIX.diff) · [结构化重复性结果](reports/CPU_REPEATABILITY_RESULT.json) · [原始工程证据](precheck/artifacts/engineering/status.json) · [外部wall账本](precheck/controller_report.json) · [42项人工测试](reports/CPU_CODE_TESTS.json)

默认只核freeze：

```sh
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python /Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-observation-20261031-r1/scripts/launch_cpu.py
```

正式入口仍未授权启用。差异复核后，父线程提供`formal_allowed:true`、本freeze SHA、`scope:Mac_CPU_fixed_492`、`max_runs:1`的GO文件，再使用：

```sh
/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-readiness/.venv/bin/python /Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-counting-exemplar-20261031/cpu-observation-20261031-r1/scripts/launch_cpu.py --formal --authorization <matching-GO.json>
```

现有formal授权模板保持false；不会自动进入正式观察。全部原失败和旧freeze继续保留。
