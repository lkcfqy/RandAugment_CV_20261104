"""Render the completed saved baseline facts; no training or model inference."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run-002'
s=json.loads((RUN/'summary.json').read_text())
v=json.loads((RUN/'independent-verification.json').read_text())
p=json.loads((ROOT/'process-timing-v2.json').read_text())
old=json.loads((ROOT/'run-001/summary.json').read_text())
assert s['status'].startswith('completed_') and v['all_passed']
t1,t2=s['task_evaluations'];a=t1['old_dev'];b=t2['old_dev'];c=t2['new_dev'];combined=t2['combined_dev']
checks=s['inherited_basic_learning_checks'];passed=checks['task1'] and checks['task2_old_and_new']
verdict='达到原定的基础学习操作门' if passed else '完成训练，但没有达到全部基础学习操作门'
report=f'''# Mammoth固定DER++两任务基础训练结果

2026-10-29 UTC。**单seed基础训练已完整完成6300步，{verdict}。** 这是独立的新基础诊断；旧760步试验依旧是基线学习不足。没有运行门A或五分支，不是增强／采样身份机制贡献证据。

## 固定任务终点

所有指标来自原官方TRAIN中预定的开发划分，每类100张，不使用test；class-incremental 10维head，无task mask。任务1和任务2均训练固定50epoch，没有用中途开发指标挑checkpoint。

| 检查点 | 旧类dev准确率 | 旧类NLL | 新类dev准确率 | 新类NLL |
|---|---:|---:|---:|---:|
| 初始化 | {s['initial_old_dev']['accuracy']:.1%} | {s['initial_old_dev']['nll']:.6f} | 未评估 | 未评估 |
| task1结束，第3150步 | {a['accuracy']:.1%} | {a['nll']:.6f} | 未评估 | 未评估 |
| task2结束，第6300步 | {b['accuracy']:.1%} | {b['nll']:.6f} | {c['accuracy']:.1%} | {c['nll']:.6f} |

task2后合并10类dev：准确率 **{combined['accuracy']:.1%}**，NLL **{combined['nll']:.6f}**。旧类准确率较task1变化{(b['accuracy']-a['accuracy'])*100:+.2f}个百分点，旧类NLL变化{b['nll']-a['nll']:+.6f}。这里只报告单seed开发结果；不能把保持／遗忘的差异归因到尚未比较的采样模块。

沿用旧操作门而未改阈值：task1旧类accuracy≥.50、NLL≤1.5且比初始化降低≥.30；task2旧/新各accuracy≥.35、NLL≤1.8。本次两项分别为 **{checks['task1']}／{checks['task2_old_and_new']}**。这些是弱基础门，达到它们不等于后续科学干预已可识别或贡献成立。

## 已执行的配方和适配

来源固定为 `aimagelab/mammoth@e75a491c69fd729edeb01431afb753d9157d9a81` 的seq-cifar100 DER++ preset：[模型参数](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/config/derpp.yaml)、[50epoch与scheduler](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/datasets/configs/seq-cifar100/xder.yaml)。

- 宽64、四stage各2block、BN ResNet18，CIFAR 3×3 stem、10类head，11,173,962参数、20个BN；无预训练。
- SGD lr=.03，momentum=0、weight_decay=0、nesterov=False；task边界重建optimizer/scheduler；每task第35/45epoch后乘.2。
- 当前batch32，末batch16；独立蒸馏/监督replay各32。buffer500，standard uniform/reservoir，从task1第二个batch就使用回放，允许同一source的不同时期观察占多个槽位。
- 当前CE + .3 mean logit MSE + .5 replay CE；严格分开当前、蒸馏、监督三次forward，保留BN更新顺序；存原图和本步更新前增强输出的logit。
- 真实torchvision padding4随机裁剪、水平翻转p=.5、标准CIFAR100归一化；无RA或Random Erasing，无残差降权、target correction或自适应alpha/beta。
- 明确适配：官方100类/10task变为固定fine labels0..9的两个5类任务；每类原TRAIN的400 fit/100 dev；不使用官方test；Mac MPS/Python3.14.10/torch2.13.0/torchvision0.28.0。使用经过校验的官方网络/observe/reservoir AST和本地buffer审计适配器，未安装完整Mammoth框架，不声称原论文硬件／全框架逐位复现。全部差异与预先设计见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。

## 步数、曝光与独立复核

optimizer hook确认 **6300个有效step**，每task3150；100个完整epoch。每个fit source恰好出现50次，当前观察 **200,000**；两类replay总 **403,136**，模型总图像观察 **603,136**。

`verify_completed.py`没有运行模型，而是从固定seed重新播放所有uniform抽样和reservoir随机数：6300步全部抽取槽位、source、arrival、接受事件，与真实日志一致；最终buffer成员和numpy RNG状态也精确一致。500槽位最终包含 **{s['exposure']['final_buffer_unique_sources']}** 个不同source，类别槽位计数为 `{s['exposure']['final_buffer_classes']}`；没有把槽位数冒称唯一图像数。

20个BN的更新次数在task1/2末分别9448/18898，与`3×optimizer_steps−2`完全一致；任务终点LR=.0012，历史targets有限，buffer原图和标签逐条匹配TRAIN来源。独立NumPy重算各终点NLL/准确率与存档相符。loss分量重构最大误差 **{v['loss_reconstruction_max_abs_error']:.3g}**。

## 时间、资源和v1中断

- v2完整子进程墙钟：**{p['whole_child_perf_seconds']:.6f}秒（{p['whole_child_perf_seconds']/60:.2f}分钟）**，从导入到退出；UTC {p['started_utc']} 至 {p['finished_utc']}。
- run内部墙钟：{s['run_perf_seconds']:.6f}秒；6300步同步计时之和{v['sum_synchronized_step_seconds']:.6f}秒。每步已在MPS同步后计时，含该步CPU变换、buffer操作和optimizer。
- task1训练阶段{t1['task_wall_seconds']:.6f}秒，task2训练阶段{t2['task_wall_seconds']:.6f}秒；这些阶段计时包含周期性安全检查点，不含各自末尾开发评估／最终保存。总进程时间含全部工作。
- 进程peak RSS {s['memory']['process_peak_RSS_bytes']:,}字节；结束MPS driver {s['memory']['mps_driver_bytes']:,}字节。统一内存指标不相加。资源guard未报告并发重Python训练或内存不足；全程本机MPS。
- 无optimizer的合成CPU/MPS验收通过：logit最大差{s['preflight']['cpu_mps_logit_max_abs']:.3g}，梯度最大差{s['preflight']['cpu_mps_gradient_max_abs']:.3g}，BN running状态最大差{s['preflight']['cpu_mps_BN_running_state_max_abs']:.3g}。

v1在第 **{old['optimizer_hook_steps']}** 步因**logger的detach共享storage**触发分量assert而停止；这是本轮新增日志代码的确定bug。v2仅改为detach.clone并使用新输出路径，同一seed/配方重启，首步记录与v1逐项相同（耗时除外）。v1两步不并入有效6300步。源文件、原freeze、run-001和中断checkpoint全部保留，详见 [LOGGER_FIX_V2_ZH.md](LOGGER_FIX_V2_ZH.md)。这不改变旧760步试验或早先健康检查。

## 可供下一协议使用的记录

`run-002/task1-final.pt`和`task2-final.pt`包含模型、BN、optimizer/scheduler、buffer、历史targets、随机状态；`steps.jsonl`、`epochs.json`、`data-and-order-plan.npz`、`exposure-counts.npz`和逐例dev输出全部保存。每10epoch安全checkpoint仅用于恢复，不用于选择结果。

下一步仍需单独冻结门A/五分支的科学协议；本轮没有执行它们。没有使用test、旧封存测试、云GPU、付费、安装、重置卡或提交/推送。旧结果、旧阈值与旧审计不修改。
'''
with (ROOT/'REPORT_ZH.md').open('x') as f:f.write(report)
readme='''# 固定Mammoth DER++基础诊断，2026-10-29

完整结果见 [REPORT_ZH.md](REPORT_ZH.md)，执行前设计见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。v2已完成6300步、两个task各50epoch；只有一个固定seed，未执行门A或五分支。

`baseline.py`/`run-001`保留2步logger中断；`baseline_v2.py`/`run-002`是独立修复后的完整运行。`LOGGER_FIX_V2_ZH.md`说明唯一修复；两个freeze文件和原始日志均保留。`supervise_v2.py`及`process-timing-v2.json`记录完整子进程时间。

`sources/manifest.json`保存官方固定提交、Git blob、SHA-256；`verify_completed.py`独立验证步数、曝光、源映射、抽样/入库随机数、checkpoint和指标，不再次运行模型。`run-002/independent-verification.json`保存结果。

所有终点固定，不重跑当前路径、不选择中间checkpoint。旧760步基线不足结果不变；本次是新的基础学习诊断，不是科研贡献结论。
'''
with (ROOT/'README.md').open('x') as f:f.write(readme)
print('Completed report and README saved')
