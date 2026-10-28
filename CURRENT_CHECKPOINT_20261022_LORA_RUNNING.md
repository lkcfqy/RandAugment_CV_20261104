# LoRA × RandAugment：运行中快照

此文件是2026-10-22T15:13:39.331994+00:00的历史快照，不是完成回执。若已有 CURRENT_CHECKPOINT_20261022_LORA_PILOT.md，应以新的完成检查点为准。

原项目研究已不限因果。本轮唯一实验为等参数前/后层Q/V-LoRA × weak/RA，四格单seed20261022、每格30epochs，共4440更新。正式协议SHA为9386fc378f876cf12c68ef5d7248f073d7313433d158350b4143942ac1b52a3f，锁定14:51:37 UTC，科学进程14:52:39 UTC启动。

工作目录：augmentation-research-20261022/pilot-20261022。协议、源码、QA、公共数据、实际资源回执及独立预审均已落盘。最终591train/589development，4重复成员全删且不补抽；确认592成员不提取/解码/评分；range传输尾部边界见data/range-transfer-scope.json。许可与3个缺前景mask的透明记录已保留。

运行：本任务exec session50369、Python PID49366，mps:0、CPU1、nice10、memory fraction.1、每步等待两倍活跃段，wall cap7200秒。front-weak已完成30epochs（42/589错误，底座SHA不变、Q/V及特征改变）；back-ra正在运行，其后back-weak、front-ra，单进程串行。实际状态以training.log、results/*/curve.json和steps.jsonl为准，不可由本快照假定已完成。

不要重复启动train.py、追加seed、换LR/epoch/主指标或解封确认图。需要停止时只对本run创建STOP.request，程序会保存incomplete并禁止拼合主I。只有results/completed.json确认四格完整后，才运行锁定analyze.py，再执行independent/verify_saved_pilot.py独立复算；其代码SHA c3dfbbf4bfd3323e100fa440db3acb33c1146c0d7fe187dc846b24b860a13720。独立脚本只读保存的开发logits/小checkpoint，CPU算术；不前向或重训。

8A最高优先，未读取或修改其文件/环境。无A100/Colab/ARiS/付费/上传/发布。旧阴性与课堂文件17+58+25份保护核验通过。Sol/max与现有Fast为请求设置，实际后端档位回执未暴露。

## 后续资源协调快照：2026-10-22T15:21:55.141568+00:00

8A正在测CPU/MPS步耗时，因此本run的PID49366已SIGSTOP原地暂停，状态T；没有重启、改协议或按性能停止。前两格front-weak/back-ra已各30epochs/1110updates，第三格back-weak已完成3epochs、133updates，正在第4epoch。暂停前最后日志距核对仅0.23秒，健康推进。累计墙钟约1756秒，7200秒cap余约5444秒，暂停仍计入cap。确认8A计时结束后才对同PID SIGCONT继续；不要再次启动train.py或读取8A文件。详细进程与步日志证据见resource-pause-01.json。

## 同进程恢复：2026-10-22T15:27:29.146174+00:00

8A计时已完成，当前仅CPU四线程pilot，没有GPU训练（来源：父线程协调，未读取8A）。已核对PID49366命令与T状态后SIGCONT，暂停334.0秒，原7200秒墙钟不重置。继续第三/四格；CPU并行期记录在resource-resume-01.json，runtime可比性需保留此边界。

## 实际持续推进确认：2026-10-22T15:34:30.852308+00:00

PID49366已于15:27:29原地SIGCONT，当前非T状态；本次核验细节见resume-progress-confirmation.json/LATEST_STATUS.json。前两格完整，第三格最新已完成epoch/updates为[('back-ra', 30, 1110), ('back-weak', 23, 855), ('front-weak', 30, 1110)]。8A由父线程核实在15:32 UTC已全部结束，CPU并行期结束时刻仅分钟精度；本项目继续原协议，原7200秒cap不变。

## 本轮完成

四格与独立复算已完成，本轮后台训练为0。当前结论、原始结果与下一步以CURRENT_CHECKPOINT_20261022_LORA_PILOT.md为准；上述运行/暂停快照作为历史记录保留。
