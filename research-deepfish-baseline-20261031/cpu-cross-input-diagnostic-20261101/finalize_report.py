"""After successful terminal and watchdog exit, archive all512 persisted outputs."""
import csv
import datetime
import hashlib
import io
import json
import pathlib
from diagnostic_contracts import P,R,PREP,METRIC_MAPPING,sha,atomic_json,complete_matrix,read_execution,tree_bytes

run=read_execution()
terminal=json.loads((P/'TERMINAL.json').read_text())
assert terminal['status']=='COMPLETE_512_ALL8_RECOMPUTED'
assert terminal['all_managed_CPU_processes_exited']
watch=json.loads((P/'WALL_WATCHDOG.json').read_text())
assert watch['status']=='DISARMED_FORMAL_TERMINAL_PRESENT'
recomputed=json.loads((P/'METRIC_RECOMPUTE_VERIFIED.json').read_text())
assert recomputed['status']=='ALL8_EXACT_FROM_PERSISTED_PREDICTIONS' and len(recomputed['conditions'])==8
matrix=complete_matrix(run)
assert matrix['manifest_sha256']==terminal['prediction_manifest_sha256']==recomputed['prediction_manifest_sha256']
values=json.loads((P/'ALL_METRICS.json').read_text())
assert len(values['conditions'])==8 and values['prediction_manifest_sha256']==matrix['manifest_sha256']
children=json.loads((P/'PROCESS_LEDGER.json').read_text())
assert len(children)==6 and all(x['status']=='EXITED' and x['exit_code']==0 for x in children)
for name,h in run['code_sha256'].items():assert sha(P/name)==h
assert sha(PREP/'ENVIRONMENT_LOCK.json')==run['environment_lock_sha256']
site=PREP/'.venv/lib/python3.11/site-packages'
for name,h in run['official_implementation_sha256'].items():assert sha(site/name)==h
assert sha(run['dev_annotation']['path'])==run['dev_annotation']['sha256']

reports=[]
for spec in run['models']:
    d=json.loads((P/'receipts'/f"model_{spec['seed']}_{spec['arm']}.json").read_text())
    assert d['status']=='MODEL_COMPLETE' and d['forwards_started']==d['forwards_completed']==d['predictions_persisted']==128
    assert d['load']['checkpoint_sha256']==spec['sha256'] and d['load']['strict'] and d['load']['weights_only']
    assert d['load']['tensor_count']==908 and d['load']['all_tensor_bytes_equal']
    reports.append(d)
totals={k:sum(d[k] for d in reports) for k in ['pipeline_seconds','forward_seconds','format_seconds','persistence_seconds','import_setup_seconds']}
totals['model_load_seconds']=sum(d['load']['seconds'] for d in reports)
totals['evaluation_seconds']=values['evaluation_seconds']
totals['persisted_recomputation_seconds']=recomputed['seconds']
totals['overall_wall_seconds']=terminal['actual_wall_seconds']
totals['other_guard_process_startup_and_bookkeeping_seconds']=terminal['actual_wall_seconds']-sum(v for k,v in totals.items() if k!='overall_wall_seconds')
totals['peak_individual_model_RSS_bytes']=max(d['peak_RSS_bytes'] for d in reports)
totals['formal_forwards']=512;totals['model_loads']=4;totals['CPU_child_processes']=6
totals['prediction_count']=512*300;totals['prediction_JSON_bytes']=sum(x['bytes'] for x in matrix['records'])
totals['actual_new_CU']=0;totals['actual_upload_bytes']=0
atomic_json(P/'ACTUAL_COSTS.json',totals,new=True)

with (P/'METRICS.csv').open('x',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['seed','training_arm','input_factor']+[f'{name}_maxDets{mdet}_percent' for name,_,mdet,_ in METRIC_MAPPING])
    for v in values['conditions']:
        writer.writerow([v['seed'],v['arm'],1.0 if v['condition']=='c100' else .8]+
            [format(v['metrics'][name]['percent'],'.17g') if v['metrics'][name]['value']>=0 else '' for name,_,_,_ in METRIC_MAPPING])
with (P/'INTERACTIONS.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(values['primary_interactions'][0]));writer.writeheader();writer.writerows(values['primary_interactions'])

def score(v,key):
    m=v['metrics'][key]
    return f"{m['percent']:.4f}" if m['value']>=0 else '未定义'

table=[]
for v in values['conditions']:
    factor='1.0' if v['condition']=='c100' else '0.8'
    table.append('| '+ ' | '.join([str(v['seed']),v['arm'],factor]+[score(v,k) for k in ['AP','AP50','AP75','AR100','AR300','AR1000']])+' |')
inter=[]
for x in values['primary_interactions']:
    inter.append(f"| {x['seed']} | {x['B_minus_A_input1_pp']:+.4f} | {x['B_minus_A_input08_pp']:+.4f} | {x['interaction_pp']:+.4f} | {x['A_input_change_pp']:+.4f} | {x['B_input_change_pp']:+.4f} |")
ints=[x['interaction_pp'] for x in values['primary_interactions']]
if all(x>0 for x in ints):
    interpretation='两个固定种子的I均为正：将输入固定改为0.8后，B相对A的AP差距向B方向变化。这个描述只针对上述人工输入条件；须同时看四格绝对AP，不能将差距变化自动等同于B绝对性能改善。'
elif all(x<0 for x in ints):
    interpretation='两个固定种子的I均为负：将输入固定改为0.8后，B相对A的AP差距进一步向A方向变化。这组结果不支持简单的“训练与输入同为0.8就会相对更好”解释。'
else:
    interpretation=(f"第一个seed的I为{ints[0]:+.4f}个百分点，数值变化很小；第二个seed的I为{ints[1]:+.4f}个百分点。"
                    '第二轮中人工0.8输入缩小B相对A的差距，但两种输入、两个seed下B的主AP仍全部低于A。'
                    '这些结果尚未支持稳定的相对颜色匹配效应，也不支持把B的劣势简单归因于训练与输入颜色不匹配；不能挑选有利种子。')
cost_table=[]
for d in reports:
    cost_table.append(f"| {d['model']} | {d['load']['seconds']:.3f} | {d['pipeline_seconds']:.3f} | {d['forward_seconds']:.3f} | {d['persistence_seconds']:.3f} | {d['elapsed_seconds']:.3f} | {d['peak_RSS_bytes']/1024**3:.3f} |")

report=f'''完整本机CPU交叉输入诊断完成：2seed×2训练臂×2输入条件×64dev，共512个唯一图像条件、153,600条未按分数筛选的top300检测。四个完整checkpoint各加载一次，没有新增训练、第三seed、预热或重跑此前8次预检。所有8条件齐全后统一评估，并由另一个无模型CPU进程从持久化结果复算；12项统计和完整precision/recall/scores数组逐项精确一致。

唯一解释问题仍是：在独立CPU环境中，训练颜色与人工输入颜色条件是否改变A/B差距。训练臂A固定Color1.0，B固定0.8；表中输入因子独立取1.0或0.8。全部四格都来自本次同一CPU实现，未混入旧GPU分数。

| seed | 训练臂 | 输入因子 | 主AP@100 | AP50@1000 | AP75@1000 | AR@100 | AR@300 | AR@1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

单位为百分数。尺度AP使用maxDets1000；AR分别使用100/300/1000，尺度AR使用1000。每图实际保留300个预测，因此maxDets1000不会产生额外预测。原COCO统计值-1表示未定义，CSV留空，不当作负性能。12项完整结果见 [METRICS.csv](METRICS.csv) / [ALL_METRICS.json](ALL_METRICS.json)，完整COCO数组保存在metrics目录。

主交互I=[AP_B−AP_A]输入0.8−[AP_B−AP_A]输入1.0，始终使用原主AP@100，单位百分点：

| seed | B−A 原图 | B−A 输入0.8 | I | A输入0.8−原图 | B输入0.8−原图 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(inter)}

{interpretation} 此为看过首轮两臂结果后开展的固定条件诊断，仅有两个已训练种子、同一64图/187GT和8个相关前缀组；前缀语义不能当成已证实的新场景。没有效用驱动选强度、选图、选checkpoint、调阈值或挑子集。结果不证明一般随机增强机制、真实域泛化或部署预处理收益，目前不足以支持可靠的新科研贡献。

执行保持官方原始MMDetection3.3.0/MMCV2.1.0 GroundingDINO Swin-T实现，隔离Python3.11.14/Torch2.1.0，CPU4线程、interop1、OpenCV顺序、FP32、eval、inference_mode。Color仍在cv2 uint8/BGR解码后、原FixScaleResize前，原LoadAnnotations顺序保留。A逐图恒等、0.8逐图匹配原cv2变换；全部512记录的几何/GT hash按图像跨8条件一致。4195/4660原bbox/area未改。官方test未用于本次推理或指标。

每个checkpoint的908张量均weights_only=True在CPU加载，strict=True，无missing/unexpected，dtype/shape/全部字节精确匹配，并与已审查CPU预检完整状态hash一致。没有unsafe pickle、任意白名单、trust_remote_code、模型/算子移植或依赖重装。数据、模型、原配置、环境与入口代码hash在 [RUN_PROTOCOL.json](RUN_PROTOCOL.json)；原始512份预测各有独立身份、payloadSHA和文件SHA，见 [ALL_512_COMPLETE.json](ALL_512_COMPLETE.json)。

逐图采用文件fsync、同目录原子发布、目录fsync和回读核对；没有checkpoint或整幅输入张量副本。全部模型进程退出后才进行指标汇总与复算。7项无模型身份/持久化/预算/人工指标映射测试和一次静态自检已先通过，见TEST_RESULTS.json、STATIC_SELF_CHECK.json。没有缺失、重复或未决partial预测被纳入结果。

实际正式运行墙钟{terminal['actual_wall_seconds']:.3f}秒（{terminal['actual_wall_seconds']/60:.2f}分钟），包含四次加载、全部前向/落盘、统一评估和独立复算；先前48.13分钟是保守估计，90分钟为上限。外部UTC watchdog在终态后解除，未触发停止。逐阶段秒数如下：

| 模型 | 加载+严格校验 | 图像管线合计 |128前向合计|逐图落盘合计|模型进程总计|峰值RSS GiB|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cost_table)}

统一8条件评估{values['evaluation_seconds']:.3f}秒，另进程8条件持久化复算{recomputed['seconds']:.3f}秒。其余解释器/进程启动、hash、资源检查和记录开销单独计入 [ACTUAL_COSTS.json](ACTUAL_COSTS.json)，不冒充前向耗时；RSS为单个模型进程实测最大值，不是物理内存容量。曾有一次外部状态查询使用系统Python3.9的monotonic基准，不能与本任务Python3.11混比较；正式监督/工作进程一直统一3.11，随后UTC与其monotonic计时核对一致，并附加独立UTC截止保护。无模型或数据因此重跑。

两项历史pipcheck元数据警告保留，不写作pipcheck通过。Ninja旧WHEEL_sha256字段是CRLF换行规范化后的文本hash，原始359字节文件的SHA与installed RECORD一致；字段意义及原始文件SHA已在 [DOCUMENTATION_CORRECTIONS.json](DOCUMENTATION_CORRECTIONS.json) 明确。旧准备报告和清单保留不改，当前AR指标映射以本次RUN_PROTOCOL为准。

全部6个受监督CPU子进程（4推理、1评估、1复算）退出码0，监督进程和外部watchdog也已结束。GPU/MPS/Colab操作、上传和新增CU均为0；历史账本继续保留1.8088330815795528CU /5443.899788473129GPU秒。研究目录仍在6GiB内，新增产物小于100MiB且保留80MiB以上余量；最终精确字节数见ARCHIVE_RECEIPT.json。未删除旧资产，未查询或重试被拒绝的物理RAM读取。

当前终态是完整诊断已完成且可从本机持久化结果复算，没有后台模型作业、未完成条件或待执行的额外seed。后续科学决策应基于全部四格和上述限域，不以交互方向代替创新性或泛化证据。
'''
assert not (P/'REPORT_ZH.md').exists();(P/'REPORT_ZH.md').write_text(report)
atomic_json(P/'FINAL_VERIFICATION.json',{'status':'PASS','unique_persisted_identities':512,'conditions':8,
    'predictions':153600,'checkpoint_loads':4,'formal_forwards':512,
    'full12_stats_and_arrays_exact_on_reread':True,'original_source_config_environment_pins_unchanged':True,
    'all_CPU_children_exited':True,'watchdog_disarmed':True,'new_CU':0,'uploads':0,
    'research_bytes_before_manifest':tree_bytes(R),'formal_bytes_before_manifest':tree_bytes(P),
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},new=True)
atomic_json(P/'STATUS.json',{'status':'COMPLETE_512_ALL8_RECOMPUTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executor_ready':True,
    'no_model_test_checks_passed':7,'formal512_completed':True,'persisted_unique_record_files':512,
    'CPU_workers':[],'all_CPU_processes_exited':True,'watchdog_disarmed':True,
    'metrics_conditions':8,'independent_recomputation_pass':True,'awaiting_GO':False,
    'new_CU':0,'uploads':0})

entries=[]
for f in sorted(P.rglob('*')):
    if f.is_file():
        assert f.name not in ('DELIVERABLE_MANIFEST.json','ARCHIVE_RECEIPT.json')
        assert not f.name.endswith('.partial')
        entries.append({'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':sha(f)})
atomic_json(P/'DELIVERABLE_MANIFEST.json',{'files':entries,'external_reused_assets':'RUN_PROTOCOL.json',
    'contains_all512_predictions':True,'no_checkpoint_or_full_image_tensor_copies':True},new=True)
atomic_json(P/'ARCHIVE_RECEIPT.json',{'status':'COMPLETE_ARCHIVED',
    'manifest_sha256':sha(P/'DELIVERABLE_MANIFEST.json'),'entries':len(entries),
    'formal_bytes_before_receipt':tree_bytes(P),'research_bytes_before_receipt':tree_bytes(R),
    'research_cap_bytes':6*1024**3,'research_margin_bytes_before_receipt':6*1024**3-tree_bytes(R),
    'formal_output_cap_bytes':100*1024**2,'new_CU':0,'uploads':0},new=True)
assert tree_bytes(P)<=100*1024**2 and tree_bytes(R)<=6*1024**3-80*1024**2
print(json.dumps({'status':'COMPLETE_ARCHIVED','manifest_sha256':sha(P/'DELIVERABLE_MANIFEST.json'),
    'files':len(entries),'formal_bytes':tree_bytes(P),'research_bytes':tree_bytes(R),
    'wall_seconds':terminal['actual_wall_seconds'],'primary_interactions':values['primary_interactions']}))
