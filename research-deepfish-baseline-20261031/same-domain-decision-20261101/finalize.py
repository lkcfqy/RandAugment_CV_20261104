"""Summarize only a complete three-arm run, no new inference or output diagnostics."""
from portable_dev.paths import project_path as _portable_project_path
import pathlib,json,hashlib,collections,datetime
P=pathlib.Path(__file__).resolve().parent;R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
status=read(P/'FINAL_STATUS.json');assert status['status']=='COMPLETE' and status['dev_predictions']==192 and status['all_child_processes_exited']
comp=read(P/'evaluation/COMPARISON.json');metricverify=read(P/'evaluation/METRIC_RECOMPUTE_VERIFIED.json');assert metricverify['status']=='ALL192_EXACT_FROM_PERSISTED'
proto=read(P/'PROTOCOL.json');historical=read(R/'cpu-paired-AM-20261101/formal_A/RESULT.json');reports={a:read(P/f'formal_{a}/RESULT.json') for a in ['A1','M1','C0']};ledger=read(P/'C0_FROZEN_SOURCE_LEDGER.json')
controls={a:{k:r[k]==historical[k] for k in ['initial','initial_RNG','trainable_names_sha256','optimizer_groups_sha256','optimizer_initial_state_empty']} for a,r in reports.items()};assert all(all(c.values()) for c in controls.values())
summary={'status':status,'controls_match_historical_A0_M0':controls,'metrics':comp,'metric_recompute':metricverify,'arms':{},'future_confirmation_pool_used':False,'official_test_used_for_model_effects':False,'historical_test_QA_images':3,'new_CU_GPU_MPS_upload_install':0}
for arm,r in reports.items():
 rows=[read(f) for f in sorted((P/f'formal_{arm}/trajectory').glob('*.json'))];assert len(rows)==200 and [x['update'] for x in rows]==list(range(1,201)) and all(x['updated'] and x['all_loss_gradients_finite'] for x in rows)
 assert r['update_call_counts']=={'clip':200,'optimizer':200,'zero_grad':200,'schedule_boundary':200} and r['scheduler_objects']==0
 if arm=='C0':assert [x['source_image_ids'] for x in rows]==[x['source_image_ids'] for x in ledger] and sum(x['k'] for x in rows)==659
 else:
  counter=collections.Counter(x['source_image_ids'][0] for x in rows);assert len(counter)==5 and set(counter.values())=={40}
 allsources=[s for row in rows for s in row['sources']];selected=[v for s in allsources for v in s['final_source_regions']];visible=[v for v in selected if v['region_xyxy'][2]>v['region_xyxy'][0] and v['region_xyxy'][3]>v['region_xyxy'][1]]
 exposure={'source_forwards':sum(x['k'] for x in rows),'selected_source_occurrences':len(selected),'visible_source_occurrences':len(visible),'individual_source_counts':dict(collections.Counter(v['image_id'] for v in selected)),'input_GT_presentations_before_pipeline':len(selected) if arm!='M1' else None,'retained_GT_presentations':sum(len(s['signature']['gt']['gt_instances']['boxes']) for s in allsources),'empty_GT_inputs':sum(not s['signature']['gt']['gt_instances']['boxes'] for s in allsources),'input_pixels_total':sum(s['signature']['shape_CHW'][1]*s['signature']['shape_CHW'][2] for s in allsources),'operator_calls':dict(collections.Counter(e['type'] for s in allsources for e in s['events'])),'Mosaic_applied':sum(e.get('applied',False) for s in allsources for e in s['events'] if e['type']=='CachedMosaic'),'MixUp_applied':sum(e.get('applied',False) for s in allsources for e in s['events'] if e['type']=='CachedMixUp'),'pipeline_GT_discard_events_total':sum(e['discarded_GT_count'] for s in allsources for e in s['events'])}
 ck=r['checkpoint'];assert sha(pathlib.Path(ck['path']))==ck['sha256'] and ck['complete_state_verified'];ev=read(P/f'evaluation/receipts/model_20261031_{arm}.json');assert ev['status']=='MODEL_COMPLETE' and ev['predictions_persisted']==64
 summary['arms'][arm]={'checkpoint':ck,'training_elapsed_seconds':r['elapsed_seconds'],'training_update_seconds':sum(x['train_step_seconds'] for x in rows),'pipeline_seconds':sum(s['pipeline_seconds'] for s in allsources),'training_peak_RSS_bytes':r['peak_RSS_bytes'],'evaluation':ev,'exposure':exposure,'step_calls':r['update_call_counts'],'source_forward_count':r['completed_source_forwards']}
assert reports['A1']['source_groups']==reports['M1']['source_groups'];summary['new_episode_anchor_order_equal']=True
summary['peak_RSS_bytes']=max(max(v['training_peak_RSS_bytes'],v['evaluation']['peak_RSS_bytes']) for v in summary['arms'].values());summary['M_effect_same_direction_across_episodes']=(comp['M1_minus_A1_pp']*comp['M0_minus_A0_pp']>0)
# The descriptive final decision is written after reading the complete common evaluation.
write(P/'SUMMARY.json',summary)
lines=['三臂同域开发性实验已完整完成。固定 seed20261031，各 U200；A1/M1 使用事先哈希选定的自构造五图，C0 使用原官方五图。三份完整 checkpoint 均先安全落盘核验，随后统一评估相同 64 张已见 dev。没有官方 test 或未来确认池效果评估。','',f'本次流程 {status["elapsed_seconds"]/60:.3f} 分钟；最高进程峰值 RSS {summary["peak_RSS_bytes"]/1024**3:.3f} GiB。新增 GPU/MPS/云费用/上传/安装均为 0。原用户累计 CU 上限未改变；研究目录上限在结果出现前按授权由 8 改为 12 GiB，未删除旧结果。','', '|指标|A0（已有）|M0（已有）|C0|A1|M1|maxDets|','|---|---:|---:|---:|---:|---:|---:|']
for key in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR100','AR300','AR1000']:
 vals=[comp['metrics_'+a][key] for a in ['A0','M0','C0','A1','M1']];lines.append('|'+key+'|'+'|'.join(f'{v["percent"]:.6f}' for v in vals)+f'|{vals[0]["maxDets"]}|')
lines+=['',f'主 AP 差（百分点）：M0−A0 = {comp["M0_minus_A0_pp"]:+.6f}；M1−A1 = {comp["M1_minus_A1_pp"]:+.6f}；C0−A0 = {comp["C0_minus_A0_pp"]:+.6f}；C0−M0 = {comp["C0_minus_M0_pp"]:+.6f}。两 episode 配方效应同向：{summary["M_effect_same_direction_across_episodes"]}。', '',
'主 AP 为 maxDets100；AP50/AP75/尺度 AP 为1000；AR100/300/1000分别对应自己的上限。未定义尺度统计原值−1保留，不解释为负准确率。原生保存池每图300框。所有新指标由完整192条持久化预测独立复算一致；没有额外排序/上界诊断。', '',
'控制：三个新模型与历史 CPU A0/M0 的完整初始 state、初始 Python/NumPy/Torch 状态、可训练参数、空 optimizer 状态、AdamW 分组与 LR 完全一致；A1/M1 anchor 顺序一致、每图40次。C0按 M0 每更新原有顺序保留来源多重集，含同图重复及被原 M0 裁空的来源；每来源恢复原图全部既有标注后独立走 A 普通几何/恒等颜色，仍允许该固定 A 随机裁剪移除框。', '',
'C0目标为每更新 k 个来源各自官方标量损失的算术均值：逐个 backward(loss/k)，最后只裁剪、AdamW step、zero_grad一次。五种 k 的合成损失/未裁剪梯度/更新结果与显式平均目标一致，k=1与官方 A 单步逐位一致；全659来源fixture和实际运行均验证200次更新。原配置 param_scheduler=[]，没有 scheduler 对象，200次 schedule-boundary 仅核恒定LR，未引入新调度。C0匹配来源次数和更新数，不匹配保留GT、像素上下文或梯度噪声，不作纯因果归因。', '', '|实际训练代价|A1|M1|C0|','|---|---:|---:|---:|']
for key in ['source_forwards','selected_source_occurrences','visible_source_occurrences','retained_GT_presentations','empty_GT_inputs','input_pixels_total','Mosaic_applied','MixUp_applied']:
 lines.append('|'+key+'|'+'|'.join(str(summary['arms'][a]['exposure'][key]) for a in ['A1','M1','C0'])+'|')
for label,key,div in [('训练进程分钟','training_elapsed_seconds',60),('训练更新累计分钟','training_update_seconds',60),('训练峰值GiB','training_peak_RSS_bytes',1024**3)]:lines.append('|'+label+'|'+'|'.join(f'{summary["arms"][a][key]/div:.3f}' for a in ['A1','M1','C0'])+'|')
lines+=['', '已有 A0/M0 整对（每臂 U200）实测合计40.057分钟，非每臂40分钟。新episode与C0耗时见上表及PHASES.json，不能只以更新数忽略额外来源计算成本。', '',
'选择与边界：新支持图 IDs 1/2009/53/1980/353保持冻结；原完整标注未改，五图只做格式和框目标QA。7623两个前缀此前各有test QA，所有三张历史test QA已在资格报告披露。数值habitat code有原官方选择器依据，完整prefix不是已证实独立视频/场景。64dev在此前探索中已见，本轮整体是探索性决策，不是确认性实验，不用两个episode作总体显著性声明，也不包装为新贡献。', '',
'产物：formal_A1、formal_M1、formal_C0中的完整模型/receipt/200条逐更新轨迹；evaluation中的192条预测、COCO完整precision/recall/scores数组和独立复算；固定协议、来源账本、测试、预算门及全部输入/代码hash。']
for arm,x in summary['arms'].items():lines+=['',f'{arm}模型 SHA256 `{x["checkpoint"]["sha256"]}`，{x["checkpoint"]["bytes"]} 字节。']
(P/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'SUMMARY_WRITTEN','M0_minus_A0_pp':comp['M0_minus_A0_pp'],'M1_minus_A1_pp':comp['M1_minus_A1_pp'],'C0_minus_A0_pp':comp['C0_minus_A0_pp'],'C0_minus_M0_pp':comp['C0_minus_M0_pp'],'minutes':status['elapsed_seconds']/60,'peak_GiB':summary['peak_RSS_bytes']/1024**3},indent=2))
