"""Post-run read-only scientific summaries from complete persisted artifacts."""
import collections,datetime,hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 status=read(P/'FINAL_STATUS.json');assert status['status']=='COMPLETE'
 reports={a:read(P/f'formal_{a}/RESULT.json') for a in ['A','M']}
 comp=read(P/'evaluation/COMPARISON.json');verified=read(P/'evaluation/METRIC_RECOMPUTE_VERIFIED.json')
 controls={k:reports['A'][k]==reports['M'][k] for k in ['initial','initial_RNG','trainable_names_sha256','optimizer_groups_sha256','optimizer_initial_state_empty','anchor_order']};assert all(controls.values())
 summary={'status':status,'paired_controls_equal':controls,'comparison':comp,'metric_recomputation':verified,'arms':{},'cloud_ledger_unchanged':{'cumulative_CU_conservative_upper':1.8207069960365825,'cumulative_GPU_seconds_upper':5481.070303295135,'new_CU':0,'new_GPU_MPS_uploads_installs':0}}
 for arm,report in reports.items():
  records=[read(p) for p in sorted((P/f'formal_{arm}/trajectory').glob('*.json'))]
  assert len(records)==200 and [x['iteration'] for x in records]==list(range(1,201))
  assert all(x['updated'] and x['all_loss_gradients_finite'] for x in records)
  exposure=dict(report['source_exposure']);anchor=collections.Counter(x['anchor_image_id'] for x in records)
  assert len(anchor)==5 and set(anchor.values())=={40}
  exposure.update(anchor_counts=dict(anchor),operator_calls=dict(collections.Counter(e['type'] for x in records for e in x['events'])),visible_unique_sources_per_input=dict(collections.Counter(x['final_unique_source_count'] for x in records)),same_image_repeated_inputs=sum(len([s for s in x['final_source_regions'] if s['region_xyxy'][2]>s['region_xyxy'][0] and s['region_xyxy'][3]>s['region_xyxy'][1]])>x['final_unique_source_count'] for x in records),same_cached_origin_repeated_inputs=sum(len([(s['image_id'],s['origin_iteration']) for s in x['final_source_regions']])>len({(s['image_id'],s['origin_iteration']) for s in x['final_source_regions']}) for x in records))
  exposure['input_pixels_total']=sum(x['signature']['shape_CHW'][1]*x['signature']['shape_CHW'][2] for x in records)
  exposure['input_pixels_mean']=exposure['input_pixels_total']/200
  exposure['selected_same_cached_origin_repeated_inputs']=exposure.pop('same_cached_origin_repeated_inputs')
  exposure['visible_same_cached_origin_repeated_inputs']=sum(len([(s['image_id'],s['origin_iteration']) for s in x['final_source_regions'] if s['region_xyxy'][2]>s['region_xyxy'][0] and s['region_xyxy'][3]>s['region_xyxy'][1]])>len({(s['image_id'],s['origin_iteration']) for s in x['final_source_regions'] if s['region_xyxy'][2]>s['region_xyxy'][0] and s['region_xyxy'][3]>s['region_xyxy'][1]}) for x in records)
  peak=max(x['peak_RSS_bytes'] for x in records)
  ck=report['checkpoint'];assert sha(pathlib.Path(ck['path']))==ck['sha256']
  ev=read(P/f'evaluation/receipts/model_20261031_{arm}.json')
  summary['arms'][arm]={'exposure':exposure,'checkpoint':ck,'pipeline_reference_matches':report['pipeline_reference_matches'],'training_step_seconds':sum(x['train_step_seconds'] for x in records),'training_pipeline_seconds':sum(x['pipeline_seconds'] for x in records),'training_process_seconds':report['elapsed_seconds'],'training_peak_RSS_bytes':max(peak,report['peak_RSS_bytes']),'evaluation':ev,'all200_finite_and_persisted':True,'initial':report['initial']}
 summary['metric_differences_pp']={k:comp['metrics_M'][k]['percent']-v['percent'] for k,v in comp['metrics_A'].items()}
 summary['peak_process_RSS_bytes']=max(max(s['training_peak_RSS_bytes'],s['evaluation']['peak_RSS_bytes']) for s in summary['arms'].values())
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 lines=['本机 CPU A/M 配对实验已完整完成。两臂均 seed=20261031、U=200、同一5-shot支持集和64张dev。主比较为CPU训练并CPU评估的M减A。','',f"总运行 {status['elapsed_seconds']/60:.2f} 分钟；最高进程峰值RSS {summary['peak_process_RSS_bytes']/1024**3:.3f} GiB；新增CU、GPU、MPS、上传、安装均为0。",'', '|指标|A (%)|M (%)|M−A (pp)|maxDets|','|---|---:|---:|---:|---:|']
 for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR100','AR300','AR1000']:
  a=comp['metrics_A'][k];m=comp['metrics_M'][k];lines.append(f"|{k}|{a['percent']:.6f}|{m['percent']:.6f}|{m['percent']-a['percent']:+.6f}|{a['maxDets']}|")
 lines+=['','COCO未定义的尺度指标原值−1（百分数表示−100）保留，不能解释为负的准确率。主AP使用maxDets100；AP50/AP75/尺度AP使用1000，AR100/300/1000分别使用对应上限。所有指标在两臂训练与完整权重安全回读成功后统一计算，并从128条持久化预测独立复算一致。','', '|实际训练数据使用|A|M|','|---|---:|---:|']
 for k in ['anchor_count','selected_occurrences','visible_occurrences','GT_presentations','empty_GT_inputs','GT_removed','Mosaic_applied','MixUp_applied','same_image_repeated_inputs','selected_same_cached_origin_repeated_inputs','visible_same_cached_origin_repeated_inputs','input_pixels_mean']:
  lines.append(f"|{k}|{summary['arms']['A']['exposure'][k]}|{summary['arms']['M']['exposure'][k]}|")
 lines+=['','|算子调用次数（不等于概率门生效次数）|A|M|','|---|---:|---:|']
 ac=summary['arms']['A']['exposure']['operator_calls'];mc=summary['arms']['M']['exposure']['operator_calls']
 for k in sorted(set(ac)|set(mc)):lines.append(f"|{k}|{ac.get(k,0)}|{mc.get(k,0)}|")
 lines+=['','同一完整初始state哈希、初始CPU/Python/NumPy随机状态、896个可训练参数张量、AdamW分组、LR、batch和200个anchor顺序全部一致；每张support作为anchor恰好40次。M额外随机算子消费和更多source曝光属于固定策略差异。每臂200次真实数据轨迹与既有冻结数据干跑记录像素/GT/几何一致。','', '模型与数据：原始官方GroundingDINO Swin-T，FP32 CPU，Torch4线程/inter-op1/OpenCV顺序；不改科学配方和标签。两完整模型、全部轨迹、128条预测、COCO完整precision/recall/score数组、协议和实现hash均保留。','', '解释范围：这是同一episode/seed、固定更新预算的整套策略探索；不是曝光匹配的纯增强因果效应，不是ETS论文复现，不证明新贡献或跨场景泛化。历史CUDA训练结果仅背景，未用于本次主差值。官方test未用于训练、选择或本次模型评估。','', '预检选择事前固定为训练轨迹A第2次、M第43/44次，覆盖正GT、大图和满缓存双混合；预检模型/优化器随子进程退出丢弃。正式两臂均独立从原始官方权重初始化。无best-checkpoint或dev选优。']
 for arm in ['A','M']:
  s=summary['arms'][arm];lines+=['',f"{arm}完整模型：`formal_{arm}/final_model.pth`，{s['checkpoint']['bytes']}字节，SHA256 `{s['checkpoint']['sha256']}`。",f"{arm}训练更新累计{s['training_step_seconds']:.3f}秒；训练进程{s['training_process_seconds']:.3f}秒；64张dev前向{s['evaluation']['forward_seconds']:.3f}秒。"]
 lines+=['','基础设施修订1：首个dev前向之前，补齐旧预测清单中缺少的几何核对引用。记录全部64张固定原图的既有CPU几何hash，未使用效果选择；训练代码/配置、模型推理函数、COCO计算函数均未改变。旧/新代码hash及5项无模型回归证据保存在`infrastructure-revision1/`。','', '统计口径：selected_occurrences包含被裁掉至空区域的来源，visible_occurrences仅计最终非空区域。重复同缓存来源分别按被选来源和可见来源统计，因此M为21和20次输入，二者不混用。']
 (P/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'SUMMARY_WRITTEN','AP_A':comp['A'],'AP_M':comp['M'],'M_minus_A_pp':comp['M_minus_A_pp'],'minutes':status['elapsed_seconds']/60,'peak_GiB':summary['peak_process_RSS_bytes']/1024**3}))
if __name__=='__main__':main()
