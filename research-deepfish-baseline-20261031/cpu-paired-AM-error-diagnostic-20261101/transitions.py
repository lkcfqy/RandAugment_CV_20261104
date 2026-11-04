"""Fixed post-hoc GT transitions and prediction-budget composition, no models."""
import collections,csv,json,pathlib
P=pathlib.Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'reports/GT_DIAGNOSTICS.csv').open()))
dets=list(csv.DictReader((P/'reports/ALL_PREDICTION_DIAGNOSTICS.csv').open()))
def rank(r,arm,label):return int(r[arm+'_coco_match'+label+'_rank']) if r[arm+'_coco_match'+label+'_rank'] else 10**9
def first(r,arm,label):return int(r[arm+'_first_rank'+label]) if r[arm+'_first_rank'+label] else 10**9
def reason(r,arm,t,label,cap):
 if float(r[arm+'_best_iou'])<t:return 'no_qualified_geometry_candidate'
 if first(r,arm,label)>cap:return 'qualified_candidate_outside_prediction_budget'
 return 'candidate_in_budget_but_unmatched_competition'
trans=[];group_rows=[]
for cap in [100,300]:
 for t,label in [(.5,'50'),(.75,'75')]:
  for r in rows:
   a=rank(r,'A',label)<=cap;m=rank(r,'M',label)<=cap
   change='retained' if a and m else 'gained' if m else 'lost' if a else 'neither'
   trans.append({'image_id':int(r['image_id']),'gt_id':int(r['gt_id']),'group':r['group'],'maxDets':cap,'IoU':t,'A_matched':a,'M_matched':m,'change':change,'missing_side_reason':reason(r,'A' if change=='gained' else 'M',t,label,cap) if change in ['gained','lost'] else '', 'A_best_iou':float(r['A_best_iou']),'M_best_iou':float(r['M_best_iou']), 'A_first_qualified_rank':None if first(r,'A',label)==10**9 else first(r,'A',label),'M_first_qualified_rank':None if first(r,'M',label)==10**9 else first(r,'M',label)})
  for group in ['ALL']+sorted({r['group'] for r in rows}):
   sub=[r for r in trans if r['maxDets']==cap and r['IoU']==t and (group=='ALL' or r['group']==group)]
   group_rows.append({'group':group,'maxDets':cap,'IoU':t,'GT':len(sub),'A_matched':sum(r['A_matched'] for r in sub),'M_matched':sum(r['M_matched'] for r in sub),**{key:sum(r['change']==key for r in sub) for key in ['gained','lost','retained','neither']},'gained_reason_counts':dict(collections.Counter(r['missing_side_reason'] for r in sub if r['change']=='gained')),'lost_reason_counts':dict(collections.Counter(r['missing_side_reason'] for r in sub if r['change']=='lost'))})
comp=[]
for arm in ['A','M']:
 for cap in [10,100,300]:
  for t,label in [(.5,'50'),(.75,'75')]:
   counts=collections.Counter()
   for r in dets:
    if r['arm']!=arm or int(r['rank'])>cap:continue
    if int(r['coco_match'+label+'_gt'])>0:kind='TP'
    elif float(r['max_gt_iou'])>=t:kind='duplicate_candidate'
    elif float(r['max_gt_iou'])>=.1:kind='localization_near_miss'
    else:kind='low_annotation_overlap'
    counts[kind]+=1
   comp.append({'arm':arm,'maxDets':cap,'IoU':t,**dict(counts)})
for name,rr in [('GT_MATCH_TRANSITIONS.csv',trans),('GROUP_MATCH_TRANSITIONS.csv',[{k:json.dumps(v,sort_keys=True) if isinstance(v,dict) else v for k,v in row.items()} for row in group_rows]),('PREDICTION_BUDGET_COMPOSITION.csv',comp)]:
 with (P/'reports'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
summary={'transition_groups':group_rows,'prediction_budget_composition':comp,'all187_GT_accounted':len(rows)==187,'matched_counts_from_original_COCO_score_greedy_assignment':True,'no_threshold_optimization':True}
(P/'reports/TRANSITION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'ALL_transitions':[x for x in group_rows if x['group']=='ALL'],'compositions':comp},indent=2))
