"""Read-only CPU analysis of frozen predictions; no model imports or calls."""
from portable_dev.paths import project_path as _portable_project_path
import contextlib, copy, csv, datetime, hashlib, importlib.metadata, io, json, math, pathlib, re
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT=pathlib.Path(__file__).resolve().parent
RUN=_portable_project_path(__file__, 'research-deepfish-baseline-20261031/recovery-v5-20261031')
ANN=RUN/'payload/data/annotations/dev64.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,data): (ROOT/'reports'/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 if not rows:return
 with (ROOT/'reports'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quiet():return contextlib.redirect_stdout(io.StringIO())
def mean_valid(x):
 x=x[x>=0];return float(x.mean()) if len(x) else None
def iou_matrix(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 ae=a[:,:2]+a[:,2:];be=b[:,:2]+b[:,2:]
 wh=np.maximum(0,np.minimum(ae[:,None,:],be[None,:,:])-np.maximum(a[:,None,:2],b[None,:,:2]))
 inter=wh[:,:,0]*wh[:,:,1];union=a[:,2,None]*a[:,3,None]+b[None,:,2]*b[None,:,3]-inter
 return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def extract(e,area,maxdet):
 ai=e.params.areaRngLbl.index(area);mi=e.params.maxDets.index(maxdet);pr=e.eval['precision'][:,:,0,ai,mi];rc=e.eval['recall'][:,0,ai,mi]
 return {'AP':mean_valid(pr),'AP50':mean_valid(pr[0]),'AP75':mean_valid(pr[5]),'AR':mean_valid(rc)}
def evaluate_subset(gt,dt,img_ids):
 # Do not pass a shortened imgIds list only to accumulate(): pycocotools
 # indexes evalImgs by the shortened list's positions, not original image IDs.
 with quiet():
  e=COCOeval(gt,dt,'bbox');e.params.imgIds=list(img_ids);e.params.maxDets=[100,300,1000];e.evaluate();e.accumulate()
 assert {r['image_id'] for r in e.evalImgs if r is not None}==set(img_ids)
 return e
assert np.allclose(iou_matrix([[0,0,2,2]],[[0,0,2,2],[4,4,1,1],[0,0,1,1]]),[[1,0,.25]])
frozen=json.loads((ROOT/'PLAN_FREEZE.json').read_text());assert sha(ROOT/'ANALYSIS_PLAN.md')==frozen['plan_sha256']
data=json.loads(ANN.read_text());images={x['id']:x for x in data['images']};ids=sorted(images)
groups={i:re.sub(r'_f\d+\.jpg$','',images[i]['file_name']) for i in ids};group_names=sorted(set(groups.values()));assert len(ids)==64 and len(group_names)==8 and len(data['annotations'])==187
gts={i:sorted([a for a in data['annotations'] if a['image_id']==i],key=lambda a:a['id']) for i in ids};gt_by_id={x['id']:x for x in data['annotations']}
area_bounds={'all':(0,1e10),'small':(0,32**2),'medium':(32**2,96**2),'large':(96**2,1e10)}
preds={};byimg={};evals={};global_rank={};dt_lookup={};input_rows=[];overall={};metric_rows=[];iou_rows=[];curve_rows=[];perimg={i:{} for i in ids}
with quiet():gt=COCO(str(ANN))
for arm in 'AB':
 path=RUN/'gpu-run/artifacts'/('evaluate_'+arm+'_predictions.bbox.json');preds[arm]=json.loads(path.read_text());assert len(preds[arm])==19200
 input_rows.append({'arm':arm,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
 byimg[arm]={i:[] for i in ids};dt_lookup[arm]={}
 for n,p in enumerate(preds[arm],1):
  assert p['category_id']==1 and math.isfinite(p['score']) and all(math.isfinite(v) for v in p['bbox']) and min(p['bbox'][2:])>=0
  d=dict(p,dt_id=n);byimg[arm][p['image_id']].append(d);dt_lookup[arm][n]=d
 assert all(len(x)==300 for x in byimg[arm].values())
 ranked=sorted(dt_lookup[arm],key=lambda d:(-dt_lookup[arm][d]['score'],dt_lookup[arm][d]['image_id'],d));global_rank[arm]={d:n+1 for n,d in enumerate(ranked)}
 with quiet():
  dt=gt.loadRes(copy.deepcopy(preds[arm]));e=COCOeval(gt,dt,'bbox');e.params.imgIds=ids;e.params.maxDets=[100,300,1000];e.evaluate();e.accumulate();e.summarize()
 observed=[float(x) for x in re.findall(r'Average (?:Precision|Recall).*?=\s*(-?\d+\.\d+)\s*$',(RUN/'gpu-run/artifacts'/('evaluate_'+arm+'.stdout.log')).read_text(),re.M)]
 assert len(observed)==12 and [float(f'{x:.3f}') for x in e.stats]==observed,(arm,e.stats,observed)
 overall[arm]={'original_12_stats':e.stats.tolist(),'original_log_12_stats':observed,'uniform100':extract(e,'all',100),'uniform300':extract(e,'all',300)};evals[arm]=e
 subsets=[('ALL',ids)]+[(g,[i for i in ids if groups[i]==g]) for g in group_names]
 for group,subids in subsets:
  sub=e if group=='ALL' else evaluate_subset(gt,dt,subids)
  for area,(lo,hi) in area_bounds.items():
   count=sum(lo<=a['area']<=hi for i in subids for a in gts[i]);row={'arm':arm,'group':group,'area':area,'images':len(subids),'GT':count}
   for cap in [100,300]:
    row.update({k+'_'+str(cap):v for k,v in extract(sub,area,cap).items()})
    ai=sub.params.areaRngLbl.index(area);mi=sub.params.maxDets.index(cap)
    for ti,t in enumerate(sub.params.iouThrs):
     pr=sub.eval['precision'][ti,:,0,ai,mi];rc=float(sub.eval['recall'][ti,0,ai,mi]);scores=sub.eval['scores'][ti,:,0,ai,mi]
     iou_rows.append({'arm':arm,'group':group,'area':area,'maxDets':cap,'IoU':float(t),'AP':mean_valid(pr),'recall':None if rc<0 else rc})
     if group=='ALL':
      for ri,recall in enumerate(sub.params.recThrs):
       curve_rows.append({'arm':arm,'area':area,'maxDets':cap,'IoU':float(t),'recall_grid':float(recall),'precision':None if pr[ri]<0 else float(pr[ri]),'score_at_recall':None if scores[ri]<0 else float(scores[ri])})
   metric_rows.append(row)
 for i in ids:
  sub=evaluate_subset(gt,dt,[i]);perimg[i][arm]=extract(sub,'all',100)
dump('INPUTS.json',{'plan_sha256':frozen['plan_sha256'],'annotation_sha256':sha(ANN),'predictions':input_rows,'pycocotools':importlib.metadata.version('pycocotools'),'numpy':np.__version__,'original_metrics_reproduced_to_logged_3_decimals':True,'model_calls':0,'GPU_allocations':0})
dump('OVERALL.json',overall);csvout('SUBGROUP_METRICS_LONG.csv',metric_rows);csvout('ALL_IOU_BINS.csv',iou_rows);csvout('FULL_INTERPOLATED_PR_CURVES.csv',curve_rows)
metric_wide=[]
for group in ['ALL']+group_names:
 for area in area_bounds:
  rows={a:next(x for x in metric_rows if x['arm']==a and x['group']==group and x['area']==area) for a in 'AB'};row={k:rows['A'][k] for k in ['group','area','images','GT']}
  for key in ['AP_100','AP50_100','AP75_100','AR_100','AP_300','AP50_300','AP75_300','AR_300']:
   for a in 'AB':row[a+'_'+key]=rows[a][key]
   row['delta_'+key]=None if rows['A'][key] is None else rows['B'][key]-rows['A'][key]
  metric_wide.append(row)
csvout('ALL_SUBGROUP_DELTAS.csv',metric_wide)
# COCO all-area image matches are computed at the largest budget. Prefixes reproduce caps100/300.
matches={arm:{row['image_id']:row for row in evals[arm].evalImgs if row is not None and row['aRng']==list(area_bounds['all'])} for arm in 'AB'}
raw_curves={'iou_thresholds':np.array(evals['A'].params.iouThrs),'recall_grid':np.array(evals['A'].params.recThrs)}
for arm in 'AB':
 for cap in [100,300]:
  rows=[matches[arm][i] for i in ids];scores=np.concatenate([r['dtScores'][:cap] for r in rows]);order=np.argsort(-scores,kind='mergesort')
  dm=np.concatenate([r['dtMatches'][:,:cap] for r in rows],axis=1)[:,order];ig=np.concatenate([r['dtIgnore'][:,:cap] for r in rows],axis=1)[:,order]
  tp=np.cumsum((dm>0)&~ig,axis=1);fp=np.cumsum((dm==0)&~ig,axis=1);recall=tp/187;precision=tp/(tp+fp+np.spacing(1))
  key=f'{arm}_cap{cap}_';raw_curves[key+'score']=scores[order];raw_curves[key+'dt_id']=np.concatenate([r['dtIds'][:cap] for r in rows])[order]
  raw_curves[key+'TP_cumulative']=tp;raw_curves[key+'FP_cumulative']=fp;raw_curves[key+'precision']=precision;raw_curves[key+'recall']=recall
  for ti in range(10):
   envelope=np.maximum.accumulate(precision[ti,::-1])[::-1];idx=np.searchsorted(recall[ti],evals[arm].params.recThrs,side='left');interp=np.zeros(101);valid=idx<len(envelope);interp[valid]=envelope[idx[valid]]
   expected=next(r['AP'] for r in iou_rows if r['arm']==arm and r['group']=='ALL' and r['area']=='all' and r['maxDets']==cap and abs(r['IoU']-evals[arm].params.iouThrs[ti])<1e-8)
   assert abs(float(interp.mean())-expected)<1e-12
np.savez_compressed(ROOT/'reports/FULL_RAW_PR_STEPS.npz',**raw_curves)
det_rows=[];pair_rows=[];gt_rows=[];error_rows=[];pair_map={arm:{} for arm in 'AB'};iou_gt={};rank_img={};best_gt={}
for i in ids:
 ga=gts[i];gb=[x['bbox'] for x in ga]
 for arm in 'AB':
  ds=byimg[arm][i];q=iou_matrix([d['bbox'] for d in ds],gb);iou_gt[arm,i]=q;ranked=sorted(ds,key=lambda x:(-x['score'],x['dt_id']));rank_img[arm,i]={x['dt_id']:n+1 for n,x in enumerate(ranked)}
  best_gt[arm,i]=[int(np.argmax(q[n])) for n in range(len(ds))]
 cross=iou_matrix([d['bbox'] for d in byimg['A'][i]],[d['bbox'] for d in byimg['B'][i]])
 candidates=np.argwhere(cross>=.5);candidates=sorted(((float(cross[a,b]),int(a),int(b)) for a,b in candidates),key=lambda x:(-x[0],x[1],x[2]));usedA=set();usedB=set()
 for ov,ai,bi in candidates:
  if ai in usedA or bi in usedB:continue
  usedA.add(ai);usedB.add(bi);pa=byimg['A'][i][ai];pb=byimg['B'][i][bi];pid=len(pair_rows)+1
  row={'pair_id':pid,'image_id':i,'group':groups[i],'A_dt_id':pa['dt_id'],'B_dt_id':pb['dt_id'],'cross_arm_iou':ov}
  for arm,idx,p in [('A',ai,pa),('B',bi,pb)]:
   qi=iou_gt[arm,i][idx];best=int(qi.argmax());row.update({arm+'_best_gt_id':ga[best]['id'],arm+'_max_gt_iou':float(qi[best]),arm+'_score':p['score'],arm+'_rank':rank_img[arm,i][p['dt_id']],arm+'_global_rank':global_rank[arm][p['dt_id']]});pair_map[arm][p['dt_id']]=pid
  row.update(delta_gt_iou=row['B_max_gt_iou']-row['A_max_gt_iou'],delta_score=row['B_score']-row['A_score'],delta_rank=row['B_rank']-row['A_rank'],delta_global_rank=row['B_global_rank']-row['A_global_rank']);pair_rows.append(row)
 for arm in 'AB':
  mat=matches[arm][i];pos={d:j for j,d in enumerate(mat['dtIds'])}
  for cap in [100,300]:
   for t,ti in [(.5,0),(.75,5)]:
    dm=mat['dtMatches'][ti,:cap];ignored=mat['dtIgnore'][ti,:cap];matched_gt=set(int(x) for x in dm if x>0)
    er={'arm':arm,'image_id':i,'group':groups[i],'maxDets':cap,'IoU':t,'GT':len(ga),'TP':int(np.sum((dm>0)&~ignored)),'FP':int(np.sum((dm==0)&~ignored)),'FN':len(ga)-len(matched_gt)};error_rows.append(er)
  for j,p in enumerate(byimg[arm][i]):
   q=iou_gt[arm,i][j];k=int(q.argmax());col=pos[p['dt_id']];tp50=bool(mat['dtMatches'][0,col]);tp75=bool(mat['dtMatches'][5,col]);mx=float(q[k]);kind='TP' if tp50 else 'duplicate_candidate' if mx>=.5 else 'localization_near_miss' if mx>=.1 else 'low_annotation_overlap'
   det_rows.append({'arm':arm,'image_id':i,'group':groups[i],'dt_id':p['dt_id'],'rank':rank_img[arm,i][p['dt_id']],'global_rank':global_rank[arm][p['dt_id']],'score':p['score'],'bbox_xywh':json.dumps(p['bbox']),'pair_id':pair_map[arm].get(p['dt_id']),'max_gt_iou':mx,'best_gt_id':ga[k]['id'],'coco_match50_gt':int(mat['dtMatches'][0,col]),'coco_match75_gt':int(mat['dtMatches'][5,col]),'TP50':tp50,'TP75':tp75,'error50_top300':kind})
 for j,g in enumerate(ga):
  row={'image_id':i,'group':groups[i],'gt_id':g['id'],'area':g['area'],'scale':'small' if g['area']<=1024 else 'medium' if g['area']<=9216 else 'large','bbox_xywh':json.dumps(g['bbox'])}
  for arm in 'AB':
   ds=byimg[arm][i];q=iou_gt[arm,i][:,j];best=int(q.argmax());row[arm+'_best_iou']=float(q[best]);row[arm+'_best_iou_dt_id']=ds[best]['dt_id'];row[arm+'_best_iou_rank']=rank_img[arm,i][ds[best]['dt_id']]
   mat=matches[arm][i];gp=mat['gtIds'].index(g['id'])
   for ti,t,label in [(0,.5,'50'),(5,.75,'75')]:
    qual=[d for k,d in enumerate(ds) if q[k]>=t];first=min(qual,key=lambda d:(-d['score'],d['dt_id'])) if qual else None
    row[arm+'_first_rank'+label]=rank_img[arm,i][first['dt_id']] if first else None;row[arm+'_first_global_rank'+label]=global_rank[arm][first['dt_id']] if first else None;row[arm+'_first_score'+label]=first['score'] if first else None
    match=int(mat['gtMatches'][ti,gp]);row[arm+'_coco_match'+label+'_dt_id']=match;row[arm+'_coco_match'+label+'_rank']=rank_img[arm,i][match] if match else None
  row['delta_best_iou']=row['B_best_iou']-row['A_best_iou'];gt_rows.append(row)
csvout('ALL_PREDICTION_DIAGNOSTICS.csv',det_rows);csvout('CROSS_ARM_PAIRS.csv',pair_rows);csvout('GT_DIAGNOSTICS.csv',gt_rows);csvout('IMAGE_ERROR_COUNTS.csv',error_rows)
image_rows=[]
for i in ids:
 row={'image_id':i,'file_name':images[i]['file_name'],'group':groups[i],'GT':len(gts[i]),'A_AP100':perimg[i]['A']['AP'],'B_AP100':perimg[i]['B']['AP']};row['delta_AP100']=row['B_AP100']-row['A_AP100'];image_rows.append(row)
csvout('PER_IMAGE_AP.csv',image_rows)
up=sorted([x for x in image_rows if x['delta_AP100']>1e-12],key=lambda x:(-x['delta_AP100'],x['image_id']))[:2]
down=sorted([x for x in image_rows if x['delta_AP100']< -1e-12],key=lambda x:(x['delta_AP100'],x['image_id']))[:2]
used={x['image_id'] for x in up+down};flat=sorted([x for x in image_rows if x['image_id'] not in used],key=lambda x:(abs(x['delta_AP100']),x['image_id']))[:2]
selected=[dict(x,selection='improvement') for x in up]+[dict(x,selection='regression') for x in down]+[dict(x,selection='unchanged' if abs(x['delta_AP100'])<=1e-12 else 'minimum_change') for x in flat];dump('SELECTED_IMAGES.json',selected)
summary={'overall':overall,'prediction_pair_count':len(pair_rows),'unpaired_per_arm':19200-len(pair_rows),'error_counts':{},'GT_coverage_rank':{},'selected_images':selected}
for arm in 'AB':
 summary['error_counts'][arm]={}
 for cap in [100,300]:
  for t in [.5,.75]:
   subset=[x for x in error_rows if x['arm']==arm and x['maxDets']==cap and x['IoU']==t];summary['error_counts'][arm][f'cap{cap}_IoU{t}']={key:sum(x[key] for x in subset) for key in ['GT','TP','FP','FN']}
 summary['error_counts'][arm]['FP300_at50_composition']={k:sum(x['arm']==arm and x['error50_top300']==k for x in det_rows) for k in ['duplicate_candidate','localization_near_miss','low_annotation_overlap']}
for t,label in [(.5,'50'),(.75,'75')]:
 both=[x for x in gt_rows if x['A_best_iou']>=t and x['B_best_iou']>=t];rchange=[x['B_first_rank'+label]-x['A_first_rank'+label] for x in both];gchange=[x['B_first_global_rank'+label]-x['A_first_global_rank'+label] for x in both]
 summary['GT_coverage_rank'][label]={'geometry_candidate_lost':sum(x['A_best_iou']>=t>x['B_best_iou'] for x in gt_rows),'geometry_candidate_gained':sum(x['A_best_iou']<t<=x['B_best_iou'] for x in gt_rows),'geometry_candidate_both':len(both),'geometry_candidate_neither':sum(x['A_best_iou']<t and x['B_best_iou']<t for x in gt_rows),'both_first_rank_B_worse':sum(x>0 for x in rchange),'both_first_rank_B_better':sum(x<0 for x in rchange),'both_first_rank_equal':sum(x==0 for x in rchange),'median_first_rank_delta':float(np.median(rchange)) if rchange else None,'median_first_global_rank_delta':float(np.median(gchange)) if gchange else None,'qualified_candidate_falls_outside_top100':sum(x['A_first_rank'+label]<=100<x['B_first_rank'+label] for x in both),'qualified_candidate_enters_top100':sum(x['B_first_rank'+label]<=100<x['A_first_rank'+label] for x in both),'global_rank_worsens_despite_nonworse_best_iou':sum(x['B_first_global_rank'+label]>x['A_first_global_rank'+label] and x['B_best_iou']>=x['A_best_iou'] for x in both)}
 for arm in 'AB':
  summary['GT_coverage_rank'][label][arm+'_candidate_exists_but_COCO_unmatched']=sum(x[arm+'_best_iou']>=t and not x[arm+'_coco_match'+label+'_dt_id'] for x in gt_rows)
summary['median_GT_best_iou_delta']=float(np.median([x['delta_best_iou'] for x in gt_rows]));summary['mean_GT_best_iou_delta']=float(np.mean([x['delta_best_iou'] for x in gt_rows]))
near=[x for x in pair_rows if x['A_best_gt_id']==x['B_best_gt_id'] and min(x['A_max_gt_iou'],x['B_max_gt_iou'])>=.5]
summary['spatial_pairs_same_GT_both_IoU50']={'pairs':len(near),'median_delta_gt_iou':float(np.median([x['delta_gt_iou'] for x in near])),'median_delta_score':float(np.median([x['delta_score'] for x in near])),'median_delta_rank':float(np.median([x['delta_rank'] for x in near])),'note':'Prediction-level duplicates are not independent observations.'}
overhang=[]
for a in data['annotations']:
 x,y,w,h=a['bbox'];im=images[a['image_id']]
 if x<0 or y<0 or x+w>im['width'] or y+h>im['height']:overhang.append({'gt_id':a['id'],'image_id':a['image_id'],'bbox':a['bbox'],'area':a['area'],'right_overhang':max(0,x+w-im['width']),'bottom_overhang':max(0,y+h-im['height'])})
summary['existing_annotation_overhangs_retained']=overhang;dump('SUMMARY.json',summary)
assert sha(ANN)==json.loads((ROOT/'reports/INPUTS.json').read_text())['annotation_sha256']
for row in input_rows:assert sha(pathlib.Path(row['path']))==row['sha256']
print(json.dumps({'status':'CPU_ANALYSIS_COMPLETE','pairs':len(pair_rows),'selected':selected,'GT_coverage_rank':summary['GT_coverage_rank']},indent=2))
