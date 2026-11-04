"""GT-assisted maximum matching on two frozen saved-prediction pools. No models."""
from portable_dev.paths import project_path as _portable_project_path
import collections,contextlib,csv,datetime,hashlib,io,json,math,pathlib,resource,time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
RUN=R/'cpu-paired-AM-20261101/evaluation'

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(name,value):(P/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvout(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quiet():return contextlib.redirect_stdout(io.StringIO())

def maximum_matching(edges):
 """Kuhn augmenting paths: one row/one column per selected edge, maximum cardinality."""
 assert edges.ndim==2
 n,m=edges.shape;neighbors=[np.flatnonzero(edges[i]).tolist() for i in range(n)];owner=[-1]*m
 def augment(pred,seen):
  for gt in neighbors[pred]:
   if seen[gt]:continue
   seen[gt]=True
   if owner[gt]<0 or augment(owner[gt],seen):owner[gt]=pred;return True
  return False
 for pred in range(n):augment(pred,[False]*m)
 pairs=[(pred,gt) for gt,pred in enumerate(owner) if pred>=0]
 assert len({x for x,y in pairs})==len(pairs)==len({y for x,y in pairs})
 assert all(bool(edges[x,y]) for x,y in pairs)
 return pairs

def self_test():
 cases=[('one prediction two GT',[[1,1]],1),('two predictions one GT',[[1],[1]],1),
        ('augmenting path avoids greedy trap',[[1,1],[1,0]],2),
        ('no edges',[[0,0],[0,0]],0),('three independent',[[1,0,0],[0,1,0],[0,0,1]],3)]
 rows=[]
 import itertools
 for name,matrix,expected in cases:
  e=np.asarray(matrix,dtype=bool);pairs=maximum_matching(e);assert len(pairs)==expected
  permutations=0
  for rr in itertools.permutations(range(e.shape[0])):
   for cc in itertools.permutations(range(e.shape[1])):
    assert len(maximum_matching(e[np.ix_(rr,cc)]))==expected;permutations+=1
  rows.append({'case':name,'matrix':matrix,'expected_maximum':expected,'pairs':pairs,'row_column_permutations_checked':permutations})
 for shape in [(0,2),(2,0),(0,0)]:assert not maximum_matching(np.zeros(shape,dtype=bool))
 write('SYNTHETIC_TESTS.json',{'status':'PASS','cases':rows,'empty_graph_cases':3,'note':'Maximum count/recall is permutation invariant; a particular optimal pairing need not be unique.'})

def main():
 began=time.monotonic();freeze=json.loads((P/'FREEZE.json').read_text())
 assert sha(P/'PROTOCOL.md')==freeze['protocol_sha256'];deadline=datetime.datetime.fromisoformat(freeze['deadline_UTC']).timestamp()
 assert time.time()<deadline
 self_test()
 run=json.loads((RUN/'RUN_PROTOCOL.json').read_text());ann=pathlib.Path(run['dev_annotation']['path']);assert sha(ann)==run['dev_annotation']['sha256']
 data=json.loads(ann.read_text());images={i['id']:i for i in data['images']};ids=sorted(images);anns=data['annotations']
 import re
 prefix={i:re.sub(r'_f\d+\.jpg$','',images[i]['file_name']) for i in ids};groups=sorted(set(prefix.values()))
 assert len(ids)==64 and len(anns)==187 and len(groups)==8
 state={'GT_count':len(anns),'images':len(ids),'categories':data['categories'],
        'crowd_values':dict(collections.Counter(str(a.get('iscrowd',0)) for a in anns)),
        'ignore_values':dict(collections.Counter(str(a.get('ignore','ABSENT')) for a in anns)),
        'all_GT_in_original_all_area_range':all(0<=a['area']<=1e10 for a in anns),
        'annotation_sha256':sha(ann)}
 write('DATA_RULES.json',state)
 if not(all(a.get('iscrowd',0)==0 and a.get('ignore',0)==0 and a['category_id']==1 for a in anns) and state['all_GT_in_original_all_area_range']):
  raise RuntimeError('Crowd/ignore/category/area state does not permit ordinary one-to-one graph matching')
 with quiet():gt=COCO(str(ann))
 inputs=[];per_image=[];pair_receipts=[];original={};actual_ignore=[]
 for arm in ['A','M']:
  predictions=[]
  for iid in run['image_order']:
   path=RUN/'predictions'/f'20261031_{arm}_c100_{iid}.json';envelope=json.loads(path.read_text());payload=envelope['payload']
   raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
   assert hashlib.sha256(raw).hexdigest()==envelope['payload_sha256']
   assert payload['image_id']==iid and payload['arm']==arm and len(payload['predictions'])==300
   assert all(p['category_id']==1 and p['image_id']==iid for p in payload['predictions'])
   predictions.extend(payload['predictions']);inputs.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
  with quiet():
   dt=gt.loadRes(predictions);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.params.catIds=[1];ev.params.maxDets=[100,300,1000]
   ev.params.iouThrs=np.linspace(.5,.95,10,endpoint=True);ev.evaluate();ev.accumulate();ev.summarize()
  saved=json.loads((RUN/'metrics'/f'20261031_{arm}_c100.json').read_text())
  assert np.allclose(ev.stats,saved['stats'],atol=1e-12,rtol=0)
  original[arm]=ev.stats.tolist()
  for row in ev.evalImgs:
   if row is None or row['aRng']!=[0,1e10]:continue
   iid=int(row['image_id']);assert row['maxDet']==1000 and len(row['dtIds'])==300
   assert not np.any(row['gtIgnore']) and not np.any(row['dtIgnore'])
   assert row['gtIds']==[a['id'] for a in ev._gts[(iid,1)]], 'Ignore-order adjustment needed'
   ious=np.asarray(ev.ious[(iid,1)]);assert ious.shape==(300,len(row['gtIds']))
   actual_ignore.append({'arm':arm,'image_id':iid,'GT_ignore_count':int(np.sum(row['gtIgnore'])),'DT_ignore_count_all_thresholds':int(np.sum(row['dtIgnore']))})
   for pool,cap,native_cap in [('native_top100',100,100),('saved_full300',300,1000)]:
    for ti,threshold in enumerate(ev.params.iouThrs):
     assert time.time()<deadline,'20-minute diagnostic deadline'
     edges=ious[:cap]>=threshold;pairs=maximum_matching(edges);upper=len(pairs)
     assert len(maximum_matching(edges[::-1,::-1]))==upper,'Permutation changed maximum count'
     native=int(np.count_nonzero(row['dtMatches'][ti,:cap]))
     assert native<=upper<=min(cap,len(row['gtIds']))
     entry={'arm':arm,'image_id':iid,'prefix':prefix[iid],'pool':pool,'saved_candidates_used':cap,'native_eval_maxDets':native_cap,'IoU':float(threshold),'GT':len(row['gtIds']),'native_matches':native,'maximum_one_to_one_matches':upper,'upper_minus_native':upper-native}
     per_image.append(entry)
     pair_receipts.append({**entry,'matching_pairs':[{'prediction_DT_id':int(row['dtIds'][d]),'GT_id':int(row['gtIds'][g]),'IoU':float(ious[d,g])} for d,g in pairs]})
 aggregated=[]
 for arm in ['A','M']:
  for group in ['ALL']+groups:
   for pool,cap,native_cap in [('native_top100',100,100),('saved_full300',300,1000)]:
    for threshold in np.linspace(.5,.95,10,endpoint=True):
     sub=[x for x in per_image if x['arm']==arm and x['pool']==pool and x['IoU']==float(threshold) and (group=='ALL' or x['prefix']==group)]
     ng=sum(x['GT'] for x in sub);n=sum(x['native_matches'] for x in sub);u=sum(x['maximum_one_to_one_matches'] for x in sub)
     aggregated.append({'arm':arm,'prefix':group,'images':len(sub),'GT':ng,'pool':pool,'saved_candidates_per_image':cap,'native_eval_maxDets':native_cap,'IoU':float(threshold),'native_matches':n,'maximum_matches':u,'native_recall':n/ng,'GT_assisted_matching_recall_upper':u/ng,'upper_minus_native_matches':u-n})
 # Reconcile original AR100 and AR1000 from the native ten-threshold counts.
 for arm in ['A','M']:
  for pool,index in [('native_top100',6),('saved_full300',8)]:
   value=np.mean([x['native_recall'] for x in aggregated if x['arm']==arm and x['prefix']=='ALL' and x['pool']==pool])
   assert abs(value-original[arm][index])<1e-12
 wide=[]
 for group in ['ALL']+groups:
  for pool,cap,native_cap in [('native_top100',100,100),('saved_full300',300,1000)]:
   for threshold in np.linspace(.5,.95,10,endpoint=True):
    both={a:next(x for x in aggregated if x['arm']==a and x['prefix']==group and x['pool']==pool and x['IoU']==float(threshold)) for a in ['A','M']}
    a,m=both['A'],both['M'];wide.append({'prefix':group,'images':a['images'],'GT':a['GT'],'pool':pool,'saved_candidates_per_image':cap,'native_eval_maxDets':native_cap,'IoU':float(threshold),'A_native':a['native_matches'],'A_maximum':a['maximum_matches'],'M_native':m['native_matches'],'M_maximum':m['maximum_matches'],'M_minus_A_native':m['native_matches']-a['native_matches'],'M_minus_A_upper_matches':m['maximum_matches']-a['maximum_matches'],'A_upper_recall_percent':100*a['GT_assisted_matching_recall_upper'],'M_upper_recall_percent':100*m['GT_assisted_matching_recall_upper']})
 truncation=[]
 for group in ['ALL']+groups:
  for threshold in np.linspace(.5,.95,10,endpoint=True):
   full=next(x for x in wide if x['prefix']==group and x['pool']=='saved_full300' and x['IoU']==float(threshold));top=next(x for x in wide if x['prefix']==group and x['pool']=='native_top100' and x['IoU']==float(threshold))
   truncation.append({'prefix':group,'GT':full['GT'],'IoU':float(threshold),'A_full300_minus_top100_maximum':full['A_maximum']-top['A_maximum'],'M_full300_minus_top100_maximum':full['M_maximum']-top['M_maximum']})
 csvout('PER_IMAGE.csv',per_image);csvout('AGGREGATED_LONG.csv',aggregated);csvout('ALL_PREFIX_ALL_THRESHOLDS.csv',wide);csvout('TRUNCATION_GAPS.csv',truncation)
 write('MATCHING_RECEIPTS.json',pair_receipts);write('ACTUAL_COCO_IGNORE_CHECKS.json',actual_ignore);write('INPUTS.json',inputs)
 write('SUMMARY.json',{'overall':[x for x in wide if x['prefix']=='ALL'],'overall_truncation':[x for x in truncation if x['prefix']=='ALL'],'original_native12_metrics':original,'native_matches_already_maximal_for_all_graphs':all(x['upper_minus_native']==0 for x in per_image),'nonzero_upper_native_gap_cases':sum(x['upper_minus_native']!=0 for x in per_image),'note':'Uses GT, independent matching at each IoU; not a deployable re-ranking or attainable AP upper bound.'})
 for path,pin in [(ann,run['dev_annotation']['sha256'])]+[(pathlib.Path(x['path']),x['sha256']) for x in inputs]:assert sha(path)==pin
 count=sum(f.stat().st_size for f in P.rglob('*') if f.is_file());total=sum(f.stat().st_size for f in R.rglob('*') if f.is_file())+count
 assert count<30*1024**2 and total<8*1024**3-80*1024**2
 write('VERIFICATION.json',{'status':'PASS','seconds':time.monotonic()-began,'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'graph_cases':len(per_image),'all_graphs_reverse_permutation_invariant':True,'single_prediction_single_GT_constraints_verified_all_matchings':True,'original_metrics_and_native_AR100_AR1000_reproduced':True,'all_actual_GT_DT_ignore_false':True,'all_sources_unchanged':True,'model_calls_installs_uploads_CU':0,'new_bytes':count,'research_plus_new_bytes':total})
 print(json.dumps({'status':'PASS','seconds':time.monotonic()-began,'overall':[x for x in wide if x['prefix']=='ALL']},indent=2))

if __name__=='__main__':main()
