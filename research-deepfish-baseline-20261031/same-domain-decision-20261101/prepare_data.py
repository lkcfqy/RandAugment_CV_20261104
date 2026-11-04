"""Freeze training-only data trajectories and bounded probe identities, no detector."""
from portable_dev.paths import project_path as _portable_project_path
import os,pathlib,json,copy,random,hashlib,importlib.util,collections,time
P=pathlib.Path(__file__).resolve().parent
R=_portable_project_path(__file__, 'research-deepfish-baseline-20261031')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import torch,numpy as np,cv2
from mmdet.datasets import CocoDataset
from mmengine.dataset import DefaultSampler
from mmdet.utils import register_all_modules
torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(0);register_all_modules(init_default_scope=True)
spec=importlib.util.spec_from_file_location('observer',R/'ets-pipeline-dryrun-20261101/dryrun.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
proto=read(P/'PROTOCOL.json');probes={};led=read(P/'C0_FROZEN_SOURCE_LEDGER.json');summary={}
for arm in ['A1','M1','C0']:
 random.seed(20261031);np.random.seed(20261031);torch.random.default_generator.manual_seed(20261031)
 data=P/'data' if arm!='C0' else R/'recovery-v5-20261031/payload/data';annfile='annotations/support1.json' if arm!='C0' else 'annotations/support5.json';shot=read(data/annfile);aid={a['image_id']:a['id'] for a in shot['annotations']}
 ds=CocoDataset(ann_file=annfile,data_root=str(data)+'/',data_prefix={'img':'images/'},metainfo={'classes':('fish',)},filter_cfg={'filter_empty_gt':False,'min_size':0},return_classes=True,pipeline=[]);index={ds.get_data_info(i)['img_id']:i for i in range(5)}
 sampler=DefaultSampler(ds,shuffle=True,seed=20261031,round_up=False);order=[]
 for ep in range(40):sampler.set_epoch(ep);order.extend(list(sampler))
 groups=[r['source_image_ids'] for r in led] if arm=='C0' else [[ds.get_data_info(i)['img_id']] for i in order]
 obs=o.Observer();pipe=o.make_pipeline('M' if arm=='M1' else 'A',obs);out=P/f'data_reference_{arm}';out.mkdir();rows=[];occ=0;total=0
 for u,ids in enumerate(groups,1):
  sources=[]
  for j,iid in enumerate(ids):
   occ+=1;info=ds.get_data_info(index[iid]);obs.begin(occ,iid,aid[iid],(info['height'],info['width']));before=o.rng();t=time.monotonic();item=pipe(info);elapsed=time.monotonic()-t;sig=o.signature(item)
   row={'source_image_id':iid,'annotation_id':aid[iid],'RNG_before':before,'signature':sig,'events':copy.deepcopy(obs.events),'final_source_regions':copy.deepcopy(obs.sources),'final_GT_lineage':copy.deepcopy(obs.boxes),'pipeline_seconds':elapsed};sources.append(row);total+=elapsed
  row={'update':u,'k':len(ids),'source_image_ids':ids,'sources':sources};write(out/f'{u:03}.json',row);rows.append(row)
 # Preflight selection sees training geometry/GT only, with no model/loss/dev.
 def positive(s):return bool(s['signature']['gt']['gt_instances']['boxes'])
 def area(s):a=s['signature']['shape_CHW'];return a[1]*a[2]
 if arm=='C0':
  maxk=max(r['k'] for r in rows);candidates=[r for r in rows if r['k']==maxk and any(positive(s) for s in r['sources'])];chosen=max(candidates,key=lambda r:(max(area(s) for s in r['sources'] if positive(s)),-r['update']));probes[arm]=[chosen['update']]
 else:
  pool=[r for r in rows if positive(r['sources'][0]) and (arm=='A1' or r['update']>40)];chosen=max(pool,key=lambda r:(area(r['sources'][0]),-r['update']));probes[arm]=[chosen['update']]
  if arm=='M1':
   mixed=[r for r in pool if all(any(e['type']==op and e.get('applied',False) for e in r['sources'][0]['events']) for op in ['CachedMosaic','CachedMixUp'])]
   if mixed:probes[arm]=sorted(set(probes[arm]+[mixed[0]['update']]))
 summary[arm]={'updates':len(rows),'individual_A_pipeline_calls' if arm=='C0' else 'anchor_pipeline_calls':occ,'pipeline_seconds':total,'anchor_order':[r['source_image_ids'][0] for r in rows],'maximum_input_area':max(area(s) for r in rows for s in r['sources']),'selected_preflight_updates':probes[arm],'selected_preflight_sources':sum(rows[i-1]['k'] for i in probes[arm]),'selected_source_occurrences':sum(len(s['final_source_regions']) for r in rows for s in r['sources'])}
 write(P/f'DATA_REFERENCE_MANIFEST_{arm}.json',[{'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.json'))])
 assert occ==(659 if arm=='C0' else 200)
proto['preflight_iterations']=probes;proto['support_sha256']['new']=hashlib.sha256((P/'data/annotations/support1.json').read_bytes()).hexdigest();proto['status']='FROZEN_READY_FOR_ENGINEERING_TESTS_AND_BOUNDED_PREFLIGHT';write(P/'PROTOCOL.json',proto);write(P/'DATA_PREPARATION.json',summary)
print(json.dumps({a:{k:v for k,v in s.items() if k!='anchor_order'} for a,s in summary.items()},indent=2))
