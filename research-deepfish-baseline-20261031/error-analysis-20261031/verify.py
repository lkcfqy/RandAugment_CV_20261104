"""Check artifact accounting and recompute PR from saved TP/FP steps, CPU only."""
from portable_dev.paths import project_path as _portable_project_path
import csv,hashlib,json,pathlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;R=ROOT/'reports'
RUN=_portable_project_path(__file__, 'research-deepfish-baseline-20261031/recovery-v5-20261031')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return list(csv.DictReader((R/name).open()))
checks=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
ann=json.loads((RUN/'payload/data/annotations/dev64.json').read_text());s=json.loads((R/'SUMMARY.json').read_text())
pins={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (RUN/'payload/SHA256SUMS').read_text().splitlines()}
paths=['data/annotations/dev64.json']+['data/images/'+i['file_name'] for i in ann['images']]
check('dev JSON and all64 original JPEG hashes match frozen payload',all(sha(RUN/'payload'/p)==pins[p] for p in paths))
expected={'A':'7644220c334bb6888a1d6e7bb6c94566753bbf4f1d7fdf59acfa327a0a296de6','B':'e2449f4116bac8c0343e25cfa149a05d0cad5e2a26e23eee3addaead7f838fcd'}
check('both original prediction SHA256 pins match',all(sha(RUN/'gpu-run/artifacts'/f'evaluate_{a}_predictions.bbox.json')==v for a,v in expected.items()))
check('analysis selection plan hash unchanged',sha(ROOT/'ANALYSIS_PLAN.md')==json.loads((ROOT/'PLAN_FREEZE.json').read_text())['plan_sha256'])
gd=rows('GT_DIAGNOSTICS.csv');dd=rows('ALL_PREDICTION_DIAGNOSTICS.csv');pp=rows('CROSS_ARM_PAIRS.csv');wide=rows('ALL_SUBGROUP_DELTAS.csv');imgs=rows('PER_IMAGE_AP.csv');bins=rows('ALL_IOU_BINS.csv');curves=rows('FULL_INTERPOLATED_PR_CURVES.csv')
check('all187 GT covered once',len(gd)==187 and {int(g['gt_id']) for g in gd}=={g['id'] for g in ann['annotations']} and len({g['gt_id'] for g in gd})==187)
check('all38400 predictions covered once',len(dd)==38400 and len({(d['arm'],d['dt_id']) for d in dd})==38400)
check('all64 images covered once',len(imgs)==64 and {int(i['image_id']) for i in imgs}=={i['id'] for i in ann['images']})
check('full9x4 group-scale matrix retained',len(wide)==36 and sum(int(r['GT']) for r in wide if r['group']!='ALL' and r['area']=='all')==187 and all(int(r['images'])==8 for r in wide if r['group']!='ALL'))
check('both caps, arms, 10 IoU bins for all group-scales',len(bins)==1440)
check('all101-point curves for2armsx2capsx4scalesx10IoUs',len(curves)==16160)
check('cross-arm greedy pairs one-to-one and threshold valid',len(pp)==8248 and len({r['A_dt_id'] for r in pp})==len(pp) and len({r['B_dt_id'] for r in pp})==len(pp) and all(float(r['cross_arm_iou'])>=.5 for r in pp))
for a in 'AB':
 check(a+' original12 stats match saved logs',all(round(x,3)==y for x,y in zip(s['overall'][a]['original_12_stats'],s['overall'][a]['original_log_12_stats'])))
 for cap in (100,300):
  for t in (.5,.75):
   ec=s['error_counts'][a][f'cap{cap}_IoU{t}'];check(f'{a} cap{cap} IoU{t} TP+FN=187, TP+FP=64xcap',ec['TP']+ec['FN']==187 and ec['TP']+ec['FP']==64*cap)
 check(a+' FP categories exhaust all top300 FPs',sum(s['error_counts'][a]['FP300_at50_composition'].values())==s['error_counts'][a]['cap300_IoU0.5']['FP'])
npz=np.load(R/'FULL_RAW_PR_STEPS.npz',allow_pickle=False)
for a in 'AB':
 for cap in (100,300):
  key=f'{a}_cap{cap}_';pr=npz[key+'precision'];rc=npz[key+'recall'];check(key+'raw steps cover every selected prediction',pr.shape==(10,64*cap) and rc.shape==pr.shape)
  for ti,t in enumerate(npz['iou_thresholds']):
   envelope=np.maximum.accumulate(pr[ti,::-1])[::-1];at=np.searchsorted(rc[ti],npz['recall_grid'],side='left');q=np.zeros(101);good=at<len(envelope);q[good]=envelope[at[good]]
   saved=[r for r in curves if r['arm']==a and r['area']=='all' and int(r['maxDets'])==cap and abs(float(r['IoU'])-t)<1e-9]
   check(f'{key}IoU{t:.2f} raw steps reproduce entire101point curve',np.allclose(q,[float(r['precision']) for r in saved],atol=1e-12,rtol=0))
sel=json.loads((R/'SELECTED_IMAGES.json').read_text());ups=sorted([r for r in imgs if float(r['delta_AP100'])>1e-12],key=lambda r:(-float(r['delta_AP100']),int(r['image_id'])))[:2];downs=sorted([r for r in imgs if float(r['delta_AP100'])< -1e-12],key=lambda r:(float(r['delta_AP100']),int(r['image_id'])))[:2];used={r['image_id'] for r in ups+downs};flat=sorted([r for r in imgs if r['image_id'] not in used],key=lambda r:(abs(float(r['delta_AP100'])),int(r['image_id'])))[:2]
check('six visual examples obey frozen deterministic selection', [r['image_id'] for r in sel]==[int(r['image_id']) for r in ups+downs+flat])
om=json.loads((R/'OVERLAY_MANIFEST.json').read_text());check('six overlays hashes and source image hashes unchanged',len(om)==6 and all(sha(ROOT/'overlays'/r['overlay'])==r['sha256'] and sha(pathlib.Path(r['source']))==r['source_sha256'] for r in om))
check('retained annotation4195/4660 overhang and original area',[(x['gt_id'],x['area'],x['right_overhang']) for x in s['existing_annotation_overhangs_retained']]==[(4195,6201.0,.5),(4660,11703.0,.5)])
out={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'check_count':len(checks),'new_model_calls':0,'new_GPU_allocations':0,'test_access_in_this_analysis':False}
(R/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks)}))
