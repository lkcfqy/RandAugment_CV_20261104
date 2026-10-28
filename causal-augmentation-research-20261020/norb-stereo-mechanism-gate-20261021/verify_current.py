"""Verify saved arithmetic and unchanged historical files, without encoding/fitting."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,numpy as np
OUT=Path(__file__).resolve().parent;OLD=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/norb-view-transfer-pilot-20261021')
def read(n):return json.loads((OUT/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=1e-9,atol=1e-10),(a,b)
for name in ['crop-lock.json','fill-lock.json','fill-training-freeze.json']:
 for f,h in read(name)['files'].items():assert sha(OUT/f)==h,(name,f)
receipt=json.loads((OLD/'delivery-receipt.json').read_text())
for r in receipt['files']:assert (OLD/r['file']).stat().st_size==r['bytes'] and sha(OLD/r['file'])==r['destination_SHA256'],r['file']
assert {str(f.relative_to(OLD)) for f in OLD.rglob('*') if f.is_file()}=={r['file'] for r in receipt['files']}|{'delivery-receipt.json'}
crop=read('crop-result.json');cr=read('crop-per-photo.json')['rows'];groups=defaultdict(list)
assert len(cr)==1080 and len({(r['row'],r['camera']) for r in cr})==1080
for r in cr:groups[(r['category'],r['instance'])].append(r)
assert len(groups)==15 and set(map(len,groups.values()))=={72}
for q in [16,32,64]:
 for a in ['X','Y']:
  for m in ['proxy_share_lost','image_area_proxy_lost','continuous_contrast_energy_share_lost']:
   val=np.mean([np.mean([r['thresholds'][str(q)][a][m] for r in values]) for values in groups.values()]);close(val,crop['threshold_summary'][str(q)][a+'_'+m])
 assert sum(r['thresholds'][str(q)]['empty_proxy'] for r in cr)==crop['empty_proxy_counts'][str(q)]
p=read('fill-protocol.json');r=read('fill-result.json');bank=np.load(OUT/'fill-feature-bank.npz');wts=np.load(OUT/'fill-readouts.npz');oldbank=np.load(OLD/'source-feature-bank.npz');z0=oldbank['clean'];ys=np.array([x['category'] for x in p['source_records']]);oldp=json.loads((OLD/'protocol.json').read_text());ev=np.load(OLD/'evaluation-features.npz');oldw=np.load(OLD/'readouts.npz');saved=np.load(OUT/'fill-development-logits.npz');objloss=np.load(OUT/'fill-development-per-object-losses.npz');train=read('fill-training-freeze.json')['source_training'];maxgrad=0.;maxlogit=0.
def loss(lg,y):return np.logaddexp.reduce(lg,axis=1)-lg[np.arange(len(y)),y]
for a in ['X','Y']:
 x=np.concatenate([z0,bank[a+'plus'],bank[a+'minus']]);y=np.tile(ys,3);w,b=wts[a+'_W'],wts[a+'_b'];lg=x@w+b
 value=loss(lg,y).mean()+.005*np.sum(w*w);pr=np.exp(lg-lg.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True);pr[np.arange(len(y)),y]-=1;g=max(np.max(abs(x.T@pr/len(y)+.01*w)),np.max(abs(pr.mean(0))));assert g<=1e-5;maxgrad=max(maxgrad,float(g));close(value,train[a]['objective']);close(g,train[a]['gradient_inf'])
for role in p['evaluation_roles']:
 rows=oldp['roles'][role];z=ev[role];y=np.array([x['category'] for x in rows]);objects=sorted({(x['category'],x['instance']) for x in rows});assert len(objects)==10
 for a in ['X','Y']:
  for mode in ['constant128','matched_gray']:
   w,b=(oldw['translate'+a+'_09_W'],oldw['translate'+a+'_09_b']) if mode=='constant128' else (wts[a+'_W'],wts[a+'_b']);lg=z@w+b;maxlogit=max(maxlogit,float(np.max(abs(lg-saved[role+'__'+a+'_'+mode]))));ll=loss(lg,y);close(ll.mean(),r['results'][role][a+'_'+mode]['LL']);close(np.mean(lg.argmax(1)==y),r['results'][role][a+'_'+mode]['accuracy']);per=[]
   for obj in objects:
    mask=np.array([(x['category'],x['instance'])==obj for x in rows]);assert mask.sum()==36;per.append(ll[mask].mean())
   close(per,objloss[role+'__'+a+'_'+mode]);close(np.mean(per),ll.mean())
 v=r['results'][role];inter=v['interaction'];D0=v['Y_constant128']['LL']-v['X_constant128']['LL'];D1=v['Y_matched_gray']['LL']-v['X_matched_gray']['LL'];dY=v['Y_constant128']['LL']-v['Y_matched_gray']['LL'];dX=v['X_constant128']['LL']-v['X_matched_gray']['LL'];close(D0,inter['original_Y_minus_X']);close(D1,inter['matched_Y_minus_X']);close(dY,inter['Y_loss_reduction']);close(dX,inter['X_loss_reduction']);close(dY-dX,inter['gap_reduction'])
v=r['results']['development_novel']['interaction'];passed=abs(v['matched_Y_minus_X'])<=.5*p['original_development_novel_gap'] and v['Y_loss_reduction']>0;assert bool(passed)==r['predeclared_explanatory_screen_passed'];assert all(n.startswith(('development_seen__','development_novel__')) for n in saved.files)
assert r['pixel_control']['total_changed_pixels_over2160']==2160*224*9
assert datetime.fromisoformat(read('fill-lock.json')['at_UTC'])<datetime.fromisoformat(read('fill-training-freeze.json')['at_UTC'])<datetime.fromisoformat(r['at_UTC'])
assert r['code_SHA256']==sha(OUT/'fill_probe.py') and r['protocol_SHA256']==sha(OUT/'fill-protocol.json')
audit={'at_UTC':datetime.now(timezone.utc).isoformat(),'status':'passed','no_encoder_or_fit_or_new_policy_selection':True,'original_pilot_files_unchanged':len(receipt['files']),'original_pilot_no_extra_files':True,'all_protocol_and_freeze_hashes_match':True,'source_crop_summary_and_object_unit_match':True,'source_physical_objects':15,'development_physical_objects':10,'camera_views_not_independent_units':True,'recomputed_max_gradient_inf':maxgrad,'saved_max_logits_difference':maxlogit,'all_object_losses_and_primary_interactions_match':True,'prespecified_absolute_gap_screen_match':True,'pixel_changed_count_matches_exact_2160by9by224':True,'new_final25_inference':False,'independent_confirmation':False,'no_original_contribution_established':True}
with (OUT/'verification-current.json').open('x') as f:json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(audit))
