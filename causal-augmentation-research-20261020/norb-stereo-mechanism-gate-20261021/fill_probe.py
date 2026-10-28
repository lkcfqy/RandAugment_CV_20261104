"""Two fixed source-trained readouts, changing only translation fill pixels."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,importlib.util,json,struct,time,argparse
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parent
OLD=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/norb-view-transfer-pilot-20261021')
def now():return datetime.now(timezone.utc).isoformat()
def read(n):return json.loads((OUT/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);args=parser.parse_args()
if args.phase=='prepare':
 assert read('fill-permission-decision.json')['allow_one_implementation_diagnostic']
 source=json.loads((OLD/'protocol.json').read_text());rs=read('crop-per-photo.json')['rows'];refs=[r['reference_gray'] for r in rs if r['camera']==0];fill=int(np.floor(np.median(refs)+.5));assert fill!=128
 rr=json.loads((OLD/'final-results.json').read_text());d0=rr['risk_table']['development_novel']['translateY_09']['LL']-rr['risk_table']['development_novel']['translateX_09']['LL'];assert d0>0
 p={'at_UTC':now(),'question':'Can augmentation fill account for at least half the previously observed Y9-X9 development-novel loss gap?','status':'posthoc_hypothesis_from_old_development_and_test; source-only new fits; no independent confirmation','only_treatment':'translation fill128 versus one source-calibrated gray; original224RGB content, retained pixel positions and clipping identical','constant_original_fill':128,'replacement_fill':fill,'replacement_rule':'round(median of original source-camera0 fourcorner gray references), labels/outcomes not used for this value','fixed_axes':['X','Y'],'translation_pixels':9,'signed_law':'identity,plus,minus each1/3','source_records':source['roles']['source_seen'],'source_objects':source['source_objects'],'evaluation_roles':['development_seen','development_novel'],'development_objects':source['development_objects'],'no_new_final25_inference_or_scoring':True,'encoder_preprocess_readout_normalizer':'reuse original fixed encoder, exact preprocessing, source scalar transform, CE+lambda=.01 and optimizer; no baseline/constant-arm refits','original_development_novel_gap':d0,'primary':'I=(oldY-oldX)-(neutralY-neutralX), equal mean over10 physical development objects','direction':'I>0','implementation_explanation_screen':'abs(neutralY_loss-neutralX_loss)<=0.5*old_development_novel_gap AND neutralY_loss<oldY_loss. I remains the signed interaction; report X absolute change and both angle regimes. Passing does not mean Y improvement accounts for half, nor that matched fill is better overall.','no_after_result_retuning':True,'intervals':'none; finite development10-object descriptive effects, not independently confirmed','source_training_convergence':'gradient infinity<=1e-5 finite; do not tune if fails','CPU_threads':1,'encoder_seconds_hard_cap':180,'new_augmented_images_encoded':2160,'new_readouts':2,'new_download_install_GPU_ARIS_A100_automation_PPT_8A':False,'source_pixel_necessary_condition_result_SHA256':sha(OUT/'crop-result.json'),'original_readouts_SHA256':sha(OLD/'readouts.npz'),'original_source_features_SHA256':sha(OLD/'source-feature-bank.npz'),'original_evaluation_features_SHA256':sha(OLD/'evaluation-features.npz'),'original_normalizer_SHA256':sha(OLD/'normalizer.npz'),'original_code_SHA256':sha(OLD/'experiment.py'),'original_training_archive_SHA256':json.loads((OLD/'training-download.json').read_text())['sha256'],'old_source_readout_fits_frozen_not_overwritten':True,'causal_limit':'Identifies this deterministic fill treatment within a fixed training pipeline; does not identify CNN internal mechanism, real stereo geometry, natural-world causal effect, general augmentation utility or originality.'}
 save('fill-protocol.json',p);save('fill-lock.json',{'at_UTC':now(),'files':{n:sha(OUT/n) for n in ['fill-protocol.json','fill_probe.py','fill-permission-decision.json','fill-mechanism-prior-gate.json']},'new_neutral_fill_features_or_models_at_lock':0});print(json.dumps({'status':'fill_protocol_locked','replacement_fill':fill,'original_development_gap':d0,'half_gap':d0/2,'new_readouts':2,'new_final_inference':False}))
else:
 for n,h in read('fill-lock.json')['files'].items():assert sha(OUT/n)==h
 start=time.perf_counter();p=read('fill-protocol.json')
 for field,name in [('original_code_SHA256','experiment.py'),('original_readouts_SHA256','readouts.npz'),('original_source_features_SHA256','source-feature-bank.npz'),('original_evaluation_features_SHA256','evaluation-features.npz'),('original_normalizer_SHA256','normalizer.npz')]:assert sha(OLD/name)==p[field]
 spec=importlib.util.spec_from_file_location('original_locked_helpers',OLD/'experiment.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
 import torch
 from torchvision.transforms import functional as TF,InterpolationMode
 model,versions=original.get_encoder();norm=np.load(OLD/'normalizer.npz');mu,scale=norm['mean'],float(norm['scale'])
 labels=np.array([r['category'] for r in p['source_records']]);oldbank=np.load(OLD/'source-feature-bank.npz');z0=oldbank['clean'];images={};requested={r['row'] for r in p['source_records']};raw=OLD/'data'/'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz';assert sha(raw)==p['original_training_archive_SHA256']
 with gzip.open(raw,'rb') as f:
  assert struct.unpack('<6i',f.read(24))==(0x1e3d4c55,4,24300,2,96,96)
  for j in range(24300):
   data=f.read(18432);assert len(data)==18432
   if j in requested:images[j]=np.frombuffer(data,dtype=np.uint8,count=9216).reshape(96,96).copy()
  assert f.read(1)==b''
 imgs=[Image.fromarray(images[r['row']]).convert('RGB').resize((224,224),Image.Resampling.BILINEAR) for r in p['source_records']]
 mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1);std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
 bank={};seconds=0.;pixel_change_max_interior=0;changed_pixels_total=0
 for axis in ['X','Y']:
  for sign in [1,-1]:
   begin=time.perf_counter();vectors=[]
   for offset in range(0,len(imgs),32):
    batch=[]
    for im in imgs[offset:offset+32]:
     translate=[sign*9,0] if axis=='X' else [0,sign*9];neutral=TF.affine(im,angle=0.,translate=translate,scale=1.,shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=[p['replacement_fill']]*3)
     op={'family':'translate'+axis,'magnitude':9};constant=original.transform(im,op,sign);aa=np.asarray(neutral);bb=np.asarray(constant);delta=np.any(aa!=bb,axis=2);pad=np.zeros((224,224),bool)
     if axis=='X':pad[:,:9 if sign>0 else 0]=sign>0;pad[:,-9:]=sign<0
     else:pad[:9 if sign>0 else 0,:]=sign>0;pad[-9:,:]=sign<0
     assert np.array_equal(delta,pad);assert np.all(aa[~pad]==bb[~pad]);assert np.all(aa[pad]==p['replacement_fill']) and np.all(bb[pad]==128)
     changed_pixels_total+=int(delta.sum());batch.append(np.asarray(neutral,dtype=np.float32).transpose(2,0,1)/255.)
    with torch.inference_mode():v=model((torch.from_numpy(np.stack(batch))-mean)/std).numpy().astype(np.float64)
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);assert np.isfinite(v).all();vectors.append((v-mu)/scale)
    assert seconds+time.perf_counter()-begin<p['encoder_seconds_hard_cap']
   seconds+=time.perf_counter()-begin;bank[axis+('plus' if sign==1 else 'minus')]=np.concatenate(vectors);print(json.dumps({'at_UTC':now(),'encoded':axis+str(sign),'seconds_cumulative':seconds}),flush=True)
 np.savez_compressed(OUT/'fill-feature-bank.npz',**bank)
 fitted={};training={}
 for axis in ['X','Y']:
  x=np.concatenate([z0,bank[axis+'plus'],bank[axis+'minus']]);y=np.tile(labels,3);w,b,info=original.fit_model(x,y);fitted[axis+'_W']=w;fitted[axis+'_b']=b;lg,loss=original.losses(z0,w,b,labels);training[axis]={**info,'source_clean_LL':float(loss.mean()),'source_clean_accuracy':float(np.mean(lg.argmax(1)==labels))}
 np.savez(OUT/'fill-readouts.npz',**fitted);save('fill-training-freeze.json',{'at_UTC':now(),'files':{n:sha(OUT/n) for n in ['fill-lock.json','fill-feature-bank.npz','fill-readouts.npz']},'source_training':training,'new_final25_inference':False})
 features=np.load(OLD/'evaluation-features.npz');oldw=np.load(OLD/'readouts.npz');oldp=json.loads((OLD/'protocol.json').read_text());summary={};perobject={};predictions={}
 for role in p['evaluation_roles']:
  z=features[role];records=oldp['roles'][role];y=np.array([r['category'] for r in records]);objects=sorted({(r['category'],r['instance']) for r in records});summary[role]={}
  for axis in ['X','Y']:
   for mode in ['constant128','matched_gray']:
    w,b=(oldw['translate'+axis+'_09_W'],oldw['translate'+axis+'_09_b']) if mode=='constant128' else (fitted[axis+'_W'],fitted[axis+'_b']);lg,loss=original.losses(z,w,b,y);name=axis+'_'+mode;summary[role][name]={'LL':float(loss.mean()),'accuracy':float(np.mean(lg.argmax(1)==y))};perobject[role+'__'+name]=np.array([loss[[(r['category'],r['instance'])==obj for r in records]].mean() for obj in objects]);predictions[role+'__'+name]=lg
  v=summary[role];gapold=v['Y_constant128']['LL']-v['X_constant128']['LL'];gapnew=v['Y_matched_gray']['LL']-v['X_matched_gray']['LL'];summary[role]['interaction']={'original_Y_minus_X':gapold,'matched_Y_minus_X':gapnew,'gap_reduction':gapold-gapnew,'Y_loss_reduction':v['Y_constant128']['LL']-v['Y_matched_gray']['LL'],'X_loss_reduction':v['X_constant128']['LL']-v['X_matched_gray']['LL']}
 np.savez(OUT/'fill-development-per-object-losses.npz',**perobject);np.savez_compressed(OUT/'fill-development-logits.npz',**predictions)
 v=summary['development_novel']['interaction'];passed=abs(v['matched_Y_minus_X'])<=.5*p['original_development_novel_gap'] and v['Y_loss_reduction']>0
 result={'at_UTC':now(),'status':'one_locked_exploratory_fill_factor_probe_complete','results':summary,'predeclared_explanatory_screen_passed':bool(passed),'no_independent_confirmation':True,'new_final25_inference':False,'no_full_RA_TA_or_end_to_end_claim':True,'no_original_contribution_established':True,'versions':versions,'encoder_seconds':seconds,'wall_seconds':time.perf_counter()-start,'new_images_encoded':2160,'new_models':2,'pixel_control':{'changed_pixels_only_in_known_new_padding':True,'original_retained_content_identical':True,'cropping_identical':True,'replacement_fill':p['replacement_fill'],'total_changed_pixels_over2160':changed_pixels_total},'protocol_SHA256':sha(OUT/'fill-protocol.json'),'code_SHA256':sha(OUT/'fill_probe.py')};save('fill-result.json',result);print(json.dumps(result),flush=True)
