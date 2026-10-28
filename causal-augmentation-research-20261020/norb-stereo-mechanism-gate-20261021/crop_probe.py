"""One exploratory source-only necessary-condition probe; no encoder or training."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,struct,time
import numpy as np
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
OLD=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/norb-view-transfer-pilot-20261021')
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def save(n,r):
 with (OUT/n).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
def read(n):return json.loads((OUT/n).read_text())
import argparse
parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);arg=parser.parse_args()
if arg.phase=='prepare':
 p=json.loads((OLD/'protocol.json').read_text()); r=json.loads((OLD/'final-results.json').read_text()); m=json.loads((OLD/'source-prediction.json').read_text()); tr=json.loads((OLD/'training-results.json').read_text())
 p2={'at_UTC':now(),'question':'Does greater support clipped off the canvas satisfy a necessary condition for the observed Y9 versus X9 training-benefit difference?','scope':'Exploratory explanation of already-observed readout results, not independent confirmation, a stereo-geometry claim or new augmentation principle.','pixel_access_after_this_lock':True,'sample_records':p['roles']['source_seen'],'source_physical_objects':p['source_objects'],'cameras':[0,1],'camera_repeat_not_independent':True,'preprocess':'uint8 gray -> PIL RGB -> bilinear224; matches original input pretransform; no model or labels used for proxy','signed_translation_pixels':9,'translation_law':'plus and minus equalweight; identity would multiply all losses by2/3 and changes no axis comparison','reference_gray':'median of four224-image12x12corner patches; proxy reference, not true background','contrast_thresholds':[16,32,64],'binary_proxy':'abs(gray-reference)>=threshold; explicitly not object/semantic mask','metrics':['fraction of proxy clipped (zero if proxy empty, separately flagged)','clipped proxy pixels/full image pixels','clipped continuous squared contrast/whole continuous squared contrast'],'clipping_convention':'X+ loses rightmost9;X- leftmost9;Y+ bottom9;Y- top9; signedlosses averaged','aggregation':'within physical object, all36 views then both cameras equally;15 objects are units; report each camera too; no sampling CI or significance claims','necessary_condition':'A simple more-content-clipped explanation predicts Y lost support > X consistently. If Y is not larger over the prechosen threshold bank and continuous contrast, do not retain that gross proxy-amount explanation. Small semantic-important pixels, proxy error and CNN boundary/aliasing effects remain unidentified.','no_after_result_threshold_or_mask_selection':True,'no_semantic_effect_or_causal_contribution_claim':True,'display_sample':'one preselected source object/category, smallest row among its frozen source grid; both cameras, raw input only','new_test_pixels':False,'new_features_or_fit':False,'CPU_threads_environment':1,'GPU':False,'new_downloads_or_installs':False,'8A_ARIS_A100_PPT_automation':False,'old_final_results_SHA256':sha(OLD/'final-results.json'),'original_protocol_SHA256':sha(OLD/'protocol.json'),'original_download_receipt_SHA256':sha(OLD/'training-download.json'),'grounding':{'final_seen':{a:r['risk_table']['final_seen'][a] for a in ['clean','translateX_09','translateY_09']},'final_novel':{a:r['risk_table']['final_novel'][a] for a in ['clean','translateX_09','translateY_09']},'development_seen':{a:r['risk_table']['development_seen'][a] for a in ['clean','translateX_09','translateY_09']},'development_novel':{a:r['risk_table']['development_novel'][a] for a in ['clean','translateX_09','translateY_09']}}}
 save('crop-protocol.json',p2);save('crop-lock.json',{'at_UTC':now(),'files':{n:sha(OUT/n) for n in ['crop-protocol.json','crop_probe.py','independent-stereo-prior-review.json']},'new_pixels_in_probe_read':0});print(json.dumps({'status':'protocol_locked_before_probe_pixels','original_question_exploratory':True,'source_rows':len(p2['sample_records']),'physical_objects':15,'cameras':2}))
else:
 for n,h in read('crop-lock.json')['files'].items():assert sha(OUT/n)==h
 start=time.perf_counter();p=read('crop-protocol.json'); oldp=json.loads((OLD/'protocol.json').read_text()); rec=json.loads((OLD/'training-download.json').read_text());raw=OLD/'data'/oldp['downloads']['training']['file'];assert raw.stat().st_size==rec['bytes'] and sha(raw)==rec['sha256']
 rows=p['sample_records'];wanted={r['row']:r for r in rows};imgs={}
 with gzip.open(raw,'rb') as f:
  assert struct.unpack('<6i',f.read(24))==(0x1e3d4c55,4,24300,2,96,96)
  for j in range(24300):
   data=f.read(18432);assert len(data)==18432
   if j in wanted:imgs[j]=np.frombuffer(data,dtype=np.uint8).reshape(2,96,96).copy()
  assert f.read(1)==b''
 assert len(imgs)==540
 measurements=[]; groups=defaultdict(list);camera_groups=defaultdict(list)
 for r in rows:
  for cam in [0,1]:
   a=np.asarray(Image.fromarray(imgs[r['row']][cam]).convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float64)[:,:,0]
   reference=float(np.median(np.concatenate([a[:12,:12].ravel(),a[:12,-12:].ravel(),a[-12:,:12].ravel(),a[-12:,-12:].ravel()])))
   contrast=np.abs(a-reference);energy=contrast**2;E=float(energy.sum()); outputs={}
   for threshold in p['contrast_thresholds']:
    mask=contrast>=threshold;den=int(mask.sum());values={}
    for axis in ['X','Y']:
     sides=[mask[:,:9],mask[:,-9:]] if axis=='X' else [mask[:9,:],mask[-9:,:]]
     lost=sum(float(x.sum()) for x in sides)/2
     es=[energy[:,:9],energy[:,-9:]] if axis=='X' else [energy[:9,:],energy[-9:,:]]
     values[axis]={'proxy_share_lost':lost/max(den,1),'image_area_proxy_lost':lost/(224*224),'continuous_contrast_energy_share_lost':sum(float(x.sum()) for x in es)/2/max(E,1e-30)}
    values['proxy_pixels']=den;values['empty_proxy']=den==0;outputs[str(threshold)]=values
    key=(r['category'],r['instance'],threshold);groups[key].append(values)
    camera_groups[(r['category'],r['instance'],cam,threshold)].append(values)
   measurements.append({'row':r['row'],'category':r['category'],'instance':r['instance'],'camera':cam,'reference_gray':reference,'thresholds':outputs})
 def summarize_group(key,values,camera=False):
  result={'category':key[0],'instance':key[1],'threshold':key[-1],'views':len(values)}
  if camera:result['camera']=key[2]
  for axis in ['X','Y']:
   for m in ['proxy_share_lost','image_area_proxy_lost','continuous_contrast_energy_share_lost']:result[axis+'_'+m]=float(np.mean([v[axis][m] for v in values]))
  result['Y_minus_X_proxy_share_lost']=result['Y_proxy_share_lost']-result['X_proxy_share_lost'];return result
 byobj=[summarize_group(k,v) for k,v in sorted(groups.items())];bycam=[summarize_group(k,v,True) for k,v in sorted(camera_groups.items())]
 summary={}
 for threshold in p['contrast_thresholds']:
  selected=[x for x in byobj if x['threshold']==threshold];assert len(selected)==15
  summary[str(threshold)]={k:float(np.mean([x[k] for x in selected])) for k in selected[0] if k.startswith(('X_','Y_'))}
  summary[str(threshold)]['objects_Y_greater_than_X']=sum(x['Y_minus_X_proxy_share_lost']>0 for x in selected)
  summary[str(threshold)]['objects_X_greater_than_Y']=sum(x['Y_minus_X_proxy_share_lost']<0 for x in selected)
 save('crop-per-photo.json',{'at_UTC':now(),'rows':measurements});save('crop-per-object.json',{'at_UTC':now(),'objects':byobj,'camera_repeated_measurements':bycam})
 display=[]
 for c in range(5):
  ob=next(tuple(x) for x in p['source_physical_objects'] if x[0]==c); rr=min((r for r in rows if (r['category'],r['instance'])==ob),key=lambda x:x['row']);display.append(rr)
 sheet=Image.new('RGB',(5*224,2*248),(245,245,245));draw=ImageDraw.Draw(sheet)
 for k,r in enumerate(display):
  for cam in [0,1]:
   im=Image.fromarray(imgs[r['row']][cam]).convert('RGB').resize((224,224),Image.Resampling.BILINEAR);sheet.paste(im,(k*224,cam*248+24));draw.text((k*224+3,cam*248+4),f"class{r['category']} object{r['instance']} row{r['row']} cam{cam}",fill=(0,0,0))
 sheet.save(OUT/'source-pair-contact-sheet.png')
 result={'at_UTC':now(),'status':'single_source_only_necessary_condition_probe_complete','original_rows':540,'camera_images':1080,'physical_objects':15,'independent_confirmation':False,'no_model_feature_or_training':True,'raw_gzip_CRC_and_SHA256_pass':True,'threshold_summary':summary,'empty_proxy_counts':{str(q):sum(r['thresholds'][str(q)]['empty_proxy'] for r in measurements) for q in p['contrast_thresholds']},'all_thresholds_Y_more_lost':all(summary[str(q)]['Y_minus_X_proxy_share_lost']>0 for q in p['contrast_thresholds']),'continuous_Y_minus_X_energy':summary['16']['Y_continuous_contrast_energy_share_lost']-summary['16']['X_continuous_contrast_energy_share_lost'],'wall_seconds':time.perf_counter()-start,'code_SHA256':sha(OUT/'crop_probe.py'),'protocol_SHA256':sha(OUT/'crop-protocol.json'),'old_training_archive_SHA256':rec['sha256'],'display_rows':display,'proxy_not_semantic_truth':True,'does_not_test_augmentation_training_causally':True}
 save('crop-result.json',result);print(json.dumps(result))
